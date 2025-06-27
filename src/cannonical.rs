use alloy::{
    eips::{BlockId, BlockNumberOrTag},
    hex::FromHex,
    primitives::map::FbBuildHasher,
    rpc::types::{
        trace::{
            common::TraceResult,
            geth::{
                AccountState, DiffMode, GethDebugBuiltInTracerType, GethDebugTracerConfig,
                GethDebugTracerType, GethDebugTracingOptions, GethDefaultTracingOptions, GethTrace,
                PreStateConfig, PreStateFrame,
            },
            parity::{Delta, TraceResultsWithTransactionHash},
        },
        AccessList, Block,
    },
};
use alloy_provider::{
    ext::{DebugApi, TraceApi},
    DynProvider, Provider,
};
use async_cell::sync::AsyncCell;
use eyre::{Context, ContextCompat};
use futures::{future::try_join_all, FutureExt, TryFutureExt};
use revm::{
    context::{BlockEnv, DBErrorMarker},
    context_interface::block::BlobExcessGasAndPrice,
    primitives::{Address, Bytes, B256, KECCAK_EMPTY, U256},
    state::{AccountInfo, Bytecode},
};
use revm::{Database, DatabaseRef};
use scc::hash_map::Entry;
use sdd::{AtomicShared, Guard, Shared, Tag};
use std::{
    collections::{BTreeMap, HashSet},
    fmt::Display,
    future::IntoFuture,
    sync::Arc,
};
use tokio::runtime::Handle;

use crate::{config::LinkType, LOGGER_TARGET_LOADS, LOGGER_TARGET_SYNC};

type AddressHasher = FbBuildHasher<20>;
type IndexHasher = FbBuildHasher<32>;

#[derive(Debug, Clone)]
pub struct Forked {
    pub cannonical: Arc<CannonicalFork>,
    pub env: BlockEnv,
    pub seconds_per_block: U256,
}

static SLOTS_TO_FETCH_PREFETCH: usize = 5;

impl Forked {
    pub fn mine(&mut self, to_mine: u64) {
        let blocks = U256::from(to_mine);
        self.env.number = self.env.number + blocks;
        self.env.timestamp += self.seconds_per_block * blocks;
        log::debug!(target: LOGGER_TARGET_SYNC, "Mined {} blocks, new block number {}", to_mine, self.env.number);
    }
    pub fn get_timestamp(&self) -> u64 {
        self.env.timestamp.to()
    }
    pub fn set_timestamp(&mut self, timestamp: u64) {
        log::debug!(target: LOGGER_TARGET_SYNC, "Setting timestamp to {}", timestamp);
        self.env.timestamp = U256::from(timestamp);
    }
    pub fn get_block_number(&self) -> u64 {
        self.env.number.to()
    }

    #[inline]
    fn block_on<F>(f: F) -> F::Output
    where
        F: core::future::Future + Send,
        F::Output: Send,
    {
        match Handle::try_current() {
            Ok(handle) => match handle.runtime_flavor() {
                // This essentially equals to tokio::task::spawn_blocking because tokio doesn't
                // allow current_thread runtime to block_in_place
                tokio::runtime::RuntimeFlavor::CurrentThread => std::thread::scope(move |s| {
                    s.spawn(move || {
                        tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap()
                            .block_on(f)
                    })
                    .join()
                    .unwrap()
                }),
                _ => tokio::task::block_in_place(move || handle.block_on(f)),
            },
            Err(_) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(f),
        }
    }
}

#[derive(Debug)]
pub struct DbError(eyre::Error);

impl Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for DbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }

    #[allow(deprecated)]
    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.0.source()
    }
}

impl DBErrorMarker for DbError {}

impl From<eyre::Error> for DbError {
    fn from(err: eyre::Error) -> Self {
        DbError(err)
    }
}

impl DatabaseRef for Forked {
    type Error = DbError;

    #[inline]
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "code_by_hash({})", code_hash);
        Forked::block_on(async {
            self.cannonical
                .clone()
                .code_by_hash(&code_hash)
                .await
                .map_err(DbError)
        })
    }

    #[inline]
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "call storage({}, {})", address, index);
        let out = Forked::block_on(async { self.cannonical.clone().storage(address, index).await });
        // log::debug!(target: LOGGER_TARGET_SYNC, "return storage({}, {}) => {:?}", address, index, out);

        out.map_err(DbError)
    }

    #[inline]
    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        Forked::block_on(async {
            self.cannonical
                .clone()
                .block_hash(number)
                .await
                .map_err(DbError)
        })
    }

    #[inline]
    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "basic({})", address);
        Forked::block_on(async {
            self.cannonical
                .clone()
                .basic(address)
                .await
                .map_err(DbError)
        })
    }
}

impl Database for Forked {
    type Error = DbError;

    #[inline]
    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        <Self as DatabaseRef>::basic_ref(self, address)
    }

    #[inline]
    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Self::Error> {
        <Self as DatabaseRef>::code_by_hash_ref(self, code_hash)
    }

    #[inline]
    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Self::Error> {
        <Self as DatabaseRef>::storage_ref(self, address, index)
    }

    #[inline]
    fn block_hash(&mut self, number: u64) -> Result<B256, Self::Error> {
        <Self as DatabaseRef>::block_hash_ref(self, number)
    }
}

#[derive(Debug, Clone)]
enum AccountData {
    Pending(Arc<AsyncCell<AccountInfo>>),
    Live(AccountInfo),
}

#[derive(Debug, Clone)]
enum StorageData {
    Pending(Arc<AsyncCell<U256>>),
    Live(U256),
}

#[derive(Debug, Clone)]
pub struct CannonicalFork {
    // Results fetched from the provider and maintained by apply_next_mainnet_block,
    // Contains the full state of the account & storage
    link_type: LinkType,

    block_hashes: scc::HashMap<u64, B256>,
    storage: scc::HashIndex<Address, scc::HashIndex<U256, StorageData, IndexHasher>, AddressHasher>,
    accounts: scc::HashIndex<Address, AccountData, AddressHasher>,
    contracts: scc::HashIndex<B256, Bytecode, IndexHasher>,
    // mappings: scc::HashMap<Address, MappingData, AddressHasher>,
    current_block: AtomicShared<Block>,
    provider: DynProvider,
}

const DEFAULT_CAPACITY_ACCOUNTS: usize = 16384;
const DEFAULT_STORAGE_PR_ACCOUNT: usize = 32;
const DEFAULT_CAPACITY_CONTRACTS: usize = 1024;

impl CannonicalFork {
    pub fn new(provider: DynProvider, fork_block: Block, config: crate::config::Config) -> Self {
        let contracts = scc::HashIndex::with_capacity_and_hasher(
            DEFAULT_CAPACITY_CONTRACTS,
            IndexHasher::default(),
        );
        if let Err(err) = contracts.insert(KECCAK_EMPTY, Bytecode::default()) {
            log::error!(target: LOGGER_TARGET_SYNC, "Failed to insert contract code hash {:?}", err);
        }

        Self {
            link_type: config.link_type,
            accounts: scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS,
                AddressHasher::default(),
            ),
            contracts: contracts,
            storage: scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS,
                AddressHasher::default(),
            ),
            block_hashes: scc::HashMap::with_capacity(1024 * 8),
            current_block: AtomicShared::new(fork_block),
            provider,
        }
    }

    pub async fn reset_state(&self, block_number: u64) -> eyre::Result<()> {
        log::info!(target: LOGGER_TARGET_SYNC, "reset_state({})", block_number);
        self.accounts.clear();
        self.storage.clear();
        self.block_hashes.clear();
        let block = self
            .provider
            .get_block_by_number(BlockNumberOrTag::Number(block_number))
            .await?
            .wrap_err(format!("Failed to fetch block {block_number} from RPC"))?;

        self.current_block.swap(
            (Some(Shared::new(block)), Tag::None),
            std::sync::atomic::Ordering::SeqCst,
        );

        Ok(())
    }

    pub fn block_env(&self) -> BlockEnv {
        let g = Guard::new();
        let block = self
            .current_block
            .load(std::sync::atomic::Ordering::Relaxed, &g)
            .as_ref()
            .unwrap();
        BlockEnv {
            number: U256::from(block.header.number),
            timestamp: U256::from(block.header.timestamp),
            gas_limit: block.header.gas_limit,
            blob_excess_gas_and_price: Some(BlobExcessGasAndPrice {
                excess_blob_gas: block.header.excess_blob_gas.unwrap_or_default(),
                blob_gasprice: block.header.blob_fee().unwrap_or_default(),
            }),
            prevrandao: Some(block.header.mix_hash),
            difficulty: block.header.difficulty,
            basefee: block.header.base_fee_per_gas.unwrap_or(1u64),

            ..Default::default()
        }
    }

    async fn apply_next_geth_block(
        &self,
        diffs: Vec<BTreeMap<Address, AccountState>>,
    ) -> eyre::Result<()> {
        for account_diffs in diffs {
            for (k, v) in account_diffs {
                let addr = Address::from(k.0);
                if !self.accounts.contains(&addr) {
                    continue;
                }
                let code_update = v
                    .code
                    .map(|v| Bytes::from_hex(&v))
                    .map(|v| v.map(|v| Bytecode::new_raw(v)));

                match self.accounts.entry_async(addr).await {
                    scc::hash_index::Entry::Occupied(entry) => {
                        let mut prev = match entry.get() {
                            AccountData::Pending(cell) => cell.get_shared().await,
                            AccountData::Live(info) => info.clone(),
                        };
                        prev.balance = match v.balance {
                            Some(value) => value,
                            _ => prev.balance,
                        };
                        if let Some(Result::Ok(code)) = code_update {
                            let code_hash = code.hash_slow();
                            self.insert_contract(&code_hash, code).await?;
                            prev.code_hash = code_hash;
                        }
                        entry.update(AccountData::Live(prev));
                    }
                    _ => {
                        continue;
                    }
                };

                if v.storage.is_empty() || !self.storage.contains(&addr) {
                    continue;
                }

                let t = self.get_account_table(&addr).await;
                for (hpos, value) in v.storage {
                    let pos: U256 = hpos.into();
                    if !t.contains(&pos) {
                        continue;
                    }
                    let value_to_insert: U256 = value.into();

                    if let scc::hash_index::Entry::Occupied(entry) = t.entry_async(pos).await {
                        log::trace!(target: LOGGER_TARGET_SYNC, "{}.{} = {}", addr, pos, value_to_insert);
                        entry.update(StorageData::Live(value_to_insert.into()));
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn get_current_block(&self) -> u64 {
        let g = Guard::new();
        self.current_block
            .load(std::sync::atomic::Ordering::Relaxed, &g)
            .as_ref()
            .unwrap()
            .header
            .number
    }

    async fn load_reth_trace_and_apply(&self) -> eyre::Result<()> {
        let block_number = self.get_current_block();
        let trace = self
            .provider
            .trace_replay_block_transactions(block_number.into())
            .await
            .wrap_err(format!("Failed to fetch trace for {}", block_number))?;

        self.apply_next_reth_block(trace).await?;
        log::trace!(target: LOGGER_TARGET_SYNC, "applied reth block {}", block_number);
        Ok(())
    }

    async fn load_geth_trace_and_apply(&self) -> eyre::Result<()> {
        let block_number = self.get_current_block();
        // log::debug!(target: LOGGER_TARGET_SYNC, "Loading geth trace for diffs");
        let prestate_config = PreStateConfig {
            diff_mode: Some(true),
            disable_code: Some(true),
            disable_storage: Some(true),
        };

        let storage_changes = self
            .provider
            .debug_trace_block_by_number(
                alloy::eips::BlockNumberOrTag::Number(block_number),
                GethDebugTracingOptions {
                    config: GethDefaultTracingOptions {
                        disable_storage: Some(true),
                        disable_stack: Some(true),
                        disable_memory: Some(true),
                        disable_return_data: Some(true),
                        ..Default::default()
                    },
                    tracer: Some(GethDebugTracerType::BuiltInTracer(
                        GethDebugBuiltInTracerType::PreStateTracer,
                    )),
                    tracer_config: GethDebugTracerConfig::from(prestate_config),
                    timeout: None,
                },
            )
            .await
            .wrap_err("Failed to fetch trace for {block_number}")?;

        let diffs = storage_changes
            .into_iter()
            .filter_map(|x| match x {
                TraceResult::Success {
                    result:
                        GethTrace::PreStateTracer(PreStateFrame::Diff(DiffMode { pre: _, post })),
                    tx_hash: _,
                } => Some(post),
                _ => None,
            })
            .collect();

        self.apply_next_geth_block(diffs).await
    }

    pub async fn apply_next_block(&self, block: Block) -> eyre::Result<()> {
        self.current_block.swap(
            (Some(Shared::new(block)), Tag::None),
            std::sync::atomic::Ordering::SeqCst,
        );
        if self.link_type == LinkType::Reth {
            self.load_reth_trace_and_apply().await?
        } else {
            self.load_geth_trace_and_apply().await?
        };
        Ok(())
    }
    async fn apply_next_reth_block(
        &self,
        diff: Vec<TraceResultsWithTransactionHash>,
    ) -> eyre::Result<()> {
        for trace in diff {
            let account_diffs = match trace.full_trace.state_diff {
                None => continue,
                Some(d) => d.0,
            };

            for (k, v) in account_diffs {
                let addr = Address::from(k.0);
                if !self.accounts.contains(&addr) {
                    continue;
                }
                match self.accounts.entry_async(addr).await {
                    scc::hash_index::Entry::Occupied(entry) => {
                        let code_update = match v.code {
                            Delta::Added(value) => Some(Bytecode::new_raw(value)),
                            Delta::Changed(value) => Some(Bytecode::new_raw(value.to)),
                            _ => None,
                        };
                        let code_update = code_update.map(|code| (code.hash_slow(), code));

                        let mut info = match entry.get() {
                            AccountData::Pending(cell) => cell.get_shared().await,
                            AccountData::Live(info) => info.clone(),
                        };
                        info.balance = match v.balance {
                            Delta::Added(value) => value,
                            Delta::Changed(value) => value.to,
                            _ => info.balance,
                        };
                        if let Some((hash, code)) = code_update.clone() {
                            self.insert_contract(&hash, code).await?;
                            info.code_hash = hash;
                        }
                        entry.update(AccountData::Live(info));
                    }
                    _ => {
                        continue;
                    }
                };

                if v.storage.is_empty() || !self.storage.contains(&addr) {
                    continue;
                }

                let t = self.get_account_table(&addr).await;
                for (index, value) in v.storage {
                    let index: U256 = index.into();
                    if !t.contains(&index) {
                        continue;
                    }
                    let value_to_insert = match value {
                        Delta::Added(value) => value,
                        Delta::Changed(t) => t.to,
                        _ => continue,
                    };
                    if let scc::hash_index::Entry::Occupied(entry) = t.entry_async(index).await {
                        log::trace!(target: LOGGER_TARGET_SYNC, "{}.{} = {}", addr, index, value_to_insert);
                        entry.update(StorageData::Live(value_to_insert.into()));
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    async fn load_storage_slots(
        &self,
        address: Address,
        index: U256,
        block_num: u64,
        slots: usize,
    ) -> eyre::Result<Vec<(U256, U256)>> {
        log::trace!(target: LOGGER_TARGET_SYNC, "(Preload) Fetching storage {} {}..{}", &address, &index, index + U256::from(slots));
        let indices = (0..slots)
            .map(|offset| index + U256::from(offset))
            .collect::<Vec<U256>>();
        let provider = self.provider.clone();
        let futs = indices.iter().map(|offset| {
            let offset = offset.clone();
            let provider = provider.clone();
            tokio::spawn(async move {
                let offset = offset.clone();
                let provider = provider.clone();

                log::trace!(target: LOGGER_TARGET_LOADS, "load_storage_slots > get_storage_at({}, {})", address, offset);
                return (
                    offset,
                    provider
                        .get_storage_at(address, offset)
                        .block_id(BlockId::from(block_num))
                        .await,
                );
            })
            .map(|v| match v {
                Ok((index, Ok(value))) => Ok((index, value)),
                e => Err(eyre::eyre!("Failed to fetch storage {:?}", e)),
            })
        });

        let out = try_join_all(futs).await?;
        Ok(out)
    }

    pub async fn load_access_list(&self, list: &AccessList) -> eyre::Result<()> {
        let block_number: u64 = self.get_current_block();
        try_join_all(list.iter().map(|item| {
            let address = item.address;
            let f = async move {
                self.basic(address).await?;

                try_join_all(item.storage_keys.iter().map(|v| {
                    let index: U256 = v.clone().into();
                    self.fetch_storage(address, index, block_number)
                }))
                .await
            };
            futures::TryFutureExt::into_future(f)
        }))
        .await?;
        Ok(())
    }

    async fn fetch_minimal_account_info(
        &self,
        address: Address,
        block_num: u64,
    ) -> eyre::Result<AccountInfo> {
        let min = Address::left_padding_from(&[255, 255]);
        let provider = self.provider.clone();
        let block_id = BlockId::from(block_num);
        let (balance, code) = if address.lt(&min) {
            (U256::from(0), Bytes::new())
        } else {
            log::trace!(target: LOGGER_TARGET_LOADS, "fetch_minimal_account_info({})", address);
            let out = futures::try_join!(
                provider
                    .get_balance(address)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_code_at(address)
                    .block_id(block_id)
                    .into_future(),
            )
            .wrap_err("Failed to fetch account info")?;

            log::trace!(target: LOGGER_TARGET_LOADS, "fetch_minimal_account_info({}) DONE", address);
            out
        };

        let code_hash = if code.is_empty() {
            KECCAK_EMPTY
        } else {
            let code = Bytecode::new_raw(code);
            let code_hash = code.hash_slow();
            self.insert_contract(&code_hash, code).await?;
            code_hash
        };

        Ok(AccountInfo {
            balance,
            code: None,
            code_hash,
            ..Default::default()
        })
    }
    async fn load_acc_info_live(
        &self,
        address: Address,
        block_num: u64,
    ) -> eyre::Result<AccountInfo> {
        Ok(self.fetch_minimal_account_info(address, block_num).await?)
    }

    async fn load_acc_info(
        &self,
        address: Address,
        block_num: u64,
        slots: usize,
    ) -> eyre::Result<(
        Address,
        AccountInfo,
        Vec<(U256, U256)>,
        Option<Vec<Address>>,
    )> {
        let mut contracts_referenced = Vec::new();
        let provider = self.provider.clone();

        let block_id = BlockId::from(block_num);

        log::trace!(target: LOGGER_TARGET_LOADS, "load_acc_info({})", address);
        let data_handle = futures::try_join!(
            provider
                .get_balance(address)
                .block_id(block_id)
                .into_future()
                .map_err(|e| eyre::eyre!("Failed to fetch balance for {}: {:?}", address, e)),
            provider
                .get_code_at(address)
                .block_id(block_id)
                .into_future()
                .map_err(|e| eyre::eyre!("Failed to fetch code for {}: {:?}", address, e)),
            self.load_storage_slots(address, U256::from(0), block_num, slots)
        )?;

        let (balance, code, preloaded_slots) = data_handle;

        let (code, code_hash) = if code.len() == 0 {
            (Bytecode::default(), KECCAK_EMPTY)
        } else {
            let code = Bytecode::new_raw(code);
            let code_hash = code.hash_slow();
            (code, code_hash)
        };

        // Run analysis

        if code_hash != KECCAK_EMPTY && !self.contracts.contains(&code_hash) {
            self.insert_contract(&code_hash, code.clone()).await?;
        }

        let contracts_referenced = if contracts_referenced.is_empty() {
            None
        } else {
            contracts_referenced.dedup();
            Some(contracts_referenced.into_iter().collect::<Vec<_>>())
        };

        Ok((
            address,
            AccountInfo {
                balance: balance,
                code: None,
                code_hash,
                ..Default::default()
            },
            preloaded_slots,
            contracts_referenced,
        ))
    }

    pub(crate) async fn load_positions(
        &self,
        positions: Vec<Address>,
    ) -> eyre::Result<Vec<Address>> {
        let block_number = self.get_current_block();
        let positions = positions
            .iter()
            .filter(|addr| !self.accounts.contains(*addr))
            .collect::<Vec<_>>();

        if positions.len() == 0 {
            return Ok(Vec::new());
        }
        log::info!(target: LOGGER_TARGET_SYNC, "load_positions: Will preload {} addresses", positions.len());

        for address in positions.iter() {
            self.get_account_table(*address).await;
        }

        let infos =
            try_join_all(positions.into_iter().map(|address| {
                self.load_acc_info(*address, block_number, SLOTS_TO_FETCH_PREFETCH)
            }))
            .await
            .wrap_err("Failed to fetch basic info for all addresses")?;

        for (addr, info, slots, _) in infos.iter() {
            if let Err(_) = self
                .accounts
                .insert_async(*addr, AccountData::Live(info.clone()))
                .await
            {
                continue;
            }
            for (index, value) in slots.iter() {
                if self.account_storage_value_initialized(addr, index).await {
                    continue;
                }
                self.init_account_storage_value(addr, index, value).await;
            }
        }
        let mut new_addreses = HashSet::<Address>::new();
        for tup in infos.iter() {
            if let Some(references) = &tup.3 {
                for addr in references.iter() {
                    if self.accounts.contains(addr) {
                        continue;
                    }
                    new_addreses.insert(addr.clone());
                }
            }
        }
        Ok(new_addreses.into_iter().collect::<Vec<_>>())
    }

    #[inline]
    async fn init_pending_account(
        &self,
        address: &Address,
    ) -> eyre::Result<Arc<AsyncCell<AccountInfo>>> {
        if let Some(_) = self.accounts.peek_with(address, |_, _| true) {
            return Err(eyre::eyre!("Account already initialized"));
        }

        log::trace!(target: LOGGER_TARGET_SYNC, "init_pending_account({})", address);
        let out = match self.accounts.entry_async(*address).await {
            scc::hash_index::Entry::Vacant(out) => {
                let cell: Arc<AsyncCell<AccountInfo>> = AsyncCell::shared();
                out.insert_entry(AccountData::Pending(cell.clone()));
                eyre::Ok(cell)
            }
            _ => {
                return Err(eyre::eyre!("Account already initialized"));
            }
        };
        out
    }

    #[inline]
    async fn insert_contract(&self, code_hash: &B256, code: Bytecode) -> eyre::Result<()> {
        if self.contracts.contains(code_hash) {
            return Ok(());
        }
        log::trace!(target: LOGGER_TARGET_SYNC, "insert_contract({})", code_hash);
        match self.contracts.entry_async(*code_hash).await {
            scc::hash_index::Entry::Vacant(out) => {
                out.insert_entry(code);
            }
            _ => {}
        }
        Ok(())
    }

    #[inline]
    async fn code_by_hash(&self, code_hash: &B256) -> eyre::Result<Bytecode> {
        if code_hash == &KECCAK_EMPTY {
            return Ok(Bytecode::default());
        }
        if let Some(code) = self.contracts.peek_with(code_hash, |_, v| v.clone()) {
            return Ok(code);
        }
        Ok(Bytecode::default())
    }

    #[inline]
    async fn basic(
        &self,
        address: Address, // dry_run: bool,
    ) -> eyre::Result<Option<AccountInfo>> {
        if let Some(value) = self.peek_account_info(&address).await {
            return Ok(Some(value));
        }
        let cell = match self.init_pending_account(&address).await {
            Err(_) => return Ok(self.peek_account_info(&address).await),
            Ok(cell) => cell,
        };

        log::trace!(target: LOGGER_TARGET_SYNC, "(Cache miss) Fetching account {}", address);
        let block_number: u64 = self.get_current_block();
        let info = self.load_acc_info_live(address, block_number).await?;
        cell.set(AccountInfo {
            balance: info.balance,
            nonce: info.nonce,
            code_hash: info.code_hash,
            code: None,
        });
        if let Some(code) = info.code {
            self.insert_contract(&info.code_hash, code).await?;
        }
        Ok(Some(AccountInfo {
            balance: info.balance,
            nonce: info.nonce,
            code_hash: info.code_hash,
            code: None,
        }))
    }
    pub async fn get_account_data(&self, address: &Address) -> eyre::Result<Option<AccountInfo>> {
        self.basic(address.clone()).await
    }
    #[inline]
    async fn peek_account_info(&self, address: &Address) -> Option<AccountInfo> {
        if let Some(value) = self.peek_account(&address).await {
            let value = match value {
                AccountData::Live(data) => data,
                AccountData::Pending(cell) => cell.get_shared().await,
            };
            return Some(value);
        }
        None
    }
    #[inline]
    async fn peek_account(&self, address: &Address) -> Option<AccountData> {
        if let Some(value) = self.accounts.peek_with(address, |_, v| v.clone()) {
            return Some(value);
        }
        None
    }

    #[inline]
    async fn account_storage_value_initialized(&self, address: &Address, index: &U256) -> bool {
        if let Some(Some(_)) = self
            .storage
            .peek_with(address, |_, v| v.peek_with(index, |_, __| true))
        {
            return true;
        }
        false
    }

    async fn init_account_storage_value(&self, address: &Address, index: &U256, value: &U256) {
        log::trace!(target: LOGGER_TARGET_SYNC, "init_account_storage_value({}, {})", address, index);

        let storage = self.get_account_table(address).await;
        if let Err(_) = storage.insert(*index, StorageData::Live(value.clone())) {}
    }

    async fn get_account_table(
        &self,
        address: &Address,
    ) -> scc::hash_index::OccupiedEntry<
        '_,
        Address,
        scc::hash_index::HashIndex<U256, StorageData, IndexHasher>,
        FbBuildHasher<20>,
    > {
        let table = if self.storage.contains(address) {
            self.storage.get_async(address).await.unwrap()
        } else {
            log::trace!(target: LOGGER_TARGET_SYNC, "init_account_storage({})", address);
            let v = &scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_STORAGE_PR_ACCOUNT,
                IndexHasher::default(),
            );
            match self.storage.entry_async(*address).await {
                scc::hash_index::Entry::Vacant(out) => {
                    out.insert_entry(v.clone());
                }
                _ => {}
            }

            return self.storage.get_async(address).await.unwrap();
        };
        return table;
    }

    async fn fetch_storage(
        &self,
        address: Address,
        index: U256,
        block_number: u64,
    ) -> eyre::Result<U256> {
        log::trace!(target: LOGGER_TARGET_LOADS, "fetch_storage({}, {})", address, index);
        if let Ok(val) = self.peek_storage(&address, &index).await {
            log::debug!(target: LOGGER_TARGET_LOADS, "cached fetch_storage({}, {})", address, index);
            return Ok(val);
        }

        let cell: Arc<AsyncCell<alloy::primitives::Uint<256, 4>>> = AsyncCell::shared();
        {
            let storage = self.get_account_table(&address).await;

            match storage.entry_async(index).await {
                scc::hash_index::Entry::Vacant(out) => {
                    out.insert_entry(StorageData::Pending(cell.clone()));
                }
                _ => return self.peek_storage(&address, &index).await,
            };
        }

        log::debug!(target: LOGGER_TARGET_LOADS, "fetching storage ({}, {})", address, index);
        let data = self
            .provider
            .get_storage_at(address, index)
            .block_id(BlockId::from(block_number))
            .await
            .wrap_err(format!(
                "Failed to fetch storage for address {address} and index {index}"
            ))?;

        cell.set(data.clone());

        return Ok(data);
    }

    #[inline]
    async fn peek_storage(&self, address: &Address, index: &U256) -> eyre::Result<U256> {
        if let Some(Some(value)) = self
            .storage
            .peek_with(address, |_, v| v.peek_with(index, |_, v| v.clone()))
        {
            match value {
                StorageData::Live(data) => return Ok(data.into()),
                StorageData::Pending(cell) => return Ok(cell.get_shared().await),
            }
        }
        Err(eyre::eyre!("Storage not found"))
    }
    #[inline]
    async fn storage(&self, address: Address, index: U256) -> eyre::Result<U256> {
        if let Ok(current_value) = self.peek_storage(&address, &index).await {
            return Ok(current_value);
        }
        log::trace!(target: LOGGER_TARGET_SYNC, "(miss) Fetching storage {} {}", address, index);
        let block_number: u64 = self.get_current_block();
        self.fetch_storage(address, index, block_number).await
    }
    async fn block_hash(&self, num: u64) -> eyre::Result<B256> {
        match self.block_hashes.entry(num) {
            Entry::Occupied(out) => Ok(out.get().clone()),
            Entry::Vacant(e) => {
                log::debug!(target: LOGGER_TARGET_SYNC, "Fetching block hash {}", num);
                let out = self
                    .provider
                    .get_block(BlockId::Number(BlockNumberOrTag::Number(num)))
                    .await
                    .wrap_err_with(|| "Failed to fetch block hash")?
                    .wrap_err_with(|| format!("Failed to fetch block hash for block {}", num))?;
                let out = out.header.hash;
                e.insert_entry(out);
                Ok(out)
            }
        }
    }
}
