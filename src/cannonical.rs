use crate::{config::LinkType, LOGGER_TARGET_SYNC};
use alloy::{
    eips::BlockId,
    hex::FromHex,
    primitives::map::FbBuildHasher,
    rpc::types::{
        trace::{
            common::TraceResult,
            geth::{
                AccountState, DiffMode, GethDebugBuiltInTracerType, GethDebugTracerConfig,
                GethDebugTracerType, GethDebugTracingOptions, GethDefaultTracingOptions, GethTrace,
                PreStateFrame,
            },
            parity::{Delta, TraceResultsWithTransactionHash, TraceType},
        },
        Block,
    },
    transports::BoxTransport,
};
use alloy_provider::{
    ext::{DebugApi, TraceApi},
    Provider,
};
use async_cell::sync::AsyncCell;
use eyre::{Context, ContextCompat};
use futures::{future::try_join_all, try_join};
use revm::primitives::{Address, Bytes, B256, KECCAK_EMPTY, U256};
use revm::{db::DatabaseRef, primitives::AccountInfo, Database};
// use rustc_hash::FxBuildHasher;
use scc::hash_map::Entry;
use std::{collections::BTreeMap, future::IntoFuture, sync::Arc};
use tokio::{runtime::Handle, sync::RwLock};

// type AddressHasher = hash_hasher::HashBuildHasher;
// type IndexHasher = FxBuildHasher;

type AddressHasher = FbBuildHasher<20>;
type IndexHasher = FbBuildHasher<32>;

#[derive(Debug, Clone)]
pub struct Forked {
    pub cannonical: Arc<CannonicalFork>,
    pub env: revm::primitives::BlockEnv,
    pub seconds_per_block: U256,
}
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

impl DatabaseRef for Forked {
    type Error = eyre::Error;

    #[inline]
    fn basic_ref(&self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        let block_number: u64 = self.env.number.to();
        Forked::block_on(async { self.cannonical.clone().basic(address, block_number).await })
    }

    #[inline]
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<revm::primitives::Bytecode, Self::Error> {
        Forked::block_on(async { self.cannonical.clone().code_by_hash(code_hash).await })
    }

    #[inline]
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        let block_number: u64 = self.env.number.to();
        Forked::block_on(async {
            self.cannonical
                .clone()
                .storage(address, index, block_number)
                .await
        })
    }

    #[inline]
    fn block_hash_ref(&self, number: u64) -> Result<B256, Self::Error> {
        Forked::block_on(async { self.cannonical.clone().block_hash(number).await })
    }
}

impl Database for Forked {
    type Error = eyre::Error;

    #[inline]
    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Self::Error> {
        <Self as DatabaseRef>::basic_ref(self, address)
    }

    #[inline]
    fn code_by_hash(&mut self, code_hash: B256) -> Result<revm::primitives::Bytecode, Self::Error> {
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

    contracts: scc::HashMap<B256, revm::primitives::Bytecode, IndexHasher>,
    block_hashes: scc::HashMap<u64, B256>,
    storage: scc::HashIndex<Address, scc::HashIndex<U256, StorageData, IndexHasher>, AddressHasher>,
    accounts: scc::HashIndex<Address, AccountData, AddressHasher>,
    current_block: Arc<RwLock<Block>>,
    provider: alloy::providers::RootProvider<BoxTransport>,
    provider_trace: alloy::providers::RootProvider<BoxTransport>,
}

const DEFAULT_CAPACITY_ACCOUNTS: usize = 1024;
const DEFAULT_SLACK: usize = 16;
const DEFAULT_STORAGE_PR_ACCOUNT: usize = 512;

impl CannonicalFork {
    pub fn new(
        provider: alloy::providers::RootProvider<BoxTransport>,
        provider_trace: alloy::providers::RootProvider<BoxTransport>,
        fork_block: Block,
        config: crate::config::Config,
    ) -> Self {
        Self {
            link_type: config.link_type,
            accounts: scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS * DEFAULT_SLACK,
                AddressHasher::default(),
            ),
            storage: scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS * DEFAULT_SLACK,
                AddressHasher::default(),
            ),
            contracts: scc::HashMap::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS * DEFAULT_SLACK,
                IndexHasher::default(),
            ),
            block_hashes: scc::HashMap::with_capacity(1024 * 8),
            current_block: Arc::new(RwLock::new(fork_block)),
            provider,
            provider_trace,
        }
    }

    pub async fn reset_state(&self, block_number: u64) -> eyre::Result<()> {
        self.accounts.clear();
        self.storage.clear();
        self.contracts.clear();
        self.block_hashes.clear();
        let block = self
            .provider
            .get_block_by_number(alloy::eips::BlockNumberOrTag::Number(block_number), false)
            .await?
            .wrap_err(format!("Failed to fetch block {block_number} from RPC"))?;

        {
            *self.current_block.write().await = block;
        }

        Ok(())
    }

    pub async fn block_env(&self) -> revm::primitives::BlockEnv {
        let block = self.current_block.read().await;
        revm::primitives::BlockEnv {
            number: U256::from(block.header.number),
            timestamp: U256::from(block.header.timestamp),
            gas_limit: U256::from(block.header.gas_limit),
            coinbase: block.header.miner,
            difficulty: block.header.difficulty,
            basefee: U256::from(1),
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
                    .map(|v| revm::primitives::Bytes::from_hex(&v))
                    .map(|v| v.map(|v| revm::primitives::Bytecode::new_raw(v)));

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
                        if let Some(Ok(code)) = code_update {
                            prev.code_hash = code.hash_slow();
                            prev.code = Some(code.clone());

                            match self.contracts.entry_async(prev.code_hash).await {
                                scc::hash_map::Entry::Vacant(entry) => {
                                    entry.insert_entry(code);
                                }
                                scc::hash_map::Entry::Occupied(mut entry) => {
                                    entry.insert(code);
                                }
                            }
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

                if let Some(table) = self.storage.get_async(&addr).await {
                    let t = table.get();
                    for (hpos, value) in v.storage {
                        let pos: U256 = hpos.into();
                        if !t.contains(&pos) {
                            continue;
                        }
                        let value_to_insert: U256 = value.into();

                        if let scc::hash_index::Entry::Vacant(entry) = t.entry_async(pos).await {
                            entry.insert_entry(StorageData::Live(value_to_insert));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    pub async fn get_current_block(&self) -> eyre::Result<u64> {
        Ok(self.current_block.read().await.header.number)
    }

    async fn load_reth_trace_and_apply(&self, block_number: u64) -> eyre::Result<()> {
        let trace = self
            .provider_trace
            .trace_replay_block_transactions(block_number.into(), &[TraceType::StateDiff])
            .await
            .wrap_err(format!("Failed to fetch trace for {}", block_number))?;

        Ok(self.apply_next_reth_block(trace).await)
    }

    async fn load_geth_trace_and_apply(&self, block_number: u64) -> eyre::Result<()> {
        // log::debug!(target: LOGGER_TARGET_SYNC, "Loading geth trace for diffs");
        let prestate_config = alloy::rpc::types::trace::geth::PreStateConfig {
            diff_mode: Some(true),
        };

        let storage_changes = self
            .provider_trace
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
        let block_number = block.header.number;
        {
            *self.current_block.write().await = block.clone();
        }

        if self.link_type == LinkType::Reth {
            self.load_reth_trace_and_apply(block_number).await?
        } else {
            self.load_geth_trace_and_apply(block_number).await?
        };
        Ok(())
    }
    async fn apply_next_reth_block(&self, diff: Vec<TraceResultsWithTransactionHash>) {
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
                            Delta::Added(value) => Some(revm::primitives::Bytecode::new_raw(value)),
                            Delta::Changed(value) => {
                                Some(revm::primitives::Bytecode::new_raw(value.to))
                            }
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
                            info.code_hash = hash;
                            info.code = Some(code.clone());

                            match self.contracts.entry_async(info.code_hash).await {
                                scc::hash_map::Entry::Vacant(entry) => {
                                    entry.insert_entry(code);
                                }
                                scc::hash_map::Entry::Occupied(mut entry) => {
                                    entry.insert(code);
                                }
                            }
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

                if let Some(table) = self.storage.get_async(&addr).await {
                    let t = table.get();
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
                        if let scc::hash_index::Entry::Vacant(entry) = t.entry_async(index).await {
                            entry.insert_entry(StorageData::Live(value_to_insert.into()));
                        }
                    }
                }
            }
        }
    }

    #[inline]
    async fn load_storage_slots(
        &self,
        address: Address,
        index: U256,
        block_num: u64,
        slots: u64,
    ) -> std::result::Result<
        Vec<U256>,
        alloy::transports::RpcError<alloy::transports::TransportErrorKind>,
    > {
        log::trace!(target: LOGGER_TARGET_SYNC, "Fetching storage {} {}..{}", &address, &index, index + U256::from(slots));
        let provider = self.provider.clone();
        let values = try_join_all((0..slots).map(|offset| {
            provider
                .get_storage_at(address, index + U256::from(offset))
                .block_id(BlockId::from(block_num))
                .into_future()
        }))
        .await;
        values
    }
    #[inline]
    async fn load_acc_info(
        &self,
        address: Address,
        block_num: u64,
        slots: u64,
    ) -> eyre::Result<(Address, AccountInfo, Vec<U256>)> {
        let min = Address::left_padding_from(&[255, 255]);
        let provider = self.provider.clone();
        let s = self.clone();
        let block_id = BlockId::from(block_num);
        let account_state = if address.lt(&min) {
            (U256::from(0), Bytes::new(), vec![])
        } else {
            try_join!(
                provider
                    .get_balance(address)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_code_at(address)
                    .block_id(block_id)
                    .into_future(),
                s.load_storage_slots(address, U256::from(0), block_num, slots)
            )?
        };

        let (code, code_hash) = match account_state.1.is_empty() {
            false => {
                let code = revm::primitives::Bytecode::new_raw(account_state.1);
                if code.is_empty() {
                    (None, KECCAK_EMPTY)
                } else {
                    (Some(code.clone()), code.hash_slow())
                }
            }
            true => (None, KECCAK_EMPTY),
        };
        Ok((
            address,
            AccountInfo {
                balance: account_state.0,
                code_hash,
                code,
                ..Default::default()
            },
            account_state.2,
        ))
    }

    pub(crate) async fn load_positions(&self, positions: Vec<Address>) -> eyre::Result<()> {
        let block_number = self.get_current_block().await?;
        let positions = positions
            .iter()
            .filter(|addr| !self.accounts.contains(*addr))
            .collect::<Vec<_>>();
        if positions.len() == 0 {
            return Ok(());
        }
        let infos = try_join_all(
            positions
                .clone()
                .into_iter()
                .map(|address| self.load_acc_info(*address, block_number, 25)),
        )
        .await
        .wrap_err("Failed to fetch basic info for all addresses")?;

        for (addr, info, slots) in infos.iter() {
            if let Some(code) = info.code.clone() {
                if let scc::hash_map::Entry::Vacant(entry) =
                    self.contracts.entry_async(info.code_hash).await
                {
                    entry.insert_entry(code);
                }
            }
            match self.accounts.entry_async(*addr).await {
                scc::hash_index::Entry::Vacant(entry) => {
                    entry.insert_entry(AccountData::Live(info.clone()));
                }
                scc::hash_index::Entry::Occupied(entry) => {
                    entry.update(AccountData::Live(info.clone()));
                }
            }
            let table = match self.storage.entry_async(*addr).await {
                scc::hash_index::Entry::Vacant(entry) => {
                    entry.insert_entry(scc::HashIndex::with_capacity_and_hasher(
                        DEFAULT_STORAGE_PR_ACCOUNT,
                        IndexHasher::default(),
                    ))
                }
                scc::hash_index::Entry::Occupied(entry) => entry,
            };
            for index in 0..slots.len() {
                if let Err(_) = table
                    .insert_async(U256::from(index), StorageData::Live(slots[index]))
                    .await
                {
                    log::error!(target: LOGGER_TARGET_SYNC, "Failed to insert storage value");
                }
            }
        }
        Ok(())
    }

    #[inline]
    async fn basic(
        &self,
        address: Address,
        block_number: u64,
    ) -> eyre::Result<Option<AccountInfo>> {
        match self.accounts.entry_async(address).await {
            scc::hash_index::Entry::Occupied(current) => Ok(Some(match current.get() {
                AccountData::Pending(cell) => cell.get_shared().await,
                AccountData::Live(info) => info.clone(),
            })),
            scc::hash_index::Entry::Vacant(accs) => {
                let cell = AsyncCell::shared();
                let entry = accs.insert_entry(AccountData::Pending(cell.clone()));
                let (_, info, slots) = self.load_acc_info(address, block_number, 10).await?;
                entry.update(AccountData::Live(info.clone()));
                cell.set(info.clone());
                let table = self.storage.entry_async(address).await.or_insert_with(|| {
                    scc::HashIndex::with_capacity_and_hasher(
                        DEFAULT_STORAGE_PR_ACCOUNT,
                        IndexHasher::default(),
                    )
                });
                for index in 0..slots.len() {
                    if let Err(_) = table.insert(U256::from(index), StorageData::Live(slots[index]))
                    {
                        log::error!(target: LOGGER_TARGET_SYNC, "Failed to insert storage value");
                    }
                }

                return Ok(Some(info));
            }
        }
    }

    async fn code_by_hash(&self, code_hash: B256) -> eyre::Result<revm::primitives::Bytecode> {
        log::info!(target: LOGGER_TARGET_SYNC, "Fetching code for {}", code_hash);
        match self.contracts.get(&code_hash) {
            Some(acc) => Ok(acc.clone()),
            None => Ok(revm::primitives::Bytecode::new()),
        }
    }

    #[inline]
    async fn storage(
        &self,
        address: Address,
        index: U256,
        block_number: u64,
    ) -> eyre::Result<U256> {
        match self
            .storage
            .entry_async(address)
            .await
            .or_insert(scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_STORAGE_PR_ACCOUNT,
                IndexHasher::default(),
            ))
            .entry_async(index)
            .await
        {
            scc::hash_index::Entry::Vacant(accs) => {
                let cell = AsyncCell::shared();
                let entry = accs.insert_entry(StorageData::Pending(cell.clone()));
                let data = self
                    .provider
                    .get_storage_at(address, index)
                    .block_id(BlockId::from(block_number))
                    .await
                    .wrap_err("Failed to fetch storage")?;
                cell.set(data.clone());
                entry.update(StorageData::Live(data.clone()));
                return Ok(data);
            }
            scc::hash_index::Entry::Occupied(previous) => {
                return Ok(match previous.get() {
                    StorageData::Pending(cell) => cell.get_shared().await,
                    StorageData::Live(value) => value.clone(),
                });
            }
        }
    }
    async fn block_hash(&self, num: u64) -> eyre::Result<B256> {
        match self.block_hashes.entry(num) {
            Entry::Occupied(out) => Ok(out.get().clone()),
            Entry::Vacant(e) => {
                log::debug!(target: LOGGER_TARGET_SYNC, "Fetching block hash {}", num);
                let out = self
                    .provider
                    .get_block(
                        BlockId::Number(alloy::eips::BlockNumberOrTag::Number(num)),
                        alloy::rpc::types::BlockTransactionsKind::Hashes,
                    )
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
