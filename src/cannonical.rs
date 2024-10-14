use crate::{analysis::perform_analysis, config::LinkType, LOGGER_TARGET_SYNC};
use alloy::{
    eips::BlockId,
    hex::FromHex,
    primitives::{map::FbBuildHasher, U160},
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
    signers::k256::elliptic_curve::rand_core::block,
    transports::BoxTransport,
};
use alloy_provider::{
    ext::{DebugApi, TraceApi},
    Provider,
};
use async_cell::sync::AsyncCell;
use eyre::{eyre, Context, ContextCompat};
use futures::{
    future::{join_all, try_join, try_join_all, TryJoinAll},
    FutureExt, TryFutureExt,
};
use itertools::Itertools;
use revm::primitives::{keccak256, Address, Bytecode, Bytes, FixedBytes, B256, KECCAK_EMPTY, U256};
use revm::{db::DatabaseRef, primitives::AccountInfo, Database};
// use rustc_hash::FxBuildHasher;
use alloy::hex;
use scc::hash_map::Entry;
use std::{
    collections::{BTreeMap, HashSet},
    future::IntoFuture,
    primitive,
    sync::Arc,
};
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
    // pub dry_run: bool,
}

static PROXY_SLOT: U256 = U256::from_be_bytes(hex!(
    "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
));
static PROXY_SLOT_2: U256 = U256::from_be_bytes(hex!(
    "7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3"
));
static ADMIN_SLOT: U256 = U256::from_be_bytes(hex!(
    "10d6a54a4754c8869d6886b5f5d7fbfa5b4522237ea5c60d11bc4e7a1ff9390b"
));

static SLOTS_TO_FETCH_ON_RUNNING: usize = 10;
static SLOTS_TO_FETCH_PREFETCH: usize = 25;

impl Forked {
    // pub fn set_dry_run(&mut self, dry_run: bool) {
    //     self.dry_run = dry_run;
    // }
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
        log::trace!(target: LOGGER_TARGET_SYNC, "basic({})", address);
        Forked::block_on(async { self.cannonical.clone().basic(address).await })
    }

    #[inline]
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<revm::primitives::Bytecode, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "code_by_hash({})", code_hash);
        Forked::block_on(async { self.cannonical.clone().code_by_hash(code_hash).await })
    }

    #[inline]
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "call storage({}, {})", address, index);
        let out = Forked::block_on(async { self.cannonical.clone().storage(address, index).await });
        log::debug!(target: LOGGER_TARGET_SYNC, "return storage({}, {}) => {:?}", address, index, out);

        out
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

    // contracts: scc::HashMap<B256, revm::primitives::Bytecode, IndexHasher>,
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
enum StorageSlotType {
    Unknown,
    // Array(U256),
    // Mapping,
    Address(Address),
}
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
            // contracts: scc::HashMap::with_capacity_and_hasher(
            //     DEFAULT_CAPACITY_ACCOUNTS * DEFAULT_SLACK,
            //     IndexHasher::default(),
            // ),
            block_hashes: scc::HashMap::with_capacity(1024 * 8),
            current_block: Arc::new(RwLock::new(fork_block)),
            provider,
            provider_trace,
        }
    }

    pub async fn reset_state(&self, block_number: u64) -> eyre::Result<()> {
        self.accounts.clear();
        self.storage.clear();
        // self.contracts.clear();
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
                            prev.code_hash = KECCAK_EMPTY;
                            prev.code = Some(code);

                            // match self.contracts.entry_async(prev.code_hash).await {
                            //     scc::hash_map::Entry::Vacant(entry) => {
                            //         entry.insert_entry(code);
                            //     }
                            //     scc::hash_map::Entry::Occupied(mut entry) => {
                            //         entry.insert(code);
                            //     }
                            // }
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
                        let code_update = code_update.map(|code| (KECCAK_EMPTY, code));

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

                            // match self.contracts.entry_async(info.code_hash).await {
                            //     scc::hash_map::Entry::Vacant(entry) => {
                            //         entry.insert_entry(code);
                            //     }
                            //     scc::hash_map::Entry::Occupied(mut entry) => {
                            //         entry.insert(code);
                            //     }
                            // }
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

    #[inline]
    async fn load_proxy_contract_code(
        &self,
        address: Address,
        block_num: u64,
    ) -> eyre::Result<Bytecode> {
        let code = if let Some(acc) = self.peek_account_info(&address).await {
            acc.code
        } else {
            let cell = match self.init_pending_account(&address) {
                Ok(cell) => cell,
                Err(_) => {
                    return match self.peek_account_info(&address).await.map(|v| v.code) {
                        Some(Some(code)) => Ok(code),
                        _ => Err(eyre::eyre!("Failed to fetch proxy contract code")),
                    };
                }
            };
            let block_id = BlockId::from(block_num);
            let (balance, code) = tokio::try_join!(
                self.provider
                    .get_balance(address)
                    .block_id(block_id)
                    .into_future(),
                self.provider
                    .get_code_at(address)
                    .block_id(block_id)
                    .into_future(),
            )?;
            let code = if code.is_empty() {
                None
            } else {
                Some(revm::primitives::Bytecode::new_raw(code))
            };
            cell.set(AccountInfo {
                balance,
                code: code.clone(),
                ..Default::default()
            });
            code
        };

        match code {
            Some(code) => Ok(code),
            None => return Err(eyre::eyre!("Proxy implementation not found {}", address)),
        }
    }

    async fn load_acc_info_analysis(
        &self,
        block_num: u64,
        slots_loaded: usize,
        proxy_slot_value: U256,
        incode: Bytecode,
        contract_address: &Address,
    ) -> eyre::Result<(Option<Vec<Address>>, Option<Vec<U256>>)> {
        log::debug!(target: LOGGER_TARGET_SYNC, "(Analysis) analyzing contract {}", contract_address);

        let data = if !proxy_slot_value.is_zero() {
            let implementation_addr: U160 = proxy_slot_value.to();
            let implementation_addr = Address::from(implementation_addr);
            let code = self
                .load_proxy_contract_code(implementation_addr, block_num)
                .await?;

            perform_analysis(&code)?
        } else {
            perform_analysis(&incode)?
        };

        let contracts_referenced = if !data.0.is_empty() {
            Some(data.0.clone())
        } else {
            None
        };

        let min = U256::from(slots_loaded);

        let additional_slots = data
            .1
            .iter()
            .cloned()
            .filter(|v| v.gt(&min))
            .collect::<Vec<_>>();

        if additional_slots.len() == 0 {
            return Ok((contracts_referenced, None));
        }
        for index in additional_slots.iter() {
            log::debug!(target: LOGGER_TARGET_SYNC, "(Analysis) slot: {}: {}", contract_address, index);
        }
        Ok((contracts_referenced, Some(additional_slots)))
    }

    async fn load_slots(
        &self,
        address: &Address,
        indices: &Vec<U256>,
        block_num: u64,
    ) -> eyre::Result<Vec<(U256, U256)>> {
        let block_id = BlockId::from(block_num);

        let address = address.clone();
        let values = try_join_all(
            indices
                .iter()
                .cloned()
                .map(|index| {
                    self.provider
                        .get_storage_at(address, index)
                        .block_id(block_id)
                        .into_future()
                })
                .collect::<Vec<_>>(),
        )
        .await?;

        Ok(indices
            .into_iter()
            .cloned()
            .zip(values.into_iter())
            .collect::<Vec<_>>())
    }

    async fn run_new_account_analysis(
        &self,
        address: Address,
        code: revm::primitives::Bytecode,
        block_number: u64,
    ) -> eyre::Result<Vec<Address>> {
        let (proxy_slot_1_value, proxy_slot_2_value, _, mut account_slots) = futures::try_join!(
            self.fetch_storage(address, PROXY_SLOT, block_number),
            self.fetch_storage(address, PROXY_SLOT_2, block_number),
            self.fetch_storage(address, ADMIN_SLOT, block_number),
            self.load_storage_slots(
                address,
                U256::from(0u64),
                block_number,
                SLOTS_TO_FETCH_ON_RUNNING
            )
        )?;

        let proxy_slot_value = if proxy_slot_1_value.is_zero() {
            proxy_slot_2_value
        } else {
            proxy_slot_1_value
        };

        let (refs, additional_slots) = self
            .load_acc_info_analysis(
                block_number,
                SLOTS_TO_FETCH_ON_RUNNING,
                proxy_slot_value,
                code.clone(),
                &address,
            )
            .await?;

        if let Some(additional_slots) = additional_slots {
            let slots = try_join_all(
                additional_slots
                    .iter()
                    .map(|index| self.fetch_storage(address, *index, block_number)),
            )
            .await?;

            for tup in additional_slots.into_iter().zip(slots.into_iter()) {
                account_slots.push(tup);
            }
        }

        let mut account_refs: Vec<Address> = Vec::new();
        if let Some(refs) = refs {
            for addr in refs.into_iter() {
                if !self.accounts.contains(&addr) {
                    account_refs.push(addr);
                }
            }
        }
        for (index, value) in account_slots.iter() {
            match self.analyze_storage_slot(&address, index, value) {
                StorageSlotType::Address(addr) => {
                    if !self.accounts.contains(&addr) {
                        account_refs.push(addr);
                    }
                }
                _ => {
                    continue;
                }
            }
        }
        Ok(account_refs.into_iter().unique().collect::<Vec<_>>())
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
            futures::try_join!(
                provider
                    .get_balance(address)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_code_at(address)
                    .block_id(block_id)
                    .into_future(),
            )
            .wrap_err("Failed to fetch account info")?
        };

        let code = if code.is_empty() {
            None
        } else {
            Some(revm::primitives::Bytecode::new_raw(code))
        };

        Ok(AccountInfo {
            balance,
            code: code,
            ..Default::default()
        })
    }
    async fn load_acc_info_live(
        &self,
        address: Address,
        block_num: u64,
    ) -> eyre::Result<AccountInfo> {
        let info: AccountInfo = self.fetch_minimal_account_info(address, block_num).await?;

        if let Some(code) = &info.code {
            let s = self.clone();
            let first_acc_code = code.clone();
            tokio::spawn(async move {
                let code = first_acc_code;
                match s.run_new_account_analysis(address, code, block_num).await {
                    Err(_) => return,
                    Ok(accounts) => accounts,
                };
            });
        };

        Ok(info)
    }

    #[inline]
    fn analyze_storage_slot(&self, _: &Address, __: &U256, value: &U256) -> StorageSlotType {
        let val = *value;

        // let maybe_array = value <= &U256::from(25u64);
        // if maybe_array {
        //     let array_data = keccak256::<[u8; 32]>(index.to_be_bytes());
        //     let array_data_offset: U256 = array_data.into();
        //     let data = self
        //         .provider
        //         .get_storage_at(address, array_data_offset)
        //         .block_id(block_id)
        //         .await?;
        //     if !data.is_zero() {
        //         val = data;
        //         log::info!(target: LOGGER_TARGET_SYNC, "array-data: {address}.{index} => {data}");
        //     }
        // }

        let len = val.byte_len();
        if len > 15 && len <= 20 {
            let z_count = val.count_zeros();
            if z_count > 150 && z_count <= 190 {
                let addr = Address::from(U160::from(val));
                return StorageSlotType::Address(addr);
            }
        }
        return StorageSlotType::Unknown;
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
        let mut contracts_referenced = HashSet::<Address>::new();
        let provider = self.provider.clone();

        let data_handle = tokio::spawn(async move {
            let provider = provider.clone();
            let block_id = BlockId::from(block_num);
            futures::try_join!(
                provider
                    .get_balance(address)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_code_at(address)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_storage_at(address, PROXY_SLOT)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_storage_at(address, PROXY_SLOT_2)
                    .block_id(block_id)
                    .into_future(),
                provider
                    .get_storage_at(address, ADMIN_SLOT)
                    .block_id(block_id)
                    .into_future()
            )
        })
        .map(|v| match v {
            Ok(Ok(tup)) => eyre::Result::Ok(tup),
            e => eyre::Result::Err(eyre::eyre!("Failed to fetch storage {:?}", e)),
        });

        let (
            (balance, code, proxy_slot_value1, proxy_slot_value2, admin_value),
            mut preloaded_slots,
        ) = futures::try_join!(
            data_handle,
            self.load_storage_slots(address, U256::from(0), block_num, slots)
        )?;

        let code = if code.len() == 0 {
            None
        } else {
            Some(revm::primitives::Bytecode::new_raw(code))
        };

        preloaded_slots.push((PROXY_SLOT, proxy_slot_value1));
        preloaded_slots.push((PROXY_SLOT_2, proxy_slot_value2));
        preloaded_slots.push((ADMIN_SLOT, admin_value));

        // Run analysis

        let proxy_slot_value = if !proxy_slot_value1.is_zero() {
            proxy_slot_value1
        } else {
            proxy_slot_value2
        };
        if let Some(code) = &code {
            let (refs, additional_slots) = self
                .load_acc_info_analysis(block_num, slots, proxy_slot_value, code.clone(), &address)
                .await?;

            if let Some(refs) = refs {
                contracts_referenced.extend(refs);
            }
            if let Some(additional_slots) = additional_slots {
                let additional_slots = self
                    .load_slots(&address, &additional_slots, block_num)
                    .await?;

                for (index, value) in additional_slots {
                    preloaded_slots.push((index, value));
                }
            }
        }

        for (index, value) in preloaded_slots.iter() {
            match self.analyze_storage_slot(&address, index, value) {
                StorageSlotType::Address(addr) => {
                    contracts_referenced.insert(addr);
                }
                _ => {
                    continue;
                }
            }
        }

        let contracts_referenced = if contracts_referenced.is_empty() {
            None
        } else {
            Some(contracts_referenced.into_iter().collect::<Vec<_>>())
        };

        Ok((
            address,
            AccountInfo {
                balance: balance,
                code,
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
        let block_number = self.get_current_block().await?;
        let positions = positions
            .iter()
            .filter(|addr| !self.accounts.contains(*addr))
            .collect::<Vec<_>>();

        if positions.len() == 0 {
            return Ok(Vec::new());
        }
        for address in positions.iter() {
            self.init_account_storage(*address);
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
                self.init_account_storage_value(addr, index, value);
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
    fn init_pending_account(&self, address: &Address) -> eyre::Result<Arc<AsyncCell<AccountInfo>>> {
        let cell = AsyncCell::shared();
        if let Err(_) = self
            .accounts
            .insert(*address, AccountData::Pending(cell.clone()))
        {
            return Err(eyre::eyre!("Account already initialized"));
        }
        Ok(cell)
    }

    #[inline]
    async fn basic(
        &self,
        address: Address, // dry_run: bool,
    ) -> eyre::Result<Option<AccountInfo>> {
        if let Some(value) = self.peek_account_info(&address).await {
            return Ok(Some(value));
        }
        let cell = match self.init_pending_account(&address) {
            Err(_) => return Ok(self.peek_account_info(&address).await),
            Ok(cell) => cell,
        };
        self.init_account_storage(&address);

        log::trace!(target: LOGGER_TARGET_SYNC, "(Cache miss) Fetching account {}", address);
        let block_number: u64 = self.get_current_block().await?;
        let info = self.load_acc_info_live(address, block_number).await?;
        cell.set(info.clone());
        Ok(Some(info))
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

    #[inline]
    fn init_account_storage_value(&self, address: &Address, index: &U256, value: &U256) {
        if let Err(_) = self
            .storage
            .get(address)
            .unwrap()
            .insert(*index, StorageData::Live(*value))
        {
            log::error!(target: LOGGER_TARGET_SYNC, "basic {}: Failed to insert storage at index {}", address, index);
        }
    }

    #[inline]
    fn init_account_storage(&self, address: &Address) {
        if self.storage.contains(address) {
            return;
        }
        if let Err(_) = self.storage.insert(
            *address,
            scc::HashIndex::with_capacity_and_hasher(
                DEFAULT_STORAGE_PR_ACCOUNT,
                IndexHasher::default(),
            ),
        ) {}
    }

    async fn code_by_hash(&self, code_hash: B256) -> eyre::Result<revm::primitives::Bytecode> {
        panic!("Code by hash not implemented");
    }

    async fn fetch_storage(
        &self,
        address: Address,
        index: U256,
        block_number: u64,
    ) -> eyre::Result<U256> {
        if let Ok(val) = self.peek_storage(&address, &index).await {
            return Ok(val);
        }

        let cell = AsyncCell::shared();

        match self
            .storage
            .get(&address)
            .unwrap()
            .insert(index, StorageData::Pending(cell.clone()))
        {
            Err(_) => {
                return self.peek_storage(&address, &index).await;
            }
            Ok(_) => {}
        };

        let data = self
            .provider
            .get_storage_at(address, index)
            .block_id(BlockId::from(block_number))
            .await
            .wrap_err(format!(
                "Failed to fetch storage for address {address} and index {index}"
            ))?;
        cell.set(data.clone());
        Ok(data)
    }

    #[inline]
    async fn peek_storage(&self, address: &Address, index: &U256) -> eyre::Result<U256> {
        if let Some(Some(value)) = self
            .storage
            .peek_with(address, |_, v| v.peek_with(index, |_, v| v.clone()))
        {
            let value = match value {
                StorageData::Live(data) => data.clone(),
                StorageData::Pending(cell) => cell.get_shared().await.clone(),
            };
            return Ok(value);
        }
        Err(eyre::eyre!("Storage not found"))
    }
    #[inline]
    async fn storage(&self, address: Address, index: U256) -> eyre::Result<U256> {
        if let Ok(current_value) = self.peek_storage(&address, &index).await {
            return Ok(current_value);
        }
        log::trace!(target: LOGGER_TARGET_SYNC, "(miss) Fetching storage {} {}", address, index);
        let block_number: u64 = self.get_current_block().await?;
        self.fetch_storage(address, index, block_number).await
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
