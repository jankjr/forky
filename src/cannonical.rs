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
    transports::BoxTransport,
};
use alloy_provider::{
    ext::{DebugApi, TraceApi},
    Provider,
};
use async_cell::sync::AsyncCell;
use bytes::Buf;
use eyre::{Context, ContextCompat};
use futures::{future::try_join_all, FutureExt, TryFutureExt};
use itertools::Itertools;
use revm::primitives::{
    keccak256, AccessList, Address, BlobExcessGasAndPrice, Bytecode, Bytes, SpecId::GRAY_GLACIER,
    B256, KECCAK_EMPTY, U256,
};
use revm::{db::DatabaseRef, primitives::AccountInfo, Database};
// use rustc_hash::FxBuildHasher;
use alloy::hex;
use rustc_hash::FxBuildHasher;
use scc::hash_map::Entry;
use sdd::{AtomicOwned, AtomicShared, Guard, Shared, Tag};
use std::{
    collections::{BTreeMap, HashSet},
    future::IntoFuture,
    sync::{atomic::AtomicBool, Arc},
};
use tokio::{runtime::Handle, sync::Semaphore};

use crate::{
    abstract_value::{ArgType, SlotType, ADDR_MASK},
    analysis::{perform_analysis, AnalysisContext, AnalyzedStoragesSlot},
    config::LinkType,
    utils::is_address_like,
    LOGGER_TARGET_LOADS, LOGGER_TARGET_SYNC,
};

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

static PROXY_SLOT: U256 = U256::from_be_bytes(hex!(
    "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
));
static PROXY_SLOT_2: U256 = U256::from_be_bytes(hex!(
    "7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3"
));
static ADMIN_SLOT: U256 = U256::from_be_bytes(hex!(
    "10d6a54a4754c8869d6886b5f5d7fbfa5b4522237ea5c60d11bc4e7a1ff9390b"
));

static SLOTS_TO_FETCH_ON_RUNNING: usize = 2;
static SLOTS_TO_FETCH_PREFETCH: usize = 5;
static MAX_ARRAY_ENTRIES_TO_PRELOAD: usize = 100;

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
        Forked::block_on(async { self.cannonical.clone().basic(address, true).await })
    }

    #[inline]
    fn code_by_hash_ref(&self, code_hash: B256) -> Result<revm::primitives::Bytecode, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "code_by_hash({})", code_hash);
        Forked::block_on(async { unimplemented!("All AccountInfo's should have a code hash") })
    }

    #[inline]
    fn storage_ref(&self, address: Address, index: U256) -> Result<U256, Self::Error> {
        log::trace!(target: LOGGER_TARGET_SYNC, "call storage({}, {})", address, index);
        let out = Forked::block_on(async { self.cannonical.clone().storage(address, index).await });
        // log::debug!(target: LOGGER_TARGET_SYNC, "return storage({}, {}) => {:?}", address, index, out);

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

type Mapper = (U256, Vec<SlotType>, SlotType, Vec<ArgType>);
#[derive(Debug, Clone)]
struct MappingData {
    consumed: scc::HashSet<u64>,
    in_use: AtomicShared<bool>,

    // Does not contain Calldata, can be cleared after 1 call
    context_slots: Vec<Mapper>,

    // Only contains args with at least 1 Calldata
    call_dependent_slots: scc::HashIndex<u32, Vec<Mapper>, FxBuildHasher>,
}

impl MappingData {
    fn consume(&mut self, addr: &U256, selector: u32) -> bool {
        let key = addr.as_limbs()[0] ^ (selector as u64);
        if self.consumed.contains(&key) {
            return false;
        }
        if let Err(_) = self.consumed.insert(key) {
            return false;
        };
        true
    }
    fn context_consumed(&mut self, addr: &Address) -> bool {
        let key: u64 = u64::from_be_bytes([
            addr.0[0], addr.0[1], addr.0[2], addr.0[3], addr.0[4], addr.0[5], addr.0[6], addr.0[7],
        ]);
        if let Some(_) = self.consumed.read(&key, |v| true) {
            return false;
        }
        if let Err(_) = self.consumed.insert(key) {
            return false;
        };
        true
    }
}

impl MappingData {
    fn new(mapping_data: &Vec<AnalyzedStoragesSlot>) -> Self {
        let mut context_slots: Vec<Mapper> = Vec::new();
        let mut slot_map = std::collections::HashMap::new();
        for slot in mapping_data {
            match slot {
                AnalyzedStoragesSlot::Mapping(slot, map, typ, args) => {
                    for args in args.iter() {
                        // context_slots
                        if args.iter().all(|v| v.selector() == 0u32) {
                            context_slots.push((
                                slot.clone(),
                                map.clone(),
                                typ.clone(),
                                args.clone(),
                            ));
                        } else {
                            let selector = args.first().unwrap().selector();
                            let entry = slot_map.entry(selector).or_insert(Vec::new());

                            entry.push((slot.clone(), map.clone(), typ.clone(), args.clone()));
                        }
                    }
                }
                _ => {}
            }
        }

        let call_dependent_slots = scc::HashIndex::with_capacity_and_hasher(
            slot_map.len() as usize,
            FxBuildHasher::default(),
        );
        for (sel, mappers) in slot_map {
            call_dependent_slots.insert(sel, mappers).unwrap();
        }
        Self {
            consumed: scc::HashSet::with_hasher(Default::default()),
            context_slots,
            in_use: AtomicShared::new(false),
            call_dependent_slots,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CannonicalFork {
    // Results fetched from the provider and maintained by apply_next_mainnet_block,
    // Contains the full state of the account & storage
    link_type: LinkType,

    block_hashes: scc::HashMap<u64, B256>,
    storage: scc::HashIndex<Address, scc::HashIndex<U256, StorageData, IndexHasher>, AddressHasher>,
    accounts: scc::HashIndex<Address, AccountData, AddressHasher>,
    mappings: scc::HashMap<Address, MappingData, AddressHasher>,

    current_block: AtomicShared<Block>,
    provider: alloy::providers::RootProvider<BoxTransport>,
    provider_trace: alloy::providers::RootProvider<BoxTransport>,
}

const DEFAULT_CAPACITY_ACCOUNTS: usize = 1024;
const DEFAULT_SLACK: usize = 8;
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
            mappings: scc::HashMap::with_capacity_and_hasher(
                DEFAULT_CAPACITY_ACCOUNTS * DEFAULT_SLACK,
                AddressHasher::default(),
            ),
            block_hashes: scc::HashMap::with_capacity(1024 * 8),
            current_block: AtomicShared::new(fork_block),
            provider,
            provider_trace,
        }
    }

    pub async fn reset_state(&self, block_number: u64) -> eyre::Result<()> {
        self.accounts.clear();
        self.storage.clear();
        self.block_hashes.clear();
        let block = self
            .provider
            .get_block_by_number(alloy::eips::BlockNumberOrTag::Number(block_number), false)
            .await?
            .wrap_err(format!("Failed to fetch block {block_number} from RPC"))?;

        self.current_block.swap(
            (Some(Shared::new(block)), Tag::None),
            std::sync::atomic::Ordering::SeqCst,
        );

        Ok(())
    }

    pub fn block_env(&self) -> revm::primitives::BlockEnv {
        let g = Guard::new();
        let block = self
            .current_block
            .load(std::sync::atomic::Ordering::Relaxed, &g)
            .as_ref()
            .unwrap();
        revm::primitives::BlockEnv {
            number: U256::from(block.header.number),
            timestamp: U256::from(block.header.timestamp),
            gas_limit: U256::from(block.header.gas_limit),
            coinbase: block.header.miner,
            blob_excess_gas_and_price: Some(BlobExcessGasAndPrice {
                excess_blob_gas: block.header.excess_blob_gas.unwrap_or_default(),
                blob_gasprice: block.header.blob_fee().unwrap_or_default(),
            }),
            prevrandao: block.header.mix_hash,
            difficulty: block.header.difficulty,
            basefee: U256::from(block.header.base_fee_per_gas.unwrap_or(1u64)),

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
            .provider_trace
            .trace_replay_block_transactions(block_number.into(), &[TraceType::StateDiff])
            .await
            .wrap_err(format!("Failed to fetch trace for {}", block_number))?;

        Ok(self.apply_next_reth_block(trace).await)
    }

    async fn load_geth_trace_and_apply(&self) -> eyre::Result<()> {
        let block_number = self.get_current_block();
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
            log::trace!(target: LOGGER_TARGET_LOADS, "load_proxy_contract_code({})", address);
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
    pub async fn load_access_list(&self, list: &AccessList) -> eyre::Result<()> {
        let block_number: u64 = self.get_current_block();
        try_join_all(list.iter().map(|item|{
            let address = item.address;
            let f = async move {
                self.basic(address, false).await?;

                try_join_all(item
                    .storage_keys
                    .iter()
                    .map(|v| {
                        let index: U256 = v.clone().into();
                        self.fetch_storage(address, index, block_number)
                    })
                ).await
            };
            futures::TryFutureExt::into_future(f)
        })).await;
        Ok(())
    }
    async fn load_acc_info_analysis(
        &self,
        block_num: u64,
        slots_loaded: usize,
        proxy_slot_value: U256,
        incode: Bytecode,
        contract_address: &Address,
    ) -> eyre::Result<(Vec<AnalyzedStoragesSlot>, Vec<Address>)> {
        log::debug!(target: LOGGER_TARGET_LOADS, "(Analysis) analyzing contract {}", contract_address);

        let (slot_data, contract_refs) = if !proxy_slot_value.is_zero() {
            let implementation_addr: U160 = proxy_slot_value.to();
            let implementation_addr = Address::from(implementation_addr);
            let code = self
                .load_proxy_contract_code(implementation_addr, block_num)
                .await?;

            perform_analysis(&code, &AnalysisContext::from(&self.block_env()))?
        } else {
            perform_analysis(&incode, &AnalysisContext::from(&self.block_env()))?
        };

        log::debug!(target: LOGGER_TARGET_LOADS, "Layout of {}:", contract_address);
        for slot in slot_data.iter() {
            log::debug!(target: LOGGER_TARGET_LOADS, "  {}", slot);
        }

        let min = U256::from(slots_loaded);

        let additional_slots = slot_data
            .iter()
            .cloned()
            .filter(|v| match v {
                AnalyzedStoragesSlot::Slot(slot, _) => slot.gt(&min),
                _ => true,
            })
            .collect::<Vec<_>>();

        for index in additional_slots.iter() {
            log::debug!(target: LOGGER_TARGET_SYNC, "(Analysis) slot: {}: {}", contract_address, index);
        }
        Ok((additional_slots, contract_refs))
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

        let (analyzed_slots, contract_refs) = self
            .load_acc_info_analysis(
                block_number,
                SLOTS_TO_FETCH_ON_RUNNING,
                proxy_slot_value,
                code.clone(),
                &address,
            )
            .await?;
        let mut account_refs: Vec<Address> = Vec::new();

        if !analyzed_slots.is_empty() {
            self.init_account_mappings(
                &address,
                &analyzed_slots
                    .iter()
                    .filter(|v| v.is_mapping())
                    .cloned()
                    .collect::<Vec<_>>(),
            );

            let slots = try_join_all(
                analyzed_slots
                    .iter()
                    .map(|index| self.fetch_analyzed_slot(&address, index, block_number)),
            )
            .await?
            .into_iter()
            .flatten()
            .unique()
            .collect::<Vec<_>>();

            for (_, slot, value) in slots.into_iter() {
                if is_address_like(&value) {
                    account_refs.push(Address::from(U160::from(value & ADDR_MASK)));
                }
                account_slots.push((slot, value));
            }
        }

        for addr in contract_refs.into_iter() {
            if !self.accounts.contains(&addr) {
                account_refs.push(addr);
            }
        }
        account_refs.dedup();
        Ok(account_refs)
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
        perform_analysis: bool,
    ) -> eyre::Result<AccountInfo> {
        let info: AccountInfo = self.fetch_minimal_account_info(address, block_num).await?;

        if let Some(code) = &info.code {
            let s = self.clone();
            let first_acc_code = code.clone();
            if perform_analysis {
                tokio::spawn(async move {
                    let code = first_acc_code;
                    match s.run_new_account_analysis(address, code, block_num).await {
                        Err(_) => return,
                        Ok(accounts) => accounts,
                    };
                });
            }
        };

        Ok(info)
    }

    pub fn call(&self, inputs: &revm::interpreter::CallInputs) -> Vec<revm::primitives::U256> {
        if inputs.input.len() < 4 {
            return Vec::new();
        }
        let selector = inputs.input.slice(0..4).get_u32();
        let to = inputs.target_address;

        let mut mappings = if let Some(e) = self.mappings.read(&to, |_, v| v.clone()) {
            e
        } else {
            return Vec::new();
        };

        let input_dep = mappings
            .call_dependent_slots
            .peek_with(&selector, |_, mappings| mappings.clone())
            .unwrap_or_default();

        let to_check: Vec<Mapper> = if mappings.context_consumed(&inputs.caller) {
            mappings
                .context_slots
                .iter()
                .chain(input_dep.iter())
                .cloned()
                .collect()
        } else {
            input_dep
        };
        if to_check.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut extra_loads = Vec::new();

        for slot_data in to_check {
            let (contract_storage_slot, _, _, args) = slot_data;

            let mut out_index = U256::ZERO;

            let mut buff: [u8; 64] = [0; 64];
            buff[32..64].copy_from_slice(&contract_storage_slot.to_be_bytes::<32>());
            let mut contains_arg = false;
            for arg_type in args.iter() {
                let mut sel = 0;
                match arg_type {
                    ArgType::MsgSender => {
                        let arg = inputs.caller.into_word();
                        let arg: U256 = arg.into();
                        buff[0..32].copy_from_slice(arg.to_be_bytes::<32>().as_slice());
                    }
                    ArgType::This => {
                        let arg = inputs.target_address.into_word();
                        let arg: U256 = arg.into();

                        buff[0..32].copy_from_slice(arg.to_be_bytes::<32>().as_slice());
                    }
                    ArgType::Calldata(_, index) => {
                        let start = 4 + (*index * 32);
                        let end = start + 32;
                        if inputs.input.len() < end {
                            out_index = U256::ZERO;
                            break;
                        }

                        contains_arg = true;

                        let arg = U256::from_be_slice(&inputs.input[start..end]);
                        sel = selector;

                        buff[0..32].copy_from_slice(arg.to_be_bytes::<32>().as_slice());
                    }
                }
                let index_bytes = keccak256(&buff);
                let storage_slot: U256 = index_bytes.into();

                if storage_slot.is_zero() {
                    out_index = U256::ZERO;
                    break;
                }

                if !mappings.consume(&storage_slot, sel) {
                    out_index = U256::ZERO;
                    break;
                }

                out_index = storage_slot;
                buff[0..32].copy_from_slice([0u8; 32].as_slice());
                buff[32..64].copy_from_slice(&storage_slot.to_be_bytes::<32>());
            }
            if out_index.is_zero() {
                continue;
            }

            match self.storage.peek_with(&inputs.target_address, |_, v| {
                v.peek_with(&out_index, |_, v| true)
            }) {
                Some(Some(true)) => continue,
                _ => {}
            }

            if contains_arg {
                extra_loads.push(out_index);
            }
            log::trace!(target: LOGGER_TARGET_LOADS, "(Analysis) get_storage_at({}, {})", inputs.target_address, out_index);
            out.push(out_index);
        }

        return out;
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
            provider
                .get_storage_at(address, PROXY_SLOT)
                .block_id(block_id)
                .into_future()
                .map_err(|e| eyre::eyre!("Failed to fetch proxy slot for {}: {:?}", address, e)),
            provider
                .get_storage_at(address, PROXY_SLOT_2)
                .block_id(block_id)
                .into_future()
                .map_err(|e| eyre::eyre!("Failed to fetch proxy slot 2 for {}: {:?}", address, e)),
            self.load_storage_slots(address, U256::from(0), block_num, slots)
        )?;

        let (balance, code, proxy_slot_value1, proxy_slot_value2, mut preloaded_slots) =
            data_handle;

        let code = if code.len() == 0 {
            None
        } else {
            Some(revm::primitives::Bytecode::new_raw(code))
        };

        preloaded_slots.push((PROXY_SLOT, proxy_slot_value1));
        preloaded_slots.push((PROXY_SLOT_2, proxy_slot_value2));

        // Run analysis

        let proxy_slot_value = if !proxy_slot_value1.is_zero() {
            proxy_slot_value1
        } else {
            proxy_slot_value2
        };
        if let Some(code) = &code {
            let (additional_slots, refs) = self
                .load_acc_info_analysis(block_num, slots, proxy_slot_value, code.clone(), &address)
                .await?;

            self.init_account_mappings(
                &address,
                &additional_slots
                    .iter()
                    .filter(|v| v.is_mapping())
                    .cloned()
                    .collect::<Vec<_>>(),
            );

            contracts_referenced.extend(refs);
            let additional_slots = try_join_all(
                additional_slots
                    .iter()
                    .map(|index| self.fetch_analyzed_slot(&address, index, block_num)),
            )
            .await?
            .into_iter()
            .flatten()
            .unique()
            .collect::<Vec<_>>();

            for (_, index, value) in additional_slots {
                if is_address_like(&value) {
                    contracts_referenced.push(Address::from(U160::from(value & ADDR_MASK)));
                }
                preloaded_slots.push((index, value));
            }
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
        perform_analysis: bool,
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
        let block_number: u64 = self.get_current_block();
        let info = self
            .load_acc_info_live(address, block_number, perform_analysis)
            .await?;
        cell.set(info.clone());
        Ok(Some(info))
    }
    pub async fn get_account_data(&self, address: &Address) -> eyre::Result<Option<AccountInfo>> {
        self.basic(address.clone(), true).await
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

    #[inline]
    fn init_account_mappings(&self, address: &Address, mappings: &Vec<AnalyzedStoragesSlot>) {
        // if self.mappings.contains(address) {
        //     return;
        // }
        // if let Err(_) = self.mappings.insert(*address, MappingData::new(mappings)) {}
    }

    async fn fetch_analyzed_slot(
        &self,
        address: &Address,
        slot: &AnalyzedStoragesSlot,
        block_number: u64,
    ) -> eyre::Result<Vec<(SlotType, U256, U256)>> {
        let v = match slot {
            AnalyzedStoragesSlot::Slot(slot, typ) => {
                log::debug!(target: LOGGER_TARGET_SYNC, "fetch_analyzed_slot({}, {})", address, slot);
                vec![(
                    *typ,
                    *slot,
                    self.fetch_storage(*address, *slot, block_number).await?,
                )]
            }
            AnalyzedStoragesSlot::Array(slot, typ) => {
                log::debug!(target: LOGGER_TARGET_SYNC, "fetch_analyzed_slot({}, {})", address, slot);
                let len = self.fetch_storage(*address, *slot, block_number).await?;

                let entries_to_fetch: usize =
                    U256::min(len.clone(), U256::from(MAX_ARRAY_ENTRIES_TO_PRELOAD)).to();
                let mut out = vec![(SlotType::Unknown, *slot, len)];
                if len.lt(&U256::from(10000)) && entries_to_fetch != 0 {
                    log::debug!(target: LOGGER_TARGET_LOADS, "fetch_analyzed_slot({}, {}): Loading {} array entries out of {}", address, slot, entries_to_fetch, len);
                    let slot = *slot;
                    let address = address.clone();
                    let array_data_offset: U256 = keccak256::<[u8; 32]>(slot.to_be_bytes()).into();
                    let slot_indices = (0..entries_to_fetch)
                        .map(|offset_from_start| array_data_offset + U256::from(offset_from_start))
                        .collect::<Vec<_>>();

                    let res =
                        try_join_all(slot_indices.iter().map(|index| {
                            self.fetch_storage(address.clone(), *index, block_number)
                        }))
                        .await?;
                    for (index, value) in slot_indices.into_iter().zip(res.into_iter()) {
                        out.push((typ.clone(), index, value));
                    }
                }
                out
            }
            AnalyzedStoragesSlot::Mapping(slot, _, _, _) => {
                vec![(SlotType::Unknown, *slot, U256::ZERO)]
            }
        };

        Ok(v)
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

        log::trace!(target: LOGGER_TARGET_LOADS, "get_storage_at({}, {})", address, index);
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
