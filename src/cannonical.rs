use alloy::{
    eips::BlockId,
    hex::FromHex,
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
use futures::future::try_join_all;

use dashmap::{try_result::TryResult, DashMap, Entry};
use eyre::{Context, ContextCompat};
use revm::primitives::{alloy_primitives::aliases as prims, Address, KECCAK_EMPTY, U256};
use revm::{db::DatabaseRef, primitives::AccountInfo, Database};
use rustc_hash::FxBuildHasher;
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};
use tokio::{runtime::Handle, sync::RwLock};

use crate::{config::LinkType, LOGGER_TARGET_SYNC};

async fn load_storage_slot_inner(
    provider: alloy::providers::RootProvider<BoxTransport>,
    address: Address,
    index: prims::U256,
    block_num: u64,
) -> eyre::Result<prims::U256> {
    log::trace!(target: LOGGER_TARGET_SYNC, "Fetching storage {} {}", &address, &index);
    let value = provider
        .get_storage_at(address, index)
        .block_id(BlockId::from(block_num))
        .await?;

    Ok(value)
}

#[derive(Debug, Clone)]
pub struct Forked {
    pub cannonical: Arc<CannonicalFork>,
    pub env: revm::primitives::BlockEnv,
    pub seconds_per_block: revm::primitives::U256,
}
impl Forked {
    pub fn mine(&mut self, to_mine: u64) {
        let blocks = revm::primitives::U256::from(to_mine);
        self.env.number = self.env.number + blocks;
        self.env.timestamp += self.seconds_per_block * blocks;
        log::debug!(target: LOGGER_TARGET_SYNC, "Mined {} blocks, new block number {}", to_mine, self.env.number);
    }
    pub fn get_timestamp(&self) -> u64 {
        self.env.timestamp.to()
    }
    pub fn set_timestamp(&mut self, timestamp: u64) {
        log::debug!(target: LOGGER_TARGET_SYNC, "Setting timestamp to {}", timestamp);
        self.env.timestamp = revm::primitives::U256::from(timestamp);
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
    fn basic_ref(
        &self,
        address: revm::primitives::Address,
    ) -> Result<Option<AccountInfo>, Self::Error> {
        Forked::block_on(async { self.cannonical.clone().basic(address).await })
    }

    #[inline]
    fn code_by_hash_ref(
        &self,
        code_hash: prims::B256,
    ) -> Result<revm::primitives::Bytecode, Self::Error> {
        Forked::block_on(async { self.cannonical.clone().code_by_hash(code_hash).await })
    }

    #[inline]
    fn storage_ref(
        &self,
        address: revm::primitives::Address,
        index: prims::U256,
    ) -> Result<prims::U256, Self::Error> {
        Forked::block_on(async { self.cannonical.clone().storage(&address, &index).await })
    }

    #[inline]
    fn block_hash_ref(&self, number: u64) -> Result<prims::B256, Self::Error> {
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
    fn code_by_hash(
        &mut self,
        code_hash: prims::B256,
    ) -> Result<revm::primitives::Bytecode, Self::Error> {
        <Self as DatabaseRef>::code_by_hash_ref(self, code_hash)
    }

    #[inline]
    fn storage(
        &mut self,
        address: Address,
        index: prims::U256,
    ) -> Result<prims::U256, Self::Error> {
        <Self as DatabaseRef>::storage_ref(self, address, index)
    }

    #[inline]
    fn block_hash(&mut self, number: u64) -> Result<prims::B256, Self::Error> {
        <Self as DatabaseRef>::block_hash_ref(self, number)
    }
}

type StorageBlock = (
    prims::U256,
    prims::U256,
    prims::U256,
    prims::U256,
    prims::U256,
);
#[derive(Debug, Clone)]
pub struct CannonicalFork {
    // Results fetched from the provider and maintained by apply_next_mainnet_block,
    // Contains the full state of the account & storage
    link_type: LinkType,

    contracts: DashMap<prims::B256, revm::primitives::Bytecode, FxBuildHasher>,
    block_hashes: DashMap<u64, prims::B256, FxBuildHasher>,
    storage: DashMap<Address, DashMap<prims::U256, prims::U256, FxBuildHasher>, FxBuildHasher>,
    accounts: DashMap<Address, AccountInfo, FxBuildHasher>,
    pending_storage_reads: DashMap<(Address, prims::U256), StorageBlock, FxBuildHasher>,
    pending_basic_reads: DashMap<Address, revm::primitives::AccountInfo, FxBuildHasher>,

    current_block: Arc<RwLock<Block>>,
    provider: alloy::providers::RootProvider<BoxTransport>,
    provider_trace: alloy::providers::RootProvider<BoxTransport>,
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
            accounts: DashMap::with_capacity_and_hasher(1024 * 256, FxBuildHasher::default()),
            storage: DashMap::with_capacity_and_hasher(1024 * 256, FxBuildHasher::default()),
            pending_basic_reads: DashMap::with_capacity_and_hasher(
                1024 * 256,
                FxBuildHasher::default(),
            ),
            pending_storage_reads: DashMap::with_capacity_and_hasher(
                1024 * 256,
                FxBuildHasher::default(),
            ),
            contracts: DashMap::with_capacity_and_hasher(1024 * 8, FxBuildHasher::default()),
            block_hashes: DashMap::with_capacity_and_hasher(1024 * 8, FxBuildHasher::default()),
            current_block: Arc::new(RwLock::new(fork_block)),
            provider,
            provider_trace,
        }
    }

    pub async fn reset_state(&self, block_number: u64) -> eyre::Result<()> {
        self.accounts.clear();
        self.storage.clear();
        self.pending_basic_reads.clear();
        self.pending_storage_reads.clear();
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
            number: prims::U256::from(block.header.number),
            timestamp: prims::U256::from(block.header.timestamp),
            gas_limit: prims::U256::from(block.header.gas_limit),
            coinbase: block.header.miner,
            difficulty: block.header.difficulty,
            basefee: prims::U256::from(1),
            ..Default::default()
        }
    }

    async fn apply_next_geth_block(
        &self,
        diffs: Vec<BTreeMap<Address, AccountState>>,
    ) -> eyre::Result<()> {
        for account_diffs in diffs {
            for (k, v) in account_diffs {
                let addr = revm::primitives::Address::from(k.0);
                if !self.accounts.contains_key(&addr) {
                    continue;
                }
                let code_update = v
                    .code
                    .map(|v| revm::primitives::Bytes::from_hex(&v))
                    .map(|v| v.map(|v| revm::primitives::Bytecode::new_raw(v)));

                match self.accounts.entry(addr) {
                    Entry::Occupied(mut prev) => {
                        let prev = prev.get_mut();
                        prev.balance = match v.balance {
                            Some(value) => value,
                            _ => prev.balance,
                        };
                        prev.nonce = match v.nonce {
                            Some(value) => value,
                            _ => prev.nonce,
                        };
                        if let Some(Ok(code)) = code_update {
                            prev.code_hash = code.hash_slow();
                            prev.code = Some(code.clone());
                            self.contracts.insert(prev.code_hash, code);
                        }
                    }
                    _ => {
                        continue;
                    }
                };

                if !self.storage.contains_key(&addr) {
                    continue;
                }

                for (index, update) in v.storage {
                    self.update_storage_value(addr, index.into(), update.into());
                }
            }
        }
        Ok(())
    }

    fn update_storage_value(&self, address: Address, index: prims::U256, value: prims::U256) {
        match self.storage.try_get_mut(&address) {
            TryResult::Present(table) => match table.entry(index) {
                Entry::Occupied(mut a) => {
                    a.insert(value);
                }
                _ => {}
            },
            _ => {}
        }
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
                        enable_memory: Some(false),
                        enable_return_data: Some(false),
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
                let addr = revm::primitives::Address::from(k.0);
                if !self.accounts.contains_key(&addr) {
                    continue;
                }
                match self.accounts.entry(addr) {
                    Entry::Occupied(mut a) => {
                        let code_update = match v.code {
                            Delta::Added(value) => Some(revm::primitives::Bytecode::new_raw(value)),
                            Delta::Changed(value) => {
                                Some(revm::primitives::Bytecode::new_raw(value.to))
                            }
                            _ => None,
                        };
                        let code_update = code_update.map(|code| (code.hash_slow(), code));

                        let info: &mut AccountInfo = a.get_mut();
                        info.balance = match v.balance {
                            Delta::Added(value) => value,
                            Delta::Changed(value) => value.to,
                            _ => info.balance,
                        };
                        info.nonce = match v.nonce {
                            Delta::Added(value) => value.to(),
                            Delta::Changed(value) => value.to.to(),
                            _ => info.nonce,
                        };
                        if let Some((hash, code)) = code_update.clone() {
                            info.code_hash = hash;
                            info.code = Some(code.clone());
                            self.contracts.insert(hash, code);
                        }
                    }
                    _ => {
                        continue;
                    }
                };

                if v.storage.is_empty() || self.storage.try_get(&addr).is_absent() {
                    continue;
                }

                for (index, change) in v.storage {
                    let update = match change {
                        Delta::Added(value) => value,
                        Delta::Changed(t) => t.to,
                        _ => continue,
                    };
                    self.update_storage_value(addr, index.into(), update.into());
                }
            }
        }
    }

    async fn load_storage_slot(
        &self,
        address: Address,
        index: U256,
        block_num: u64,
    ) -> eyre::Result<(Address, U256, StorageBlock)> {
        let provider = self.provider.clone();
        let out = tokio::spawn(async move {
            let joined: Result<(U256, U256, U256, U256, U256), eyre::Error> = tokio::try_join!(
                load_storage_slot_inner(provider.clone(), address, index.clone(), block_num),
                load_storage_slot_inner(
                    provider.clone(),
                    address,
                    index + U256::from(1),
                    block_num
                ),
                load_storage_slot_inner(
                    provider.clone(),
                    address,
                    index + U256::from(2),
                    block_num
                ),
                load_storage_slot_inner(
                    provider.clone(),
                    address,
                    index + U256::from(3),
                    block_num
                ),
                load_storage_slot_inner(
                    provider.clone(),
                    address,
                    index + U256::from(4),
                    block_num
                )
            );
            joined
        })
        .await?;

        let out = out?;
        Ok((address, index.clone(), out))
    }

    async fn load_acc_info(
        &self,
        address: Address,
        block_num: u64,
    ) -> eyre::Result<(Address, AccountInfo)> {
        log::trace!(target: LOGGER_TARGET_SYNC, "Fetching account {}", address);
        let provider = self.provider.clone();
        let account_state = tokio::spawn(async move {
            tokio::try_join!(
                provider
                    .get_balance(address)
                    .block_id(BlockId::from(block_num)),
                provider
                    .get_transaction_count(address)
                    .block_id(BlockId::from(block_num)),
                provider
                    .get_code_at(address)
                    .block_id(BlockId::from(block_num))
            )
        })
        .await
        .wrap_err("Failed to fetch account info")
        .wrap_err("Failed to fetch account info")??;

        let (balance, nonce, code) = account_state;
        let (code, code_hash) = match code.is_empty() {
            false => {
                let code = revm::primitives::Bytecode::new_raw(code.0.into());
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
                balance: balance,
                nonce: nonce,
                code_hash,
                code,
            },
        ))
    }

    pub(crate) async fn load_positions(
        &self,
        positions: Vec<(Address, Vec<revm::primitives::U256>)>,
    ) -> eyre::Result<()> {
        let block_number = self.get_current_block().await?;
        let positions = positions
            .iter()
            .filter(|addr| !self.accounts.contains_key(&addr.0))
            .collect::<Vec<_>>();
        if positions.len() == 0 {
            return Ok(());
        }
        let infos = try_join_all(
            positions
                .clone()
                .into_iter()
                .map(|(address, _)| self.load_acc_info(*address, block_number)),
        )
        .await
        .wrap_err("Failed to fetch basic info for all addresses")?;

        for (addr, info) in infos.iter() {
            if let Some(code) = info.code.clone() {
                self.contracts.insert(info.code_hash, code);
            }
            self.accounts.insert(*addr, info.clone());
        }

        let addr_slot_pairs = positions
            .iter()
            .zip(infos)
            .filter(|(_, info)| info.1.code.is_some())
            .flat_map(|((addr, positions), _)| {
                positions
                    .iter()
                    .map(|index| index / U256::from(5))
                    .filter(|pos| match self.storage.try_get(addr) {
                        TryResult::Present(acc) => !acc.contains_key(&pos),
                        _ => true,
                    })
                    .map(|pos| (addr, pos))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .map(|(a, s)| (*a, s, s * U256::from(5)))
                    .collect::<Vec<_>>()
            });

        let res = try_join_all(
            addr_slot_pairs
                .map(|(address, _, index)| self.load_storage_slot(address, index, block_number))
                .collect::<Vec<_>>(),
        )
        .await?;

        for res in res.iter() {
            self.insert_block_to_storage(&res.0, &res.1, &res.2);
        }

        Ok(())
    }

    async fn fetch_basic_from_remote(&self, address: Address) -> eyre::Result<AccountInfo> {
        let block_number = self.get_current_block().await?;
        match self.pending_basic_reads.entry(address) {
            Entry::Occupied(read) => {
                return Ok(read.get().clone());
            }
            Entry::Vacant(accs) => {
                let mut pending_account_info = accs.insert(Default::default());

                let (_, info) = self.load_acc_info(address, block_number).await?;

                let info = info.clone();
                if let Some(code) = info.code.clone() {
                    self.contracts.insert(info.code_hash, code);
                }
                let info = info.clone();
                pending_account_info.balance = info.balance;
                pending_account_info.nonce = info.nonce;
                pending_account_info.code_hash = info.code_hash;
                pending_account_info.code = info.code;

                self.accounts.insert(address, pending_account_info.clone());
                return Ok(pending_account_info.clone());
            }
        };
    }

    pub async fn basic(&self, address: Address) -> eyre::Result<Option<AccountInfo>> {
        if let TryResult::Present(acc) = self.accounts.try_get(&address) {
            return Ok(Some(acc.clone()));
        }
        Ok(Some(self.fetch_basic_from_remote(address).await?))
    }
    async fn code_by_hash(
        &self,
        code_hash: prims::B256,
    ) -> eyre::Result<revm::primitives::Bytecode> {
        match self.contracts.get(&code_hash) {
            Some(acc) => Ok(acc.clone()),
            None => Ok(revm::primitives::Bytecode::new()),
        }
    }
    async fn try_fetch_storage(
        &self,
        address: &Address,
        table_index: &U256,
        block_number: u64,
    ) -> eyre::Result<StorageBlock> {
        let table_aligned = table_index * revm::primitives::U256::from(5);
        let key = (address.clone(), *table_index);
        let data: eyre::Result<(U256, U256, U256, U256, U256)> =
            match self.pending_storage_reads.entry(key) {
                Entry::Vacant(entry) => {
                    let data = self
                        .load_storage_slot(key.0, table_aligned, block_number)
                        .await?
                        .2;
                    entry.insert(data);
                    Ok(data.clone())
                }
                Entry::Occupied(acc) => Ok(acc.get().clone()),
            };

        let data = data?;
        return Ok(data);
    }

    fn insert_block_to_storage(
        &self,
        address: &Address,
        aligned_index: &prims::U256,
        block: &StorageBlock,
    ) {
        let storage = self.storage.entry(*address).or_default();
        storage.insert(aligned_index + U256::from(1), block.1);
        storage.insert(aligned_index + U256::from(2), block.2);
        storage.insert(aligned_index + U256::from(3), block.3);
        storage.insert(aligned_index + U256::from(4), block.4);
        storage.insert(*aligned_index, block.0);
    }
    async fn fetch_storage(&self, address: &Address, index: &U256) -> eyre::Result<prims::U256> {
        let block_number = self.get_current_block().await?;

        let (table_index, rem) = index.div_rem(U256::from(5));
        let data = self
            .try_fetch_storage(address, &table_index, block_number)
            .await?;

        // if !data.1.is_zero() || !data.2.is_zero() || !data.3.is_zero() || !data.4.is_zero() {
        //     let s = self.clone();
        //     let address = address.clone();
        //     let table_index = table_index + U256::from(1);
        //     if self
        //         .pending_storage_reads
        //         .try_get(&(address, table_index))
        //         .is_absent()
        //     {
        //         tokio::spawn(async move {
        //             let mut table_index = table_index;
        //             let data = s
        //                 .try_fetch_storage(&address, &table_index, block_number)
        //                 .await;
        //             if let Ok(data) = data {
        //                 s.insert_block_to_storage(&address, &table_index, &data);
        //             }
        //         });
        //     }
        // }

        let aligned_index = &table_index * U256::from(5);
        self.insert_block_to_storage(address, &aligned_index, &data);
        match rem.to() {
            0 => Ok(data.0),
            1 => Ok(data.1),
            2 => Ok(data.2),
            3 => Ok(data.3),
            4 => Ok(data.4),
            _ => Err(eyre::eyre!("Invalid storage index {}", rem)),
        }
    }

    pub async fn storage(&self, address: &Address, index: &U256) -> eyre::Result<prims::U256> {
        if let TryResult::Present(acc) = self.storage.try_get(address) {
            if let TryResult::Present(acc) = acc.try_get(index) {
                return Ok(acc.clone().into());
            }
        };
        return Ok(self.fetch_storage(address, index).await?);
    }
    async fn block_hash(&self, num: u64) -> eyre::Result<prims::B256> {
        match self.block_hashes.entry(num) {
            dashmap::Entry::Occupied(out) => Ok(out.get().clone()),
            dashmap::Entry::Vacant(e) => {
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
