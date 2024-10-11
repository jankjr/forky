use alloy::eips::BlockId;
use alloy::hex::FromHex;
use alloy::transports::BoxTransport;

use alloy_provider::Provider;
use dashmap::DashSet;
use eyre::ContextCompat;
use eyre::Ok;
use neon::handle::Handle;
use neon::object::Object;
use neon::prelude::{Context, FunctionContext, ModuleContext, TaskContext};
use neon::result::{self, JsResult, NeonResult};
use neon::types::JsUndefined;
use neon::types::Value;
use neon::types::{JsBigInt, JsFunction, JsNumber, JsObject, JsPromise, JsString, JsValue};

use revm::db::CacheDB;
use revm::primitives::ExecutionResult;
use revm::primitives::U256;
use revm::primitives::{hex, TransactTo};
use revm::DatabaseRef;
use revm::{primitives::Address, Evm};

use once_cell::sync::OnceCell;
use tokio::sync::RwLock;

use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::DerefMut;
use std::{str::FromStr, sync::Arc};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

pub mod cannonical;
pub mod config;
use cannonical::{CannonicalFork, Forked};

use config::{Config, LinkType};
// pub mod errors;

pub const LOGGER_TARGET_MAIN: &str = "slot0";
pub const LOGGER_TARGET_SYNC: &str = "slot0::sync";
pub const LOGGER_TARGET_API: &str = "slot0::api";
pub const LOGGER_TARGET_SIMULATION: &str = "slot0::sim";

#[derive(Debug, Clone)]
pub struct TransactionRequest {
    pub from: revm::primitives::Address,
    pub to: revm::primitives::Address,
    pub data: Vec<u8>,
    pub value: revm::primitives::U256
}
pub struct ApplicationState {
    pub desynchronized: Arc<RwLock<bool>>,
    pub cannonical: Arc<CannonicalFork>,
    pub preload: Arc<DashSet<Address>>,
    pub provider: alloy::providers::RootProvider<BoxTransport>,
    pub provider_trace: alloy::providers::RootProvider<BoxTransport>,
    pub config: Config,
}

fn bigint_to_u256<'a>(
    value: Handle<'a, JsBigInt>,
    cx: &mut FunctionContext<'a>,
) -> neon::result::NeonResult<revm::primitives::U256> {
    let mut limbs: [u64; 4] = [0; 4];

    let value = value.read_digits_le(cx, &mut limbs);

    let o = revm::primitives::U256::from_limbs_slice(&limbs);
    if value.0 == JsBigInt::NEGATIVE {
        return cx.throw_error("Cannot convert negative bigint to U256");
    }
    std::result::Result::Ok(o)
}


const RTOKEN_SLOTS: [revm::primitives::U256; 20] = [
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d06fc")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d06fb")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0706")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0707")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0700")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0704")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("cb")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("cd")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("97")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("b53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d06ff")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d06fd")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0701")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0703")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0702")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d0705")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!("a7ce836d032b2bf62b7e2097a8e0a6d8aeb35405ad15271e96d3b0188a1d06fe")),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!(
        "7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3"
    )),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!(
        "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"
    )),
    revm::primitives::U256::from_be_slice(&revm::primitives::hex_literal::hex!(
        "10d6a54a4754c8869d6886b5f5d7fbfa5b4522237ea5c60d11bc4e7a1ff9390b"
    ))
];
impl ApplicationState {
    pub async fn create(
        config: Config,
        provider: alloy::providers::RootProvider<BoxTransport>,
        provider_trace: alloy::providers::RootProvider<BoxTransport>,
    ) -> eyre::Result<Self> {
        use eyre::WrapErr;

        let forking_from = config.start_block.map(|b|BlockId::from(b)).unwrap_or(BlockId::latest());
        log::debug!(target: LOGGER_TARGET_SYNC, "Creating cannonical fork. Start block {}", forking_from);
        let fork_block = provider
            .get_block(forking_from, alloy::rpc::types::BlockTransactionsKind::Hashes)
            .await
            .wrap_err("Failed to fetch latest block")?
            .wrap_err("Network contains no blocks")?;

        let out = Self {
            desynchronized: Arc::new(RwLock::new(false)),
            cannonical: Arc::new(CannonicalFork::new(
                provider.clone(),
                provider_trace.clone(),
                fork_block,
                config.clone(),
            )),
            preload: Arc::new(DashSet::new()),
            provider: provider.clone(),
            provider_trace: provider_trace.clone(),
            config,
        };

        Ok(out)
    }

    async fn fork_db<'a>(&self) -> eyre::Result<ActiveForkRef> {
        if *self.desynchronized.read().await == true {
            return Err(eyre::eyre!("Simulator desynchronized, please create new simulator"));
        }
        // Forks the current state of the cannonical fork into into a new EVM instance with
        // it's own CacheDB that does the following:
        // All read that are not found in the fork, get loaded from the cannonical fork, which will either pull
        // it from the provider if the value is not present, or give it back a value it has in it's own cache

        // The cannonical fork will keep track of all previously read values from simulations
        // The cannonical depends on a RPC it can trust to provice accurate storage slot diffs between each block

        let reader = self.cannonical.clone();
        let block_env = reader.block_env().await;

        let out = Arc::new(Mutex::new(ActiveFork {
            db: revm::db::CacheDB::<Forked>::new(Forked {
                cannonical: reader,
                env: block_env,
                seconds_per_block: U256::from(self.config.seconds_per_block),
            }),
            checkpoints: HashMap::new(),
        }));
        Ok(out)
    }

    pub async fn on_block(
        &self,
        latest_block: u64,
    ) -> eyre::Result<()> {
        if *self.desynchronized.read().await == true {
            return Err(eyre::eyre!("Simulator desynchronized, please create new simulator"));
        }
        
        let mut current_syncced_block = self.cannonical
            .get_current_block()
            .await?;

        if current_syncced_block >= latest_block {
            return Ok(());
        }

        let delta = latest_block - current_syncced_block;

        if delta > self.config.max_blocks_behind {
            log::info!(target: LOGGER_TARGET_SYNC, "Simulator is far behind ({delta} blocks), will reset state");
            self.cannonical.reset_state(latest_block).await?;
            return Ok(());
        }

        if delta > 1 {
            log::info!(target: LOGGER_TARGET_SYNC, "We are behind by {delta} blocks - synccing");
        }
        while current_syncced_block < latest_block {
            let block_number: u64 = current_syncced_block + 1;
            let block = self
                .provider
                .get_block(BlockId::Number(alloy::eips::BlockNumberOrTag::Number(block_number)), alloy::rpc::types::BlockTransactionsKind::Hashes)
                .await?
                .wrap_err(format!("Failed to fetch block {block_number} from RPC"))?;

            self.cannonical
                .apply_next_block(block)
                .await?;
            current_syncced_block = block_number;
        }
        Ok(())
    }
}

type ApplicationStateRef = Arc<ApplicationState>; 

struct ActiveFork {
    pub db: CacheDB<Forked>,
    pub checkpoints: HashMap<u32, CacheDB<Forked>>,
}
impl ActiveFork {
    pub fn checkpoint(&mut self) -> u32 {
        let id = self.checkpoints.len() as u32;
        self.checkpoints.insert(id as u32, self.db.clone());
        id
    }

    pub fn revert_to(&mut self, id: u32) -> eyre::Result<()> {
        self.db = self
            .checkpoints
            .remove(&id)
            .wrap_err(format!("Invalid checkpoint id {}", id))?
            .clone();
        eyre::Result::Ok(())
    }
}

type ActiveForkRef = Arc<Mutex<ActiveFork>>;

fn runtime<'a, C: Context<'a>>(cx: &mut C) -> NeonResult<&'static Runtime> {
    static RUNTIME: OnceCell<Runtime> = OnceCell::new();

    RUNTIME.get_or_try_init(|| {
        if std::env::var_os("RUST_LOG").is_none() {
            std::env::set_var("RUST_LOG", "slot0::api");
        }
        log::set_max_level(log::LevelFilter::Debug);
        pretty_env_logger::init();
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .global_queue_interval(8)
            .build()
            .or_else(|err| cx.throw_error(err.to_string()))
    })
}


async fn provider_from_string(url: &String) -> eyre::Result<alloy::providers::RootProvider<BoxTransport>> {
    let builder = alloy::providers::ProviderBuilder::new();
    let out = if url.starts_with("http") {
        builder.on_http(url.as_str().parse()?).boxed()
    } else if url.starts_with("ws") {
        builder.on_ws(alloy::providers::WsConnect::new(url.as_str())).await?.boxed()
    } else {
        return Err(eyre::eyre!("Invalid provider type"))
    };
    
    Ok(out)
}

async fn init_application_state(config: Config) -> eyre::Result<ApplicationStateRef> {
    use eyre::WrapErr;
    


    let provider = provider_from_string(&config.fork_url).await?;
    let trace_provider = provider_from_string(&config.trace_fork_url).await?;

    let app_state = ApplicationState::create(config, provider, trace_provider)
        .await
        .wrap_err("Failed to create application state")?;

    Ok(Arc::new(app_state))
}

fn to_neon<'a, T, C: Context<'a>>(cx: &mut C, e: eyre::Result<T>) -> neon::result::NeonResult<T> {
    match e {
        eyre::Result::Ok(v) => neon::result::NeonResult::Ok(v),
        eyre::Result::Err(e) => cx.throw_error(e.to_string()),
    }
}

fn js_value_to_address<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<Address> {
    use eyre::WrapErr;
    let down_casted = js_val.to_string(cx)?;
    let out = Address::from_str(down_casted.value(cx).as_str());
    to_neon(cx, out.wrap_err("Failed to parse"))
}

fn js_value_to_bytes<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<Vec<u8>> {
    use eyre::WrapErr;
    let down_casted = js_val.to_string(cx)?;
    let out = hex::decode(down_casted.value(cx).as_str()).wrap_err("Failed to parse");
    to_neon(cx, out)
}
fn js_value_to_uint256<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<revm::primitives::U256> {
    let down_casted: Handle<'a, JsBigInt> =
        js_val.downcast_or_throw::<JsBigInt, neon::context::FunctionContext<'a>>(cx)?;
    bigint_to_u256(down_casted, cx)
}

fn js_obj_tx_to_transaction<'a>(
    mut cx: FunctionContext<'a>,
    js_obj_tx: Handle<JsObject>,
) -> neon::result::NeonResult<TransactionRequest> {
    let from = js_obj_tx.get_value(&mut cx, "from")?;
    let to = js_obj_tx.get_value(&mut cx, "to")?;
    let data = js_obj_tx.get_value(&mut cx, "data")?;
    let value = js_obj_tx.get_value(&mut cx, "value")?;

    let from = js_value_to_address(&mut cx, from)?;
    let to = js_value_to_address(&mut cx, to)?;
    let data = js_value_to_bytes(&mut cx, data)?;
    let value = js_value_to_uint256(&mut cx, value)?;
    neon::result::NeonResult::Ok(TransactionRequest {
        from,
        to,
        data,
        value,
    })
}

struct LogTracer {
    pub memory_reads: HashMap<revm::primitives::Address, HashSet<revm::primitives::U256>>,
    pub logs: Vec<(
        revm::primitives::Address,
        Vec<revm::primitives::FixedBytes<32>>,
        Vec<u8>,
    )>,
}

impl LogTracer {
    pub fn new() -> Self {
        Self {
            logs: Vec::new(),
            memory_reads: HashMap::new(),
        }
    }

    pub fn collect_memory_reads(
        &self,
    ) -> Vec<(revm::primitives::Address, Vec<revm::primitives::U256>)> {
        self.memory_reads
            .iter()
            .map(|f| {
                (
                    f.0.clone(),
                    f.1.clone().into_iter().map(|v| v.clone()).collect(),
                )
            })
            .collect()
    }
}


impl revm::Inspector<&mut CacheDB<Forked>> for LogTracer {
    fn step_end(
        &mut self,
        interp: &mut revm::interpreter::Interpreter,
        _: &mut revm::EvmContext<&mut CacheDB<Forked>>,
    ) {
        let opcode: u8 = interp.current_opcode();
        if opcode == 0x54 {
            let address = interp.contract().target_address;
            let loc = interp.stack.peek(0).unwrap_or_default();
            if RTOKEN_SLOTS.contains(&loc) {
                return;
            }
            self.memory_reads
                .entry(address)
                .and_modify(|set| {
                    set.insert(loc);
                })
                .or_insert(HashSet::from_iter(vec![loc]));
            return;
        }
    }
    fn log(
        &mut self,
        _: &mut revm::interpreter::Interpreter,
        _context: &mut revm::EvmContext<&mut CacheDB<Forked>>,
        log: &revm::primitives::Log,
    ) {
        let address = log.address;
        let payload = log.data.data.to_vec();
        let topics = log.topics().to_vec();
        self.logs.push((address, topics, payload));
    }
}

fn instantiate_js_log<'a>(
    context: &mut TaskContext<'a>,
    addr: Address,
    topics: Vec<revm::primitives::FixedBytes<32>>,
    data: Vec<u8>,
) -> JsResult<'a, JsObject> {
    let js_log = context.empty_object();
    let log_address = context.string(addr.to_checksum(None));
    let log_data = context.string(hex::encode_prefixed(data));
    let log_topics = context.empty_array();
    let mut ii = 0;
    for topic in topics.iter() {
        let topic = context.string(hex::encode_prefixed(topic.0.as_slice()));
        log_topics.set(context, ii, topic)?;
        ii += 1;
    }
    js_log.set(context, "address", log_address)?;
    js_log.set(context, "data", log_data)?;
    js_log.set(context, "topics", log_topics)?;

    JsResult::Ok(js_log)
}
fn instantiate_run_tx<'a>(
    cx: &mut TaskContext<'a>,
    app_state: ApplicationStateRef,
    db: ActiveForkRef,
    commit: bool,
) -> JsResult<'a, JsFunction> {
    JsFunction::new(cx, move |mut fnctx: FunctionContext| {
        let on_step = fnctx.argument::<JsFunction>(1)?.root(&mut fnctx);
        
        let (deferred, promise) = fnctx.promise();
        let rt = runtime(&mut fnctx)?;
        let channel = fnctx.channel();

        let js_obj_tx: Handle<'_, JsObject> = fnctx.argument::<JsObject>(0)?;
        let tx = js_obj_tx_to_transaction(fnctx, js_obj_tx)?;

        log::trace!(target: LOGGER_TARGET_MAIN, "executing transaction: {:?}", tx);

        let app_state = app_state.clone();
        let db = db.clone();
        rt.spawn(async move {
            let app_state = app_state.clone();
            let out = {
                let out: Vec<Address> = app_state.preload.iter().map(|v| v.clone()).collect();
                app_state.preload.clear();
                out
            };
            if out.len() != 0 {
                let out: Vec<(Address, Vec<revm::primitives::U256>)> = out
                    .into_iter()
                    .map(|v: Address| {
                        (
                            v,
                            (0u64..10u64)
                                .map(|i| revm::primitives::U256::from(i))
                                .chain(RTOKEN_SLOTS.iter().map(|v| v.clone()).collect::<Vec<revm::primitives::U256>>())
                                .collect::<Vec<revm::primitives::U256>>()
                        )
                    })
                    .collect();
                log::info!(target: LOGGER_TARGET_MAIN, "Preloading {} addresses", out.len());

                let state = app_state.cannonical.clone();
                match state
                    .load_positions(out)
                    .await
                {
                    eyre::Result::Err(e) => {
                        log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}", e)
                    }
                    _ => {}
                }
            }

            let mut db = db.lock_owned().await;

            let active_fork = db.deref_mut();
            let checkpoint_to_revert_to = if commit == false {
                Some(active_fork.checkpoint())
            } else {
                None
            };

            let result = {

                let tracer = LogTracer::new();
                let env = active_fork.db.db.env.clone();
                log::debug!(target: LOGGER_TARGET_MAIN, "Simulator env {:?}", env);
                let gas_limit = env.gas_limit;
                let inner_db =  active_fork.db.borrow_mut();
                let out: Evm<'_, LogTracer, &mut CacheDB<Forked>> = revm::Evm::builder()
                    .with_block_env(env)
                    .with_db(inner_db)
                    .with_external_context(tracer)
                    .append_handler_register(revm::inspector_handle_register)
                    .with_spec_id(revm::primitives::SpecId::CANCUN)
                    .build();

                let start_time = std::time::Instant::now();
                let mut simulation_runner = out
                    .modify()
                    .modify_tx_env(|env| {
                        let tx = tx.clone();
                        env.caller = tx.from;
                        if !tx.data.is_empty() {
                            env.data = revm::primitives::Bytes::from(tx.data);
                        }
                        env.value = tx.value;
                        env.gas_limit = gas_limit.to();
                        env.gas_price = revm::primitives::U256::from(1u64);
                        env.transact_to = TransactTo::Call(tx.to);
                    })
                    .build();

                let result = simulation_runner.transact_commit();

                let end_time = std::time::Instant::now();
                log::info!(target: LOGGER_TARGET_MAIN, "Simulation took {:?}", end_time - start_time);

                match result {
                    std::result::Result::Err(e) => {
                        log::error!(target: LOGGER_TARGET_MAIN, "Failed to revert to checkpoint: {}", e);
                        deferred.settle_with(&channel, move |mut cx: TaskContext<'_>| {
                            let err = cx.error(e.to_string())?;
                            cx.throw::<_, Handle<'_, JsUndefined>>(err)
                        });
                        return;
                    }
                    std::result::Result::Ok(v) => {
                        match &v {
                            ExecutionResult::Revert { gas_used, output } => {
                                log::debug!(target: LOGGER_TARGET_MAIN, "Simulation reverted with error {} gas used {}", output, gas_used);

                            }
                            ExecutionResult::Halt { reason, gas_used } => {
                                log::debug!(target: LOGGER_TARGET_MAIN, "Simulation halted with reason {:?} gas used {}", reason, gas_used);
                            }
                            _ => {}
                        };
                        let logs = simulation_runner.context.external.logs.clone();
                        (
                            v,
                            logs,
                            simulation_runner.context.external.collect_memory_reads(),
                        )
                    }
                }
            };

            match checkpoint_to_revert_to.map(|id| db.revert_to(id)).transpose() {
                Err(e) => {
                    log::error!(target: LOGGER_TARGET_MAIN, "Failed to revert to checkpoint: {}", e);
                    deferred.settle_with(&channel, move |mut cx: TaskContext<'_>|  {
                        let err = cx.error(e.to_string())?;
                        cx.throw::<_, Handle<'_, JsUndefined>>(err)
                    });
                    return;
                }
                _ => {}
            };

            let (res, logs, reads) = result;


            let res = (
                res.output().unwrap_or_default().to_vec(),
                res.gas_used(),
                res.into_logs(),
                reads,
            );

            deferred.settle_with(&channel, move |mut cx: TaskContext<'_>| {
                let on_step = on_step.into_inner(&mut cx);
                for (address, topics, payload) in logs {
                    let js_log = instantiate_js_log(&mut cx, address, topics, payload).unwrap();
                    
                    on_step
                        .call_with(&mut cx)
                        .arg(js_log)
                        .apply::<JsValue, TaskContext>(&mut cx)?;
                }
                let obj = cx.empty_object();

                let js_logs = cx.empty_array();

                let result_bytes = res.0;
                let comitted = res.1 != 0u64;
                let gas_used = res.1;
                let logs = res.2;
                let receipt: Result<Handle<'_, JsObject>, result::Throw> = {
                    let total_gas_spent = cx.number(gas_used as f64);

                    obj.set(&mut cx, "logs", js_logs)?;
                    obj.set(&mut cx, "gasUsed", total_gas_spent)?;
                    obj.set(&mut cx, "cumulativeGasUsed", total_gas_spent)?;

                    let mut i = 0;
                    for log in logs {
                        let js_log: Handle<'_, JsObject> = instantiate_js_log(&mut cx, log.address, log.topics().to_vec(), log.data.data.to_vec())?;
                        js_logs.set(&mut cx, i, js_log)?;
                        i += 1;
                    }
                    let comitted_value = if comitted {
                        cx.number(1.0)
                    } else {
                        cx.number(0.0)
                    };
                    obj.set(&mut cx, "status", comitted_value)?;
                    JsResult::Ok(obj)
                };
                let exec_result: Result<Handle<'_, JsObject>, result::Throw> = {
                    let cx = &mut cx;
                    let result_bytes = cx.string(hex::encode_prefixed(result_bytes));
                    let obj = cx.empty_object();
                    obj.set(cx, "returnValue", result_bytes)?;
                    if !comitted {
                        let cx = cx;
                        let err_obj = cx.empty_object();
                        let error_desc = cx.string("EVM Error");
                        let error_type = cx.string("EVM Error");
                        err_obj.set(cx, "error", error_type)?;
                        err_obj.set(cx, "errorType", error_desc)?;
                        obj.set(cx, "exceptionError", err_obj)?;
                    }
                    JsResult::Ok(obj)
                };
                let access_list = cx.empty_array();
                {
                    let mut i = 0u32;
                    let cx = &mut cx;
                    for (addr, reads) in res.3 {
                        let out_js = cx.empty_array();
                        let addr = cx.string(addr.to_checksum(None));
                        let reads_js = cx.empty_array();
                        let mut ii = 0u32;

                        for read in reads {
                            let slot = cx
                                .string(hex::encode_prefixed(read.to_be_bytes::<32>().as_slice()));
                            reads_js.set(cx, ii, slot)?;
                            ii += 1;
                        }
                        out_js.set(cx, 0, addr)?;
                        out_js.set(cx, 1, reads_js)?;
                        access_list.set(cx, i, out_js)?;
                        i += 1;
                    }
                }

                let execution = {
                    let cx = &mut cx;
                    let total_gas_spent = cx.number(gas_used as f64);
                    let obj = cx.empty_object();
                    obj.set(cx, "execResult", exec_result?)?;
                    obj.set(cx, "gasUsed", total_gas_spent)?;
                    obj.set(cx, "receipt", receipt?)?;
                    obj.set(cx, "accessList", access_list)?;

                    JsResult::Ok(obj)
                };

                execution
            });
        });

        return JsResult::Ok(promise);
    })
}

fn instantiate_checkpoint_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let id = forked.checkpoint();
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                JsResult::Ok(cx.number(id))
            });
        });

        JsResult::Ok(promise)
    })
}
fn instantiate_revert_to_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();
        let id = cx.argument::<JsNumber>(0)?.value(&mut cx) as u32;
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let res = forked.revert_to(id);
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                match res {
                    Err(e) => return cx.throw_error(e.to_string()),
                    _ => {}
                };
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_mine_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();
        let blocks = cx.argument::<JsNumber>(0)?.value(&mut cx);

        if blocks < 0.0 {
            return cx.throw_error("Cannot mine a negative number of blocks");
        }
        if blocks.floor() != blocks {
            return cx.throw_error("Cannot mine a non-integer number of blocks");
        }
        let blocks = blocks as u64;
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            forked.db.db.mine(blocks);
            let block_number = forked.db.db.get_block_number();

            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                NeonResult::Ok(cx.number(block_number as f64))
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_get_block_number_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let forked = db.lock().await;
            let block_number = forked.db.db.get_block_number();
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                NeonResult::Ok(cx.number(block_number as f64))
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_get_timestamp_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let forked = db.lock().await;
            let timestamp = forked.db.db.get_timestamp();
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                NeonResult::Ok(cx.number(timestamp as f64))
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_set_timestamp_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();
        let timestamp = cx.argument::<JsNumber>(0)?.value(&mut cx);
        if timestamp < 0.0 {
            return cx.throw_error("Cannot set negative timestamp");
        }
        if timestamp.floor() != timestamp {
            return cx.throw_error("Cannot set non-integer timestamp");
        }
        let timestamp = timestamp as u64;
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;

            forked.db.db.set_timestamp(timestamp);
            let timestamp = forked.db.db.get_timestamp();
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                NeonResult::Ok(cx.number(timestamp as f64))
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_set_balance_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;
        let new_balance = bigint_to_u256(cx.argument::<JsBigInt>(1)?, &mut cx)?;
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let acc = forked
                .db
                .load_account(address)
                .map(|v| v.info.balance = new_balance);

            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                to_neon(&mut cx, acc)?;
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_get_balance_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.lock().await;
            let acc = forked.db.load_account(address).map(|v| v.info.balance);
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                let acc  = to_neon(&mut cx, acc)?;
                
                let n = acc.as_limbs();
                let out = JsBigInt::from_digits_le(&mut cx,neon::types::bigint::Sign::Positive, n.as_slice()); 
                NeonResult::Ok(out)
            })
        });

        JsResult::Ok(promise)
    })
}


fn instantiate_get_storage_at_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;
        let position = cx.argument(1)?;
        let position = js_value_to_uint256(&mut cx, position)?;

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let forked = db.lock().await;
            let value = forked.db.storage_ref(address, position);
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                let acc  = to_neon(&mut cx, value)?;
                
                let n = acc.as_limbs();
                let out = JsBigInt::from_digits_le(&mut cx,neon::types::bigint::Sign::Positive, n.as_slice()); 
                NeonResult::Ok(out)
            })
        });

        JsResult::Ok(promise)
    })
}

fn init_preload_addr_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;
        let rt = runtime(&mut cx)?;
        rt.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            match forked.db.load_account(address) {
                eyre::Result::Err(e) => {
                    log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}: {}", address, e)
                }
                _ => {}
            }
        });

        JsResult::Ok(cx.undefined())
    })
}

fn create_preload_fn<'a>(
    app_state: ApplicationStateRef,
    cx: &mut TaskContext<'a>,
) -> JsResult<'a, JsFunction> {
    let db = app_state.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();

        let address = cx.argument(0)?;

        let address = js_value_to_address(&mut cx, address)?;
        let rt: &Runtime = runtime(&mut cx)?;

        rt.spawn(async move {
            let db = db.clone();
            db.preload.insert(address);
        });

        JsResult::Ok(cx.undefined())
    })
}

fn instantiate_set_storage_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db: ActiveForkRef = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;

        let key = bigint_to_u256(cx.argument::<JsBigInt>(1)?, &mut cx)?;
        let new_value = bigint_to_u256(cx.argument::<JsBigInt>(2)?, &mut cx)?;

        log::info!(target: LOGGER_TARGET_MAIN, "set_storage({}, {}, {})", address, key, new_value);

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;

            let acc = forked.db.insert_account_storage(address, key, new_value);
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                to_neon(&mut cx, acc)?;
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_set_code_fn<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    use eyre::WrapErr;
    let db: ActiveForkRef = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;
        let code = cx.argument::<JsString>(1)?.value(&mut cx);
        log::info!(target: LOGGER_TARGET_MAIN, "Setting code for {}", address);

        let code = to_neon(
            &mut cx,
            revm::primitives::Bytes::from_hex(code).wrap_err("Invalid hex"),
        )?;
        let code = if code.len() == 0 {
            None
        } else {
            let code = revm::primitives::Bytecode::new_raw(code);

            Some(code)
        };

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let acc = forked.db.load_account(address).map(|v| v.info.code = code);
            log::info!(target: LOGGER_TARGET_MAIN, "Account code set");
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                to_neon(&mut cx, acc)?;
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn init_fork_js_obj<'a>(
    cx: &mut TaskContext<'a>, // evm: Arc<Evm<'a,(), CacheDB<Forked>>>,
    app_state: ApplicationStateRef,
    db: ActiveForkRef,
) -> JsResult<'a, JsObject> {
    let db = db.clone();
    let obj = cx.empty_object();

    let fn_ref = { instantiate_run_tx(cx, app_state.clone(), db.clone(), true)? };
    obj.set(cx, "commitTx", fn_ref)?;

    let fn_ref = { instantiate_run_tx(cx, app_state.clone(), db.clone(), false)? };
    obj.set(cx, "simulateTx", fn_ref)?;

    let fork_fn = { instantiate_checkpoint_fn(cx, db.clone())? };
    obj.set(cx, "checkpoint", fork_fn)?;

    let revert_fn = { instantiate_revert_to_fn(cx, db.clone())? };
    obj.set(cx, "revertTo", revert_fn)?;

    let mine_fn = { instantiate_mine_fn(cx, db.clone())? };
    obj.set(cx, "mine", mine_fn)?;

    let get_block_number_fn = { instantiate_get_block_number_fn(cx, db.clone())? };
    obj.set(cx, "getBlockNumber", get_block_number_fn)?;

    let get_timestamp_fn = { instantiate_get_timestamp_fn(cx, db.clone())? };
    obj.set(cx, "getTimestamp", get_timestamp_fn)?;

    let set_timestamp_fn = { instantiate_set_timestamp_fn(cx, db.clone())? };
    obj.set(cx, "setTimestamp", set_timestamp_fn)?;

    let set_balance_fn = { instantiate_set_balance_fn(cx, db.clone())? };
    obj.set(cx, "setBalance", set_balance_fn)?;

    let get_balance_fn = { instantiate_get_balance_fn(cx, db.clone())? };
    obj.set(cx, "getBalance", get_balance_fn)?;
    
    let get_storage_fn = { instantiate_get_storage_at_fn(cx, db.clone())? };
    obj.set(cx, "getStorageAt", get_storage_fn)?;

    let set_storage_fn: Handle<'_, JsFunction> = { instantiate_set_storage_fn(cx, db.clone())? };
    obj.set(cx, "setContractStorage", set_storage_fn)?;

    let set_code_fn = { instantiate_set_code_fn(cx, db.clone())? };
    obj.set(cx, "setAccountCode", set_code_fn)?;

    let preload_fn = { init_preload_addr_fn(cx, db.clone())? };
    obj.set(cx, "preload", preload_fn)?;

    JsResult::Ok(obj)
}

fn create_fork_fn<'a>(
    state: ApplicationStateRef,
    cx: &mut TaskContext<'a>,
) -> JsResult<'a, JsFunction> {
    JsFunction::new(cx, move |mut cx: FunctionContext<'_>| {
        let app_state = state.clone();
        let rt = runtime(&mut cx)?;
        let channel = cx.channel();
        let (deferred, promise) = cx.promise();

        rt.spawn(async move {
            let db = {
                let app_state = app_state.clone();
                app_state.fork_db().await
            };

            let app_state = app_state.clone();
            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                match db {
                    eyre::Result::Ok(db) => {
                        init_fork_js_obj(&mut cx, app_state.clone(), db.clone())
                    }
                    Err(e) => {
                        let err = cx.error(e.to_string())?;
                        cx.throw(err)
                    }
                }
            });
        });

        JsResult::Ok(promise)
    })
}

fn create_on_block_fn<'a>(
    state: ApplicationStateRef,
    cx: &mut TaskContext<'a>,
) -> JsResult<'a, JsFunction> {
    JsFunction::new(cx, move |mut cx| {
        let rt = runtime(&mut cx)?;
        let channel = cx.channel();
        let (deferred, promise) = cx.promise();
        let block_number = cx.argument::<JsNumber>(0)?.value(&mut cx) as u64;
        let state = state.clone();
        rt.spawn(async move {
            let state = state.clone();

            match state.on_block(block_number).await {
                Err(e) => {
                    {
                        *state.desynchronized.write().await = true;
                    }
                    deferred.settle_with(&channel, move |mut cx| {
                        let err = cx.error(e.to_string())?;
                        cx.throw::<_, Handle<'_, JsUndefined>>(err)
                    });
                    return;
                }
                _ => {
                    
                }
            };
            deferred.settle_with(&channel, move |mut cx| JsResult::Ok(cx.boolean(true)));
        });
        JsResult::Ok(promise)
    })
}

fn create_simulator(mut cx: FunctionContext) -> JsResult<JsPromise> {
    let fork_url = cx.argument::<JsString>(0)?.value(&mut cx);
    let link: Result<LinkType, eyre::Error> = LinkType::from_str(cx.argument::<JsString>(1)?.value(&mut cx).as_str());
    let (seconds_per_block, start_block, trace_fork_url, max_blocks_behind) = match cx.argument_opt(2) {
        None => (12u64, None, fork_url.clone(), 50u64),
        Some(obj) => {
            let opts = obj.downcast_or_throw::<JsObject, FunctionContext>(&mut cx)?;
            let seconds_pr_block = opts.get_opt(&mut cx, "secondsPerBlock")?.unwrap_or(cx.number(12u32)).downcast_or_throw::<JsNumber, FunctionContext>(&mut cx)?.value(&mut cx) as u64;
            let fork_block_number = if let Some(num) = opts.get_opt::<JsNumber, FunctionContext, _>(&mut cx, "forkBlock")? {
                let num = num.downcast_or_throw::<JsNumber, FunctionContext>(&mut cx)?.value(&mut cx) as u64;
                if num == 0 {
                    None
                } else {
                    Some(num)
                }
            } else {
                None
            };
            let trace_provider_url = if let Some(val) = opts.get_opt::<JsString, FunctionContext, _>(&mut cx, "traceProvider")? {
                val.downcast_or_throw::<JsString, FunctionContext>(&mut cx)?.value(&mut cx)
            } else {
                fork_url.clone()
            };
            let max_blocks_behind = opts.get_opt(&mut cx, "maxBlocksBehind")?.unwrap_or(cx.number(50u32)).downcast_or_throw::<JsNumber, FunctionContext>(&mut cx)?.value(&mut cx) as u64;
            
            (seconds_pr_block, fork_block_number, trace_provider_url, max_blocks_behind)
        }
    };

    let config = Config {
        fork_url,
        trace_fork_url,
        seconds_per_block,
        link_type: to_neon(&mut cx, link)?,
        start_block,
        max_blocks_behind,
    };

    log::info!(target: LOGGER_TARGET_MAIN, "Initializing simulator with config {:?}", config);

    let rt: &Runtime = runtime(&mut cx)?;
    let (deferred, promise) = cx.promise();
    let channel = cx.channel();
    rt.spawn(async move {
        let app_state = init_application_state(config.clone()).await;
        log::info!(target: LOGGER_TARGET_MAIN, "instantiated simulator");

        deferred.settle_with(&channel, move |mut cx| {
            let app_state = match app_state {
                eyre::Result::Ok(v) => v,
                Err(e) => {
                    return cx.throw_error(e.to_string())
                }
            };
            let api_obj = cx.empty_object();

            let fork_fn = { create_fork_fn(app_state.clone(), cx.borrow_mut())? };
            api_obj.set(&mut cx, "fork", fork_fn)?;

            let onblock_fn = { create_on_block_fn(app_state.clone(), cx.borrow_mut())? };
            api_obj.set(&mut cx, "onBlock", onblock_fn)?;

            let preload_fn = { create_preload_fn(app_state.clone(), cx.borrow_mut())? };
            api_obj.set(&mut cx, "preload", preload_fn)?;

            JsResult::Ok(api_obj)
        });
    });

    JsResult::Ok(promise)
}

#[neon::main]
pub fn main(mut cx: ModuleContext) -> NeonResult<()> {
    runtime(&mut cx)?;
    cx.export_function("createSimulator", create_simulator)?;
    NeonResult::Ok(())
}
