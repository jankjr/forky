use alloy::eips::BlockId;
use alloy::hex::FromHex;
use alloy::rpc::types::AccessList;

use alloy::rpc::types::AccessListItem;
use alloy_provider::DynProvider;
use alloy_provider::Provider;
use neon::prelude::Context;
use revm::context::result::ExecutionResult;
use revm::context::TxEnv;
use revm::interpreter::interpreter_types::InputsTr;
use revm::interpreter::interpreter_types::Jumps;
use revm::interpreter::Interpreter;
use revm::interpreter::InterpreterTypes;
use revm::primitives::Bytes;
use revm::primitives::FixedBytes;
use revm::primitives::Log;
use revm::state::Bytecode;
use revm::ExecuteEvm;
use revm::InspectEvm;
use revm::MainBuilder;
use revm::MainContext;
use scc::hash_set::HashSet;
use eyre::ContextCompat;
use eyre::Ok;
use neon::handle::Handle;
use neon::object::Object;
use neon::prelude::{FunctionContext, ModuleContext, TaskContext};
use neon::result::{self, JsResult, NeonResult};
use neon::types::JsArray;
use neon::types::JsUndefined;
use neon::types::Value;
use neon::types::{JsBigInt, JsFunction, JsNumber, JsObject, JsPromise, JsString, JsValue};

use revm::database::CacheDB;
use revm::primitives::U256;
use revm::primitives::{hex};
use revm::DatabaseRef;
use revm::{primitives::Address};

use once_cell::sync::OnceCell;
use utils::provider_from_string;

use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::{str::FromStr, sync::Arc};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
pub mod utils;
pub mod cannonical;
pub mod config;

use cannonical::{CannonicalFork, Forked};
use config::{Config, LinkType};
// pub mod errors;

pub const LOGGER_TARGET_MAIN: &str = "forky";
pub const LOGGER_TARGET_SYNC: &str = "forky::sync";
pub const LOGGER_TARGET_LOADS: &str = "forky::sync::loads";
pub const LOGGER_TARGET_API: &str = "forky::api";
pub const LOGGER_TARGET_SIMULATION: &str = "forky::sim";

#[derive(Debug, Clone)]
pub struct TransactionRequest {
    pub from: Address,
    pub to: Address,
    pub data: Vec<u8>,
    pub value: U256,
    pub gas_price: Option<U256>,
    pub gas_priority_fee: Option<U256>,
    pub max_fee_per_gas: Option<U256>,
    pub gas_limit: Option<U256>,

    pub access_list: Option<AccessList>
}
pub struct ApplicationState {
    pub cannonical: Arc<CannonicalFork>,
    pub preload: Arc<HashSet<Address>>,
    pub provider: DynProvider,
    pub config: Config,
}

fn bigint_to_u256<'a>(
    value: Handle<'a, JsBigInt>,
    cx: &mut FunctionContext<'a>,
) -> neon::result::NeonResult<revm::primitives::U256> {
    let decimal = value.to_string(cx)?.value(cx);
    let o = match U256::from_str_radix(&decimal, 10) {
        std::result::Result::Ok(v) => v,
        Err(e) => {
            return cx.throw_error(e.to_string());
        }
    };


    std::result::Result::Ok(o)
}


const DEFAULT_SLOTS: [revm::primitives::U256; 3] = [
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
        provider: DynProvider,
    ) -> eyre::Result<Self> {
        use eyre::WrapErr;

        let forking_from = config.start_block.map(|b|BlockId::from(b)).unwrap_or(BlockId::latest());
        log::debug!(target: LOGGER_TARGET_SYNC, "Creating cannonical fork. Start block {}", forking_from);
        let fork_block = provider
            .get_block(forking_from)
            .await
            .wrap_err("Failed to fetch latest block")?
            .wrap_err("Network contains no blocks")?;

        let out = Self {
            cannonical: Arc::new(CannonicalFork::new(
                provider.clone(),
                fork_block,
                config.clone(),
            )),
            preload: Arc::new(HashSet::new()),
            provider: provider.clone(),
            config,
        };

        Ok(out)
    }

    async fn fork_db<'a>(&self) -> eyre::Result<ActiveForkRef> {
        // Forks the current state of the cannonical fork into into a new EVM instance with
        // it's own CacheDB that does the following:
        // All read that are not found in the fork, get loaded from the cannonical fork, which will either pull
        // it from the provider if the value is not present, or give it back a value it has in it's own cache

        // The cannonical fork will keep track of all previously read values from simulations
        // The cannonical depends on a RPC it can trust to provice accurate storage slot diffs between each block

        let reader = self.cannonical.clone();
        let block_env = reader.block_env();

        let out = Arc::new(Mutex::new(ActiveFork {
            db: CacheDB::<Forked>::new(Forked {
                cannonical: reader,
                env: block_env,
                seconds_per_block: U256::from(self.config.seconds_per_block)
                // dry_run: false,
            }),
            checkpoints: HashMap::new(),
        }));
        Ok(out)
    }

    pub async fn load_positions(
        &self,
        addresses: Vec<Address>,
    ) -> eyre::Result<()> {
        let run = addresses;
        let next = match self.cannonical.load_positions(
            run
        ).await {
            eyre::Result::Ok(v) => v,
            Err(e) => {
                log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}", e);
                return Err(e);
            }
        };
        if next.is_empty() {
            return Ok(());
        }
        let mut run = next;
        let s = self.cannonical.clone();
        for _ in 0..2 {
            log::info!(target: LOGGER_TARGET_MAIN, "Loading {} more addresses", run.len());
            let next = match s.load_positions(
                run
            ).await {
                eyre::Result::Ok(v) => v,
                Err(e) => {
                    log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}", e);
                    return Err(e);
                }
            };
            if next.len() < 10 {
                break;
            }
            run = next;
        }
        Ok(())

        
    }

    pub async fn on_block(
        &self,
        latest_block: u64,
    ) -> eyre::Result<()> {
        let mut current_syncced_block = self.cannonical.get_current_block();

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
                .get_block(BlockId::Number(alloy::eips::BlockNumberOrTag::Number(block_number)))
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
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .event_interval(8)
            .max_io_events_per_tick(4096)
            .build()
            .or_else(|err| cx.throw_error(err.to_string()))
    })
}

async fn init_application_state(config: Config) -> eyre::Result<ApplicationStateRef> {
    use eyre::WrapErr;
    

    let provider = provider_from_string(&config.fork_url).await?;

    let app_state = ApplicationState::create(config, provider)
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
fn js_value_to_b256<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<revm::primitives::B256> {
    use eyre::WrapErr;
    let down_casted: Handle<'a, JsString> =
        js_val.downcast_or_throw::<JsString, neon::context::FunctionContext<'a>>(cx)?;
    let value = down_casted.value(cx);
    let out = revm::primitives::B256::from_str(
        &value
    );
    
    to_neon(cx, out.wrap_err("Failed to parse"))
}
fn get_optional_bigint<'a>(
    js_val: NeonResult<Option<Handle<'a, JsBigInt>>>,
    cx: &mut neon::context::FunctionContext<'a>,
) -> neon::result::NeonResult<Option<revm::primitives::U256>> {
    match js_val {
        eyre::Result::Ok(Some(v)) => match bigint_to_u256(v, cx) {
            eyre::Result::Ok(v) => NeonResult::Ok(Some(v)),
            Err(e) => cx.throw_error(e.to_string()),
        },
        eyre::Result::Ok(None) => NeonResult::Ok(None),
        Err(e) => cx.throw_error(e.to_string()),
    }
}

fn js_obj_tx_to_transaction<'a>(
    mut cx: FunctionContext<'a>,
    js_obj_tx: Handle<JsObject>,
) -> neon::result::NeonResult<TransactionRequest> {
    let from = js_obj_tx.get_value(&mut cx, "from")?;
    let to = js_obj_tx.get_value(&mut cx, "to")?;
    let data = js_obj_tx.get_value(&mut cx, "data")?;
    let gas_price = get_optional_bigint(js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "gasPrice"), &mut cx)?;
    let gas_limit = get_optional_bigint(js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "gasLimit"), &mut cx)?;

    let gas_priority_fee = get_optional_bigint(js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "maxPriorityFeePerGas"), &mut cx)?;
    let max_fee_per_gas = get_optional_bigint(js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "maxFeePerGas"), &mut cx)?;
    let value = get_optional_bigint(js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "value"), &mut cx)?.unwrap_or_default();

    let access_list = if let Some(v) = js_obj_tx.get_opt::<JsArray, _, _>(&mut cx, "accessList")? {
        let mut out = Vec::new();
        let length = v.len(&mut cx);
        for i in 0..length {
            let item: Handle<JsObject> = v.get(&mut cx, i)?;
            let address = item.get(&mut cx, "address")?;
            let address = js_value_to_address(&mut cx, address)?;
            let storage_keys = item.get::<JsArray, _, _>(&mut cx, "storageKeys").map(|v|{
                let mut out = Vec::new();
                let length = v.len(&mut cx);
                for i in 0..length {
                    let item = v.get::<>(&mut cx, i)?;
                    let item = js_value_to_b256(&mut cx, item)?;
                    out.push(item);
                }
                neon::result::NeonResult::Ok(out)
            })??;
            
            out.push(AccessListItem {
                address,
                storage_keys: storage_keys
            });
        }
        Some(AccessList(out))
    } else {
        None
    };
    let from = js_value_to_address(&mut cx, from)?;
    let to = js_value_to_address(&mut cx, to)?;
    let data = js_value_to_bytes(&mut cx, data)?;
    neon::result::NeonResult::Ok(TransactionRequest {
        from,
        to,
        data,
        value,
        gas_limit,
        gas_price,
        gas_priority_fee,
        max_fee_per_gas,
        access_list
    })
}

struct LogTracer {
    pub memory_reads: HashMap<Address, std::collections::HashSet<U256>>,
    pub logs: Vec<(
        Address,
        Vec<FixedBytes<32>>,
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
                    f.1.iter().map(|v| v.clone()).collect(),
                )
            })
            .collect()
    }
}


impl<CTX, INTR: InterpreterTypes> revm::Inspector<CTX, INTR> for LogTracer {
    #[inline]
    fn log(
        &mut self,
        _: &mut Interpreter<INTR>,
        _context: &mut CTX,
        log: alloy::primitives::Log
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
    topics: Vec<FixedBytes<32>>,
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

        let app_state = app_state.clone();
        let db = db.clone();
        rt.spawn(async move {
            let app_state = app_state.clone();
            let out = {
                let mut out: Vec<Address> = Vec::new();
                app_state.preload.scan_async(|v|out.push(v.clone())).await;
                app_state.preload.clear();
                out
            };
            if out.len() != 0 {
                match app_state
                    .load_positions(out)
                    .await
                {
                    eyre::Result::Err(e) => {
                        log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}", e)
                    }
                    _ => {}
                }
            }

            if let Some(list) = &tx.access_list {
                if let Err(e) = app_state.cannonical.load_access_list(list).await {
                    log::error!(target: LOGGER_TARGET_MAIN, "Failed to load access list {}", e);
                };
            };

            let mut db = db.lock_owned().await;

            let active_fork = db.deref_mut();
            let checkpoint_to_revert_to = if commit == false {
                Some(active_fork.checkpoint())
            } else {
                None
            };

            let result = {
                let start_time = std::time::Instant::now();

                

                let block_env = active_fork.db.db.env.clone();
                let gas_limit = tx.gas_limit.unwrap_or(U256::from(block_env.gas_limit));
                let inner_db =  active_fork.db.borrow_mut();
                let mut simulation_runner = revm::Context::mainnet()
                    .with_block(block_env.clone())
                    .with_db(inner_db)
                    .modify_cfg_chained(|f|{
                        f.memory_limit = 1024 * 1024 * 64;
                        f.disable_eip3607 = true;
                        f.disable_balance_check = !commit;
                    })
                    .build_mainnet_with_inspector(LogTracer::new());


                let tx = TxEnv::builder()
                .caller(tx.from)
                .to(tx.to)
                .data(Bytes::from(tx.data))
                .value(tx.value)
                .gas_limit(gas_limit.to())
                .gas_price(tx.gas_price.map(|i|i.to()).unwrap_or(block_env.basefee as u128 + 1u128))
                .build()
                .unwrap();

                let result = simulation_runner.inspect(tx.clone(), LogTracer::new());

                let end_time = std::time::Instant::now();
                log::info!(target: LOGGER_TARGET_MAIN, "Simulation took {:?}", end_time - start_time);

                match result {
                    Result::Err(e) => {
                        log::error!(target: LOGGER_TARGET_MAIN, "EVM error: {}", e);
                        deferred.settle_with(&channel, move |mut cx: TaskContext<'_>| {
                            let err = cx.error(e.to_string())?;
                            cx.throw::<_, Handle<'_, JsUndefined>>(err)
                        });
                        return;
                    }
                    Result::Ok(v) => {
                        match &v.result {
                            ExecutionResult::Revert { gas_used, output } => {
                                log::debug!(target: LOGGER_TARGET_MAIN, "Simulation reverted with error {} gas used {}", output, gas_used);

                            }
                            ExecutionResult::Halt { reason, gas_used } => {
                                log::debug!(target: LOGGER_TARGET_MAIN, "Simulation halted with reason {:?} gas used {}", reason, gas_used);
                            }
                            _ => {}
                        };
                        let tracer = simulation_runner.into_inspector();
                        (
                            v,
                            tracer.logs.clone(),
                            tracer.collect_memory_reads(),
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
                res.result.output().unwrap_or_default().to_vec(),
                res.result.gas_used(),
                res.result.into_logs(),
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
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
    cx: &mut TaskContext<'a>,
    db: ActiveForkRef,
) -> JsResult<'a, JsFunction> {
    let db = db.clone();
    JsFunction::new(cx, move |mut cx: FunctionContext| {
        let db = db.clone();
        let channel = cx.channel();

        let address = cx.argument(0)?;
        let address = js_value_to_address(&mut cx, address)?;
        let new_balance = bigint_to_u256(cx.argument::<JsBigInt>(1)?, &mut cx)?;

        log::debug!(target: LOGGER_TARGET_MAIN, "setBalance({}, {})", address, new_balance);
        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let acc = forked
                .db
                .load_account(address)
                .map(|v| v.info.balance = new_balance)
                .map_err(|i|i.into());

            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                to_neon(&mut cx, acc)?;
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_get_balance_fn<'a>(
    cx: &mut TaskContext<'a>,
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
            let acc = forked.db.load_account(address).map(|v| v.info.balance).map_err(|i|i.into());
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
    cx: &mut TaskContext<'a>,
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
            let value = forked.db.storage_ref(address, position).map_err(|i|i.into());
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
    cx: &mut TaskContext<'a>,
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

        let addresses = cx.argument::<JsArray>(0)?;

        let length = addresses.len(&mut cx);
        let mut addreses = Vec::with_capacity(length as usize);
        for i in 0..length {
            let address = addresses.get(&mut cx, i)?;
            let address = js_value_to_address(&mut cx, address)?;
            addreses.push(address);
        }
        let rt: &Runtime = runtime(&mut cx)?;
        let channel = cx.channel();
        let (deferred, p) = cx.promise();
        rt.spawn(async move {
            let db = db.clone();
            
            if addreses.len() == 1 {
                if let Err(_) = db.preload.insert_async(addreses[0]).await {}
                
                deferred.settle_with(&channel, |mut cx|{
                    JsResult::Ok(cx.undefined())
                });
            } else if addreses.len() > 1 {
                let res = db.load_positions(
                    addreses
                ).await;
                deferred.settle_with(&channel, |mut cx|{
                    match res {
                        eyre::Result::Ok(_) => {
                            JsResult::Ok(cx.undefined())
                        }
                        eyre::Result::Err(e) => {
                            log::error!(target: LOGGER_TARGET_MAIN, "Failed to preload {}", e);
                            let err = cx.error(e.to_string())?;
                            cx.throw::<_, Handle<'_, JsUndefined>>(err)
                        }
                    }
                });
            }
        });
        JsResult::Ok(p)
    })
}



fn instantiate_set_storage_fn<'a>(
    cx: &mut TaskContext<'a>,
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
            if let Err(_) = forked.db.db.cannonical.get_account_data(&address).await {}
            let acc = forked.db.insert_account_storage(address, key, new_value).map_err(|i|i.into());

            deferred.settle_with(&channel, move |mut cx: TaskContext| {
                to_neon(&mut cx, acc)?;
                NeonResult::Ok(cx.undefined())
            })
        });

        JsResult::Ok(promise)
    })
}

fn instantiate_set_code_fn<'a>(
    cx: &mut TaskContext<'a>,
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
            let code = Bytecode::new_raw(code);

            Some(code)
        };

        let (deferred, promise) = cx.promise();
        runtime(&mut cx)?.spawn(async move {
            let mut forked = db.clone().lock_owned().await;
            let acc = forked.db.load_account(address).map(|v| v.info.code = code).map_err(|i|i.into());
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
    cx: &mut TaskContext<'a>,
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
    let (seconds_per_block, start_block, max_blocks_behind) = match cx.argument_opt(2) {
        None => (12u64, None, 50u64),
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
            let max_blocks_behind = opts.get_opt(&mut cx, "maxBlocksBehind")?.unwrap_or(cx.number(50u32)).downcast_or_throw::<JsNumber, FunctionContext>(&mut cx)?.value(&mut cx) as u64;
            
            (seconds_pr_block, fork_block_number, max_blocks_behind)
        }
    };

    let config = Config {
        fork_url,
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
    // if std::env::var_os("RUST_LOG").is_none() {
    //     std::env::set_var("RUST_LOG", "slot0=info");
    // }
    // log::set_max_level(log::LevelFilter::Debug);
    pretty_env_logger::init();

    cx.export_function("createSimulator", create_simulator)?;
    NeonResult::Ok(())
}
