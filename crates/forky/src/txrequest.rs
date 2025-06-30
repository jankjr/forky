use crate::utils::*;
use alloy::rpc::types::{AccessList, AccessListItem};
use neon::{prelude::*, types::JsBigInt};
use revm::primitives::{Address, U256};
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

    pub access_list: Option<AccessList>,
}

pub fn js_obj_tx_to_transaction<'a>(
    mut cx: FunctionContext<'a>,
    js_obj_tx: Handle<JsObject>,
) -> neon::result::NeonResult<TransactionRequest> {
    let from = js_obj_tx.get_value(&mut cx, "from")?;
    let to = js_obj_tx.get_value(&mut cx, "to")?;
    let data = js_obj_tx.get_value(&mut cx, "data")?;
    let gas_price = get_optional_bigint(
        js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "gasPrice"),
        &mut cx,
    )?;
    let gas_limit = get_optional_bigint(
        js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "gasLimit"),
        &mut cx,
    )?;

    let gas_priority_fee = get_optional_bigint(
        js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "maxPriorityFeePerGas"),
        &mut cx,
    )?;
    let max_fee_per_gas = get_optional_bigint(
        js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "maxFeePerGas"),
        &mut cx,
    )?;
    let value = get_optional_bigint(
        js_obj_tx.get_opt::<JsBigInt, _, _>(&mut cx, "value"),
        &mut cx,
    )?
    .unwrap_or_default();

    let access_list = if let Some(v) = js_obj_tx.get_opt::<JsArray, _, _>(&mut cx, "accessList")? {
        let mut out = Vec::new();
        let length = v.len(&mut cx);
        for i in 0..length {
            let item: Handle<JsObject> = v.get(&mut cx, i)?;
            let address = item.get(&mut cx, "address")?;
            let address = js_value_to_address(&mut cx, address)?;
            let storage_keys = item
                .get::<JsArray, _, _>(&mut cx, "storageKeys")
                .map(|v| {
                    let mut out = Vec::new();
                    let length = v.len(&mut cx);
                    for i in 0..length {
                        let item = v.get(&mut cx, i)?;
                        let item = js_value_to_b256(&mut cx, item)?;
                        out.push(item);
                    }
                    neon::result::NeonResult::Ok(out)
                })??;

            out.push(AccessListItem {
                address,
                storage_keys: storage_keys,
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
        access_list,
    })
}
