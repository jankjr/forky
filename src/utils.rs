use std::str::FromStr;

use alloy::hex;
use alloy::transports::ws::WebSocketConfig;
use alloy_provider::{DynProvider, Provider};
use neon::prelude::*;
use neon::{handle::Handle, types::JsBigInt};
use revm::primitives::{Address, U256};

pub async fn provider_from_string(url: &String) -> eyre::Result<DynProvider> {
    let builder = alloy::providers::ProviderBuilder::new();
    let out = if url.starts_with("http") {
        builder.connect_http(url.as_str().parse()?)
    } else if url.starts_with("ws") {
        let config = WebSocketConfig::default()
            .max_frame_size(Some(128 << 20))
            .max_message_size(Some(128 << 20));
        let ws = builder
            .connect_ws(alloy::providers::WsConnect::new(url.as_str()).with_config(config))
            .await?;

        ws
    } else {
        return Err(eyre::eyre!("Invalid provider type"));
    };

    Ok(out.erased())
}

pub fn to_neon<'a, T, C: Context<'a>>(
    cx: &mut C,
    e: eyre::Result<T>,
) -> neon::result::NeonResult<T> {
    match e {
        eyre::Result::Ok(v) => neon::result::NeonResult::Ok(v),
        eyre::Result::Err(e) => cx.throw_error(e.to_string()),
    }
}

pub fn js_value_to_address<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<Address> {
    use eyre::WrapErr;
    let down_casted = js_val.to_string(cx)?;
    let out = Address::from_str(down_casted.value(cx).as_str());
    to_neon(cx, out.wrap_err("Failed to parse"))
}

pub fn js_value_to_bytes<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<Vec<u8>> {
    use eyre::WrapErr;
    let down_casted = js_val.to_string(cx)?;
    let out = hex::decode(down_casted.value(cx).as_str()).wrap_err("Failed to parse");
    to_neon(cx, out)
}

pub fn js_value_to_uint256<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<revm::primitives::U256> {
    let down_casted: Handle<'a, JsBigInt> =
        js_val.downcast_or_throw::<JsBigInt, neon::context::FunctionContext<'a>>(cx)?;
    bigint_to_u256(down_casted, cx)
}
pub fn js_value_to_b256<'a>(
    cx: &mut neon::context::FunctionContext<'a>,
    js_val: Handle<'a, JsValue>,
) -> neon::result::NeonResult<revm::primitives::B256> {
    use eyre::WrapErr;
    let down_casted: Handle<'a, JsString> =
        js_val.downcast_or_throw::<JsString, neon::context::FunctionContext<'a>>(cx)?;
    let value = down_casted.value(cx);
    let out = revm::primitives::B256::from_str(&value);

    to_neon(cx, out.wrap_err("Failed to parse"))
}
pub fn get_optional_bigint<'a>(
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

pub fn bigint_to_u256<'a>(
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
