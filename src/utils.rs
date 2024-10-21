use alloy::{hex, transports::BoxTransport};
use revm::primitives::U256;

use crate::abstract_value::ADDR_MASK;


pub async fn provider_from_string(url: &String) -> eyre::Result<alloy::providers::RootProvider<BoxTransport>> {
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
static MIN_ADDR_VAL: U256 = U256::from_be_slice(&hex!("c0d7d3017b342ff039b55b0879"));


pub fn is_address_like(value: &U256) -> bool {
    let zeros: usize = value.count_zeros();
    value.lt(&ADDR_MASK) && value.gt(&MIN_ADDR_VAL) && zeros < 200 && zeros > 160
}
