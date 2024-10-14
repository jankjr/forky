use alloy::transports::BoxTransport;


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
