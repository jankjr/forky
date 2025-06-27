use alloy::transports::ws::WebSocketConfig;
use alloy_provider::{DynProvider, Provider};

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
