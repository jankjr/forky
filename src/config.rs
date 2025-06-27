#[derive(Debug, Clone)]
pub struct Config {
    pub link_type: LinkType,
    pub fork_url: String,
    pub seconds_per_block: u64,
    pub start_block: Option<u64>,
    pub max_blocks_behind: u64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub enum LinkType {
    Reth,
    Geth,
}

impl std::str::FromStr for LinkType {
    type Err = eyre::Report;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "Reth" => Ok(LinkType::Reth),
            "Geth" => Ok(LinkType::Geth),
            _ => Err(eyre::eyre!("Invalid LinkType {}", s)),
        }
    }
}
