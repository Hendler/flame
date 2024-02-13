





 # git@github.com:HumanAssistedIntelligence/flame.git

# first time

setup

    brew uninstall rust
    brew install rustup rustup-init
    rustup-init
    rustup default nightly && rustup update


install https://github.com/pgcentralfoundation/pgrx

     brew install git icu4c pkg-config
     cargo install --locked cargo-pgrx
     cargo pgrx init


install candle

    cargo add --git https://github.com/huggingface/candle.git candle-core --rename candle --features "cuda"
    cargo add --git https://github.com/huggingface/candle.git candle-nn --rename candle_nn --features "cuda"
    cargo add hf-hub --rename hf_hub --features "tokio"
    cargo add tokenizers --features="hf-hub"
    cargo add serde_json

