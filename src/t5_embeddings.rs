#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::path::PathBuf;

use candle_transformers::models::t5;

use anyhow::{Error as E, Result};
use candle::{DType, Device};
use candle_nn::VarBuilder;
// use candle_transformers::generation::LogitsProcessor;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const DTYPE: DType = DType::F32;

use lazy_static::lazy_static;
use std::sync::Mutex;
use clap::Parser;
use candle::utils::{cuda_is_available, metal_is_available};



#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    tracing: bool,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    revision: Option<String>,
    #[arg(long)]
    decode: bool,
    #[arg(long, default_value = "false")]
    disable_cache: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    decoder_prompt: Option<String>,
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

static ARGS: Args = Args {
    cpu: true, // Assuming you want to run on CPU
    tracing: false, // Tracing typically not needed in production
    model_id: None, // Must be None, as String cannot be constructed statically
    revision: None, // Must be None for the same reason
    decode: false, // Set to true if you need decoding
    disable_cache: false, // Assuming caching is enabled
    prompt: None, // Must be None, as String cannot be constructed statically
    decoder_prompt: None, // Must be None for the same reason
    normalize_embeddings: true, // Default value provided
    temperature: 0.8, // Default value provided
    top_p: None, // Must be None, as f64 cannot be constructed statically
    repeat_penalty: 1.1, // Default value provided
    repeat_last_n: 64, // Default value provided
};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

lazy_static! {
    pub static ref MODEL: Mutex<Option<(T5ModelBuilder, Tokenizer)>> = Mutex::new(None);
}

// Initializes the model and tokenizer, if they haven't been already
pub fn init_model() -> Result<(), E> {
    let mut model = MODEL.lock().map_err(|_| E::msg("Mutex is poisoned"))?;
    if model.is_none() {
        // Perform the loading here
        *model = Some(T5ModelBuilder::load(&ARGS)?);
    }
    Ok(())
}

pub struct T5ModelBuilder {
    pub device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<(Self, Tokenizer)> {
        let device = device(args.cpu)?;
        let default_model = "t5-small".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2"
        {
            hub_load_safetensors(&api, "model.safetensors.index.json")?
        } else {
            vec![api.get("model.safetensors")?]
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }

    pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }
}


/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle::bail!("no weight map in {json_file:?}").into(),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle::bail!("weight map in {json_file:?} is not a map").into(),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors_files)
}

