mod fast_embeddings;
use fast_embeddings::FAST_EMBEDDING;
use pgrx::prelude::*;
pgrx::pg_module_magic!();

use fastembed::EmbeddingBase;

#[pg_extern]
fn fast_embed(sentence: String) -> Vec<Option<f32>> {
    let documents = vec![sentence];
    let lock = FAST_EMBEDDING.lock().expect("Mutex is poisoned");
    let model = lock.as_ref().expect("Model is not initialized");

    match model.embed(documents, None) {
        Ok(embeddings) => {
            // Convert the embeddings to Vec<Option<f32>>
            embeddings[0].iter().map(|&e| Some(e)).collect()
        }
        Err(e) => {
            eprintln!("Embedding failed: {}", e);
            vec![]
        }
    }
}

// #[cfg(any(test, feature = "pg_test"))]
// #[pg_schema]
// mod tests {
//     use pgrx::prelude::*;

//     #[pg_test]
//     fn test_hello_flame() {
//         let sentences: Vec<String> = vec![String::from("Hello"), String::from("World")];
//         let embeddings = crate::hf_embeddings::create_embeddings(sentences);
//     }

// }

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
