mod t5_embeddings;
use candle::Tensor;
use t5_embeddings::{MODEL, init_model};

use pgrx::prelude::*;
pgrx::pg_module_magic!();

// #[pg_extern]
// fn tf_embeddings(sentence: String) -> Vec<Option<f32>> {

// }


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
