# FLAME

Flame is named for its ambition to use HuggingFace's Candle library.

Currently this is a proof of concept Postgresql plugin for text embeddings.
Goal is to use Flame with PGVector for faster RAG applications in Postgresql.
Model chosen is  `MiniLM-L6-v2`, since I was using it already with 768 length.

The proof of concept (v0.0.1) uses [PGRX](https://github.com/pgcentralfoundation/pgrx) and [fastembed-rs](https://github.com/Anush008/fastembed-rs) Rust implementation of [fastembed](https://github.com/qdrant/fastembed)

## contributing

I don't know Rust, or Candle, but would love to see more embeddings... embedded.

## usage

see [SETUP.md](./SETUP.md)

    cargo pgrx run
    # [everything just works...]
    flame=# CREATE EXTENSION flame;
    flame=# SELECT fast_embed('hello world');

# Release Notes

## v0.0.1

- singleton for instatiation
- default model

# Roadmap

## v0.1.0

- tests (long text strings, peformance, multi threading situations, with pgvector)
- choose models at runtime

## v0.2.0

- auto create pgvector
- more tests


