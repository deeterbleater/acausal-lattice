# Acausal Lattice Prototype Memorandum

This document maps the current Rust implementation to the technical specifications provided in the papers "Orthogonal Architecture: An Acausal Ontological Model" and "Toward an Acausal Ontological Framework".

## Core Components

### 1. Lattice Topology Engine (LTE)
- **Status:** Implemented in `lattice-core`.
- **Mechanism:** Uses `roaring` bitmaps for bitset tracking of precipitated Achronons.
- **Eligibility:** Implements the $P_a \subseteq R$ check using `RoaringBitmap::is_superset`.
- **Achronon Triple:** Represented by the `Achronon` struct (Antecedents, Orthogonals, Transformation).

### 2. Tensor Transformation Engine (TTE)
- **Status:** Implemented in `lattice-tensor` (CPU fallback).
- **Mechanism:** Uses `ndarray` for matrix-vector multiplication.
- **State Space:** A latent vector of dimension $N$ (currently 4 in the CLI demo).
- **Batching:** Supports applying a batch of orthogonal Achronons.
- **Future Work:** Integration with CUDA/ROCm via `candle` or `burn` for true parallel tensor contraction.

### 3. Cognitive Context Engine (CCE)
- **Status:** Implemented in `lattice-cce`.
- **Mechanism:** Flattens the lattice state into a structured text prompt.
- **Information Density:** Groups events by status (Precipitated vs. Potentiality), omitting chronological timestamps to emphasize structural relations.

## Demo Scenario

The `lattice-cli` implements a sample "Aion" (event continuum) with the following structure:
- **Sequential Chain:** Inquiry (1) -> Research (2) -> Strategy (3).
- **Parallel Branch:** LTE Implementation (4) and TTE Implementation (5) are orthogonal and precipitate simultaneously in Step 4.
- **Synthesis:** Integration (6) requires both 4 and 5 to have precipitated.

## Theoretical Alignment

- **Acausal Execution:** The system does not use a fixed `while True` loop with a clock. Instead, it advances by "precipitating" all currently eligible events from the Aion.
- **Commutativity:** The batch execution in `TTE::apply_batch` assumes that orthogonal events commute, as specified in the "Commutativity Invariant" section of the paper.
- **Emergent Causality:** Causality is derived purely from the `antecedents` bitsets, not from the order of creation or timestamps.

## Next Steps

1. **Persistence:** Add JSON/TOML serialization for the Aion to allow loading complex lattices from disk.
2. **LLM Loop:** Integrate an actual LLM client (e.g., Anthropic/OpenAI) into the CCE loop to allow events to be generated dynamically based on the "Current Potentiality" prompt.
3. **Advanced TTE:** Implement non-identity transformations (e.g., word embeddings as state vectors) to see how the latent space "warps" as events precipitate.
