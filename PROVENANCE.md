# Fly Window data provenance

## Dataset

The connectome source for this project is **MaleCNS v1.0**, exposed in neuPrint as `male-cns:v1.0` and published by the Male CNS Connectome project.

The three bulk source files used by the v1 pipeline are pinned to the official MaleCNS Google Cloud Storage mirror over HTTPS:

- `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather`
- `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather`

The official MaleCNS download page states that the Male CNS dataset is licensed under **CC-BY**. The project is a collaboration involving FlyEM at HHMI Janelia, the University of Cambridge Department of Zoology, the MRC Laboratory of Molecular Biology, and Google Research. Users of derived artifacts should retain appropriate dataset attribution.

Raw dataset files are intentionally excluded from git. `scripts/fetch_malecns.py` writes `data/raw/source-manifest.json`, which records the dataset id, retrieval time, source URL, file size, and SHA256 for every downloaded source file.

## Task-subgraph selection

Fly Window v1 does **not** claim to simulate the entire MaleCNS. It extracts a deterministic task-focused subgraph from the real directed weighted connectome.

The pinned selection rule is stored in `configs/subgraph_v1.json`:

1. descending outputs are cells annotated with `superclass == "descending_neuron"`;
2. visual input seeds are typed cells whose type begins with one of `LC`, `LPLC`, `LT`, `MC`, `Mi`, `Tm`, `T4`, or `T5`;
3. connections with weight below 3 are removed;
4. bounded forward reachability from visual seeds and bounded reverse reachability from descending outputs are computed for at most 4 hops;
5. intermediate cells must lie in both reachable sets; visual and descending seed cells are retained explicitly;
6. if the intermediate population exceeds the configured capacity, intermediates are ranked deterministically by path depth, retained incident connection weight, and body id; input/output seeds are never dropped;
7. only real MaleCNS edges between surviving cells are retained.

The configured `max_nodes=6000` therefore acts as an intermediate-node budget rather than a hard cap on the total graph size. In the verified v1 extraction the mandatory seed populations alone contain 56,093 visual inputs and 1,314 descending outputs, so the final graph necessarily exceeds 6,000 nodes.

The visual type prefixes are an **engineering proxy for visually relevant populations**, not a claim that every selected cell is a photoreceptor or that the synthetic visual encoder reproduces the biological fly eye exactly.

## Verified Checkpoint A

A clean GitHub-hosted Ubuntu runner executed the full pipeline on 2026-09-14: source acquisition, streamed connectivity loading, thresholding, task-subgraph selection, and NPZ/Parquet export. The run completed successfully and produced the following graph summary:

- visual input seeds: **56,093**
- descending output seeds: **1,314**
- final nodes: **57,407**
- final directed weighted edges: **1,543,613**
- semantic graph SHA256: `69af9e02aac7d276ea33b2529d7cdcaa83c297b17385dd9d8867d9819cd8288e`

Pinned source fingerprints from that run:

- annotations: `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` (14,483,314 bytes)
- neurotransmitters: `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` (43,282,834 bytes)
- connectivity weights: `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` (1,051,241,946 bytes)

Serialized export fingerprints from the same run:

- `subgraph_v1_nodes.parquet`: `12dda4218d522b9204a776fd63c422cbc6e6089d1f94bbaa184891d9f8311655`
- `subgraph_v1_edges.parquet`: `fc21a73d120026c42b777712c2e62ab7cc7ea4c5c03598c0f6fa858b04868f50`
- `subgraph_v1.npz`: `e39d1327f30b61aceb551516f11ea28f6bce46541f3c661f049dc0cabe868989`

These values define the first verified real-data checkpoint. A future change to source files, selection rules, or graph semantics is expected to change one or more of these fingerprints and must be recorded as a new checkpoint rather than silently replacing this one.

## Export integrity

`src/fly_window/data/export.py` writes sorted node/edge Parquet files plus a compact NPZ graph. The manifest records ordinary SHA256 hashes of the serialized files.

A separate `artifact_sha256` is a semantic graph hash. It is computed from canonical source/config metadata and the logical graph arrays (`body_ids`, `src_idx`, `dst_idx`, `weights`, `input_mask`, `output_mask`), including dtype and shape. It therefore does not depend on incidental ZIP or Parquet serialization metadata.

## Scientific boundary

The real-data claim in Fly Window is limited to the published MaleCNS neuron identities, annotations, neurotransmitter predictions, directed topology, and connection-strength table used to constrain the model. Neural dynamics, synthetic vision, flight physics, reward, and the eventual sensor/readout mappings are simplified engineering models and must be described as such in public material.
