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
6. if the graph exceeds 6000 nodes, intermediates are ranked deterministically by path depth, retained incident connection weight, and body id;
7. only real MaleCNS edges between surviving cells are retained.

The visual type prefixes are an **engineering proxy for visually relevant populations**, not a claim that every selected cell is a photoreceptor or that the synthetic visual encoder reproduces the biological fly eye exactly.

## Export integrity

`src/fly_window/data/export.py` writes sorted node/edge Parquet files plus a compact NPZ graph. The manifest records ordinary SHA256 hashes of the serialized files.

A separate `artifact_sha256` is a semantic graph hash. It is computed from canonical source/config metadata and the logical graph arrays (`body_ids`, `src_idx`, `dst_idx`, `weights`, `input_mask`, `output_mask`), including dtype and shape. It therefore does not depend on incidental ZIP or Parquet serialization metadata.

## Scientific boundary

The real-data claim in Fly Window is limited to the published MaleCNS neuron identities, annotations, neurotransmitter predictions, directed topology, and connection-strength table used to constrain the model. Neural dynamics, synthetic vision, flight physics, reward, and the eventual sensor/readout mappings are simplified engineering models and must be described as such in public material.
