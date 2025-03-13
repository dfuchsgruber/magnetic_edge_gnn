# Graph Neural Networks for Edge Signals: Orientation Equivariance and Invariance

Reference implementation of our paper [Graph Neural Networks for Edge Signals: Orientation Equivariance and Invariance](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=K-egQS0AAAAJ&citation_for_view=K-egQS0AAAAJ:9yKSN-GCB0IC) by Dominik Fuchsgruber, Tim Postuvan, Stephan Günnemann and Simon Geisler pubished at ICLR 2025.


## Installation

This code and its dependencies can be installed as the package `magnetic_edge_gnn` via

```bash
pip install -e .
```

You can also install optional developer dependencies (e.g. `pytest`) via:

```bash
.[dev]
```

## Running Experiments

This code is built on the [sacred](https://sacred.readthedocs.io/en/stable/) and [seml](https://github.com/TUM-DAML/seml/tree/master) frameworks. The latter is only for convience to run experiments on a HPC cluster and only the former is required to run experiments. We use sacred's `named configurations` to configure experiments conveniently.

### Configurations

The default configurations for datasets and models are found in `magnetic_edge_gnn/configs`. Named default configurations are decorated with `@experiment.named_config` and are found in `magnetic_edge_gnn/configs/data` and `magnetic_edge_gnn/configs/model.py` respectively. They can be used like:

```bash
python train.py with traffic_flow_simulation_barcelona eign
```

The named configurations for datasets are:
- `traffic_flow_[denoising|interpolation|simulation]_[anaheim|barcelona|chicago|winnipeg]`: Traffic flow datasets for different cities and the respective task.
- `electrical_circuits_[denoising|interpolation|simulation]`: Electrical circuits and the simulated current.
- `random_walk_denoising`: The synthetic Random Walk Completion (RW Comp) task.
- `mixed_longest_cycle_identification`: The synthetic Longest Directed Cycle Prediction (LD Cycles) task.
- `typed_triangles_orientation`: The synthetic Triangle Flow Orientation (Tri-Flow) task.

The named configurations for the models are:
- `mlp`: An MLP.
- `line_graph_gnn`: A Line-Graph GNN.
- `equivariant_hodge_gnn`: Hodge GNN.
- `concatenated_hodge_gnn`: A Hodge GNN that concatenates orientation equivariant and orientation invariant features (HodgeGNN+Inv).
- `directed_hodge_gnn`: A Hodge GNN that breaks orientation equivariance to become direction aware (HodgeGNN+Dir).
- `transformer`: An edge-level transformer that uses the eigenvectors of the Magnetic Edge Laplacian as positional encodings (Line-MagNet).
- `dir_gnn`: An edge-level GNN with different message passing for edges in each direction (DirGNN).
- `eign`: Our proposed model, EIGN.

The model's hyperparameters can be set through the command line as well, for example:

```bash
python train.py with traffic_flow_simulation_barcelona eign model.hidden_dim=32 model.num_layers=8 optimization.lr=0.003
```

We also provide the optimal hyperparameters found through cross-validation in `hyperparameters/`. They can be loaded using the corresponding named configs in `magnetic_edge_gnn/configs/best_hyperparameters.py` like:


```bash
python train.py with traffic_flow_simulation_barcelona eign traffic_flow_simulation_best_hyperparameters
```

The respective named configs are:
- `traffic_flow_[denoising|interpolation|simulation]_best_hyperparameters`: Traffic datasets.
- `electrical_circuits_[denoising|interpolation|simulation]_best_hyperparameters`: Electrical circuits dataset.
- `random_walk_denoising_best_hyperparameters`: RW Comp.
- `mixed_longest_cycle_identification_best_hyperparameters`: LD Cycles.
- `typed_triangles_orientation_best_hyperparameters`: Tri-Flow.


One can also print the resulting full configuration by running:

```bash
python train.py print_config with ...
```

### Datasets

All real-world datasets are provided together with the code while synthetic datasets are generated on-the-fly. We take the data for the traffic datasets from the [TNTP Repository](https://github.com/bstabler/TransportationNetworks). They are located in `data/`.

Real-world datasets are pre-processed as described in our paper and augmented, for example with positional encodings of the Magnetic Edge Laplacian for the Line-MagNet baseline. These computations are cached in a dataset registry which uses `TinyDB` in a thread-safe way to organize these cached datasets. By default, this registry will be located in `data/registry`, but the path can be adapted through the command line arguments `data.registry.[database_path|lockfile_path|storage_path]`. If you, for some reason, want to override an already precomputed dataset, you can use `dataset.registry.force_rebuild=True`.

### Logging and Results

By default, results are logged onto the hard drive to `output/`. You can override this with the command line argument `save_dir=my_path`. For each run, it will create a directory according to the following scheme:

```bash
{output_dir}/runs/{task}/{collection_name}_{experiment_id}/{uuid}-{seed}/{run_id}/{split_id}/
```

Where each component is as follows:
- `{output_dir}`: The output directory (`output/` by default, overridden by `save_dir=...`).
- `{task}`: The dataset and task, e.g. `traffic-flow-simluation-barcelona`.
- `{collection_name}`, `{experiment_id}`: If using [seml](https://github.com/TUM-DAML/seml/tree/master) to organize experiments, these refer to the collection and experiment id within the collection. Otherwise, both will be set to `"None"`.
- `{uuid}`: A random uuid that identifies the run.
- `{seed}`: The random seed for this run.
- `{run_id}`: Corresponds to the `run_idx` field in the config and can be used to enumerate different runs.
- `{split_idx}`: As one execution of `train.py` can run multiple splits (using the `num_splits` command line argument), this identifies which split was run.

Within the directory associated with a run all files logged by `pytorch_lightning` are stored, such as model checkpoints. There will also be a `running_metrics.pt` object that stores the metrics over the training for later inspection. A file `config.yaml` contains the configuration used to run this experiment. Furthermore, the training method returns the evaluation metrics on the validation and test sets respectively for each dataset split such that these can be used when querying results using [seml](https://github.com/TUM-DAML/seml/tree/master).


### Weights & Biases

You can also use W&B to monitor your runs. To that end append `wandb` as the last named config for the command line interface like:

```bash
python train.py with traffic_flow_simulation_barcelona eign traffic_flow_simulation_best_hyperparameters wandb
```

It will create a new run in W&B for you. You can also configure how the run should be logged by overriding the following command line options:

- `logging.wandb.project`: The project name.
- `logging.wandb.name`: The run name.
- `logging.wandb.log_internal_dir`: The log internal dir for W&B. We recommend using something that links to `/dev/null` to not clutter your hard drive. By default it is set to `output/wandb`.
- `logging.wandb.cache_dir`: The wandb cache directory, by default `output/wandb_cache`.
- `logging.wandb.api_token_path`: Path to an optional W&B API token to authorize logging. Defaults to `./wandb_api_token`.

## Unit Tests

We provide rudimentary unit tests that can check whether a model is orientation equivariant or invariant w.r.t. to direction-consistent orientations. These are found in `test/`. They can be run using:

```bash
pytest test/test.py
```


## Credits

A large portion of the code is adapted by a version of [Tim Postuvan](https://github.com/timpostuvan). Thank you for the amazing work!

## Citation
Please cite our paper if you use our method or code in your own works:
```
@article{fuchsgruber2024graph,
  title={Graph Neural Networks for Edge Signals: Orientation Equivariance and Invariance},
  author={Fuchsgruber, Dominik and Po{\v{s}}tuvan, Tim and G{\"u}nnemann, Stephan and Geisler, Simon},
  journal={arXiv preprint arXiv:2410.16935},
  year={2024}
}
```