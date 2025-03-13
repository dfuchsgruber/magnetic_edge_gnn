from collections import defaultdict
from os.path import join

import numpy as np
import pandas as pd

from magnetic_edge_gnn.datasets.dataset_utils import (
    combine_edges,
    create_pyg_graph_transductive,
    normalize_features,
    normalize_flows,
    relabel_nodes,
)
from magnetic_edge_gnn.datasets.registry import DatasetRegistryKey

from .transductive_dataset import TransductiveDataset


class TNTPFlowDenoisingInterpolationDataset(TransductiveDataset):
    """Class for traffic datasets for TNTP flows."""
    
    supported_tasks = ["denoising", "interpolation", "simulation"]
    supported_datasets = {
        "traffic-anaheim": "Anaheim",
        "traffic-barcelona": "Barcelona",
        "traffic-chicago": "ChicagoSketch",
        "traffic-winnipeg": "Winnipeg",
    }

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset_registry_database_path: str,
        dataset_registry_lockfile_path: str,
        dataset_registry_storage_path: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.5,
        seed: float | None = None,
        arbitrary_orientation: bool = True,
        orientation_equivariant_labels: bool = False,
        interpolation_label_size: float = 0.75,
        max_num_positional_laplacian_encodings: int = 32,
        laplacian_encodings_phase_shift: float = 0.0,
        dataset_registry_force_rebuild: bool = False,
    ):
        """
        Dataset class for the edge flow denoising and edge flow interpolation tasks for the traffic datasets from the
        TransportationNetworks GitHub repository (https://github.com/bstabler/TransportationNetworks).

        Args:
            split (str): Data split to load. Inconsequential, since all datasets have the same base graph.
            dataset_name (str): Name of the dataset.
            dataset_path (str): Path to the dataset.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.1.
            test_ratio (float, optional): Ratio of test data. Defaults to 0.5.
            seed (float, optional): Random seed. Defaults to 0.
            arbitrary_orientation (bool, optional): Whether to arbitrarily orient the edges.
                Defaults to False.
            orientation_equivariant_labels (bool, optional): Whether the labels are orientation-equivariant or not.
                Defaults to False.
        """

        dataset, task = dataset_name.rsplit("-", 1)
        if dataset not in self.supported_datasets:
            raise ValueError(
                f"The dataset should be in {self.supported_datasets.keys()}. The dataset {dataset} is not supported!"
            )
        if task not in self.supported_tasks:
            raise ValueError(
                f"The task should be in {self.supported_tasks.keys()}. The task {task} is not supported!"
            )

        self.dataset = self.supported_datasets[dataset]
        self.task = task

        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_registry_database_path=dataset_registry_database_path,
            dataset_registry_lockfile_path=dataset_registry_lockfile_path,
            dataset_registry_storage_path=dataset_registry_storage_path,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            arbitrary_orientation=arbitrary_orientation,
            orientation_equivariant_labels=orientation_equivariant_labels,
            interpolation_label_size=interpolation_label_size,
            add_noisy_flow_to_input=(self.task == "denoising"),
            add_interpolation_flow_to_input=(self.task == "interpolation"),
            add_zeros_to_flow_input=(self.task == "simulation"),
            max_num_positional_laplacian_encodings=max_num_positional_laplacian_encodings,
            laplacian_encodings_phase_shift=laplacian_encodings_phase_shift,
            dataset_registry_force_rebuild=dataset_registry_force_rebuild,
        )

    def preprocess(self):
        # Read the features and flows.
        key = DatasetRegistryKey(
            name=f"TNTPFlowDenoisingInterpolationDatasetv2_{self.dataset}",
            q=self.laplacian_encodings_phase_shift,
            num_laplacian_eigenvectors=self.max_num_positional_laplacian_encodings,
        )
        if key not in self.dataset_registry or self.dataset_registry_force_rebuild:
            features_file = join(self.dataset_path, f"{self.dataset}_net.tntp")
            features_df = pd.read_csv(features_file, skiprows=8, sep="\t")
            features_df.columns = [s.strip().lower() for s in features_df.columns]
            # Drop useless columns.
            features_df.drop(["~", ";"], axis=1, inplace=True)

            flows_file = join(self.dataset_path, f"{self.dataset}_flow.tntp")
            flows_df = pd.read_csv(flows_file, sep="\t")
            flows_df.columns = [s.strip().lower() for s in flows_df.columns]

            # All links types.
            all_link_types = set(features_df["link_type"])
            link_type2idx = {
                link_type: idx for idx, link_type in enumerate(all_link_types)
            }

            # Convert the dataframes to dictionaries.
            features = {}
            for _, row in features_df.iterrows():
                u, v = int(row["init_node"]), int(row["term_node"])
                numerical_features = np.array(
                    [
                        row["capacity"],
                        row["length"],
                        row["free_flow_time"],
                        row["b"],
                        row["power"],
                        # row["speed"],
                        row["toll"],
                    ]
                )

                # One-hot encode the link types.
                link_type = int(row["link_type"])
                link_type_features = np.zeros(len(link_type2idx))
                link_type_features[link_type2idx[link_type]] = 1

                features[(u, v)] = np.concatenate(
                    [numerical_features, link_type_features]
                )

            flows = {
                (int(row["from"]), int(row["to"])): row["volume"]
                for _, row in flows_df.iterrows()
            }

            # Divergence at every node
            divergence = defaultdict(float)
            for (src, dst), flow in flows.items():
                divergence[dst] += flow
                divergence[src] -= flow
            has_divergence = {
                v: not np.isclose(div, 0) for v, div in divergence.items()
            }

            # Pre-process the graph.
            features, flows, undirected_edges = combine_edges(
                features=features, flows=flows
            )

            # Determine which edges have divergence
            closed_flow = {
                (u, v): ((not has_divergence[v]) and (not has_divergence[u]))
                for (u, v) in flows.keys()
            }
            features = {
                e: np.concatenate((np.array([int(closed_flow[e])]), f))
                for e, f in features.items()
            }

            features, flows, undirected_edges, node_mapping = relabel_nodes(
                features=features, flows=flows, undirected_edges=undirected_edges
            )
            features, flows, undirected_edges = normalize_flows(
                features=features, flows=flows, undirected_edges=undirected_edges
            )
            features = normalize_features(features)

            inv_features = {k: v for k, v in features.items()}
            train_edges = {e for e, closed in closed_flow.items() if not closed}
            equi_features = {k: np.zeros(0) for k in features}

            data = create_pyg_graph_transductive(
                equi_features=equi_features,
                inv_features=inv_features,
                undirected_edges=undirected_edges,
                labels=flows,
                max_num_positional_laplacian_encodings=self.max_num_positional_laplacian_encodings,
                laplacian_encodings_phase_shift=self.laplacian_encodings_phase_shift,
                train_edges=train_edges,
            )
            self.dataset_registry[key] = [data]

        return self.dataset_registry[key]
