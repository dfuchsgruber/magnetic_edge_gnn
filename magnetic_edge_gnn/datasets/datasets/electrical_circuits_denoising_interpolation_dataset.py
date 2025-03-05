import glob
from os.path import join

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from magnetic_edge_gnn.datasets.dataset_utils import (
    create_pyg_graph_transductive,
    normalize_flows,
    relabel_nodes,
)

from .transductive_dataset import TransductiveDataset
from magnetic_edge_gnn.datasets.registry import DatasetRegistryKey

class ElectricalCircuitsDenoisingInterpolationDataset(TransductiveDataset):
    """Electrical circuits dataset."""
    supported_tasks = ["denoising", "interpolation", "simulation"]

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset_registry_database_path: str,
        dataset_registry_lockfile_path: str,
        dataset_registry_storage_path: str,
        dataset_registry_force_rebuild: bool = False,
        val_ratio: float = 0.1,
        test_ratio: float = 0.5,
        seed: float | None = None,
        arbitrary_orientation: bool = True,
        orientation_equivariant_labels: bool = False,
        interpolation_label_size: float = 0.75,
        max_num_positional_laplacian_encodings: int = 32,
        laplacian_encodings_phase_shift: float = 0.0,
        include_non_source_voltages: bool = False,
        current_relative_to_voltage: bool = False,
    ):
        """
        Dataset class for the edge flow denoising and edge flow interpolation tasks for the electrical circuits dataset.

        Args:
            split (str): Data split to load. Should be one of: ["train", "val", "test"].
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
        if task not in self.supported_tasks:
            raise ValueError(
                f"The task should be in {self.supported_tasks.keys()}. The task {task} is not supported!"
            )

        self.dataset = dataset
        self.task = task
        self.include_non_source_voltages = include_non_source_voltages
        self.current_relative_to_voltage = current_relative_to_voltage
        
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            dataset_registry_database_path=dataset_registry_database_path,
            dataset_registry_lockfile_path=dataset_registry_lockfile_path,
            dataset_registry_storage_path=dataset_registry_storage_path,
            dataset_registry_force_rebuild=dataset_registry_force_rebuild,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            arbitrary_orientation=arbitrary_orientation,
            orientation_equivariant_labels=orientation_equivariant_labels,
            interpolation_label_size=interpolation_label_size,
            add_noisy_flow_to_input=(self.task == "denoising"),
            add_interpolation_flow_to_input=(self.task == "interpolation"),
            add_zeros_to_flow_input=(
                self.task == "simulation"
            ),
            max_num_positional_laplacian_encodings = max_num_positional_laplacian_encodings,
            laplacian_encodings_phase_shift = laplacian_encodings_phase_shift,
        )

    def preprocess(self):
        
        key = DatasetRegistryKey(
            name=f'{self.dataset}',
            q=self.laplacian_encodings_phase_shift,
            num_laplacian_eigenvectors=self.max_num_positional_laplacian_encodings,
        )
        if key not in self.dataset_registry:
            
            data = []

            component_type2idx = {"r": 0, "d": 1, "v": 2}

            # Process all files in the directory.
            for file_name in tqdm(glob.glob(join(self.dataset_path, "*.json")), desc='Preprocessing circuits...'):
                df = pd.read_json(file_name)

                # Skip the graph if all edge currents are 0.
                if df["edge_current"].abs().max() == 0:
                    continue

                # Replace NaN values with 0.
                df.fillna(0, inplace=True)

                features_df = df[
                    ["row", "col", "components", "resistances", "edge_voltage"]
                ]
                flows_df = df[["row", "col", "edge_current"]]

                # Convert the dataframes to dictionaries.
                features = {}
                undirected_edges = {}
                source_voltage = np.nan

                for _, row in features_df.iterrows():
                    u, v = int(row["row"]), int(row["col"])
                    voltage = row["edge_voltage"]

                    # One-hot encode the component types.
                    component_type = row["components"]
                    component_type_features = np.zeros(len(component_type2idx))
                    component_type_features[component_type2idx[component_type]] = 1

                    if component_type != "v" and not self.include_non_source_voltages:
                        voltage = 0  # Set voltage to 0 for non-source components
                    if component_type == "v":
                        source_voltage = voltage
                    numerical_features = np.array([voltage, row["resistances"]])

                    features[(u, v)] = np.concatenate(
                        [numerical_features, component_type_features]
                    )
                    undirected_edges[(u, v)] = 0 if component_type == "d" else 1

                flows = {
                    (int(row["row"]), int(row["col"])): row["edge_current"]
                    for _, row in flows_df.iterrows()
                }
                if self.current_relative_to_voltage:
                    assert not np.isnan(source_voltage)
                    for (u, v), current in flows.items():
                        flows[(u, v)] = current / source_voltage

                # Pre-process the graph.
                features, flows, undirected_edges, node_mapping = relabel_nodes(
                    features=features, flows=flows, undirected_edges=undirected_edges
                )

                # In this inductive setting, we can not normalize flows per graph, they are prediction targets
                features, flows, undirected_edges = normalize_flows(
                    features=features,
                    flows=flows,
                    undirected_edges=undirected_edges,
                    normalize_by_max_flow=False,
                )

                inv_features = {
                    k: v[1:] for k, v in features.items()
                }  # everything but voltage
                equi_features = {k: v[:1] for k, v in features.items()}  # only voltage

                current_data = create_pyg_graph_transductive(
                    equi_features=equi_features,
                    inv_features=inv_features,
                    undirected_edges=undirected_edges,
                    labels=flows,
                    max_num_positional_laplacian_encodings=self.max_num_positional_laplacian_encodings,
                    laplacian_encodings_phase_shift=self.laplacian_encodings_phase_shift,
                )
                data.append(current_data)

            max_flow = torch.cat([d.y for d in data]).abs().max().item()
            std_flow = torch.cat([d.y for d in data]).abs().std().item()
            

            equi_edge_attr = torch.cat([d.equi_edge_attr for d in data])
            inv_edge_attr = torch.cat([d.inv_edge_attr for d in data])
            assert (
                inv_edge_attr.size(1) == 1 + max(component_type2idx.values()) + 1
            )  # resistances + component types
            assert equi_edge_attr.size(1) == 1, (
                equi_edge_attr.size(1)
            )

            # inv edge attribute:
            # resistance, *component types
            # equi edge attributes:
            # voltage

            inv_edge_attr_scaler = StandardScaler()
            inv_edge_attr_scaler.fit(inv_edge_attr.numpy()[:, :1])

            # The edge voltage should also be normalized by its max value
            voltages = equi_edge_attr[:, 0:]
            assert voltages.size(1) == 1, "Only voltage should be left"
            voltages = voltages.flatten()
            voltages = voltages[voltages != 0]  # most voltages we supply may be zero
            voltages = voltages.unsqueeze(1)

            equi_normalizer = voltages.abs().std(dim=0, keepdim=True)[0]

            for d in data:
                # Normalize all flows
                d.y_original = d.y.clone()
                d.y /= std_flow
                d.equi_edge_attr[:, 0:] /= equi_normalizer
                d.inv_edge_attr[:, :1] = torch.from_numpy(
                    inv_edge_attr_scaler.transform(d.inv_edge_attr[:, :1].numpy())
                ).to(d.inv_edge_attr.dtype)
                
            self.dataset_registry[key] = data
        
        return self.dataset_registry[key]


