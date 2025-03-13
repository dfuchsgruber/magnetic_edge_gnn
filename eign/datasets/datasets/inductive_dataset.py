

from magnetic_edge_gnn.datasets.dataset_utils import (
    random_orientation,
    split_pyg_graph_inductive,
)
from magnetic_edge_gnn.datasets.datasets.base_dataset import BaseDataset


class InductiveDataset(BaseDataset):
    """Base class for datasets for inductive edge level problems."""
    def split(self, seed: int | None = None):
        if seed is None:
            seed = self.seed
        graphs = split_pyg_graph_inductive(
            self.base_data,
            seed=seed,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            interpolation_label_size=self.interpolation_label_size,
            add_noisy_flow_to_input=self.add_noisy_flow_to_input,
            add_interpolation_flow_to_input=self.add_interpolation_flow_to_input,
            add_zeros_to_flow_input=self.add_zeros_to_flow_input,
        )

        if self.arbitrary_orientation:
            graphs = {
                split: [
                    random_orientation(
                        graph,
                        orientation_equivariant_labels=self.orientation_equivariant_labels,
                        seed=seed,
                    )
                    for graph in graphs_split
                ]
                for split, graphs_split in graphs.items()
            }

        return graphs
