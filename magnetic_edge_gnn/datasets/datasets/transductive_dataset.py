

from magnetic_edge_gnn.datasets.dataset_utils import (
    random_orientation,
    split_pyg_graph_transductive,
)

from .base_dataset import BaseDataset


class TransductiveDataset(BaseDataset):
    """Base class for datasets for transductive edge level problems."""
    def split(self, seed: int | None = None):
        if seed is None:
            seed = self.seed
        base_data = split_pyg_graph_transductive(
            self.base_data,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=seed,
            interpolation_label_size=self.interpolation_label_size,
            add_noisy_flow_to_input=self.add_noisy_flow_to_input,
            add_interpolation_flow_to_input=self.add_interpolation_flow_to_input,
            add_zeros_to_flow_input=self.add_zeros_to_flow_input,
        )
        # All datasets have the same graphs, each of which splits the edges into train, val, and test sets
        graphs = {split: base_data for split in ["train", "val", "test"]}

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
