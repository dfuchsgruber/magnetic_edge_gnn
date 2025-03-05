from torch_geometric.data import Data

from magnetic_edge_gnn.datasets.registry import DatasetRegistry


class BaseDataset:
    """Base Dataset class for graphs."""

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        dataset_registry_database_path: str,
        dataset_registry_lockfile_path: str,
        dataset_registry_storage_path: str,
        val_ratio: float = 0.1,
        test_ratio: float = 0.8,
        seed: int | None = None,
        arbitrary_orientation: bool = True,
        orientation_equivariant_labels: bool = False,
        interpolation_label_size: float = 0.75,
        add_noisy_flow_to_input: bool = False,
        add_interpolation_flow_to_input: bool = False,
        add_zeros_to_flow_input: bool = False,
        max_num_positional_laplacian_encodings: int = 32,
        laplacian_encodings_phase_shift: float = 0.0,
        dataset_registry_force_rebuild: bool = False,
    ):
        """
        Abstract dataset class for transductive tasks.

        Args:
            dataset_name (str): Name of the dataset.
            dataset_path (str): Path to the dataset.
            val_ratio (float, optional): Ratio of validation data. Defaults to 0.1.
            test_ratio (float, optional): Ratio of test data. Defaults to 0.8.
            seed (float, optional): Random seed. Defaults to 0.
            arbitrary_orientation (bool, optional): Whether to arbitrarily orient the edges.
                Defaults to False.
            orientation_equivariant_labels (bool, optional): Whether the labels are orientation-equivariant or not.
                Defaults to False.
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.arbitrary_orientation = arbitrary_orientation
        self.orientation_equivariant_labels = orientation_equivariant_labels
        self.add_noisy_flow_to_input = add_noisy_flow_to_input
        self.add_interpolation_flow_to_input = add_interpolation_flow_to_input
        self.add_zeros_to_flow_input = add_zeros_to_flow_input
        self.interpolation_label_size = interpolation_label_size
        self.max_num_positional_laplacian_encodings = (
            max_num_positional_laplacian_encodings
        )
        self.laplacian_encodings_phase_shift = laplacian_encodings_phase_shift

        self.dataset_registry = DatasetRegistry(
            dataset_registry_database_path,
            dataset_registry_lockfile_path,
            dataset_registry_storage_path,
        )
        self.dataset_registry_force_rebuild = dataset_registry_force_rebuild

        self.base_data = self.preprocess()

    def split(self) -> list[Data]:
        """Splits the dataset into train, validation, and test sets."""
        raise NotImplementedError

    def preprocess(self) -> list[Data]:
        """Preprocessing to return the (base) graphs for the dataset."""
        raise NotImplementedError
