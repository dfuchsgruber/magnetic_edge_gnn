"""Data module for edge-level tasks."""

import itertools

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from .datasets import (
    DAGDenoisingDataset,
    ElectricalCircuitsDenoisingInterpolationDataset,
    LongestCycleIdentificationDataset,
    MixedLongestCycleIdentificationDataset,
    RandomWalkDenoisingDataset,
    TNTPFlowDenoisingInterpolationDataset,
    TypedTrianglesOrientationDataset,
)


class EdgeLevelDataset(Dataset):
    """Base dataset class for edge level tasks."""
    def __init__(self, graphs: list[Data]):
        super().__init__()
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class EdgeLevelTaskDataModule(LightningDataModule):
    def __init__(
        self,
        config: DictConfig,
        batch_size: int,
        seed: int | None = None,
        arbitrary_orientation: bool = True,
    ):
        """
        PyTorch Lightning datamodule class for edge-level tasks.

        Args:
            config (DictConfig): Configuration file for the dataset.
            batch_size (int): Batch size.
            seed (int, optional): Random seed. Defaults to 0.
            arbitrary_orientation (bool, optional): Whether to arbitrarily orient the edges.
                Defaults to False.
        """
        super().__init__()

        self.name = config.name
        self.dataset_path = config.dataset_path
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.arbitrary_orientation = arbitrary_orientation
        self.orientation_equivariant_labels = config.orientation_equivariant_labels
        self.config = config
        self.dataset_kwargs = {}

        self.supported_datasets = list(
            "-".join(tpl)
            for tpl in itertools.product(
                [
                    "traffic-anaheim",
                    "traffic-barcelona",
                    "traffic-chicago",
                    "traffic-winnipeg",
                    "electrical-circuits",
                ],
                ["denoising", "interpolation", "simulation"],
            )
        ) + [
            "traffic-LA",
            "DAG-denoising",
            "longest-cycle-identification",
            "random-walk-denoising",
            "mixed-longest-cycle-identification",
            "typed-triangles-orientation",
        ]

        if self.name not in self.supported_datasets:
            raise ValueError(f"The dataset {self.name} is not supported!")

        dataset_cls = None
        if self.name in tuple(
            "-".join([dataset, task])
            for dataset, task in itertools.product(
                [
                    "traffic-anaheim",
                    "traffic-barcelona",
                    "traffic-chicago",
                    "traffic-winnipeg",
                ],
                ["denoising", "interpolation", "simulation"],
            )
        ):
            dataset_cls = TNTPFlowDenoisingInterpolationDataset
            self.dataset_kwargs |= dict(
                interpolation_label_size=self.config.get(
                    "interpolation_label_size", 0.75
                ),
            )
        elif self.name in tuple(
            "-".join([dataset, task])
            for dataset, task in itertools.product(
                [
                    "electrical-circuits",
                ],
                ["denoising", "interpolation", "simulation"],
            )
        ):
            dataset_cls = ElectricalCircuitsDenoisingInterpolationDataset
            self.dataset_kwargs |= dict(
                include_non_source_voltages=self.config.include_non_source_voltages,
                current_relative_to_voltage=self.config.current_relative_to_voltage,
                interpolation_label_size=self.config.get(
                    "interpolation_label_size", 0.75
                ),
            )
        elif self.name == "DAG-denoising":
            dataset_cls = DAGDenoisingDataset
        elif self.name == "longest-cycle-identification":
            dataset_cls = LongestCycleIdentificationDataset
        elif self.name == "random-walk-denoising":
            dataset_cls = RandomWalkDenoisingDataset
        elif self.name == "mixed-longest-cycle-identification":
            dataset_cls = MixedLongestCycleIdentificationDataset
        elif self.name == "typed-triangles-orientation":
            dataset_cls = TypedTrianglesOrientationDataset
        else:
            raise ValueError(f"The dataset {self.name} is not supported!")

        self.base_data = dataset_cls(
            dataset_name=self.name,
            dataset_path=self.dataset_path,
            dataset_registry_database_path=self.config.registry.database_path,
            dataset_registry_lockfile_path=self.config.registry.lockfile_path,
            dataset_registry_storage_path=self.config.registry.storage_path,
            dataset_registry_force_rebuild=self.config.registry.force_rebuild,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            seed=self.seed,
            arbitrary_orientation=self.arbitrary_orientation,
            orientation_equivariant_labels=self.orientation_equivariant_labels,
            laplacian_encodings_phase_shift=self.config.laplacian_encodings_phase_shift,
            max_num_positional_laplacian_encodings=self.config.num_laplacian_encodings,
            **self.dataset_kwargs,
        )

    def split(self, seed: int | None = None):
        """Splits the dataset into train, validation, and test sets."""
        graphs = self.base_data.split(seed=seed)
        self.train_dataset = EdgeLevelDataset(graphs["train"])
        self.val_dataset = EdgeLevelDataset(graphs["val"])
        self.test_dataset = EdgeLevelDataset(graphs["test"])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
