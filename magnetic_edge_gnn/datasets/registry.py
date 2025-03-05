"""Dataset registry for storing pre-computed datasets and the eigendecompositions of the Magnetic Laplacians."""

from magnetic_edge_gnn.registry import StorageRegistry
from torch_geometric.data import Data
from pathlib import Path
from typing import NamedTuple
import torch

class DatasetRegistryKey(NamedTuple):
    """Key in the dataset registry for a precomputed dataset."""
    name: str
    q: float
    num_laplacian_eigenvectors: int
    
    def __str__(self):
        q = f"{self.q}".replace(".", "_")
        return f"{self.name}_q={q}_num_laplacian_eigenvectors={self.num_laplacian_eigenvectors}"



class DatasetRegistry(StorageRegistry[DatasetRegistryKey, Data]):
    """Class for storing pre-computed datasets."""

    def __init__(self, database_path: str, lockfile_path: str, storage_path: str):
        super().__init__(
            database_path=database_path,
            lockfile_path=lockfile_path,
            storage_path=storage_path,
        )

    def serialize(self, value: Data, path: Path):
        torch.save(
            value,
            path,
        )

    def deserialize(self, path: Path) -> Data:
        deserialized = torch.load(path)
        return deserialized

