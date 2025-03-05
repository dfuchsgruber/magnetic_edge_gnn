import torch

from .dag_denoising_dataset import DAGDenoisingDataset
from .electrical_circuits_denoising_interpolation_dataset import (
    ElectricalCircuitsDenoisingInterpolationDataset,
)
from .inductive_dataset import InductiveDataset
from .longest_cycle_identification_dataset import LongestCycleIdentificationDataset
from .mixed_longest_cycle_identification_dataset import (
    MixedLongestCycleIdentificationDataset,
)
from .random_walk_denoising_dataset import RandomWalkDenoisingDataset
from .tntp_flow_denoising_interpolation_dataset import (
    TNTPFlowDenoisingInterpolationDataset,
)
from .transductive_dataset import TransductiveDataset
from .typed_triangles_orientation_dataset import TypedTrianglesOrientationDataset

from .base_dataset import BaseDataset

torch.serialization.add_safe_globals(
    [
        InductiveDataset,
        TransductiveDataset,
        TNTPFlowDenoisingInterpolationDataset,
        ElectricalCircuitsDenoisingInterpolationDataset,
        RandomWalkDenoisingDataset,
        DAGDenoisingDataset,
        LongestCycleIdentificationDataset,
        MixedLongestCycleIdentificationDataset,
        TypedTrianglesOrientationDataset,
    ]
)

__all__ = [
    "InductiveDataset",
    "TransductiveDataset",
    "TNTPFlowDenoisingInterpolationDataset",
    "ElectricalCircuitsDenoisingInterpolationDataset",
    "RandomWalkDenoisingDataset",
    "DAGDenoisingDataset",
    "LongestCycleIdentificationDataset",
    "MixedLongestCycleIdentificationDataset",
    "TypedTrianglesOrientationDataset",
    "BaseDataset",
]
