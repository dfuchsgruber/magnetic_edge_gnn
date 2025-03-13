from torch import nn


class BaseModel(nn.Module):
    """Base class for all models."""

    @property
    def can_return_embeddings(self) -> bool:
        False
        
    @property
    def requires_laplacian_eigenvectors(self) -> bool:
        return False
