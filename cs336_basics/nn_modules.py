import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
    """
    A linear transformation module that performs y = xW^T without bias.
    
    This implementation uses einsum for batched matrix multiplication and 
    supports arbitrary leading batch dimensions.
    """
    
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Initialize the Linear layer.
        
        Args:
            in_features (int): Size of the input dimension
            out_features (int): Size of the output dimension
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Create weight parameter W of shape (out_features, in_features)
        # Note: We store W (not W^T) for memory ordering reasons as specified
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the weight parameter using truncated normal distribution."""
        torch.nn.init.trunc_normal_(self.W)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """
        Apply the linear transformation to the input.
        
        Args:
            x (Float[Tensor, "... d_in"]): Input tensor with arbitrary leading dimensions
        
        Returns:
            Float[Tensor, "... d_out"]: Transformed output tensor
        """
        # Use einsum to perform batched matrix multiplication
        # x has shape (..., in_features), W has shape (out_features, in_features)
        # Result should have shape (..., out_features)
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """
    A custom embedding layer that maps integer token IDs to vectors.
    
    This implementation performs embedding lookup by indexing into an embedding
    matrix of shape (vocab_size, d_model).
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Initialize the Embedding layer.
        
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors (d_model)
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None): Data type of the parameters
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding matrix as a parameter
        # Shape: (vocab_size, d_model) with d_model as the final dimension
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        
        # Initialize weights using truncated normal distribution
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize the embedding weights using truncated normal distribution."""
        torch.nn.init.trunc_normal_(self.weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.
        
        Args:
            token_ids (torch.Tensor): Token IDs with shape (batch_size, sequence_length)
                                    or any shape (...,)
        
        Returns:
            torch.Tensor: Embedding vectors with shape (..., embedding_dim)
        """
        # Use indexing to select embedding vectors for each token ID
        # token_ids shape: (...,), weight shape: (num_embeddings, embedding_dim)
        # Result shape: (..., embedding_dim)
        return self.weight[token_ids]