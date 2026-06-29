import torch
from cs336_basics.nn_modules import Embedding

def test_embedding_custom():
    """Custom test for Embedding class with small matrices"""
    
    # Test parameters
    vocab_size = 5
    d_model = 3
    
    # Create embedding layer
    embedding = Embedding(vocab_size, d_model)
    
    # Set known weights for predictable testing
    # Shape: (5, 3)
    test_weights = torch.tensor([
        [1.0, 2.0, 3.0],  # token 0
        [4.0, 5.0, 6.0],  # token 1
        [7.0, 8.0, 9.0],  # token 2
        [10.0, 11.0, 12.0],  # token 3
        [13.0, 14.0, 15.0]   # token 4
    ])
    embedding.weight.data = test_weights
    
    print("=== Custom Embedding Test ===")
    print(f"Vocab size: {vocab_size}, Embedding dim: {d_model}")
    print(f"Weight matrix shape: {embedding.weight.shape}")
    print(f"Weight matrix:\n{embedding.weight}")
    
    # Test 1: Single token lookup
    print("\n--- Test 1: Single token lookup ---")
    token_id = torch.tensor(2)
    result = embedding(token_id)
    expected = test_weights[2]
    print(f"Token ID: {token_id}")
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {torch.allclose(result, expected)}")
    
    # Test 2: Batch of tokens
    print("\n--- Test 2: Batch of tokens ---")
    token_ids = torch.tensor([0, 3, 1])
    result = embedding(token_ids)
    expected = torch.stack([test_weights[0], test_weights[3], test_weights[1]])
    print(f"Token IDs: {token_ids}")
    print(f"Result shape: {result.shape}")
    print(f"Result:\n{result}")
    print(f"Expected:\n{expected}")
    print(f"Match: {torch.allclose(result, expected)}")
    
    # Test 3: 2D input (batch_size, sequence_length)
    print("\n--- Test 3: 2D input (batch_size=2, seq_len=3) ---")
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 0]])
    result = embedding(token_ids)
    expected = torch.stack([
        torch.stack([test_weights[0], test_weights[1], test_weights[2]]),  # batch 0
        torch.stack([test_weights[3], test_weights[4], test_weights[0]])   # batch 1
    ])
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Token IDs:\n{token_ids}")
    print(f"Result shape: {result.shape}")
    print(f"Expected shape: {expected.shape}")
    print(f"Match: {torch.allclose(result, expected)}")
    
    # Test 4: Check parameter properties
    print("\n--- Test 4: Parameter properties ---")
    print(f"Weight is parameter: {isinstance(embedding.weight, torch.nn.Parameter)}")
    print(f"Weight requires grad: {embedding.weight.requires_grad}")
    print(f"Module has {sum(p.numel() for p in embedding.parameters())} parameters")
    
    # Test 5: Gradient flow
    print("\n--- Test 5: Gradient flow ---")
    token_ids = torch.tensor([1, 2])
    result = embedding(token_ids)
    loss = result.sum()
    loss.backward()
    print(f"Gradient shape: {embedding.weight.grad.shape}")
    print(f"Gradient sum: {embedding.weight.grad.sum().item()}")
    print(f"Gradient exists: {embedding.weight.grad is not None}")
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    test_embedding_custom()