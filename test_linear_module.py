import torch
from cs336_basics.nn_modules import Linear

def test_linear_basic():
    """Test basic functionality of the Linear module."""
    # Create a simple linear layer
    linear = Linear(3, 2)
    
    # Test with a single input
    x = torch.randn(3)
    output = linear(x)
    assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
    
    # Test with batched input
    x_batch = torch.randn(5, 3)
    output_batch = linear(x_batch)
    assert output_batch.shape == (5, 2), f"Expected shape (5, 2), got {output_batch.shape}"
    
    # Test with multiple batch dimensions
    x_multi_batch = torch.randn(2, 4, 3)
    output_multi_batch = linear(x_multi_batch)
    assert output_multi_batch.shape == (2, 4, 2), f"Expected shape (2, 4, 2), got {output_multi_batch.shape}"
    
    print("All tests passed!")

def test_linear_manual_computation():
    """Test that the linear layer computes the correct result."""
    # Create a linear layer with known weights
    linear = Linear(2, 3)
    
    # Set specific weights
    weights = torch.tensor([
        [1.0, 2.0],  # First output neuron: 1*x1 + 2*x2
        [3.0, 4.0],  # Second output neuron: 3*x1 + 4*x2
        [5.0, 6.0],  # Third output neuron: 5*x1 + 6*x2
    ])
    linear.W.data = weights
    
    # Test input
    x = torch.tensor([1.0, 0.5])  # x1=1, x2=0.5
    
    # Expected output:
    # y1 = 1*1 + 2*0.5 = 2.0
    # y2 = 3*1 + 4*0.5 = 5.0
    # y3 = 5*1 + 6*0.5 = 8.0
    expected = torch.tensor([2.0, 5.0, 8.0])
    
    output = linear(x)
    
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-7)
    print("Manual computation test passed!")

if __name__ == "__main__":
    test_linear_basic()
    test_linear_manual_computation()
    print("Linear module implementation is working correctly!")