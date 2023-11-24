import torch
from nanodpo.causal_transformer import CausalTransformer

def test_initialization():
    # Test initialization with different configurations
    model = CausalTransformer(d_feature=10, d_model=32, n_head=4, n_layer=2)
    assert model is not None, "Failed to initialize CausalTransformer"

def test_forward_pass():
    # Create a dummy input
    dummy_input = torch.rand(5, 10, 10)  # Batch size = 5, Sequence length = 10, Feature size = 10
    model = CausalTransformer(d_feature=10, d_model=32, n_head=4, n_layer=2)

    # Test forward pass
    try:
        output = model(dummy_input)
        assert output is not None, "Forward pass returned None"
    except Exception as e:
        assert False, f"Forward pass raised an exception: {e}"

def test_output_shape():
    # Define input parameters
    batch_size = 5
    seq_length = 10
    d_feature = 10
    num_actions = 3  # As defined in the CausalTransformer class

    # Create a dummy input
    dummy_input = torch.rand(batch_size, seq_length, d_feature)
    model = CausalTransformer(d_feature=d_feature, d_model=32, n_head=4, n_layer=2, num_actions=num_actions)

    # Test output shape
    output = model(dummy_input)
    expected_shape = (batch_size, num_actions)
    assert output.shape == expected_shape, f"Output shape is incorrect. Expected: {expected_shape}, Got: {output.shape}"

