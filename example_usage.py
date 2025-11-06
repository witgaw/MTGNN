"""
Minimal example showing external library usage of MTGNN.

Demonstrates:
1. Using MTGNN as a library dependency
2. Training with injected data and explicit config
3. Saving the trained model
4. Loading the model with exact state
5. Running inference on loaded model
"""

import numpy as np
import sys
import os

# For running without installing package
sys.path.insert(0, os.path.dirname(__file__))

from train_multi_step import train_injected, TrainingConfig
from net import MTGNNModel


def generate_sample_data(num_nodes=207, num_samples=1000, seq_len=12, num_features=2):
    """Generate sample training data."""
    np.random.seed(42)

    data = {
        'x_train': np.random.randn(num_samples, seq_len, num_nodes, num_features).astype(np.float32),
        'y_train': np.random.randn(num_samples, seq_len, num_nodes, num_features).astype(np.float32),
        'x_val': np.random.randn(200, seq_len, num_nodes, num_features).astype(np.float32),
        'y_val': np.random.randn(200, seq_len, num_nodes, num_features).astype(np.float32),
        'x_test': np.random.randn(200, seq_len, num_nodes, num_features).astype(np.float32),
        'y_test': np.random.randn(200, seq_len, num_nodes, num_features).astype(np.float32)
    }

    adjacency = np.random.rand(num_nodes, num_nodes).astype(np.float32)
    adjacency = (adjacency + adjacency.T) / 2  # symmetric

    return data, adjacency


if __name__ == "__main__":
    # 1. Import as library dependency (done above)

    # 2. Training with explicit config and injected data
    config = TrainingConfig(
        num_nodes=207,
        epochs=2,  # Small for demo
        batch_size=64,
        learning_rate=0.001,
        device='cpu'
    )

    data, adj = generate_sample_data()

    print("Training...")
    model = train_injected(config, data, adj)
    print(f"Training complete. Validation MAE: {model.metrics['vmae']:.4f}")

    # 3. Save model
    model_path = './save/example_model.pth'
    model.save_model(model_path)

    # 4. Load model with exact state
    print("\nLoading model...")
    loaded_model = MTGNNModel.load_model(model_path, device='cpu')

    # 5. Run inference with loaded model
    print("\nRunning inference...")
    test_input = data['x_test'][:5].transpose(0, 3, 2, 1)  # (batch, features, nodes, seq_len)
    predictions = loaded_model.predict(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Learned adjacency available: {loaded_model.learned_adj is not None}")
    if loaded_model.learned_adj is not None:
        print(f"Learned adjacency shape: {loaded_model.learned_adj.shape}")

    print("\nDone!")
