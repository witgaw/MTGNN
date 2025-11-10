import numpy as np
import torch
from safetensors.torch import save_file, load_file
import json
import os

from .layer import *


class gtnet(nn.Module):
    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        device,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=2,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.gc = graph_constructor(
            num_nodes,
            subgraph_size,
            node_dim,
            device,
            alpha=tanhalpha,
            static_feat=static_feat,
        )

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )
                    self.gconv2.append(
                        mixprop(
                            conv_channels,
                            residual_channels,
                            gcn_depth,
                            dropout,
                            propalpha,
                        )
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.seq_length - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (
                                residual_channels,
                                num_nodes,
                                self.receptive_field - rf_size_j + 1,
                            ),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, (
            "input sequence length not equal to preset sequence length"
        )

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(
                input, (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class MTGNNModel:
    """
    High-level wrapper for MTGNN model with serialization and inference capabilities.
    """

    def __init__(self, config=None, model=None):
        """
        Initialize MTGNNModel.

        Args:
            config (dict): Model configuration parameters
            model (gtnet): Pre-trained model instance
        """
        if model is not None:
            self.model = model
            self.config = config or {}
        else:
            if config is None:
                raise ValueError("Either config or model must be provided")
            self.config = config
            self.model = self._create_model_from_config(config)

        self.scaler = None
        self.learned_adj = None
        self.device = config.get("device", "cpu") if config else "cpu"

    def _create_model_from_config(self, config):
        """Create gtnet model from configuration."""
        device = torch.device(config.get("device", "cpu"))

        # Handle both TrainingConfig keys (seq_in_len, seq_out_len) and old keys (seq_length, out_dim)
        seq_length = config.get("seq_in_len", config.get("seq_length", 12))
        out_dim = config.get("seq_out_len", config.get("out_dim", 12))

        return gtnet(
            gcn_true=config.get("gcn_true", True),
            buildA_true=config.get("buildA_true", True),
            gcn_depth=config.get("gcn_depth", 2),
            num_nodes=config["num_nodes"],
            device=device,
            predefined_A=config.get("predefined_A"),
            static_feat=config.get("static_feat"),
            dropout=config.get("dropout", 0.3),
            subgraph_size=config.get("subgraph_size", 20),
            node_dim=config.get("node_dim", 40),
            dilation_exponential=config.get("dilation_exponential", 1),
            conv_channels=config.get("conv_channels", 32),
            residual_channels=config.get("residual_channels", 32),
            skip_channels=config.get("skip_channels", 64),
            end_channels=config.get("end_channels", 128),
            seq_length=seq_length,
            in_dim=config.get("in_dim", 2),
            out_dim=out_dim,
            layers=config.get("layers", 3),
            propalpha=config.get("propalpha", 0.05),
            tanhalpha=config.get("tanhalpha", 3),
            layer_norm_affline=config.get("layer_norm_affline", True),
        ).to(device)

    def save_model(self, path):
        """
        Save model and configuration to disk.

        Args:
            path (str): Path to save the model
        """
        # Prepare tensors for safetensors
        tensors = self.model.state_dict().copy()
        
        # Add scaler and learned_adj as tensors if they exist
        if self.scaler is not None:
            if hasattr(self.scaler, 'mean') and hasattr(self.scaler, 'std'):
                tensors['scaler_mean'] = torch.tensor(self.scaler.mean)
                tensors['scaler_std'] = torch.tensor(self.scaler.std)
        
        if self.learned_adj is not None:
            if isinstance(self.learned_adj, np.ndarray):
                tensors['learned_adj'] = torch.from_numpy(self.learned_adj)
            else:
                tensors['learned_adj'] = self.learned_adj
        
        # Save tensors with safetensors
        save_file(tensors, path)
        
        # Save config as JSON separately
        config_path = path.replace('.safetensors', '_config.json')
        if not config_path.endswith('_config.json'):
            config_path = path + '_config.json'
        
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Model saved to {path}")
        print(f"Config saved to {config_path}")
        if self.learned_adj is not None:
            print(
                f"  - Includes learned adjacency matrix of shape {self.learned_adj.shape}"
            )

    @classmethod
    def load_model(cls, path, device=None):
        """
        Load model from disk.

        Args:
            path (str): Path to the saved model
            device (str): Device to load model on

        Returns:
            MTGNNModel: Loaded model instance
        """
        # Load tensors with safetensors
        tensors = load_file(path, device=device)
        
        # Load config from JSON
        config_path = path.replace('.safetensors', '_config.json')
        if not config_path.endswith('_config.json'):
            config_path = path + '_config.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)

        if device is not None:
            config["device"] = device

        model_wrapper = cls(config=config)
        
        # Extract model state dict (excluding scaler and learned_adj tensors)
        model_state_dict = {}
        scaler_mean = None
        scaler_std = None
        learned_adj = None
        
        for key, tensor in tensors.items():
            if key == 'scaler_mean':
                scaler_mean = tensor
            elif key == 'scaler_std':
                scaler_std = tensor
            elif key == 'learned_adj':
                learned_adj = tensor
            else:
                model_state_dict[key] = tensor
        
        model_wrapper.model.load_state_dict(model_state_dict)
        
        # Reconstruct scaler if available
        if scaler_mean is not None and scaler_std is not None:
            class SimpleScaler:
                def __init__(self, mean, std):
                    self.mean = mean.item() if mean.dim() == 0 else mean
                    self.std = std.item() if std.dim() == 0 else std
                
                def inverse_transform(self, data):
                    return data * self.std + self.mean
            
            model_wrapper.scaler = SimpleScaler(scaler_mean, scaler_std)
        
        model_wrapper.learned_adj = learned_adj

        print(f"Model loaded from {path}")
        if model_wrapper.learned_adj is not None:
            print(
                f"  - Loaded learned adjacency matrix of shape {model_wrapper.learned_adj.shape}"
            )

        return model_wrapper

    def predict(self, input_data):
        """
        Make predictions on input data.

        Args:
            input_data (np.ndarray or torch.Tensor): Input sequences
                Shape: (batch_size, features, nodes, seq_length)

        Returns:
            np.ndarray: Predictions
        """
        self.model.eval()

        # Convert input to tensor if needed
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()

        # Move to device
        input_tensor = input_tensor.to(self.device)

        # Ensure correct shape: (batch, features, nodes, seq_len)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(1)

        # Apply input normalization if scaler is available
        # During training, inputs are normalized, so we must do the same for inference
        if self.scaler is not None:
            # Normalize first feature (speed) only - shape: (batch, features, nodes, seq_len)
            input_tensor[:, 0, :, :] = (
                input_tensor[:, 0, :, :] - self.scaler.mean
            ) / self.scaler.std

        with torch.no_grad():
            predictions = self.model(input_tensor)

        # Convert back to numpy
        predictions = predictions.cpu().numpy()

        # Apply inverse scaling if scaler is available
        if self.scaler is not None:
            # Model outputs shape: (batch, horizon, nodes, features)
            # Scaler operates on last dimension (features are normalized/denormalized)
            # Since scaler.mean and scaler.std are scalars, we can directly apply them
            predictions = self.scaler.inverse_transform(predictions)

        return predictions

    def set_scaler(self, scaler):
        """Set the data scaler for inverse transformation."""
        self.scaler = scaler

    def set_learned_adj(self, learned_adj):
        """Set the learned adjacency matrix."""
        self.learned_adj = learned_adj

    @classmethod
    def from_training_results(cls, results):
        """
        Create MTGNNModel from training results dictionary.

        Args:
            results (dict): Results from train_multi_step.main()

        Returns:
            MTGNNModel: Wrapped model ready for inference and saving
        """
        config = results.get("config", {})

        model_wrapper = cls(config=config, model=results["model"])
        model_wrapper.scaler = results.get("scaler")
        model_wrapper.learned_adj = results.get("learned_adj")

        return model_wrapper
