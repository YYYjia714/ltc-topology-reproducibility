from __future__ import annotations

import torch
from torch import nn

from .topology import get_edges


class LiquidTimeConstantCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_linear = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([x_t, h_t], dim=-1)
        tau = torch.sigmoid(self.tau_linear(joined))
        candidate = torch.tanh(
            self.candidate_linear(joined)
            + self.input_linear(x_t)
            + self.hidden_linear(h_t)
        )
        return h_t + tau * (candidate - h_t)


class LTCEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LiquidTimeConstantCell(
                    input_size=input_size if layer_index == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = x.shape
        states = [
            x.new_zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)
        ]
        outputs = []
        for step in range(steps):
            layer_input = x[:, step]
            for layer_index, cell in enumerate(self.layers):
                states[layer_index] = cell(layer_input, states[layer_index])
                layer_input = states[layer_index]
            outputs.append(layer_input.unsqueeze(1))
        return torch.cat(outputs, dim=1)


def build_normalized_adjacency(
    num_joints: int,
    skeleton_edges: list[list[int]] | list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    edges = get_edges(num_joints) if skeleton_edges is None else skeleton_edges
    for src, dst in edges:
        if not (0 <= src < num_joints and 0 <= dst < num_joints):
            raise ValueError(f"Invalid skeleton edge ({src}, {dst}) for {num_joints} joints")
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


class SkeletonGraphBlock(nn.Module):
    def __init__(self, features: int, adjacency: torch.Tensor, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.self_linear = nn.Linear(features, features)
        self.neighbor_linear = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-2], original_shape[-1])
        neighbors = torch.einsum("ij,bjf->bif", self.adjacency, x_flat)
        updated = self.self_linear(x_flat) + self.neighbor_linear(neighbors)
        updated = torch.nn.functional.gelu(updated)
        updated = self.dropout(updated)
        updated = self.norm(updated + x_flat)
        return updated.reshape(*original_shape)


class TopologyAwareLTCModel(nn.Module):
    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_topology_encoder: bool = True,
        use_topology_decoder: bool = True,
        use_root_guidance: bool = True,
        skeleton_edges: list[list[int]] | list[tuple[int, int]] | None = None,
        guidance_joint_index: int = 0,
    ):
        super().__init__()
        self.joints = joints
        self.history = history
        self.future = future
        self.hidden_size = hidden_size
        self.node_hidden = max(64, hidden_size // 2)
        self.use_topology_encoder = use_topology_encoder
        self.use_topology_decoder = use_topology_decoder
        self.use_root_guidance = use_root_guidance
        if not 0 <= guidance_joint_index < joints:
            raise ValueError(f"Invalid guidance joint {guidance_joint_index} for {joints} joints")
        self.guidance_joint_index = guidance_joint_index

        adjacency = build_normalized_adjacency(joints, skeleton_edges=skeleton_edges)
        self.input_projection = nn.Linear(3, self.node_hidden)
        self.spatial_encoder = nn.ModuleList(
            [
                SkeletonGraphBlock(self.node_hidden, adjacency, dropout=dropout),
                SkeletonGraphBlock(self.node_hidden, adjacency, dropout=dropout),
            ]
        )
        self.temporal_encoder = LTCEncoder(
            input_size=self.node_hidden * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.temporal_to_future = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, future * self.node_hidden),
        )
        self.joint_embedding = nn.Parameter(torch.randn(1, 1, joints, self.node_hidden) * 0.02)
        self.decoder_blocks = nn.ModuleList(
            [
                SkeletonGraphBlock(self.node_hidden, adjacency, dropout=dropout),
                SkeletonGraphBlock(self.node_hidden, adjacency, dropout=dropout),
            ]
        )
        self.output_projection = nn.Linear(self.node_hidden, 3)

    def encode_spatial(self, inputs: torch.Tensor) -> torch.Tensor:
        nodes = self.input_projection(inputs)
        if self.use_topology_encoder:
            for block in self.spatial_encoder:
                nodes = block(nodes)
        pooled_mean = nodes.mean(dim=2)
        if self.use_root_guidance:
            root_feature = nodes[:, :, self.guidance_joint_index, :]
        else:
            root_feature = torch.zeros_like(pooled_mean)
        return torch.cat([pooled_mean, root_feature], dim=-1)

    def decode_future(self, features: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        future_context = self.temporal_to_future(features).reshape(
            batch_size, self.future, 1, self.node_hidden
        )
        joint_tokens = future_context + self.joint_embedding
        if self.use_topology_decoder:
            for block in self.decoder_blocks:
                joint_tokens = block(joint_tokens)
        deltas = self.output_projection(joint_tokens)
        return last_frame[:, None, :, :] + deltas

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        temporal_inputs = self.encode_spatial(inputs)
        encoded = self.temporal_encoder(temporal_inputs)
        features = encoded[:, -1]
        last_frame = inputs[:, -1]
        return self.decode_future(features, last_frame)


class MotionForecastModel(nn.Module):
    def __init__(
        self,
        model_type: str,
        joints: int,
        history: int,
        future: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_topology_encoder: bool = True,
        use_topology_decoder: bool = True,
        use_root_guidance: bool = True,
        skeleton_edges: list[list[int]] | list[tuple[int, int]] | None = None,
        guidance_joint_index: int = 0,
    ):
        super().__init__()
        if model_type not in {"ltc", "lstm", "gru", "ltc_topology"}:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.model_type = model_type
        self.joints = joints
        self.history = history
        self.future = future
        self.input_size = joints * 3
        self.output_size = future * joints * 3
        self.hidden_size = hidden_size

        if model_type == "ltc_topology":
            self.encoder = TopologyAwareLTCModel(
                joints=joints,
                history=history,
                future=future,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_topology_encoder=use_topology_encoder,
                use_topology_decoder=use_topology_decoder,
                use_root_guidance=use_root_guidance,
                skeleton_edges=skeleton_edges,
                guidance_joint_index=guidance_joint_index,
            )
            self.head = None
        elif model_type == "ltc":
            self.encoder = LTCEncoder(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
            )
        elif model_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.encoder = nn.GRU(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        if model_type != "ltc_topology":
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.output_size),
            )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model_type == "ltc_topology":
            return self.encoder(inputs)
        batch_size = inputs.shape[0]
        x = inputs.reshape(batch_size, self.history, self.input_size)
        if self.model_type == "ltc":
            encoded = self.encoder(x)
            features = encoded[:, -1]
        else:
            encoded, hidden = self.encoder(x)
            if isinstance(hidden, tuple):
                features = hidden[0][-1]
            else:
                features = hidden[-1]
        outputs = self.head(features)
        return outputs.reshape(batch_size, self.future, self.joints, 3)
