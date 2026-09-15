from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .models import MotionForecastModel, build_normalized_adjacency


def orthonormal_dct(length: int) -> torch.Tensor:
    matrix = torch.empty(length, length, dtype=torch.float32)
    scale0 = math.sqrt(1.0 / length)
    scale = math.sqrt(2.0 / length)
    for frequency in range(length):
        factor = scale0 if frequency == 0 else scale
        for position in range(length):
            matrix[frequency, position] = factor * math.cos(
                math.pi * (position + 0.5) * frequency / length
            )
    return matrix


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class SiMLPeCommon18(nn.Module):
    """DCT temporal-MLP adaptation with the public siMLPe residual design."""

    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        hidden_size: int = 256,
        depth: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.history = history
        self.future = future
        self.total = history + future
        self.register_buffer("dct", orthonormal_dct(self.total))
        self.input_projection = nn.Linear(self.total, hidden_size)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_size, dropout) for _ in range(depth)
        )
        self.output_projection = nn.Linear(hidden_size, self.total)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        last = inputs[:, -1:]
        padded = torch.cat([inputs, last.expand(-1, self.future, -1, -1)], dim=1)
        batch = padded.shape[0]
        coordinates = padded.reshape(batch, self.total, self.joints * 3)
        coefficients = torch.einsum("ft,btn->bnf", self.dct, coordinates)
        hidden = self.input_projection(coefficients)
        for block in self.blocks:
            hidden = block(hidden)
        predicted_coefficients = self.output_projection(hidden)
        reconstructed = torch.einsum(
            "tf,bnf->btn", self.dct.transpose(0, 1), predicted_coefficients
        )
        return reconstructed[:, self.history :].reshape(
            batch, self.future, self.joints, 3
        )


class GraphResidualBlock(nn.Module):
    def __init__(self, width: int, adjacency: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.self_linear = nn.Linear(width, width)
        self.neighbor_linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjf->bif", self.adjacency, nodes)
        update = self.self_linear(nodes) + self.neighbor_linear(neighbors)
        return self.norm(nodes + self.dropout(torch.nn.functional.gelu(update)))


def common18_group_assignment(joints: int) -> torch.Tensor:
    if joints != 18:
        raise ValueError("The frozen common18 MSR-GCN adapter requires 18 joints")
    groups = (
        (2, 5, 8, 11),
        (9, 12, 14, 16),
        (10, 13, 15, 17),
        (0, 3, 6),
        (1, 4, 7),
    )
    assignment = torch.zeros(len(groups), joints, dtype=torch.float32)
    for group_index, members in enumerate(groups):
        assignment[group_index, list(members)] = 1.0 / len(members)
    return assignment


class MSRGCNCommon18(nn.Module):
    """Two-scale residual GCN adaptation for the frozen common18 skeleton."""

    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        skeleton_edges: list[list[int]],
        hidden_size: int = 256,
        depth: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.history = history
        self.future = future
        self.total = history + future
        self.register_buffer("dct", orthonormal_dct(self.total))
        fine_adjacency = build_normalized_adjacency(joints, skeleton_edges)
        assignment = common18_group_assignment(joints)
        coarse_adjacency = assignment @ fine_adjacency @ assignment.transpose(0, 1)
        coarse_adjacency = coarse_adjacency / coarse_adjacency.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        self.register_buffer("assignment", assignment)
        self.input_projection = nn.Linear(3 * self.total, hidden_size)
        self.fine_blocks = nn.ModuleList(
            GraphResidualBlock(hidden_size, fine_adjacency, dropout)
            for _ in range(depth)
        )
        self.coarse_blocks = nn.ModuleList(
            GraphResidualBlock(hidden_size, coarse_adjacency, dropout)
            for _ in range(max(2, depth // 2))
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
        )
        self.output_projection = nn.Linear(hidden_size, 3 * self.total)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        last = inputs[:, -1:]
        padded = torch.cat([inputs, last.expand(-1, self.future, -1, -1)], dim=1)
        coefficients = torch.einsum("ft,btjc->bjcf", self.dct, padded)
        fine = self.input_projection(coefficients.flatten(2))
        for block in self.fine_blocks:
            fine = block(fine)
        coarse = torch.einsum("gj,bjf->bgf", self.assignment, fine)
        for block in self.coarse_blocks:
            coarse = block(coarse)
        coarse_up = torch.einsum(
            "jg,bgf->bjf", self.assignment.transpose(0, 1), coarse
        )
        fused = self.fusion(torch.cat([fine, coarse_up], dim=-1))
        predicted_coefficients = self.output_projection(fused).reshape(
            inputs.shape[0], self.joints, 3, self.total
        )
        reconstructed = torch.einsum(
            "tf,bjcf->btjc", self.dct.transpose(0, 1), predicted_coefficients
        )
        return reconstructed[:, self.history :]


class STTransformerCommon18(nn.Module):
    """Decoupled spatial-temporal Transformer adaptation for 3D motion prediction."""

    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        hidden_size: int = 256,
        depth: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.joints = joints
        self.history = history
        self.future = future
        self.input_projection = nn.Linear(3, hidden_size)
        self.joint_embedding = nn.Parameter(torch.randn(1, 1, joints, hidden_size) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(1, history, 1, hidden_size) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout, batch_first=True, norm_first=True
        )
        temporal_layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout, batch_first=True, norm_first=True
        )
        self.spatial = nn.TransformerEncoder(spatial_layer, depth)
        self.temporal = nn.TransformerEncoder(temporal_layer, depth)
        self.future_embedding = nn.Parameter(torch.randn(1, future, 1, hidden_size) * 0.02)
        future_layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout, batch_first=True, norm_first=True
        )
        self.future_decoder = nn.TransformerEncoder(future_layer, max(1, depth // 2))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        tokens = self.input_projection(inputs) + self.joint_embedding + self.time_embedding
        spatial = self.spatial(tokens.reshape(batch * self.history, self.joints, -1))
        spatial = spatial.reshape(batch, self.history, self.joints, -1)
        temporal = spatial.permute(0, 2, 1, 3).reshape(
            batch * self.joints, self.history, -1
        )
        temporal = self.temporal(temporal)[:, -1]
        context = temporal.reshape(batch, self.joints, -1)[:, None]
        future_tokens = context + self.future_embedding + self.joint_embedding
        future_tokens = future_tokens.permute(0, 2, 1, 3).reshape(
            batch * self.joints, self.future, -1
        )
        future_tokens = self.future_decoder(future_tokens).reshape(
            batch, self.joints, self.future, -1
        ).permute(0, 2, 1, 3)
        offsets = self.output(future_tokens)
        return inputs[:, -1:, :, :] + offsets


def timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < width:
        embedding = torch.nn.functional.pad(embedding, (0, width - embedding.shape[-1]))
    return embedding


class DiffusionDenoiser(nn.Module):
    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        hidden_size: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.future = future
        self.context = nn.GRU(joints * 3, hidden_size, batch_first=True)
        self.noisy_projection = nn.Linear(3, hidden_size)
        self.joint_embedding = nn.Parameter(torch.randn(1, 1, joints, hidden_size) * 0.02)
        self.future_embedding = nn.Parameter(torch.randn(1, future, 1, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout, batch_first=True, norm_first=True
        )
        self.denoiser = nn.TransformerEncoder(layer, depth)
        self.output = nn.Linear(hidden_size, 3)

    def forward(
        self, inputs: torch.Tensor, noisy_future: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        batch = inputs.shape[0]
        _, hidden = self.context(inputs.reshape(batch, inputs.shape[1], -1))
        context = hidden[-1][:, None, None, :]
        time = timestep_embedding(timesteps, context.shape[-1])[:, None, None, :]
        tokens = (
            self.noisy_projection(noisy_future)
            + self.joint_embedding
            + self.future_embedding
            + context
            + time
        )
        tokens = self.denoiser(tokens.reshape(batch, self.future * self.joints, -1))
        return self.output(tokens).reshape(batch, self.future, self.joints, 3)


class HumanMACCommon18(nn.Module):
    """Masked-completion diffusion adaptation with deterministic DDIM validation."""

    def __init__(
        self,
        joints: int,
        history: int,
        future: int,
        hidden_size: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 50,
    ) -> None:
        super().__init__()
        self.joints = joints
        self.history = history
        self.future = future
        self.diffusion_steps = diffusion_steps
        self.denoiser = DiffusionDenoiser(
            joints, history, future, hidden_size, depth, heads, dropout
        )
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def training_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        timesteps = torch.randint(
            0, self.diffusion_steps, (batch,), device=inputs.device
        )
        noise = torch.randn_like(targets)
        alpha_bar = self.alpha_bars[timesteps].view(batch, 1, 1, 1)
        noisy = alpha_bar.sqrt() * targets + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(inputs, noisy, timesteps)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, inputs: torch.Tensor, initial_noise: torch.Tensor | None = None) -> torch.Tensor:
        values = (
            torch.zeros(
                inputs.shape[0], self.future, self.joints, 3,
                device=inputs.device, dtype=inputs.dtype
            )
            if initial_noise is None
            else initial_noise
        )
        for step in range(self.diffusion_steps - 1, -1, -1):
            timesteps = torch.full(
                (inputs.shape[0],), step, device=inputs.device, dtype=torch.long
            )
            predicted_noise = self.denoiser(inputs, values, timesteps)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            previous_bar = self.alpha_bars[step - 1] if step > 0 else alpha_bar.new_tensor(1.0)
            predicted_clean = (
                values - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt().clamp_min(1e-6)
            values = previous_bar.sqrt() * predicted_clean + (
                1.0 - previous_bar
            ).sqrt() * predicted_noise
        return values

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sample(inputs)


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    family: str
    implementation_status: str


BASELINE_SPECS = {
    "lstm": BaselineSpec("Controlled LSTM", "recurrent", "project_native"),
    "msrgcn": BaselineSpec(
        "MSR-GCN common18 adaptation", "graph", "controlled_reimplementation"
    ),
    "simlpe": BaselineSpec(
        "siMLPe common18 adaptation", "mlp", "controlled_reimplementation"
    ),
    "st_transformer": BaselineSpec(
        "ST-Transformer common18 adaptation", "transformer", "controlled_reimplementation"
    ),
    "humanmac": BaselineSpec(
        "HumanMAC common18 adaptation", "diffusion", "controlled_reimplementation"
    ),
}


def build_reviewer_baseline(
    kind: str,
    joints: int,
    history: int,
    future: int,
    skeleton_edges: list[list[int]],
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    if kind == "lstm":
        return MotionForecastModel(
            model_type="lstm",
            joints=joints,
            history=history,
            future=future,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if kind == "msrgcn":
        return MSRGCNCommon18(
            joints, history, future, skeleton_edges, hidden_size, num_layers + 2, dropout
        )
    if kind == "simlpe":
        return SiMLPeCommon18(
            joints, history, future, hidden_size, max(4, num_layers * 4), dropout
        )
    if kind == "st_transformer":
        return STTransformerCommon18(
            joints, history, future, hidden_size, max(2, num_layers), 8, dropout
        )
    if kind == "humanmac":
        return HumanMACCommon18(
            joints, history, future, hidden_size, max(2, num_layers), 8, dropout
        )
    raise ValueError(f"Unsupported reviewer baseline: {kind}")
