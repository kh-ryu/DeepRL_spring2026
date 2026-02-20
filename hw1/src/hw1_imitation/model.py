"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.model(state)
        pred = pred.view(state.size(0), self.chunk_size, self.action_dim)
        loss = nn.functional.mse_loss(pred, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred = self.model(state)
        pred = pred.view(state.size(0), self.chunk_size, self.action_dim)
        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim + chunk_size * action_dim + 1  # state + flattened action chunk + timestep
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.model = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Sample from prior distribution on the correct device
        prior_sample = torch.randn(state.size(0), self.chunk_size, self.action_dim, device=state.device)

        # Sample a random time step for each sample in the batch
        time_step = torch.rand(state.size(0), 1, 1, device=state.device)  # (batch, 1, 1) for broadcasting
        interpolation = time_step * action_chunk + (1 - time_step) * prior_sample  # (batch, chunk_size, action_dim)

        # Ground truth velocity
        velocity = action_chunk - prior_sample  # (batch, chunk_size, action_dim)

        # Model prediction
        time_step_flat = time_step.view(state.size(0), 1)  # (batch, 1)
        interp_flat = interpolation.view(state.size(0), -1)  # (batch, chunk_size * action_dim)
        model_input = torch.cat([state, interp_flat, time_step_flat], dim=1)  # (batch, state_dim + chunk_size*action_dim + 1)
        pred = self.model(model_input)  # (batch, chunk_size * action_dim)
        pred = pred.view(state.size(0), self.chunk_size, self.action_dim)  # (batch, chunk_size, action_dim)

        # Compute flow matching loss
        loss = torch.mean((pred - velocity) ** 2)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Sample initial noise from the prior distribution on the correct device
        batch_size = state.size(0)
        sample = torch.randn(batch_size, self.chunk_size, self.action_dim, device=state.device)  # (batch, chunk_size, action_dim)

        # Integrate the flow over time steps for steps=0 to 1
        for step in range(num_steps):
            time_step = torch.full((batch_size, 1), step / num_steps, device=state.device)  # (batch, 1)
            sample_flat = sample.view(batch_size, -1)  # (batch, chunk_size * action_dim)
            model_input = torch.cat([state, sample_flat, time_step], dim=1)  # (batch, state_dim + chunk_size*action_dim + 1)
            pred = self.model(model_input)  # (batch, chunk_size * action_dim)
            pred = pred.view(batch_size, self.chunk_size, self.action_dim)  # (batch, chunk_size, action_dim)

            # Euler integration step
            sample = sample + pred * (1 / num_steps)

        return sample


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
