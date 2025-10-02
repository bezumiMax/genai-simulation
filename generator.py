"""генерируем MDP для определенного уровня"""
import torch
from torch import nn, layout
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim=10, hidden_dim=128):
        super(Generator,  self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.GreenNet = nn.Sequential(
            nn.Linear(latent_dim + 3 + 1, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.RedNet = nn.Sequential(
            nn.Linear(latent_dim + 3 + 1, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=1.0),
            nn.Linear(hidden_dim, 1)
        )

    def smooth_sigmoid(self, x, temperature=500.0):
        return torch.sigmoid(x / temperature)

    def get_valid_transitions(self, state) -> list:
        i, j = state // self.grid_size, state % self.grid_size
        valid_transitions = []
        movements = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for di, dj in movements:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                next_state = ni * self.grid_size + nj
                valid_transitions.append(next_state)
        valid_transitions.append(state)
        return valid_transitions

    def forward(self, z, level_data: torch.tensor):
        z_extended = torch.cat([z, level_data])
        coords_green = []
        for num_green in range(level_data[1]):
            torch_num_green = torch.tensor([num_green])
            coord_green = self.GreenNet(torch.cat([z_extended, torch_num_green]))
            coord_green = self.smooth_sigmoid(coord_green) * (level_data[0] ** 2 - 1)
            coord_green = coord_green.int()
            coords_green.append(coord_green.item())
        coords_red = []
        for num_red in range(level_data[2]):
            torch_num_red = torch.tensor([num_red])
            coord_red = self.RedNet(torch.cat([z_extended, torch_num_red]))
            coord_red = self.smooth_sigmoid(coord_red) * (level_data[0] ** 2 - 1)
            coord_red = coord_red.int()
            coords_red.append(coord_red.item())
        return {
            'coords_green': coords_green,
            'coords_red': coords_red
        }
