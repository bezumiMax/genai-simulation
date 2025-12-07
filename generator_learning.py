import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from create_mdp import create_mdp
from policy_iteration import policy_improvement, policy_evaluation, policy_iteration
from double_q_learning import double_q_learning


class GeneratorTrainer:
    def __init__(self, grid_size, material_point, generator, discriminator, latent_dim=10, lr_g=0.0002, lr_d=0.0001):
        self.grid_size = grid_size
        self.material_point = material_point
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss_weight = 0.6
        self.difficulty_weight = 0.3
        self.learnability_weight = 0.3

    def train_generator(self):
        self.generator.train()
        self.discriminator.eval()
        z = torch.randn(self.latent_dim)
        level_data = torch.tensor([self.grid_size, torch.randint(1, self.grid_size, (1,)).item(),
                                       torch.randint(0, self.grid_size, (1,)).item()])
        fake_data = self.generator(z, level_data)
        coords_green = fake_data['coords_green']
        coords_red = fake_data['coords_red']
        fake_data = double_q_learning(self.material_point, level_data[0], level_data[1], level_data[2], coords_green, coords_red)
        validity_score, difficulty_score, learnability_score = self.discriminator(fake_data)
        target_valid = torch.ones_like(validity_score)
        target_diff = torch.ones_like(difficulty_score)
        target_learn = torch.ones_like(learnability_score)
        adversarial_loss = self.adversarial_loss(validity_score,target_valid)
        difficulty_loss = self.adversarial_loss(difficulty_score, target_diff)
        learnability_loss = self.adversarial_loss(learnability_score, target_learn)
        auxiliary_loss = self.discriminator.calculate_auxiliary_loss(fake_data)
        total_batch_loss = (
                adversarial_loss +
                self.difficulty_weight * difficulty_loss +
                self.learnability_weight * learnability_loss +
                self.auxiliary_loss_weight * auxiliary_loss
        )
        self.optimizer_G.zero_grad()
        total_batch_loss.backward()
        self.optimizer_G.step()
        return adversarial_loss.item(), auxiliary_loss.item(), total_batch_loss.item()
