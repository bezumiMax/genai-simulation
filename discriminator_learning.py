"""
дискриминатор получает на вход info: качество прохождения уровня агентом rl
info:
    time_learning (время обучения),
    total_reward (награда за лучшую политику),
    time_passage (время прохождения лучшей политики)
"""
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from discriminator import Discriminator
import random


class DiscriminatorTrainer:
    def __init__(self, generator, discriminator, lr=0.0001, beta1=0.5, beta2=0.999):
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr, betas=(beta1, beta2)
        )
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss_weight = 0.6
        self.difficulty_weight = 0.3
        self.learnability_weight = 0.3

    def generate_fake_samples(self):
        total_reward = random.uniform(-200.0, 200.0)
        time_passage = random.uniform(0.0, 40.0)
        fake_metrics = torch.tensor([total_reward, time_passage])
        fake_labels = torch.zeros(1, 1)
        return fake_metrics, fake_labels

    def prepare_dataset(self, metrics: torch.tensor):
        real_samples = metrics
        real_labels = torch.ones(1, 1)
        fake_samples, fake_labels = self.generate_fake_samples()
        all_samples = torch.cat([real_samples, fake_samples], dim=0).reshape(2, 2)
        all_labels = torch.cat([real_labels, fake_labels], dim=0)
        indices = torch.randperm(all_samples.size(0))
        return all_samples[indices], all_labels[indices]

    def train_epoch(self, real_metrics: torch.tensor):
        metrics, labels = self.prepare_dataset(real_metrics)
        dataset = TensorDataset(metrics, labels)
        total_adversarial_loss = 0
        total_auxiliary_loss = 0
        total_loss = 0
        validity_score, difficulty_score, learnability_score = self.discriminator(metrics)
        validity_score = torch.sigmoid(validity_score)
        difficulty_score = torch.sigmoid(difficulty_score)
        learnability_score = torch.sigmoid(learnability_score)
        adversarial_loss = self.adversarial_loss(validity_score, labels)
        difficulty_loss = self.adversarial_loss(difficulty_score, labels)
        learnability_loss = self.adversarial_loss(learnability_score, labels)
        auxiliary_loss = self.discriminator.calculate_auxiliary_loss(metrics.T)
        total_batch_loss = (
                adversarial_loss +
                self.difficulty_weight * difficulty_loss +
                self.learnability_weight * learnability_loss +
                self.auxiliary_loss_weight * auxiliary_loss
        )
        self.optimizer.zero_grad()
        total_batch_loss.backward()
        self.optimizer.step()
        return adversarial_loss.item(), auxiliary_loss.item(), total_batch_loss.item()

    def train_discriminator(self, real_metrics, epochs=100):
        losses = []
        for epoch in range(epochs):
            adv_loss, aux_loss, total_loss = self.train_epoch(real_metrics)
            losses.append((adv_loss, aux_loss, total_loss))
        return losses
