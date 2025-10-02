"""
дискриминатор получает на вход info: качество прохождения уровня агентом rl
info:
    time_learning (время обучения),
    total_reward (награда за лучшую политику),
    time_passage (время прохождения лучшей политики)
"""
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, dropout_rate=0.3):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(alpha=1.0, inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.learnability_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.scaler = StandardScaler()

    def preprocess(self, metrics: torch.tensor):
        """Предобработка и нормализация входных данных"""
        if metrics.dim() == 1:
            metrics = metrics.unsqueeze(0)
        if metrics.size(0) == 3 and metrics.size(1) != 3:
            metrics = metrics.T
        log_time_learning = torch.log(metrics[:, 0] + 1e-8)
        log_time_passage = torch.log(metrics[:, 2] + 1e-8)
        normalized_reward = (metrics[:, 1] - metrics[:, 1].min()) / (
                metrics[:, 1].max() - metrics[:, 1].min() + 1e-8)
        inputs = torch.stack([log_time_learning, normalized_reward, log_time_passage], dim=1)
        inputs_np = inputs.detach().numpy()
        return inputs

    def forward(self, metrics: torch.tensor):
        metrics = metrics.T
        x = self.preprocess(metrics)
        features = self.feature_extractor(x)
        validity_score = self.classifier(features)
        difficulty_score = self.difficulty_head(features)
        learnability_score = self.learnability_head(features)
        return validity_score, difficulty_score, learnability_score

    def get_discrimination_criteria(self):
        return {
            'optimal_time_learning': (10.0, 100.0),
            'optimal_total_reward': (-1000.0, 0.0),
            'optimal_time_passage': (5.0, 100.0)
        }

    def calculate_auxiliary_loss(self, metrics: torch.Tensor):
        criteria = self.get_discrimination_criteria()

        if metrics.dim() == 1:
            metrics = metrics.unsqueeze(0)

        if metrics.size(0) == 3 and metrics.size(1) != 3:
            metrics = metrics.T

        time_learning_loss = torch.mean(
            torch.relu(criteria['optimal_time_learning'][0] - metrics[:, 0]) +
            torch.relu(metrics[:, 0] - criteria['optimal_time_learning'][1])
        )
        reward_loss = torch.mean(
            torch.relu(criteria['optimal_total_reward'][0] - metrics[:, 1]) +
            torch.relu(metrics[:, 1] - criteria['optimal_total_reward'][1])
        )
        time_passage_loss = torch.mean(
            torch.relu(criteria['optimal_time_passage'][0] - metrics[:, 2]) +
            torch.relu(metrics[:, 2] - criteria['optimal_time_passage'][1])
        )
        return time_learning_loss + reward_loss + time_passage_loss
