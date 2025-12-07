import pygame, controls
import torch
import sys
import numpy as np
from generator import Generator
from discriminator import Discriminator
from material_point import Material_point
from policy_iteration import policy_iteration, policy_improvement, policy_evaluation
from run_hero import run_hero
from generator_learning import GeneratorTrainer
from discriminator_learning import DiscriminatorTrainer
from double_q_learning import double_q_learning
from visualization import TrainingVisualizer
import matplotlib.pyplot as plt


def run(grid_size, n_green, n_red, rerun):
    "здесь контролируется весь процесс игры"
    pygame.init()
    screen = pygame.display.set_mode((grid_size*50, grid_size*50))
    pygame.display.set_caption("GenLab: Simulator")
    bg_color = (0, 0, 0)
    material_point = Material_point(screen)

    generator = Generator()
    discriminator = Discriminator()
    generator_trainer = GeneratorTrainer(grid_size, material_point, generator, discriminator)
    discriminator_trainer = DiscriminatorTrainer(generator, discriminator)
    #visualizer = TrainingVisualizer()

    for level in range(10):
        print(f'\n🚀 Проходим уровень {level + 1}')
        level_successes = 0
        level_attempts = 0
        for run in range(1 + rerun):
            z = torch.randn(generator.latent_dim)
            level_data = torch.tensor([grid_size,
                                       torch.randint(n_green + level, n_green + level + 2, (1,)).item(),
                                       torch.randint(2 * (n_green + level), 2 * (n_green + level + 1), (1,)).item()])
            all_coords = generator(z, level_data)
            metrics_for_level = double_q_learning(material_point,
                                                  level_data[0],
                                                  level_data[1],
                                                  level_data[2],
                                                  all_coords['coords_green'],
                                                  all_coords['coords_red'],
                                                  gamma=0.98,
                                                  init_alpha=0.5,
                                                  min_alpha=0.01,
                                                  alpha_decay_ratio=0.5,
                                                  init_eps=1.0,
                                                  min_eps=0.1,
                                                  eps_decay_ratio=0.9,
                                                  episodes=10000)
            if hasattr(metrics_for_level, 'success_count'):
                level_successes += metrics_for_level.success_count
            level_attempts += 1


            gen_loss = discriminator_trainer.train_discriminator(metrics_for_level, epochs=1)
            dis_loss = generator_trainer.train_generator()

            # Расчет успеваемости
            success_rate = level_successes / level_attempts if level_attempts > 0 else 0

            # Визуализация
            #visualizer.update_plots(level + 1, gen_loss, dis_loss, success_rate)
            #visualizer.print_status(level + 1, gen_loss, dis_loss, success_rate)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    plt.ioff()
                    plt.show()
                    return

