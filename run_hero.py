from policy_iteration import policy_evaluation, policy_improvement, policy_iteration
import time
import pygame
from create_mdp import create_mdp
from create_mdp import get_state
from tqdm import tqdm
import torch


def run_hero(material_point, grid_size, n_green, n_red, coords_green, coords_red):
    print(n_green, n_red)
    """Возвращает метрики"""
    pygame.init()
    screen = pygame.display.set_mode((grid_size * 50, grid_size * 50))
    pygame.display.set_caption("GenLab: Simulator")
    bg_color = (0, 0, 0)
    green_square = pygame.Surface((50, 50))
    green_square.fill((0, 255, 0))  # ЗЕЛЕНЫЙ квадрат
    red_square = pygame.Surface((50, 50))
    red_square.fill((255, 0, 0))  # КРАСНЫЙ квадрат
    action_to_key = {
        0: 'W',  # Вверх
        1: 'D',  # Вправо
        2: 'S',  # Вниз
        3: 'A'  # Влево
    }
    material_point.rect.centerx = material_point.screen_rect.centerx
    material_point.rect.centery = material_point.screen_rect.centery
    cur_MDP = create_mdp(grid_size, n_green, n_red, coords_green, coords_red)
    start_time = time.time()
    print(f'Начало итерации политик')
    V, cur_pi = policy_iteration(cur_MDP, grid_size, n_green, n_red, coords_green, coords_red)
    print('Конец итерации политик')
    end_time = time.time()
    time_learning = end_time - start_time
    total_reward = 0
    state = int(get_state(material_point, grid_size))
    font = pygame.font.Font(None, 18)
    start_time_simulation = time.time()
    for e in tqdm(range(10000), desc='Episodes', leave=False):
        action = cur_pi(get_state(material_point, grid_size))
        key = action_to_key[action]
        material_point.controller.simulate_key_press(key)
        for i in range(10):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            material_point.update_point()
            screen.fill(bg_color)
            for j in range(n_green):
                x = (coords_green[j] % grid_size) * 50
                y = (coords_green[j] // grid_size) * 50
                screen.blit(green_square, (x, y))
            for j in range(n_red):
                x = (coords_red[j] % grid_size) * 50
                y = (coords_red[j] // grid_size) * 50
                screen.blit(red_square, (x, y))
            material_point.output()
            info_lines = [
                f'Time learning: {time_learning:.2f}s',
                f'Total Reward: {total_reward}',
                f'Time Passage: {time.time() - start_time_simulation:.2f}s'
            ]
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (50, 5 + i * 15))
            pygame.display.flip()
            pygame.time.delay(30)
        material_point.controller.simulate_key_release(key)
        new_state = get_state(material_point, grid_size)
        check_terminal = False
        for i in range(n_green):
            if material_point.rect.colliderect(pygame.Rect(
                    (coords_green[i] % grid_size) * 50,
                    (coords_green[i] // grid_size) * 50,
                    50, 50)):
                check_terminal = True
                break
        if check_terminal:
            total_reward += 200.0
            return torch.tensor([
                time_learning,
                total_reward,
                time.time() - start_time_simulation
            ])
        elif material_point.rect.colliderect(pygame.Rect(50, 0, 50, 50)):
            reward = -100.0
        else:
            reward = -1.0
        total_reward += reward
        screen.fill(bg_color)
        for j in range(n_green):
            x = (coords_green[j] % grid_size) * 50
            y = (coords_green[j] // grid_size) * 50
            screen.blit(green_square, (x, y))
        for j in range(n_red):
            x = (coords_red[j] % grid_size) * 50
            y = (coords_red[j] // grid_size) * 50
            screen.blit(red_square, (x, y))
        material_point.output()
        info_lines = [
            f'Time learning: {time_learning:.2f}s',
            f'Total Reward: {total_reward}',
            f'Time Passage: {time.time() - start_time_simulation:.2f}s'
        ]
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (50, 5 + i * 15))
        pygame.display.flip()
        pygame.time.delay(2000)
    return torch.tensor([
        time_learning,
        total_reward,
        time.time() - start_time_simulation
    ])
