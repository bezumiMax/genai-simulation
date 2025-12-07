"""теперь двойное q-обучение"""
import torch
from decay_shedule import decay_schedule
import numpy as np
import pygame
from tqdm import tqdm
import time


def double_q_learning(material_point,
                      grid_size,
                      n_green,
                      n_red,
                      coords_green,
                      coords_red,
                      gamma=0.98,
                      init_alpha=0.5,
                      min_alpha=0.01,
                      alpha_decay_ratio=0.5,
                      init_eps=1.0,
                      min_eps=0.1,
                      eps_decay_ratio=0.9,
                      episodes=10000):
    n_green = n_green.item()
    n_red = n_red.item()
    for i in range(n_green):
        print(f"{i}: {coords_green[i]}")
    for i in range(n_red):
        print(f"{i}: {coords_red[i]}")
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
    start_time = time.time()
    epsilons = decay_schedule(init_eps, min_eps, eps_decay_ratio, episodes)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, episodes)
    state = int(get_state(material_point, grid_size))
    available_actions = [0, 1, 2, 3]
    total_reward = 0
    success_count = 0
    n_states = grid_size ** 2
    n_actions = 4
    Q1 = np.zeros((n_states, n_actions), dtype=np.float64)
    Q2 = np.zeros((n_states, n_actions), dtype=np.float64)
    N = np.zeros((n_states, n_actions), dtype=np.float64)
    Qe = [None] * episodes
    returns = np.zeros(episodes, dtype=np.float64)
    actions = np.zeros(episodes, dtype=np.int32)
    font = pygame.font.Font(None, 18)
    for e in tqdm(range(episodes), desc='Episodes', leave=False):
        eps = epsilons[e]
        alpha = alphas[e]
        state = get_state(material_point, grid_size)
        Q_avg = (Q1[state] + Q2[state]) / 2
        if np.random.random() > eps:
            available_q_values = [Q_avg[a] for a in available_actions]
            best_action_index = np.argmax(available_q_values)
            action = int(available_actions[best_action_index])
        else:
            action = int(np.random.choice(available_actions))
        key = action_to_key[action]
        material_point.controller.simulate_key_press(key)
        for i in range(10):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            material_point.update_point()
            screen.fill(bg_color)
            for j in range(n_red):
                x = (coords_red[j] % grid_size) * 50
                y = (coords_red[j] // grid_size) * 50
                screen.blit(red_square, (x, y))
            for j in range(n_green):
                x = (coords_green[j] % grid_size) * 50
                y = (coords_green[j] // grid_size) * 50
                screen.blit(green_square, (x, y))
            material_point.output()
            info_lines = [
                f'Total Reward: {total_reward}',
                f'Time Passage: {time.time() - start_time:.2f}s'
            ]
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (50, 5 + i * 15))
            pygame.display.flip()
            pygame.time.delay(30)

        material_point.controller.simulate_key_release(key)
        new_state = get_state(material_point, grid_size)
        agent_x = material_point.rect.topleft[0] // 50
        agent_y = material_point.rect.topleft[1] // 50
        agent_pos = agent_y * grid_size + agent_x
        if agent_pos in coords_green:
            reward = 200
            success_count += 1
            total_reward += reward
            coords_green.remove(agent_pos)
            n_green -= 1
            if len(coords_green) == 0:
                print("Все зеленые квадраты собраны!")
                end_time = time.time()
                return torch.tensor([total_reward, end_time - start_time])
        elif agent_pos in coords_red:
            reward = -50
            total_reward += reward
        else:
            reward = -1
            total_reward += reward
        if np.random.random() < 0.5:
            best_action = np.argmax(Q1[new_state])
            Q1[state, action] += alpha * (reward + gamma * Q2[new_state, best_action] - Q1[state, action])
        else:
            best_action = np.argmax(Q2[new_state])
            Q2[state, action] += alpha * (reward + gamma * Q1[new_state, best_action] - Q2[state, action])
        N[state, action] += 1
        Q_combined = (Q1 + Q2) / 2
        Qe[e] = Q_combined.copy()
        returns[e] = reward
        actions[e] = action
        screen.fill(bg_color)
        for j in range(n_red):
            x = (coords_red[j] % grid_size) * 50
            y = (coords_red[j] // grid_size) * 50
            screen.blit(red_square, (x, y))
        for j in range(n_green):
            x = (coords_green[j] % grid_size) * 50
            y = (coords_green[j] // grid_size) * 50
            screen.blit(green_square, (x, y))
        material_point.output()
        info_lines = [
            f'Total Reward: {total_reward}',
            f'Time Passage: {time.time() - start_time:.2f}s'
        ]
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (50, 5 + i * 15))
        pygame.display.flip()
        pygame.time.delay(500)
    return torch.tensor([sum(returns), time.time() - start_time])

def get_state(material_point, grid_size):
    """Преобразует позицию в состояние"""
    screen_width = material_point.screen_rect.width
    screen_height = material_point.screen_rect.height
    x_bin = min(int(material_point.rect.x / screen_width * grid_size), grid_size - 1)
    y_bin = min(int(material_point.rect.y / screen_height * grid_size), grid_size - 1)
    return y_bin * grid_size + x_bin
