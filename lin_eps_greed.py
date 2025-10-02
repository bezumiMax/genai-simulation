"""для начала применим эпсилон-жадную стратегию"""

import time
import pygame
from tqdm import tqdm
import numpy as np

def lin_eplison_greedy(material_point,
                       init_eps=1.0,
                       min_eps=0.01,
                       decay_ratio=0.05,
                       episodes=1000):
    pygame.init()
    screen = pygame.display.set_mode((1100, 1100))
    pygame.display.set_caption("Inside AI")
    bg_color = (0, 0, 0)

    grid_size = 11
    num_states = grid_size * grid_size
    num_actions = 4
    Q = np.zeros((num_states, num_actions))
    N = np.zeros((num_states, num_actions))

    Qe = np.empty((episodes, num_states, num_actions))
    returns = np.empty(episodes)
    actions = np.empty(episodes)

    # Создаем белый квадрат для отрисовки
    white_square = pygame.Surface((100, 100))
    white_square.fill((255, 255, 255))

    # Соответствие действий клавишам
    action_to_key = {
        0: 'W',  # Вверх
        1: 'D',  # Вправо
        2: 'S',  # Вниз
        3: 'A'  # Влево
    }

    material_point.rect.centerx = material_point.screen_rect.centerx
    material_point.rect.centery = material_point.screen_rect.centery

    # Переменные для статистики
    total_reward = 0
    success_count = 0

    for e in tqdm(range(episodes), desc='Episodes', leave=False):
        # Расчет epsilon
        decay_episodes = episodes * decay_ratio
        eps = 1 - e / decay_episodes
        eps *= init_eps - min_eps
        eps += min_eps
        eps = np.clip(eps, min_eps, init_eps)
        # Получаем текущее состояние
        state = get_state(material_point, grid_size)

        # Выбор действия
        if np.random.random() > eps:
            action = np.argmax(Q[state])
        else:
            action = np.random.randint(num_actions)

        # Выполняем действие
        key = action_to_key[action]
        material_point.controller.simulate_key_press(key)

        # Обновляем и отрисовываем в течение нескольких кадров
        for _ in range(10):
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return Qe[:e], returns[:e], actions[:e]

            # Обновляем позицию (это вызовет движение)
            material_point.update_point()

            # Отрисовка
            screen.fill(bg_color)
            screen.blit(white_square, (0, 0))
            material_point.output()

            # Отображаем информацию
            font = pygame.font.Font(None, 36)

            # Основная информация
            info_lines = [
                f'Episode: {e + 1}/{episodes}',
                f'Action: {action} ({key})',
                f'Position: {material_point.rect.x},{material_point.rect.y}',
                f'Epsilon: {eps:.3f}',
                f'State: {state}',
                f'Total Reward: {total_reward}',
                f'Successes: {success_count}'
            ]

            # Отображаем все строки информации
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (10, 10 + i * 30))

            pygame.display.flip()
            pygame.time.delay(30)

        # Отпускаем клавишу
        material_point.controller.simulate_key_release(key)

        # Получаем новое состояние и награду
        new_state = get_state(material_point, grid_size)

        # Проверяем достижение цели
        if (material_point.rect.colliderect(pygame.Rect(0, 0, 100, 100))):
            reward = 100
            success_count += 1
            # Сбрасываем позицию при достижении цели
            material_point.rect.centerx = material_point.screen_rect.centerx
            material_point.rect.centery = material_point.screen_rect.centery
        else:
            reward = -1

        # Обновляем общую награду
        total_reward += reward

        # Обновление Q-значения
        alpha = 0.1
        Q[state, action] += alpha * (reward + 0.9 * np.max(Q[new_state]) - Q[state, action])
        N[state, action] += 1

        # Сохранение результатов
        Qe[e] = Q.copy()
        returns[e] = reward
        actions[e] = action

        # Отображаем награду после действия
        screen.fill(bg_color)
        screen.blit(white_square, (0, 0))
        material_point.output()

        font = pygame.font.Font(None, 36)

        # Информация с наградой
        info_lines = [
            f'Episode: {e + 1}/{episodes}',
            f'Action: {action} ({key})',
            f'Reward: {reward}',
            f'Total Reward: {total_reward}',
            f'Successes: {success_count}',
            f'Epsilon: {eps:.3f}',
            f'State: {state} -> {new_state}'
        ]

        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 30))

        # Цвет текста награды в зависимости от значения
        reward_color = (0, 255, 0) if reward > 0 else (255, 0, 0) if reward < 0 else (255, 255, 255)
        reward_text = font.render(f'Reward: {reward}', True, reward_color)
        screen.blit(reward_text, (10, 70))

        pygame.display.flip()
        pygame.time.delay(500)  # Пауза чтобы увидеть награду

    pygame.quit()
    return Qe, returns, actions


def get_state(material_point, grid_size):
    """Преобразует позицию в состояние"""
    screen_width = material_point.screen_rect.width
    screen_height = material_point.screen_rect.height

    x_bin = min(int(material_point.rect.x / screen_width * grid_size), grid_size - 1)
    y_bin = min(int(material_point.rect.y / screen_height * grid_size), grid_size - 1)

    return y_bin * grid_size + x_bin
