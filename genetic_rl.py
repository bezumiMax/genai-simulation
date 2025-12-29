import torch
import numpy as np
import pygame
from tqdm import tqdm
import time
import random

def genetic_algorithm_rl(material_point,
                         agent,
                         is_fake,
                         grid_size,
                         n_green,
                         n_red,
                         coords_green,
                         coords_red,
                         episodes,
                         gamma=0.98):
    n_green = n_green.item()
    n_red = n_red.item()
    
    pygame.init()
    screen = pygame.display.set_mode((grid_size * 50, grid_size * 50))
    pygame.display.set_caption("GenLab: Genetic Algorithm Simulator")
    
    bg_color = (0, 0, 0)
    green_square = pygame.Surface((50, 50))
    green_square.fill((0, 255, 0))
    red_square = pygame.Surface((50, 50))
    red_square.fill((255, 0, 0))
    
    action_to_key = {
        0: 'W',  # Вверх
        1: 'D',  # Вправо
        2: 'S',  # Вниз
        3: 'A'   # Влево
    }
    
    # Параметры генетического алгоритма
    POPULATION_SIZE = 20
    ELITE_SIZE = 4
    MUTATION_RATE = 0.1
    MUTATION_STRENGTH = 0.3
    MAX_STEPS_PER_EVALUATION = 100
    GENERATIONS = episodes // 100  # Конвертируем эпизоды в поколения
    
    n_states = grid_size ** 2
    n_actions = 4
    
    # Инициализация популяции
    population = []
    for _ in range(POPULATION_SIZE):
        # Если у агента уже есть веса политики, используем их как основу
        if agent.policy_weights is not None and not is_fake:
            weights = agent.policy_weights.copy() + np.random.normal(0, 0.1, (n_states, n_actions))
        else:
            weights = np.random.randn(n_states, n_actions) * 0.1
        population.append(weights)
    
    def evaluate_individual(weights, max_steps=MAX_STEPS_PER_EVALUATION):
        """Оценивает качество политики (особи)"""
        total_reward = 0
        success_count = 0
        
        temp_green = coords_green.copy()
        temp_red = coords_red.copy()
        
        material_point.rect.centerx = material_point.screen_rect.centerx
        material_point.rect.centery = material_point.screen_rect.centery
        
        for step in range(max_steps):
            state = get_state(material_point, grid_size)
            
            # Softmax policy
            logits = weights[state] - np.max(weights[state])
            action_probs = np.exp(logits) / np.sum(np.exp(logits))
            action = np.random.choice(range(n_actions), p=action_probs)
            
            key = action_to_key[action]
            material_point.controller.simulate_key_press(key)
            
            for i in range(10):
                material_point.update_point()
            
            material_point.controller.simulate_key_release(key)
            
            agent_x = material_point.rect.topleft[0] // 50
            agent_y = material_point.rect.topleft[1] // 50
            agent_pos = agent_y * grid_size + agent_x
            
            if agent_pos in temp_green:
                reward = 200
                success_count += 1
                total_reward += reward
                temp_green.remove(agent_pos)
                if len(temp_green) == 0:
                    total_reward += 1000 * (max_steps - step) / max_steps
                    break
            elif agent_pos in temp_red:
                reward = -100
                total_reward += reward
                break
            else:
                reward = -1
                total_reward += reward
        
        return total_reward, success_count
    
    start_time = time.time()
    best_fitness = -float('inf')
    best_individual = None
    font = pygame.font.Font(None, 18)
    
    for generation in tqdm(range(GENERATIONS), desc='Generations'):
        fitness_scores = []
        for idx, individual in enumerate(population):
            fitness, _ = evaluate_individual(individual)
            fitness_scores.append((fitness, idx))
        
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = population[fitness_scores[0][1]].copy()
            # Обновляем агента если это реальное обучение
            if not is_fake:
                agent.update(policy_weights=best_individual, fitness=best_fitness)
        
        elite_indices = [idx for _, idx in fitness_scores[:ELITE_SIZE]]
        elites = [population[idx].copy() for idx in elite_indices]
        
        new_population = elites.copy()
        
        while len(new_population) < POPULATION_SIZE:
            tournament_size = 3
            tournament = random.sample(fitness_scores[:POPULATION_SIZE//2], tournament_size)
            parent1_idx = max(tournament, key=lambda x: x[0])[1]
            
            tournament = random.sample(fitness_scores[:POPULATION_SIZE//2], tournament_size)
            parent2_idx = max(tournament, key=lambda x: x[0])[1]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            child = parent1.copy()
            crossover_mask = np.random.rand(*parent1.shape) > 0.5
            child[crossover_mask] = parent2[crossover_mask]
            
            mutation_mask = np.random.rand(*child.shape) < MUTATION_RATE
            mutation_values = np.random.randn(*child.shape) * MUTATION_STRENGTH
            child[mutation_mask] += mutation_values[mutation_mask]
            
            new_population.append(child)
        
        population = new_population
        
        # Визуализация (остается без изменений)
        # ... [код визуализации]
    
    # Возвращаем результат
    end_time = time.time()
    return torch.tensor([best_fitness, end_time - start_time])

def get_state(material_point, grid_size):
    """Преобразует позицию в состояние"""
    screen_width = material_point.screen_rect.width
    screen_height = material_point.screen_rect.height
    x_bin = min(int(material_point.rect.x / screen_width * grid_size), grid_size - 1)
    y_bin = min(int(material_point.rect.y / screen_height * grid_size), grid_size - 1)
    return y_bin * grid_size + x_bin