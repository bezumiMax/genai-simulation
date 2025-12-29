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
    
    POPULATION_SIZE = 20
    ELITE_SIZE = 4
    MUTATION_RATE = 0.1
    MUTATION_STRENGTH = 0.3
    
    n_states = grid_size ** 2
    n_actions = 4
    
    population = []
    for _ in range(POPULATION_SIZE):
        if hasattr(agent, 'policy_weights'):
            weights = agent.policy_weights.copy() + np.random.normal(0, 0.1, (n_states, n_actions))
        else:
            weights = np.random.randn(n_states, n_actions) * 0.1
        population.append(weights)
    
    def evaluate_individual(weights, max_steps=100):
        total_reward = 0
        success_count = 0
        
        temp_green = coords_green.copy()
        temp_n_green = n_green
        
        material_point.rect.centerx = material_point.screen_rect.centerx
        material_point.rect.centery = material_point.screen_rect.centery
        
        for step in range(max_steps):
            state = get_state(material_point, grid_size)
            
            action_probs = np.exp(weights[state]) / np.sum(np.exp(weights[state]))
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
                temp_n_green -= 1
                if len(temp_green) == 0:
                    # Бонус за быстрый успех
                    total_reward += 1000 * (max_steps - step) / max_steps
                    break
            elif agent_pos in coords_red:
                reward = -100
                total_reward += reward
                break  # Заканчиваем при столкновении с красным
            else:
                reward = -1
                total_reward += reward
        
        return total_reward, success_count
    
    # Основной цикл генетического алгоритма
    start_time = time.time()
    best_fitness = -float('inf')
    best_individual = None
    
    font = pygame.font.Font(None, 18)
    
    for generation in tqdm(range(episodes // 100), desc='Generations'):
        # Оценка всей популяции
        fitness_scores = []
        for idx, individual in enumerate(population):
            fitness, _ = evaluate_individual(individual)
            fitness_scores.append((fitness, idx))
        
        # Сортировка по фитнесу
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Сохраняем лучшую особь
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = population[fitness_scores[0][1]].copy()
        
        # Отбор элитных особей
        elite_indices = [idx for _, idx in fitness_scores[:ELITE_SIZE]]
        elites = [population[idx].copy() for idx in elite_indices]
        
        # Создание нового поколения
        new_population = elites.copy()  # Сохраняем элит
        
        # Скрещивание и мутация для заполнения популяции
        while len(new_population) < POPULATION_SIZE:
            # Выбор родителей (турнирный отбор)
            tournament_size = 3
            tournament = random.sample(fitness_scores[:POPULATION_SIZE//2], tournament_size)
            parent1_idx = max(tournament, key=lambda x: x[0])[1]
            
            tournament = random.sample(fitness_scores[:POPULATION_SIZE//2], tournament_size)
            parent2_idx = max(tournament, key=lambda x: x[0])[1]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Скрещивание (равномерное)
            child = parent1.copy()
            crossover_mask = np.random.rand(*parent1.shape) > 0.5
            child[crossover_mask] = parent2[crossover_mask]
            
            # Мутация
            mutation_mask = np.random.rand(*child.shape) < MUTATION_RATE
            mutation_values = np.random.randn(*child.shape) * MUTATION_STRENGTH
            child[mutation_mask] += mutation_values[mutation_mask]
            
            new_population.append(child)
        
        population = new_population
        
        if generation % 5 == 0 and best_individual is not None:
            material_point.rect.centerx = material_point.screen_rect.centerx
            material_point.rect.centery = material_point.screen_rect.centery
            
            temp_green = coords_green.copy()
            temp_red = coords_red.copy()
            temp_n_green = n_green
            temp_n_red = n_red
            
            total_reward = 0
            success_count = 0
            
            for step in range(100):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                
                state = get_state(material_point, grid_size)
                action_probs = np.exp(best_individual[state]) / np.sum(np.exp(best_individual[state]))
                action = np.argmax(action_probs)
                
                key = action_to_key[action]
                material_point.controller.simulate_key_press(key)
                
                for i in range(10):
                    material_point.update_point()
                    
                    screen.fill(bg_color)
                    for j in range(temp_n_red):
                        x = (temp_red[j] % grid_size) * 50
                        y = (temp_red[j] // grid_size) * 50
                        screen.blit(red_square, (x, y))
                    for j in range(temp_n_green):
                        x = (temp_green[j] % grid_size) * 50
                        y = (temp_green[j] // grid_size) * 50
                        screen.blit(green_square, (x, y))
                    
                    material_point.output()
                    
                    info_lines = [
                        f'Generation: {generation}',
                        f'Best Fitness: {best_fitness:.1f}',
                        f'Total Reward: {total_reward}',
                        f'Green Squares Left: {temp_n_green}',
                        f'Time: {time.time() - start_time:.1f}s'
                    ]
                    for i, line in enumerate(info_lines):
                        text = font.render(line, True, (255, 255, 255))
                        screen.blit(text, (50, 5 + i * 15))
                    
                    pygame.display.flip()
                    pygame.time.delay(30)
                
                material_point.controller.simulate_key_release(key)
                
                agent_x = material_point.rect.topleft[0] // 50
                agent_y = material_point.rect.topleft[1] // 50
                agent_pos = agent_y * grid_size + agent_x
                
                if agent_pos in temp_green:
                    reward = 200
                    success_count += 1
                    total_reward += reward
                    temp_green.remove(agent_pos)
                    temp_n_green -= 1
                    if len(temp_green) == 0:
                        break
                elif agent_pos in temp_red:
                    reward = -100
                    total_reward += reward
                    break
                else:
                    reward = -1
                    total_reward += reward
                
                pygame.time.delay(100)
            
            if time.time() - start_time > 50.0:
                end_time = time.time()
                return torch.tensor([total_reward, 51.0])
    
    material_point.rect.centerx = material_point.screen_rect.centerx
    material_point.rect.centery = material_point.screen_rect.centery
    
    temp_green = coords_green.copy()
    temp_red = coords_red.copy()
    temp_n_green = n_green
    temp_n_red = n_red
    
    total_reward = 0
    success_count = 0
    
    for step in range(200):  # Увеличиваем время для финальной демонстрации
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        state = get_state(material_point, grid_size)
        action_probs = np.exp(best_individual[state]) / np.sum(np.exp(best_individual[state]))
        action = np.argmax(action_probs)
        
        key = action_to_key[action]
        material_point.controller.simulate_key_press(key)
        
        for i in range(10):
            material_point.update_point()
            
            screen.fill(bg_color)
            for j in range(temp_n_red):
                x = (temp_red[j] % grid_size) * 50
                y = (temp_red[j] // grid_size) * 50
                screen.blit(red_square, (x, y))
            for j in range(temp_n_green):
                x = (temp_green[j] % grid_size) * 50
                y = (temp_green[j] // grid_size) * 50
                screen.blit(green_square, (x, y))
            
            material_point.output()
            
            info_lines = [
                f'FINAL RUN - Best Policy',
                f'Total Reward: {total_reward}',
                f'Green Squares Left: {temp_n_green}',
                f'Success Count: {success_count}',
                f'Total Time: {time.time() - start_time:.1f}s'
            ]
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (50, 5 + i * 15))
            
            pygame.display.flip()
            pygame.time.delay(30)
        
        material_point.controller.simulate_key_release(key)
        
        agent_x = material_point.rect.topleft[0] // 50
        agent_y = material_point.rect.topleft[1] // 50
        agent_pos = agent_y * grid_size + agent_x
        
        if agent_pos in temp_green:
            reward = 200
            success_count += 1
            total_reward += reward
            temp_green.remove(agent_pos)
            temp_n_green -= 1
            if len(temp_green) == 0:
                end_time = time.time()
                return torch.tensor([total_reward, end_time - start_time])
        elif agent_pos in temp_red:
            reward = -100
            total_reward += reward
            break
        else:
            reward = -1
            total_reward += reward
        
        pygame.time.delay(100)
    
    end_time = time.time()
    return torch.tensor([total_reward, end_time - start_time])

def get_state(material_point, grid_size):
    """Преобразует позицию в состояние"""
    screen_width = material_point.screen_rect.width
    screen_height = material_point.screen_rect.height
    x_bin = min(int(material_point.rect.x / screen_width * grid_size), grid_size - 1)
    y_bin = min(int(material_point.rect.y / screen_height * grid_size), grid_size - 1)
    return y_bin * grid_size + x_bin