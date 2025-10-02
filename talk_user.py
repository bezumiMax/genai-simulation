import numpy as np
from main import run

print("Выберите размер сетки (нечетное число от 14 до 20):")
grid_size = int(input())
assert grid_size < 21, "Incorrect number!"
assert grid_size > 13, "Incorrect number!"
assert grid_size % 2 == 1, "Incorrect number!"
print(f'Кол-во зеленых квадратов (число от 1 до {(grid_size ** 2 - 2) // 30}):')
n_green = int(input())
assert n_green <= (grid_size ** 2 - 167) // 3, "Incorrect number!"
assert n_green > 0, "Incorrect number!"
n_red = np.random.randint(2 * n_green - 1, 2 * n_green + 1)
print("Выведите сколько раз генератор генерирует один и тот же уровень (число от 1 до 10000):")
rerun = int(input())
assert rerun >= 0, "Incorrect number!"
assert rerun <= 10000, "Incorrect number!"


run(grid_size, n_green, n_red, rerun)
