def create_mdp(grid_size, n_green, n_red, coords_green, coords_red):
    P = {}
    n_states = grid_size ** 2
    coords_red_numeric = []
    for coord in coords_red:
        if hasattr(coord, 'item'):
            coords_red_numeric.append(coord.item())
        else:
            coords_red_numeric.append(int(coord))

    coords_green_numeric = []
    for coord in coords_green:
        if hasattr(coord, 'item'):
            coords_green_numeric.append(coord.item())
        else:
            coords_green_numeric.append(int(coord))

    for state in range(n_states):
        actions = {}
        if state >= grid_size:  # Вверх
            actions[0] = [[1.0, state - grid_size, -1.0, False]]  # ← Обернули в список
        else:
            actions[0] = [[1.0, state, -1.0, False]]  # ← Обернули в список

        if state % grid_size != grid_size - 1:  # Вправо
            actions[1] = [[1.0, state + 1, -1.0, False]]  # ← Обернули в список
        else:
            actions[1] = [[1.0, state, -1.0, False]]  # ← Обернули в список

        if state < n_states - grid_size:  # Вниз
            actions[2] = [[1.0, state + grid_size, -1.0, False]]  # ← Обернули в список
        else:
            actions[2] = [[1.0, state, -1.0, False]]  # ← Обернули в список

        if state % grid_size != 0:  # Влево
            actions[3] = [[1.0, state - 1, -1.0, False]]  # ← Обернули в список
        else:
            actions[3] = [[1.0, state, -1.0, False]]  # ← Обернули в список

        P[state] = actions

    for s in coords_red_numeric:
        s = int(s)
        neighbors = [
            (s - grid_size, 2),  # Сверху (действие Down)
            (s + 1, 3),  # Справа (действие Left)
            (s + grid_size, 0),  # Снизу (действие Up)
            (s - 1, 1)  # Слева (действие Right)
        ]
        for neighbor_state, action in neighbors:
            neighbor_state = int(neighbor_state)
            if 0 <= neighbor_state < n_states and action in P[neighbor_state]:
                # Обновляем награду в существующем списке
                P[neighbor_state][action][0][2] = -50.0  # ← Теперь обращаемся к [0][2]

    for s in coords_green_numeric:
        s = int(s)
        neighbors = [
            (s - grid_size, 2),  # Сверху (действие Down)
            (s + 1, 3),  # Справа (действие Left)
            (s + grid_size, 0),  # Снизу (действие Up)
            (s - 1, 1)  # Слева (действие Right)
        ]
        for neighbor_state, action in neighbors:
            neighbor_state = int(neighbor_state)
            if 0 <= neighbor_state < n_states and action in P[neighbor_state]:
                P[neighbor_state][action][0][2] = 200.0  # ← [0][2]
                P[neighbor_state][action][0][3] = True  # ← [0][3]

        for action in P[s]:
            P[s][action][0][2] = 0.0  # ← [0][2]
            P[s][action][0][3] = True  # ← [0][3]

    return P


def get_state(material_point, grid_size):
    """Преобразует позицию в состояние"""
    screen_width = material_point.screen_rect.width
    screen_height = material_point.screen_rect.height
    x_bin = min(int(material_point.rect.x / screen_width * grid_size), grid_size - 1)
    y_bin = min(int(material_point.rect.y / screen_height * grid_size), grid_size - 1)
    state = y_bin * grid_size + x_bin

    # Преобразуем тензор в число, если это тензор
    if hasattr(state, 'item'):
        return state.item()
    else:
        return int(state)
