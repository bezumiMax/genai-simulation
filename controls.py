import pygame, sys

class InputController:
    def __init__(self):
        self.key_states = {
            'W': False,
            'A': False,
            'S': False,
            'D': False
        }

    def set_key_state(self, key, state):
        """Устанавливает состояние клавиши программно (для агента)"""
        if key in self.key_states:
            self.key_states[key] = state

    def simulate_key_press(self, key):
        """Симулирует нажатие клавиши (удобный метод для агента)"""
        self.set_key_state(key, True)

    def simulate_key_release(self, key):
        """Симулирует отпускание клавиши"""
        self.set_key_state(key, False)

    def get_key_state(self, key):
        """Возвращает текущее состояние клавиши"""
        return self.key_states.get(key, False)

def events(material_point):
    """обработка события"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                material_point.mright = True
            if event.key == pygame.K_a:
                material_point.mleft = True
            if event.key == pygame.K_s:
                material_point.mdown = True
            if event.key == pygame.K_w:
                material_point.mup = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_d:
                material_point.mright = False
            if event.key == pygame.K_a:
                material_point.mleft = False
            if event.key == pygame.K_s:
                material_point.mdown = False
            if event.key == pygame.K_w:
                material_point.mup = False


white_square = pygame.Surface((50, 50))
white_square.fill((255, 255, 255)) # финиш
def update(bg_color, screen, material_point):
    screen.fill(bg_color)
    screen.blit(white_square, (0, 0))
    material_point.output()
    pygame.display.flip()
