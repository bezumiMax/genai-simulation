import pygame
from controls import InputController


class Material_point:
    def __init__(self, screen):
        """инициализация главного героя"""
        self.screen = screen
        self.image = pygame.image.load('images/pixil-frame-0.png')
        self.rect = self.image.get_rect()
        self.screen_rect = screen.get_rect()
        self.rect.centerx = self.screen_rect.centerx
        self.rect.centery = self.screen_rect.centery
        self.rect.topleft = self.screen_rect.topleft
        self.rect.topright = self.screen_rect.topright
        self.rect.bottomleft = self.screen_rect.bottomleft
        self.rect.bottomright = self.screen_rect.bottomright
        self.rect.bottom = self.screen_rect.bottom
        self.mright = False
        self.mleft = False
        self.mdown = False
        self.mup = False
        self.controller = InputController()

    def output(self):
        """рисование"""
        self.screen.blit(self.image, self.rect)

    def update_point(self):
        """обновление позиции"""
        self.mright = self.controller.get_key_state('D')
        self.mleft = self.controller.get_key_state('A')
        self.mdown = self.controller.get_key_state('S')
        self.mup = self.controller.get_key_state('W')

        if self.mright and self.rect.right < self.screen_rect.right:
            self.rect.centerx += 5
        if self.mleft and self.rect.left > self.screen_rect.left:
            self.rect.centerx -= 5
        if self.mdown and self.rect.bottom < self.screen_rect.bottom:
            self.rect.centery += 5
        if self.mup and self.rect.top > self.screen_rect.top:
            self.rect.centery -= 5

    def get_coord(self):
        x = self.rect.topleft[0]
        y = self.rect.topleft[1]
        return x, y
