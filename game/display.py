import pygame
import sys
import numpy as np

class FighterGameDisplay:
    def __init__(self, size, real_size, origin):
        
        if not isinstance(size, tuple):
            size = tuple(size)

        self.size = size

        self.arena_size = real_size
        self.origin = origin

        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Fighter game")

    def draw(self, fighters):
        self.screen.fill((255,255,255))
        for fighter in fighters:
            screen_pos = self.conv_space_to_screen_coord(fighter.pos)
            self.screen = fighter.draw(self.screen, screen_pos)

            thrust_draw = self.conv_space_to_screen_coord(fighter.thrust)

            pygame.draw.line(self.screen, (0,255,0), screen_pos-thrust_draw, screen_pos, width = 5)

        pygame.display.flip()  # Update the full display Surface to the screen


    def conv_space_to_screen_coord(self, pos):
        normalised_position = np.divide((pos - self.origin), self.arena_size)
        position_on_screen = np.multiply(normalised_position, self.size)
        return position_on_screen