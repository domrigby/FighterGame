import numpy as np
import pygame

from game.item import Item

class Bullet(Item):

    def __init__(self, mass=1, init_pos=..., init_vel=..., time_step=0.01, drag=0.000001, draw_shape=None, colour=...):
        self.ent_type = 'bullet'
        super().__init__(mass, init_pos, init_vel, time_step, drag, draw_shape, colour)


    def update_state(self, deltaT=None):
        if deltaT is None:
            deltaT = self.time_step
        self.pos = self.pos + np.multiply(self.vel,deltaT)

    def draw(self, screen, screen_pos): # Define the triangle's vertices
        # Draw the triangle
        pygame.draw.circle(screen, self.colour, screen_pos, 2)
        return screen
