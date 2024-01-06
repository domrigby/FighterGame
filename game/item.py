import numpy as np
import numpy.linalg as la

import pygame

class Item:

    def __init__(self, mass=1, init_pos=[0,0], init_vel = [0,0], time_step=0.01, drag=0.000001, draw_shape=None, colour=(255,0,0)):

        self.mass = mass
        self.pos = init_pos
        self.vel = init_vel

        self.time_step = time_step
        self.drag = drag

        self.force = np.array([0,0])

        self.draw_shape = draw_shape
        self.colour = colour


    def update_state(self,deltaT=None):
        
        if deltaT is None:
            deltaT = self.time_step

        acc = np.multiply(self.vel,-self.drag*la.norm(self.vel)) + np.divide(self.force, self.mass)

        self.pos = self.pos + np.multiply(self.vel,deltaT)+np.multiply(acc,deltaT**(2)/2)
        self.vel = self.vel + np.multiply(acc,deltaT)

    def draw(self, screen, screen_pos): # Define the triangle's vertices
        draw_shape = self.rotate()
        vertices = draw_shape + screen_pos
        # Draw the triangle
        pygame.draw.polygon(screen, self.colour, vertices)
        return screen
    
    def rotate(self):
        angle = self.forward_angle_world
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],\
                            [-np.sin(angle), np.cos(angle)]])
        # ALERT PYGAME DOES NOT OBEY RIGHT HAND RULE
        
        new_shape_points = np.dot(self.draw_shape, rot_mat)

        return new_shape_points
    
    @property
    def forward_angle_world(self):
        return np.arctan2(self.vel[1], self.vel[0])
    
    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    @staticmethod
    def scalar_and_angle_to_vec(scalar, angle):
        return np.multiply([np.cos(angle), np.sin(angle)], scalar)