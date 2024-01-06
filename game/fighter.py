from game.item import Item
from game.bullet import Bullet

import numpy as np
import numpy.linalg as la

from copy import deepcopy

import pygame

class Fighter(Item):

    def __init__(self, team, mass=1, init_pos=..., init_vel=[10,10],draw_shape=None, colour=(255,0,0)):
        self.ent_type = 'fighter'
        super().__init__(mass, init_pos, init_vel, draw_shape=draw_shape, colour=colour)

    def apply_thrust(self, vector):
        self.force = np.array(vector)

    def point_thruster(self, angle, force=100):
        """
        This is in the fighters frame of reference with 0 pi directly backwards and anti clockwise
        """

        angle = np.clip(angle, -np.pi/3, np.pi/3)

        # must convert to world frame of reference
        thrust_angle_world = self.forward_angle_world + angle # thrust is in forward direction
        self.force = self.scalar_and_angle_to_vec(force, thrust_angle_world)
    
    def shoot(self):
        # deep copy so thet dont have the same memory address
        return Bullet(init_pos=deepcopy(self.pos), init_vel=250*self.vel/la.norm(self.vel), colour=(0,0,0))

    @property
    def thrust_0_angle_world(self):
        return self.forward_angle_world - np.pi



    