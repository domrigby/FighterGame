from game.item import Item
from game.bullet import Bullet

import numpy as np
import numpy.linalg as la

from copy import deepcopy

import pygame

class Fighter(Item):

    def __init__(self, team, mass=0.1, init_pos=..., init_vel=[10,10],draw_shape=None, colour=(255,0,0)):
        self.team = team
        self.hit_box = 100
        self.radar_bins = 72
        self.ent_type = 'fighter'
        self.magazine_size = 0

        self.rudder_area = 1000

        super().__init__(mass, init_pos, init_vel, draw_shape=draw_shape, colour=colour)

    def apply_thrust(self, vector):
        self.thrust = np.array(vector)

    def point_thruster(self, angle, force=100):
        """
        This is in the fighters frame of reference with 0 pi directly backwards and anti clockwise
        """

        angle = np.clip(angle, -np.pi/6, np.pi/6)

        # must convert to world frame of reference
        thrust_angle_world = self.orientation + angle # thrust is in forward direction
        self.thrust= self.scalar_and_angle_to_vec(force, thrust_angle_world)

    def apply_break(self, amount):
        scale_for_misalign = abs(np.cos(self.forward_vel_angle_world - self.orientation))
        self.brake = - 0.5* amount * (self.vel/la.norm(self.vel))* scale_for_misalign
        if self.brake.dot(self.vel) > 0:
            raise ValueError("Brake cannot be in direction of velocity")
    
    def activate_adjuster(self, amount):
        # amount is in range [-pi/2, pi/2]
        # this is the angle of the rudder

        # This gives us the relative angle of the rudder the velocity 
        rudder_angle_to_vel = np.pi - (self.orientation - self.forward_vel_angle_world) - amount

        length_in_vel_dir = abs(self.rudder_area * np.sin(rudder_angle_to_vel))

        force_on_rudder = self.drag_const_at_speed() * la.norm(self.vel) * length_in_vel_dir

        self.adjuster_torque = force_on_rudder * np.sin(amount)

    def shoot(self):
        # deep copy so thet dont have the same memory address
        if self.magazine_size <= 0:
            return
        self.magazine_size -= 1
        vel_norm = self.vel/la.norm(self.vel)
        return Bullet(init_pos=deepcopy(self.pos)+vel_norm*(self.hit_box*2), init_vel=250*vel_norm, colour=(0,0,0))
    
    def sensors(self, other_ents):
        radar = np.zeros(self.radar_bins)
        threat_id = np.zeros(self.radar_bins)
        bin_width = (2*np.pi/self.radar_bins)
        for other in other_ents:
            if other is self:
                continue
            rel_pos = self.pos - other.pos
            rel_angle_world = np.arctan2(rel_pos[1], rel_pos[0])
            rel_angle_fighter_frame = rel_angle_world - self.forward_angle_world

            rel_angle_fighter_frame = self.conv_to_angle_range(rel_angle_fighter_frame)

            #TODO: fix the below... keeps giving 
            angle_bin = int(rel_angle_fighter_frame//bin_width + self.radar_bins//2)-1
            radar[angle_bin] = la.norm(rel_pos)
            
            threat_true = 1 if self.team != other.team else 0
            threat_id[angle_bin] = threat_true

        sensor_return = np.append(radar, threat_id)
        return sensor_return
    
    def stabiliser(self):
        stab_torque=-0.1*self.vel_vs_orientation_ang
        return stab_torque

    
    def state(self, other_fighters):
        sensor_return = self.sensors(other_fighters)
        state = np.concatenate((self.pos, [la.norm(self.vel)], [self.ang_vel], [self.vel_vs_orientation_ang]))
        #state = np.concatenate((self.pos, self.vel, [self.orientation], [self.ang_vel] ) )#, sensor_return))
        return state
    
    @staticmethod
    def conv_to_angle_range(angle):
        # np mod is for 0 to 2pi so have to shift by pi then take pi away
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    
    @property
    def thrust_0_angle_world(self):
        return self.forward_angle_world - np.pi
    



    