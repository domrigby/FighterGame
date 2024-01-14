import numpy as np
import numpy.linalg as la

import pygame
from copy import deepcopy

class Item:

    def __init__(self, mass=1, init_pos=[0,0], init_vel = [0,0], orientation=None, mom_of_inertia=None,time_step=0.1, drag=0.5, draw_shape=None, colour=(255,0,0)):

        self.mass = mass
        self.pos = init_pos
        self.vel = init_vel

        self.ang_vel = 0
        if orientation is None:
            self.orientation = deepcopy(self.forward_vel_angle_world)
        else:
            self.orientation = orientation

        if mom_of_inertia is None:
            self.mom_of_inertia=100*self.mass*2**2

        self.time_step = time_step
        self.drag = drag

        self.thrust = np.array([0,0])
        self.brake = np.array([0,0])

        self.thruster_pos = np.array([0, 0])

        self.adjuster_torque = 0.0
        self.adjuster_pos =np.array([0, 0])

        self.draw_shape = draw_shape
        self.colour = colour

        self.dead = False

        self.debug = False

    def random_reset(self, arena_size):
        #TODO carry on from here
        # need to reset the fighters

        # current starts in top and is fired towards middle
        self.pos = np.random.uniform(0, arena_size[0], size=2)
        vec_to_middle = (np.array(arena_size)/2) - self.pos
        self.vec = np.random.uniform(20,40)*vec_to_middle/la.norm(vec_to_middle)

        #self.vel= np.random.uniform(20, 30, size=2)

        self.orientation = self.forward_vel_angle_world + np.random.uniform(-0.1, 0.1)

        # below allows you set custom range in either direction
        self.ang_vel = np.random.uniform(0, np.pi/24)
        self.ang_vel *= np.random.choice([-1,1])

        self.dead = False

    def update_state(self, deltaT=None):
        
        if deltaT is None:
            deltaT = self.time_step

        lin_force, thrust_torque = self.split_translat_and_rotat(self.thrust, self.thruster_pos)


        # apply forces
        lin_drag = np.multiply(self.vel,-self.drag*la.norm(self.vel))

        stabiliser_force = self.stabiliser()

        lin_acc = np.divide(lin_drag+lin_force+self.drag*self.brake, self.mass)
        ang_acc = np.multiply(1/self.mom_of_inertia, thrust_torque+self.adjuster_torque)

        self.pos += np.multiply(self.vel,deltaT)+np.multiply(lin_acc,deltaT**(2)/2)
        self.vel += np.multiply(lin_acc,deltaT)

        self.orientation += np.multiply(self.ang_vel,deltaT)+np.multiply(ang_acc,deltaT**(2)/2)
        self.orientation = self.normalize_angle(self.orientation)
        self.ang_vel += ang_acc*deltaT

        if self.debug:
            print(f"Lin a: {lin_acc}")
            print(f"Lin drag {lin_drag} Lin thrust: {lin_force} Lin brake {self.brake}")
            print(f"Ang T: {thrust_torque+self.adjuster_torque} Ang A: {ang_acc}")
            print(f"pos: {self.pos} vel: {self.vel} ori: {self.orientation} ang vel: {self.ang_vel}")


    def split_translat_and_rotat(self, force, origin_of_force):
        pos_norm = la.norm(origin_of_force)
        if pos_norm > 0.001:
            pos_unit = origin_of_force/pos_norm
            LoA_force = np.multiply(np.dot(force, pos_unit), pos_unit)
        else:
            LoA_force = force

        if len(force) < 3 or len(origin_of_force) < 3:
            force = np.append(force, 0)
            origin_of_force = np.append(origin_of_force, 0)

        torque = np.cross(origin_of_force, force)[2]

        return LoA_force, torque

    def draw(self, screen, screen_pos): # Define the triangle's vertices
        draw_shape = self.rotate(self.orientation)
        vertices = draw_shape + screen_pos
        # Draw the triangle
        pygame.draw.polygon(screen, self.colour, vertices)
        return screen
    
    def rotate(self, angle):
        rot_mat = np.array([[np.cos(angle), np.sin(angle)],\
                            [-np.sin(angle), np.cos(angle)]])
        # ALERT PYGAME DOES NOT OBEY RIGHT HAND RULE
        
        new_shape_points = np.dot(self.draw_shape, rot_mat)

        return new_shape_points
    
    def stabiliser(self):
        return np.array([0,0])
    
    @property
    def state(self):
        state = np.append(self.pos, self.vel, self.orientation, self.ang_vel)
        return state
    
    @property
    def forward_vel_angle_world(self):
        return np.arctan2(self.vel[1], self.vel[0])
    
    @property
    def vel_vs_orientation_ang(self):
        return self.normalize_angle(self.forward_vel_angle_world - self.orientation)
    
    @staticmethod
    def normalize_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    @staticmethod
    def scalar_and_angle_to_vec(scalar, angle):
        return np.multiply([np.cos(angle), np.sin(angle)], scalar)