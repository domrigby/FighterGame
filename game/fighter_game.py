from typing import List
import pandas as pd
import numpy as np
import time
import json

import numpy.linalg as la

from game.fighter import Fighter
from game.display import FighterGameDisplay

def str_to_list(str):
    str_list = str.split(',')
    return np.array([float(element) for element in str_list])


class FighterGame:

    def __init__(self, input_file, render=False, real_time=False) -> None:

        # create fighter list
        self.active_fighters: List[Fighter] = []
        self.active_weapons = []

        with open('inputs/world.json', 'r') as file:
            world_inputs = json.load(file)

        self.arena_size = world_inputs['arena_size']
        self.origin = np.array([0,0])

        self.input_file = input_file
        self.set()

        # Rendering set up
        self.render = render
        if self.render:
            self.screen_size = world_inputs['draw_size']
            self.render_env = FighterGameDisplay(self.screen_size, self.arena_size, self.origin)

        self.time_step = 0.5
        self.last_time = time.time_ns()

        self.real_time = real_time

    def set(self):
        # load input files
        player_inputs = pd.read_csv(self.input_file)#, skiprows=1)

        for i, row in player_inputs.iterrows():
            team = row['team']
            mass = float(row['mass'])
            init_pos = str_to_list(row['initial position'])
            init_vel = str_to_list(row['initial velocity'])
            colour = tuple(str_to_list(row['colour']))

            self.active_fighters.append(Fighter(team, mass, init_pos, init_vel, draw_shape=np.array([[0,-15],[0,15],[15,0]]), colour=colour))
        
        return self.system_state

    def reset(self):
        #TODO carry on from here
        # need to reset the fighters
        self.active_fighters[0].random_reset(self.arena_size)

        #self.active_fighters[1].pos = np.random.uniform(0, self.arena_size, size=2)
        #self.active_fighters[1].vel = np.zeros(2)

        self.prev_ang_vel = self.active_fighters[0].ang_vel

        return self.system_state

    def inf_run(self):

        while True:
            #self.one_game_iter()
            action = np.array([np.pi/7, 100, 0.2, 0.001])
            self.one_game_iter(action)

    def step(self, action):
        # aircraft can:
        #   1. Point and adjust thruster
        #   2. Adjust rotation
        #   3. Brake
        #   4. Shoot
        # Vector for control:
        #   [Angle of thruster, thrust, perpendicular adjustment, brake]

        #TODO integrate multiple fighters
        if self.render and self.real_time:
            time_now = time.time_ns()
            time_step = (time_now - self.last_time) / 10e9
            self.last_time = time_now
        else:
            time_step = self.time_step

        self.one_game_iter([action, [0,0,0,0]], time_step=time_step)
        
        reward = self.reward_function()
        reward += 1
        
        done=False
        for obj in self.active_fighters:
            self.check_in_area(obj)
            if obj.dead:
                done = True

            if abs(obj.vel_vs_orientation_ang) > np.pi/2:
                done = True

        truncated = False

        return self.system_state, reward, done, truncated
    
    def reward_function(self):
        reward = 0
        return reward

    def one_game_iter(self, action_set, time_step=None):
        if time_step is None:
            time_step = self.time_step

        for action, obj in zip(action_set, self.active_fighters):

            obj.update_state(time_step)

            # apply control
            obj.point_thruster(action[0],action[1])
            obj.activate_adjuster(action[2])
            obj.apply_break(action[3])
            if np.random.random() < 0:
                self.active_weapons.append(obj.shoot())

            # reinstate when multiple agents
            if obj.dead:
                self.active_fighters.remove(obj)
        
        for bul in self.active_weapons:
            bul.update_state(time_step)
            self.check_in_range(bul, self.active_fighters)
            self.check_in_area(bul)
            if bul.dead:
                self.active_weapons.remove(bul)

        if self.render:
            self.render_env.draw(self.active_fighters+self.active_weapons)

    def check_in_range(self, item, list_to_check):
        for i, agent in enumerate(list_to_check):
            if la.norm(agent.pos-item.pos) < agent.hit_box:
                list_to_check[i].dead = True
                item.dead = True
    
    def check_in_area(self, item):
        lower_bound_out = np.any(item.pos < 0)
        higher_bound_out = np.any((item.pos-self.arena_size)>0)
        if lower_bound_out or higher_bound_out:
            item.dead = True

    @property
    def system_state(self):
        system_state = []
        for agent in self.active_fighters:
                system_state.append(agent.state(self.active_fighters))
        return system_state


        

            
    