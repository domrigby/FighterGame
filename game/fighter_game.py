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

    def __init__(self, input_file, render=False) -> None:

        # create fighter list
        self.active_fighters: List[Fighter] = []

        with open('inputs/world.json', 'r') as file:
            world_inputs = json.load(file)

        self.arena_size = world_inputs['arena_size']
        self.origin = np.array([0,0])

        # load input files
        player_inputs = pd.read_csv(input_file)#, skiprows=1)

        for i, row in player_inputs.iterrows():
            team = row['team']
            mass = float(row['mass'])
            init_pos = str_to_list(row['initial position'])
            init_vel = str_to_list(row['initial velocity'])
            colour = tuple(str_to_list(row['colour']))

            self.active_fighters.append(Fighter(team, mass, init_pos, init_vel, draw_shape=np.array([[0,-15],[0,15],[15,0]]), colour=colour))

        # Rendering set up
        self.render = render
        if self.render:
            self.screen_size = world_inputs['draw_size']
            self.render_env = FighterGameDisplay(self.screen_size, self.arena_size, self.origin)

    def run(self):
        active_weapons = []

        while True:
            for obj in self.active_fighters:
                obj.update_state(0.01)

                # apply control
                obj.point_thruster(0,100)
                obj.apply_break(1)
                if np.random.random() < 0.002:
                    active_weapons.append(obj.shoot())

                if obj.dead:
                    self.active_fighters.remove(obj)
            
            for bul in active_weapons:
                bul.update_state(0.01)
                self.check_in_range(bul, self.active_fighters)
                self.check_in_area(bul)
                if bul.dead:
                    active_weapons.remove(bul)

            if self.render:
                self.render_env.draw(self.active_fighters+active_weapons)

    def check_in_range(self, item, list_to_check):
        for i, agent in enumerate(list_to_check):
            print(la.norm(agent.pos-item.pos))
            if la.norm(agent.pos-item.pos) < agent.hit_box:
                list_to_check[i].dead = True
                item.dead = True
    
    def check_in_area(self, item):
        lower_bound_out = np.any(item.pos < 0)
        higher_bound_out = np.any((item.pos-self.arena_size)>0)
        if lower_bound_out or higher_bound_out:
            item.dead = True

    def get_state(self):
        pass


        

            
    