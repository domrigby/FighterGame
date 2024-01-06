from typing import List
import pandas as pd
import numpy as np
import time
import json

from game.fighter import Fighter
from game.display import FighterGameDisplay

def str_to_list(str):
    str_list = str.split(',')
    return np.array([float(element) for element in str_list])


class FighterGame:

    def __init__(self, input_file, render=False) -> None:


        # create fighter list
        self.active_list: List[Fighter] = []

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

            self.active_list.append(Fighter(team, mass, init_pos, init_vel, draw_shape=np.array([[0,-15],[0,15],[15,0]]), colour=colour))

        # Rendering set up
        self.render = render
        if self.render:
            self.screen_size = world_inputs['draw_size']
            self.render_env = FighterGameDisplay(self.screen_size, self.arena_size, self.origin)

    def run(self):
        while True:
            new_entities_to_add = []
            for obj in self.active_list:
                
                if obj.ent_type == 'fighter': # quick fix i dont like
                    obj.point_thruster(np.pi/6,100)
                    if np.random.random() < 0.002:
                        new_entities_to_add.append(obj.shoot())

                obj.update_state(0.01)

            self.active_list += new_entities_to_add

            if self.render:
                self.render_env.draw(self.active_list)


        

            
    