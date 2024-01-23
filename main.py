

from game.fighter_game import FighterGame
from RL.agent import RLAgent

env = FighterGame('inputs/fighter.csv', render=True) #, real_time=True)

agent = RLAgent(env, learning=False) #, load_actor_file=True, num_runs=10000)

for i in range(100000):
    #print(i)
    agent.explore_one_episode()
    #print(agent.scores[-1])
