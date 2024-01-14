from RL.networks import ActorNetwork, CriticNetwork, ValueNetwork
from RL.memory import ReplayBuffer
import torch
from torch import nn

from itertools import count

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RLAgent:

    def __init__(self, env, learning=True, temperature=2, load_actor_file=None, num_rums=1000) -> None:
        self.env = env

        state_dim = 5

        # temperature is how much entropy of policy matters
        self.temperature = temperature

        # learning params
        self.learning = learning
        self.batch_size= 32
        self.tau = 0.05
        self.num_runs = num_rums

        self.gamma = 0.9

        self.replay = ReplayBuffer(10000, state_dim, 4)

        # TODO: add action dims and state dims to game env
        # TODO: expand for multipy agents

        self.actor_net = ActorNetwork(state_dim=state_dim, action_dim=4, name="actor")

        if load_actor_file:
            self.load_everything()

        # Two critic networks for stability
        self.critic_1 = CriticNetwork(state_dim=state_dim, action_dim=4, name="critic_1")
        self.critic_2 = CriticNetwork(state_dim=state_dim, action_dim=4, name="critic_2")

        # Value network
        self.value_net = ValueNetwork(state_dim=state_dim, name='value')
        self.target_value_net = ValueNetwork(state_dim=state_dim, name='target')
        self.val_loss = nn.MSELoss()

        self.conversative_value_update()

        self.scores = []

        self.plot_scores = True
        if self.plot_scores:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots()
            scores = []  # List to store scores
            self.line, = self.ax.plot(scores)
            plt.xlabel('Runs')
            plt.ylabel('Score')
            plt.title('RL Algorithm Score Over Time')

        self.counter = 0
    
    def get_action(self, game_state, fighter):
        # In future polciy will be contained within fighter
        action, _ = self.actor_net.sample_actions(torch.tensor(game_state[0], dtype=torch.float32, device=self.actor_net.device))
        return action.cpu().numpy()

    def explore_one_episode(self):

        state = self.env.reset()

        score = 0

        for t in count():

            action = self.get_action(state, None)
            new_state, reward, done, _ = self.env.step(action)
            score += reward

            if (t%self.num_runs*10==0 and t!=0) or done:
                done = True
            else:
                done = False

            self.replay.store_transition(state[0], action, reward, new_state[0], done)

            if self.learning:
                self.learn()

            if done:
                state = self.env.reset()
                break
            
            state = new_state

        self.scores.append(score)
        if self.plot_scores:
            self.line.set_ydata(self.scores)
            self.line.set_xdata(range(len(self.scores)))
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.counter%10==0:
            self.save_everything()
        
        self.counter += 1

    def learn(self):
        if self.replay.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
                self.replay.sample_buffer(self.batch_size)

        # create tensors from numpy arrays
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor_net.device)
        done = torch.tensor(done).to(self.actor_net.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.actor_net.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor_net.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor_net.device)


        # COMPUTE VALUE ERROR
        value = self.value_net(state).view(-1) # values of the current states
        
        critic_val, log_probs = self.criticise(state)

        self.value_net.optimiser.zero_grad()
        # next is the value function with entropy included
        value_target = critic_val - log_probs
        # error is value observed by critic and entropy - value of curent state
        value_loss = 0.5*self.val_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value_net.optimiser.step()

        critic_val, log_probs = self.criticise(state, reparam=True)
        # Actor wants entropy to be high and value to be high
        # Entropy of probs is always negative therefore min log probs and negative crtic value
        actor_loss = log_probs - critic_val
        actor_loss = torch.mean(actor_loss)
        self.actor_net.optimiser.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_net.optimiser.step()

        # OPTIMISE CRITICS WITH CLASSIC Q LEARNING
        target_values = self.target_value_net(new_state).view(-1) # target values of the next state
        target_values[done] = 0.0 # def of value func

        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()
        q_hat = self.temperature*reward + self.gamma*target_values
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * self.val_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * self.val_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        self.conversative_value_update()

    def criticise(self, states, reparam=False):
        # see what actions we would do in current state
        actions, log_probs = self.actor_net.sample_actions(states, reparam)
        log_probs = log_probs.view(-1) # need log probs for entropy calc
        q1_of_new_pol = self.critic_1.forward(states, actions)
        q2_of_new_pol = self.critic_2.forward(states, actions)
        critic_val = torch.min(q1_of_new_pol, q2_of_new_pol) # minimum taken for stability
        critic_val= critic_val.view(-1)
        return critic_val, log_probs

    def conversative_value_update(self):
        V_target_params= self.target_value_net.state_dict()
        V_learned_params = self.value_net.state_dict()
        for key in V_learned_params :
            V_target_params[key] = V_learned_params[key]*self.tau + V_target_params[key]*(1-self.tau)
        self.target_value_net.load_state_dict(V_target_params)

    def save_everything(self):
        # find all networks... not loss
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.save_network()

    def load_everything(self):
        for name, attribute in self.__dict__.items():
            if isinstance(attribute, nn.Module) and not isinstance(attribute, nn.modules.loss._Loss):
                attribute.load_network()

    def animate(self):
        plt.cla()  # Clear the current axes
        plt.plot(self.scores)

    