from torch import nn, optim
from torch.distributions.normal import Normal
import torch

from numpy import pi

PI = pi

# Custom weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.normal_(m.bias, mean=0.0, std=0.01)

class GenericNetwork:
    def __init__(self, name="", save_path=None) -> None:

        self.name = name

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimiser = optim.Adam(self.parameters(), lr=0.1)
        self.to(self.device)
        
        if save_path is None:
            self.save_path = f"saved_networks/{name}_{self.__class__.__name__}_save.pt"
        else:
            self.save_path = save_path

        # TODO: Over complicated?
        # send all torch tensors to cuda
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(self.device))

    def save_network(self):
        with open(self.save_path, 'wb') as f:
            torch.save(self.state_dict(),f)

    def load_network(self):
        self.load_state_dict(torch.load(self.save_path))

class ActorNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, action_dim, name="") -> None:

        # No Super so can initialise at different times
        nn.Module.__init__(self)

        hidden_lay_1 = 128
        hidden_lay_2 = 128

        self.fc1 = nn.Linear(state_dim, hidden_lay_1)
        self.fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.mean = nn.Linear(hidden_lay_2, action_dim)
        self.sigma = nn.Linear(hidden_lay_2, action_dim)

        self.noise = 1e-6
        
        self.action_min= torch.tensor([-PI/6, 0, -500, 0])
        self.action_max = torch.tensor([PI/6, 3089, 500, 10])

        self.act = nn.ReLU()

        GenericNetwork.__init__(self, name, None)

        self.apply(init_weights)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        # Get mean and std 
        mean = self.mean(x)
        std = self.sigma(x)

        std = torch.clamp(std, min=self.noise, max=1)

        return mean, std
    
    def sample_actions(self, state, reparameterize=False):

        # run network
        mean, std = self.forward(state)
        action_distribution = Normal(mean, std)

        # sample the dsitribtion
        if reparameterize:
            actions = action_distribution.rsample()
        else:
            actions = action_distribution.sample()
        
        # action: ranges fro
        tanh_actions = torch.tanh(actions)
        action = ((tanh_actions+1)*(self.action_max-self.action_min)/2) + self.action_min

        # take log probs for entropy
        log_probs = action_distribution.log_prob(actions)
        log_probs -= torch.log(((self.action_max-self.action_min)/2)*(1-tanh_actions.pow(2))+self.noise)
        log_probs = log_probs.sum()

        actions = ((actions+1)*(self.action_max-self.action_min)/2) + self.action_min

        return action, log_probs

class ValueNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, name="") -> None:
        nn.Module.__init__(self)

        hidden_lay_1 = 128
        hidden_lay_2 = 128

        self.fc1 = nn.Linear(state_dim, hidden_lay_1)
        self.fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.value = nn.Linear(hidden_lay_2, 1)

        self.act = nn.ReLU()

        GenericNetwork.__init__(self, name, None)

    def forward(self, state):
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        value = self.value(x)
        return value
    

class CriticNetwork(nn.Module, GenericNetwork):

    def __init__(self, state_dim, action_dim, name="") -> None:
        nn.Module.__init__(self)

        hidden_lay_1 = 128
        hidden_lay_2 = 128

        self.fc1 = nn.Linear(state_dim+action_dim, hidden_lay_1)
        self.fc2 = nn.Linear(hidden_lay_1, hidden_lay_2)

        self.q_val = nn.Linear(hidden_lay_2, 1)

        self.act = nn.ReLU()

        GenericNetwork.__init__(self, name, None)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        q = self.q_val(x)
        return q