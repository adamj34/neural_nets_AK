import numpy as np
import random
from collections import namedtuple, deque
import os

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_graph_mat(n=10, size=1):
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

def plot_graph(coords):
    n = len(coords)
    plt.scatter(coords[:,0], coords[:,1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'b', alpha=0.7)

State = namedtuple('State', ('W', 'coords', 'partial_solution'))

def state2tens(state: State) -> torch.Tensor:
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords = state.coords
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),
           (1 if i == sol_first_node else 0),
           (1 if i == sol_last_node else 0),
           coords[i,0],
           coords[i,1]
          ] for i in range(nr_nodes)]
    
    return torch.tensor(xv, dtype=torch.float32, device=device)

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)
        return x

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
LR = 0.001
NUM_EPISODES = 1000
MEMORY_CAPACITY = 10000
N_NODES = 10  # Fixed number of nodes

# Initialize networks
q_net = QNetwork().to(device)
target_q_net = QNetwork().to(device)
target_q_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

epsilon = EPS_START

def select_action(state: State, q_net: QNetwork, epsilon: float, n_nodes: int):
    if random.random() < epsilon:
        valid_actions = [i for i in range(n_nodes) if i not in state.partial_solution]
        return random.choice(valid_actions) if valid_actions else None
    else:
        with torch.no_grad():
            state_tensor = state2tens(state).unsqueeze(0)
            q_values = q_net(state_tensor).squeeze()
        mask = torch.ones(n_nodes, dtype=torch.bool, device=device)
        mask[list(state.partial_solution)] = False
        q_values[~mask] = -float('inf')
        return q_values.argmax().item()

# Training Loop
for episode in range(NUM_EPISODES):
    coords, W_np = get_graph_mat(n=N_NODES)
    current_state = State(W=W_np, coords=coords, partial_solution=[])
    total_reward = 0
    done = False

    while not done:
        action = select_action(current_state, q_net, epsilon, N_NODES)
        if action is None:
            done = True
            break

        next_partial = current_state.partial_solution + [action]
        prev_partial = current_state.partial_solution

        if len(next_partial) == N_NODES:
            done = True
            if len(prev_partial) == 0:
                reward = 0.0
            else:
                prev_city = prev_partial[-1]
                step_reward = -W_np[prev_city, action]
                return_reward = -W_np[action, next_partial[0]]
                reward = step_reward + return_reward
        else:
            if len(prev_partial) == 0:
                reward = 0.0
            else:
                prev_city = prev_partial[-1]
                reward = -W_np[prev_city, action]
            done = False

        next_state = State(W=W_np, coords=coords, partial_solution=next_partial)
        memory.push(current_state, action, reward, next_state, done)
        current_state = next_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = [state2tens(s) for s in batch.state]
            state_batch = torch.stack(state_batch)
            action_batch = torch.tensor(batch.action, dtype=torch.long, device=device)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
            next_state_batch = [state2tens(s) for s in batch.next_state]
            next_state_batch = torch.stack(next_state_batch)
            done_batch = torch.tensor(batch.done, dtype=torch.bool, device=device)

            current_q_values = q_net(state_batch)
            current_q = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_q_net(next_state_batch)
                mask = torch.ones_like(next_q_values, dtype=torch.bool, device=device)
                for i, next_state in enumerate(batch.next_state):
                    mask[i, list(next_state.partial_solution)] = False
                next_q_values[~mask] = -float('inf')
                max_next_q = next_q_values.max(dim=1)[0]
                target_q = reward_batch + GAMMA * max_next_q * (~done_batch).float()

            loss = F.mse_loss(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE == 0:
        target_q_net.load_state_dict(q_net.state_dict())

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    print(f'Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}')