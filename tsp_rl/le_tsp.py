import numpy as np
import random
import math
from collections import namedtuple, deque

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Device configuration (note: the code is not fully optimized for GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#############################################
# Helper functions (from your starter code)
#############################################
def get_graph_mat(n=10, size=1):
    """Creates n nodes randomly in a square and returns the (n,2) coordinates and (n,n) distance matrix."""
    coords = size * np.random.uniform(size=(n, 2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

def plot_graph(coords):
    """Plots the fully connected graph."""
    n = len(coords)
    plt.scatter(coords[:, 0], coords[:, 1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'b', alpha=0.7)
    plt.show()

# The state contains the full distance matrix, node coordinates, and the current tour (partial_solution)
State = namedtuple('State', ('W', 'coords', 'partial_solution'))

def state2tens(state: State) -> torch.Tensor:
    """
    Convert a State into a (N, 5) tensor where for each node we store:
      - Whether it is already in the tour.
      - Whether it is the first node.
      - Whether it is the last visited node.
      - Its x-coordinate.
      - Its y-coordinate.
    """
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if state.partial_solution else -1
    sol_first_node = state.partial_solution[0] if state.partial_solution else -1
    coords = state.coords
    nr_nodes = coords.shape[0]
    xv = [[1 if i in solution else 0,
           1 if i == sol_first_node else 0,
           1 if i == sol_last_node else 0,
           coords[i, 0],
           coords[i, 1]]
          for i in range(nr_nodes)]
    return torch.tensor(xv, dtype=torch.float32, device=device)

#############################################
# Define the Q-Network (a simple MLP)
#############################################
class QNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # One output per node (a scalar Q value)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state_tensor):
        # state_tensor is of shape (N, 5)
        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)  # shape: (N, 1)
        return q.squeeze(-1)  # shape: (N,)

#############################################
# Replay Memory for Experience Replay
#############################################
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Save a transition."""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

#############################################
# Environment Step and Policy Functions
#############################################
def step(state: State, action: int):
    """
    Given a state and an action (choosing the next city), compute:
      - The immediate reward (negative distance traveled).
      - The next state.
      - Whether the episode is done (i.e. the tour is complete).
    When the tour is complete, we add the cost to return to the starting city.
    """
    current_city: int = state.partial_solution[-1]
    reward: float = - state.W[current_city, action]
    new_partial: list[int] = state.partial_solution + [action]
    done = False
    if len(new_partial) == state.W.shape[0]:
        # Completed tour; add cost for returning to the starting city.
        reward += - state.W[action, state.partial_solution[0]]
        done = True
    next_state: State = State(state.W, state.coords, new_partial)
    return next_state, reward, done

def select_action(state: State, policy_net, epsilon: float):
    """
    Selects the next action using an epsilon-greedy policy. Valid actions are the cities not
    already in the partial solution.
    """
    valid_actions = list(set(range(state.coords.shape[0])) - set(state.partial_solution))
    if random.random() < epsilon:
        return random.choice(valid_actions)
    else:
        with torch.no_grad():
            state_tensor = state2tens(state)
            q_values = policy_net(state_tensor) # state tensor contains all nodes embeddings at time t
            # Mask out already visited nodes by assigning them -infinity.
            q_values_masked = q_values.clone()
            for idx in state.partial_solution:
                q_values_masked[idx] = -float('inf')
            action = q_values_masked.argmax().item() # retern an index of the max value (it's action aka node label)
        return action

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma):
    """
    Samples a batch from replay memory and performs one optimization step.
    The target is computed using the target network.
    """
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    loss = 0.0
    for trans in transitions:
        s, a, r, next_s, done = trans
        s_tensor = state2tens(s)
        q_val = policy_net(s_tensor)[a]  # it's the 
        if done:
            target = torch.tensor(r, device=device)
        else:
            ns_tensor = state2tens(next_s)
            q_next = target_net(ns_tensor)
            # Mask out visited nodes in next state.
            q_next_masked = q_next.clone()
            for idx in next_s.partial_solution:
                q_next_masked[idx] = -float('inf')
            target = torch.tensor(r, device=device) + gamma * q_next_masked.max()
        loss += (q_val - target) ** 2
    loss = loss / batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#############################################
# Main Training Loop for DQL on TSP
#############################################
def train_tsp_dql(num_episodes=500, batch_size=32, gamma=0.99, 
                  epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200,
                  target_update=10, memory_capacity=10000, n_cities=10):
    # Generate a random TSP instance.
    coords, W_np = get_graph_mat(n=n_cities)
    W = W_np  # distance matrix (using numpy for reward computation)
    
    # A helper to reset the environment: here we fix the starting city to 0.
    def reset():
        return State(W, coords, [0])
    
    policy_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(memory_capacity)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = reset()
        total_reward = 0.0
        
        # Epsilon decays over episodes.
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay)
        
        while True:
            action = select_action(state, policy_net, epsilon)
            next_state, reward, done = step(state, action)
            total_reward += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma)
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Update the target network periodically.
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
    
    return policy_net, target_net, coords, W

#############################################
# Testing and Plotting the Greedy Solution
#############################################
def get_greedy_solution(policy_net, W, coords):
    """Extracts a tour by always choosing the best action (epsilon=0)."""
    state = State(W, coords, [0])
    solution = state.partial_solution.copy()
    while True:
        with torch.no_grad():
            state_tensor = state2tens(state)
            q_values = policy_net(state_tensor)
            q_values_masked = q_values.clone()
            for idx in state.partial_solution:
                q_values_masked[idx] = -float('inf')
            action = q_values_masked.argmax().item()
        state, reward, done = step(state, action)
        solution.append(action)
        if done:
            break
    return solution

def plot_tour(coords, tour):
    """Plots the TSP tour as a cycle connecting the nodes."""
    # Append the starting city to complete the cycle.
    tour_cycle = tour + [tour[0]]
    tour_coords = coords[tour_cycle]
    plt.figure()
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'ro-', label='Tour')
    plt.scatter(coords[:, 0], coords[:, 1], c='blue')
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i))
    plt.title("TSP Tour")
    plt.legend()
    plt.show()

#############################################
# Run the Training and Test the Solution
#############################################
if __name__ == '__main__':
    # For reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_episodes = 5000  # Adjust the number of episodes as needed.
    policy_net, target_net, coords, W = train_tsp_dql(num_episodes=num_episodes, n_cities=10)
    
    solution = get_greedy_solution(policy_net, W, coords)
    print("Greedy Solution (node order):", solution)
    # plot_tour(coords, solution)
