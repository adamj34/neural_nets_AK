import numpy as np

from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_graph_mat(n=10, size=1):
    """Creates n nodes randomly in a square and returns the (n,2) coordinates and (n,n) distance matrix."""
    coords = size * np.random.uniform(size=(n, 2))
    dist_mat = distance_matrix(coords, coords)
    dist_mat = torch.tensor(dist_mat, dtype=torch.float32, requires_grad=False, device=device)

    return coords, dist_mat

def moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

def plot_graph(coords):
    """Plots the fully connected graph."""
    n = len(coords)
    plt.scatter(coords[:, 0], coords[:, 1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], 'b', alpha=0.7)
    plt.show()

def plot_rewards(episode_rewards):
    """Plot the total rewards over episodes."""
    plt.figure(figsize=(10, 5))
    # plt.plot(moving_avg(episode_rewards))
    plt.semilogy(moving_avg(episode_rewards))
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

def plot_solution(coords, solution, tour_len, tour_type):
    plt.title(f"${tour_type}$ Tour Length: {tour_len:.2f}")
    plt.scatter(coords[:,0], coords[:,1])
    n = len(coords)
    
    for idx in range(n-1):
        i, next_i = solution[idx], solution[idx+1]
        plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    
    i, next_i = solution[-1], solution[0]
    plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    plt.plot(coords[solution[0], 0], coords[solution[0], 1], 'x', markersize=10)
    plt.show()
