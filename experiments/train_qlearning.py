import argparse
import os
from rl_spatial.envs.grid_spatial_env import GridSpatialEnv
from rl_spatial.agents.tabular_q import TabularQLearner
from rl_spatial.viz.plots import plot_learning_curve, save_q_values

def run(episodes: int, grid: int, seed: int):
    env = GridSpatialEnv(grid_size=grid, seed=seed, max_steps=100)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = TabularQLearner(n_states, n_actions, seed=seed)

    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(s)
            sp, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.update(s, a, r, sp, done)
            s = sp
            total += r
        agent.decay_epsilon()
        returns.append(total)
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {ep+1}/{episodes} | Return={total:.2f} | eps={agent.eps:.3f}")

    os.makedirs('artifacts', exist_ok=True)
    plot_learning_curve(returns, 'artifacts/learning_curve.png')
    save_q_values(agent.Q, 'artifacts/q_values.npy')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--grid', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    run(args.episodes, args.grid, args.seed)
