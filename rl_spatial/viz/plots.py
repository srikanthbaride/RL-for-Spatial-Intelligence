import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def plot_learning_curve(returns: list[float], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    x = np.arange(len(returns))
    plt.plot(x, returns, label='Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def save_q_values(Q: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    np.save(out_path, Q)
