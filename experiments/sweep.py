import itertools
import subprocess

grids = [8, 10]
episodes = [200, 400]
seeds = [13, 42]

for g, e, s in itertools.product(grids, episodes, seeds):
    print(f"\n=== Running grid={g}, episodes={e}, seed={s} ===")
    subprocess.run(['python', 'experiments/train_qlearning.py', '--grid', str(g), '--episodes', str(e), '--seed', str(s)], check=True)
