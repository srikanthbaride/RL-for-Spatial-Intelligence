# RL-for-Spatial-Intelligence

**Purpose:** Bridge Reinforcement Learning with Spatial Data Mining.  
Agents learn **spatial co-location discovery policies** in toy grid environments and (optionally) integrate with `colocationpy`.

> Novel AI × Geospatial theme → relevant to NVIDIA Spatial AI, KAUST/MBZUAI, and applied research labs.

---

## ✨ Features
- **Gymnasium environments** for spatial grids with Points-of-Interest (POIs)
- **Tabular Q-learning agent** (starter baseline) with ε-greedy exploration
- **Matplotlib visualizations** (learning curves, spatial heatmaps)
- **Streamlit dashboard** (prototype) under `rl_spatial/viz/dashboard.py`
- **Integration adapter** for `colocationpy` (placeholder API in `integration/colocationpy_adapter.py`)
- **Tests + CI** with GitHub Actions (`pytest -q` on push)

---

## 🧱 Project Layout
```
RL-for-Spatial-Intelligence/
├─ rl_spatial/
│  ├─ envs/
│  │  └─ grid_spatial_env.py         # Gymnasium Env for spatial co-location rewards
│  ├─ agents/
│  │  └─ tabular_q.py                # Minimal Q-learning baseline
│  ├─ viz/
│  │  ├─ plots.py                    # Matplotlib helper functions
│  │  └─ dashboard.py                # Streamlit prototype
│  └─ integration/
│     └─ colocationpy_adapter.py     # Optional adapter for colocationpy
├─ experiments/
│  ├─ train_qlearning.py             # Train & plot learning curve
│  └─ sweep.py                       # Simple hyperparameter sweep
├─ tests/
│  ├─ test_env.py
│  └─ test_agent.py
├─ .github/workflows/ci.yml          # PyTest workflow
├─ requirements.txt
├─ pyproject.toml
├─ LICENSE
└─ README.md
```

---

## 🚀 Quickstart

```bash
# (Recommended) Use Python 3.10+ in a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run a quick training session (tabular Q-learning on GridSpatialEnv)
python experiments/train_qlearning.py --episodes 300 --grid 10 --seed 42

# (Optional) Streamlit dashboard (prototype)
streamlit run rl_spatial/viz/dashboard.py
```

This will produce:
- `artifacts/learning_curve.png`
- `artifacts/q_values.npy`

---

## 🧪 Tests

```bash
pytest -q
```

---

## 🧩 Colocation Integration (Optional)

If you have `colocationpy` available (or this repo in development), use the adapter:

```python
from rl_spatial.integration.colocationpy_adapter import ColocationAdapter

adapter = ColocationAdapter()
pairs = adapter.mine_colocations(points=[(1,2,'A'), (3,4,'B')], radius=2.0, min_prev=0.2)
print(pairs)
```

> The adapter is a **shim** so you can later drop in a real implementation (API names noted inside the file).

---

## 📄 Citation

If you use this repo in research/teaching, please cite:

```bibtex
@misc{rlspatial-2025,
  title   = {RL-for-Spatial-Intelligence},
  author  = {Baride, Srikanth},
  year    = {2025},
  url     = {https://github.com/srikanthbaride/RL-for-Spatial-Intelligence},
  note    = {RL agents for spatial co-location discovery; Gymnasium environments and Q-learning baseline.}
}
```

---

## 📜 License
MIT License — see `LICENSE`.

---

## 🙌 Acknowledgments
- Gymnasium (formerly OpenAI Gym) for environment interfaces
- Inspiration from spatial data mining and co-location pattern mining literature
