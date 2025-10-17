from rl_spatial.envs.grid_spatial_env import GridSpatialEnv

def test_env_shapes():
    env = GridSpatialEnv(grid_size=5, n_pois=10, n_types=3, max_steps=5, seed=0)
    s, info = env.reset()
    assert isinstance(s, int)
    assert env.observation_space.contains(s)
    for _ in range(3):
        sp, r, done, trunc, info = env.step(env.action_space.sample())
        assert isinstance(sp, int)
        assert env.observation_space.contains(sp)
        assert isinstance(r, float)
