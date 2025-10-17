from rl_spatial.agents.tabular_q import TabularQLearner

def test_q_learner_update():
    agent = TabularQLearner(n_states=4, n_actions=2, alpha=0.5, gamma=0.9, seed=0)
    s, a, r, sp, done = 0, 1, 1.0, 2, False
    before = agent.Q[s, a]
    agent.update(s, a, r, sp, done)
    after = agent.Q[s, a]
    assert after != before
