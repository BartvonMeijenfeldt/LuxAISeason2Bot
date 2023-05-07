import numpy as np
import random

from itertools import count
from luxai_s2.env import LuxAI_S2


def interact(env: LuxAI_S2, agents, nr_steps: int, seed=None):
    if not seed:
        seed = random.randint(0, 10000)

    obs = env.reset(seed=seed)
    np.random.seed(0)

    setup_step = 0

    while env.state.real_env_steps < 0:
        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(setup_step, o)
            actions[player] = a

        setup_step += 1
        obs, _, _, _ = env.step(actions)

    for step in count(start=setup_step):
        if step >= nr_steps + setup_step:
            break

        actions = {}
        for player in env.agents:
            o = obs[player]

            # env_step = o["real_env_steps"]
            # if env_step in [999] and player == "player_0":
            #     import cProfile

            #     cProfile.runctx("a = agents[player].act(step, o)", None, locals(), f"prof/step_{env_step}.prof")

            a = agents[player].act(step, o)
            actions[player] = a

        obs, _, dones, _ = env.step(actions)
        done = dones["player_0"] and dones["player_1"]
        if done:
            break

    return
