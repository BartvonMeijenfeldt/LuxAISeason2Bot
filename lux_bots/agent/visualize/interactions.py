import numpy as np
import cv2
import random

from datetime import datetime
from luxai_s2.env import LuxAI_S2


def animate(imgs):
    file_name = _get_file_name()
    height, width, _ = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    video = cv2.VideoWriter(file_name, fourcc, 1, (width, height))

    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.write(img)

    video.release()


def _get_file_name() -> str:
    datetime_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f'matches/{datetime_now}.webm'
    return file_name


def interact(env: LuxAI_S2, agents, nr_steps: int, seed=None):
    if not seed:
        seed = random.randint(0, 10000)

    obs = env.reset(seed=seed)
    np.random.seed(0)

    imgs = []
    step = 0

    while env.state.real_env_steps < 0:
        if step >= nr_steps:
            break

        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].early_setup(step, o)
            actions[player] = a

        step += 1
        obs, _, dones, _ = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]

    done = False
    while not done:
        if step >= nr_steps:
            break

        actions = {}
        for player in env.agents:
            o = obs[player]
            a = agents[player].act(step, o)
            actions[player] = a

        step += 1
        obs, _, dones, _ = env.step(actions)
        imgs += [env.render("rgb_array", width=640, height=640)]
        done = dones["player_0"] and dones["player_1"]

    return animate(imgs)
