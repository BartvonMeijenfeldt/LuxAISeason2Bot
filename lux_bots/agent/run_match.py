from luxai_s2.env import LuxAI_S2

from agent import Agent
from visualize.interactions import interact

env = LuxAI_S2()
env.reset()

# recreate our agents and run
agents = {player: Agent(player, env.state.env_cfg) for player in env.agents}
interact(env, agents, 1000)
