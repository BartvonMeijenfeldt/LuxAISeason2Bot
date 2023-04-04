import numpy as np

from collections import defaultdict
from objects.cargo import UnitCargo
from typing import Dict, List

from lux.config import EnvConfig
from lux.team import Team
from objects.actors.unit import Unit
from objects.game_state import GameState
from objects.board import Board
from objects.coordinate import TimeCoordinate
from objects.actors.factory import Factory
from objects.actions.unit_action import UnitAction
from logic.goals.unit_goal import Goal


def process_action(action):
    return to_json(action)


def to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj


def from_json(state):
    if isinstance(state, list):
        return np.array(state)
    elif isinstance(state, dict):
        out = {}
        for k in state:
            out[k] = from_json(state[k])
        return out
    else:
        return state


def process_obs(player, game_state, step, obs):
    if step == 0:
        # at step 0 we get the entire map information
        game_state = from_json(obs)
    else:
        # use delta changes to board to update game state
        obs = from_json(obs)
        for k in obs:
            if k != "board":
                game_state[k] = obs[k]
            else:
                if "valid_spawns_mask" in obs[k]:
                    game_state["board"]["valid_spawns_mask"] = obs[k]["valid_spawns_mask"]
        for item in ["rubble", "lichen", "lichen_strains"]:
            for k, v in obs["board"][item].items():
                k = k.split(",")
                x, y = int(k[0]), int(k[1])
                game_state["board"][item][x, y] = v

    return game_state


def obs_to_game_state(step, env_cfg: EnvConfig, obs, player: str, opp: str, prev_steps_goals: Dict[str, Goal]):
    units = create_units(obs=obs, env_cfg=env_cfg, t=obs["real_env_steps"], prev_step_goals=prev_steps_goals)
    factories = create_factories(obs=obs, env_cfg=env_cfg, t=obs["real_env_steps"])
    factory_occupancy_map = create_factory_occupancy_map(factories, obs["board"]["rubble"].shape)

    player_team = Team(**obs["teams"][player], agent=player) if player in obs["teams"] else None
    opp_team = Team(**obs["teams"][opp], agent=opp) if opp in obs["teams"] else None

    board = Board(
        rubble=obs["board"]["rubble"],
        ice=obs["board"]["ice"],
        ore=obs["board"]["ore"],
        lichen=obs["board"]["lichen"],
        lichen_strains=obs["board"]["lichen_strains"],
        factory_occupancy_map=factory_occupancy_map,
        factories_per_team=obs["board"]["factories_per_team"],
        valid_spawns_mask=obs["board"]["valid_spawns_mask"],
        player_units=units[player],
        opp_units=units[opp],
        player_factories=factories[player],
        opp_factories=factories[opp],
    )

    return GameState(
        env_cfg=env_cfg,
        env_steps=step,
        board=board,
        player_team=player_team,
        opp_team=opp_team,
    )


def create_units(obs, env_cfg: EnvConfig, t: int, prev_step_goals: Dict[str, Goal]) -> Dict[str, List[Unit]]:
    units = defaultdict(list)

    for agent in obs["units"]:
        for unit_data in obs["units"][agent].values():
            unit_data = unit_data.copy()
            unit_data["tc"] = TimeCoordinate(*unit_data["pos"], t=t)
            del unit_data["pos"]
            unit_data["cargo"] = UnitCargo(**unit_data["cargo"])
            unit_data["unit_cfg"] = env_cfg.get_unit_config(unit_data["unit_type"])
            unit_data["action_queue"] = [UnitAction.from_array(action) for action in unit_data["action_queue"]]

            if unit_data["unit_id"] in prev_step_goals:
                unit_data["prev_step_goal"] = prev_step_goals[unit_data["unit_id"]]
            else:
                unit_data["prev_step_goal"] = None

            unit = Unit(**unit_data)
            units[agent].append(unit)

    return units


def create_factories(obs, env_cfg, t: int) -> Dict[str, List[Factory]]:
    factories = defaultdict(list)

    for agent in obs["factories"]:
        for factory_data in obs["factories"][agent].values():
            factory_data = factory_data.copy()
            factory_data["center_tc"] = TimeCoordinate(*factory_data["pos"], t=t)
            del factory_data["pos"]
            factory_data["cargo"] = UnitCargo(**factory_data["cargo"])
            factory = Factory(**factory_data, env_cfg=env_cfg)
            factories[agent].append(factory)

    return factories


def create_factory_occupancy_map(factories: Dict[str, List[Factory]], board_shape):
    factory_occupancy_map = np.ones(board_shape, dtype=int) * -1

    for agent_factories in factories.values():
        for factory in agent_factories:
            factory_occupancy_map[factory.pos_slice] = factory.strain_id

    return factory_occupancy_map
