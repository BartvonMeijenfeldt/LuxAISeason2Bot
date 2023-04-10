import numpy as np

from collections import defaultdict
from objects.cargo import Cargo
from typing import Dict, List

from lux.config import EnvConfig
from lux.team import Team
from objects.actors.actor import Actor
from objects.actors.unit import Unit
from objects.game_state import GameState
from objects.board import Board
from objects.coordinate import TimeCoordinate
from objects.actors.factory import Factory
from objects.actions.unit_action import UnitAction


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


def obs_to_game_state(
    step, env_cfg: EnvConfig, obs, player: str, opp: str, prev_step_actors: Dict[str, Actor]
) -> GameState:

    units = create_units(obs=obs, env_cfg=env_cfg, t=obs["real_env_steps"], prev_step_actors=prev_step_actors)
    factories = create_factories(obs=obs, env_cfg=env_cfg, t=obs["real_env_steps"], prev_step_actors=prev_step_actors)
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

    return GameState(env_cfg=env_cfg, env_steps=step, board=board, player_team=player_team, opp_team=opp_team)


def create_units(obs, env_cfg: EnvConfig, t: int, prev_step_actors: Dict[str, Actor]) -> Dict[str, List[Unit]]:
    units = defaultdict(list)

    for agent in obs["units"]:
        for unit_data in obs["units"][agent].values():
            unit_id = unit_data["unit_id"]
            power = unit_data["power"]
            cargo = Cargo(**unit_data["cargo"])
            tc = TimeCoordinate(*unit_data["pos"], t=t)

            if unit_id in prev_step_actors:
                unit: Unit = prev_step_actors[unit_id]  # type: ignore
                action_queue = [UnitAction.from_array(action) for action in unit_data["action_queue"]]
                unit.update_state(tc=tc, power=power, cargo=cargo, action_queue=action_queue)
            else:
                team_id = unit_data["team_id"]
                unit_type = unit_data["unit_type"]
                unit_cfg = env_cfg.get_unit_config(unit_data["unit_type"])

                unit = Unit(
                    team_id=team_id,
                    unit_id=unit_id,
                    power=power,
                    cargo=cargo,
                    unit_type=unit_type,
                    tc=tc,
                    unit_cfg=unit_cfg,
                )

            units[agent].append(unit)

    return units


def create_factories(obs, env_cfg, t: int, prev_step_actors: Dict[str, Actor]) -> Dict[str, List[Factory]]:
    factories = defaultdict(list)

    for agent in obs["factories"]:
        for factory_data in obs["factories"][agent].values():
            team_id = factory_data["team_id"]
            unit_id = factory_data["unit_id"]
            power = factory_data["power"]
            cargo = Cargo(**factory_data["cargo"])
            strain_id = factory_data["strain_id"]
            center_tc = TimeCoordinate(*factory_data["pos"], t=t)

            if unit_id in prev_step_actors:
                factory: Factory = prev_step_actors[unit_id]  # type: ignore
                factory.update_state(center_tc, power, cargo)

            else:

                factory = Factory(
                    team_id=team_id,
                    unit_id=unit_id,
                    power=power,
                    cargo=cargo,
                    strain_id=strain_id,
                    center_tc=center_tc,
                    env_cfg=env_cfg,
                )

            factories[agent].append(factory)

    return factories


def create_factory_occupancy_map(factories: Dict[str, List[Factory]], board_shape):
    factory_occupancy_map = np.ones(board_shape, dtype=int) * -1

    for agent_factories in factories.values():
        for factory in agent_factories:
            factory_occupancy_map[factory.pos_slice] = factory.strain_id

    return factory_occupancy_map
