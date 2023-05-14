from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from logic.early_setup import get_factory_spawn_loc
from logic.goal_resolution.scheduler import Scheduler
from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState

if TYPE_CHECKING:
    from objects.actors.actor import Actor


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig, debug_mode: bool = False) -> None:
        datetime_now = datetime.now().strftime("%Y%m_%d_%H_%M_%S")
        logging.basicConfig(level=logging.INFO, filename=f"data/{datetime_now}_{player}.log")
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.prev_step_actors: dict[str, Actor] = {}
        self.DEBUG_MODE = debug_mode

    def early_setup(self, step: int, obs, remaing_overage_time: int = 60):
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_step_actors)

        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            if is_my_turn_to_place_factory(game_state, step):
                spawn_loc = get_factory_spawn_loc(game_state.board, obs["board"]["valid_spawns_mask"])
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remaining_overage_time: int = 60):

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        self._set_time()

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_step_actors)
        self._schedule_goals(game_state=game_state)
        self._store_actors(game_state=game_state)
        actions = self.get_actions(game_state=game_state)

        self._log_time_taken(game_state.real_env_steps, game_state.player_team.team_id)

        return actions

    def _set_time(self) -> None:
        self.start_time = time.time()

    def _schedule_goals(self, game_state: GameState) -> None:
        scheduler = Scheduler(self.start_time, self.DEBUG_MODE, game_state)
        scheduler.schedule_goals()

    def _get_time_taken(self) -> float:
        return time.time() - self.start_time

    def _log_time_taken(self, real_env_steps: int, team_id: int) -> None:
        time_taken = self._get_time_taken()
        if time_taken < 1:
            logging.info(f"{real_env_steps}: player {team_id} {time_taken: 0.1f}")
        else:
            logging.warning(f"{real_env_steps}: player {team_id} {time_taken: 0.1f}")

    def get_actions(self, game_state: GameState) -> Dict[str, Any]:
        actions = {}

        for factory in game_state.player_factories:
            if factory.private_action_plan:
                actions[factory.unit_id] = factory.private_action_plan.to_lux_output()

        for unit in game_state.player_units:
            if not unit.can_update_action_queue:
                unit.set_send_no_action_queue()
                continue

            if not unit.private_action_plan:
                unit.set_send_no_action_queue()
                continue

            if not unit.action_queue and unit.private_action_plan.is_first_action_move_center():
                unit.set_send_no_action_queue()
                continue

            if unit.action_queue and unit.first_action_of_queue_and_private_action_plan_same:
                unit.set_send_no_action_queue()
                continue

            if (
                unit.has_too_little_power_for_first_action_in_queue(game_state)
                and unit.private_action_plan.is_first_action_move_center()
            ):
                unit.set_send_no_action_queue()
                continue

            actions[unit.unit_id] = unit.private_action_plan.to_lux_output()
            unit.set_send_action_queue(unit.private_action_plan)

        return actions

    def _store_actors(self, game_state: GameState) -> None:
        self.prev_step_actors = {actor.unit_id: actor for actor in game_state.actors}
