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


logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig, debug_mode: bool = False) -> None:
        np.random.seed(0)
        datetime_now = datetime.now().strftime("%Y%m_%d_%H_%M_%S")
        logging.basicConfig(level=logging.INFO, filename=f"data/{datetime_now}_{player}.log")

        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"

        self.env_cfg: EnvConfig = env_cfg
        self.prev_step_actors: dict[str, Actor] = {}
        self.DEBUG_MODE = debug_mode

    def early_setup(self, step: int, obs: dict, remaing_overage_time: int = 60):
        """Method called during the first set up turns of the game in which the players and select the locations of
        their bases

        Args:
            step: Step number of game.
            obs: Observation of the game state.
            remaing_overage_time: Extra time in seconds to decide, in addition to the 3s per turn.

        Returns:
            dict indicating the bid, location to pick base, or nothing (if opponent turn).
        """
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_step_actors)

        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            if is_my_turn_to_place_factory(game_state, step):
                spawn_loc = get_factory_spawn_loc(game_state.board, obs["board"]["valid_spawns_mask"])
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs: dict, remaining_overage_time: int = 60) -> Dict[str, Any]:
        """This function schedules the actions of all actors. In the case of a Factory an explicit action to carry out.
        In the case of a unit that means updating the action queue with multiple actions. Units will update their
        action plan (which might be longer than the action queue) and only update the action queue if the next action
        would be different than the current action queue (to save power, every update costs power).

        Args:
            step: Step number of game.
            obs: Observation of the game state.
            remaining_overage_time: Extra time in seconds to decide, in addition to the 3s per turn.

        Returns:
            Actions of all actors.
        """
        self._set_time()

        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player, self.prev_step_actors)
        self._schedule_goals(game_state=game_state)
        self._store_actors(game_state=game_state)
        actions = self._get_actions(game_state=game_state)

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
            logger.info(f"{real_env_steps}: player {team_id} {time_taken: 0.1f}")
        else:
            logger.warning(f"{real_env_steps}: player {team_id} {time_taken: 0.1f}")

    def _get_actions(self, game_state: GameState) -> Dict[str, Any]:
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

            if unit.action_queue and unit.are_first_action_of_queue_and_private_action_plan_same:
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
