from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from dataclasses import replace
from collections.abc import Iterator
from search import get_actions_a_to_b

if TYPE_CHECKING:
    from search import Graph
    from objects.unit import Unit
    from objects.action import Action
    from objects.board import Board
    from objects.game_state import GameState


class ActionPlan:
    def __init__(self, actions: list[Action], unit: Unit) -> None:
        self.original_actions = actions
        self.unit = unit
        self.value: float = None

        self.actions = self._get_condensed_action_plan(self.original_actions)
        self.primitive_actions = self._get_primitive_actions(self.original_actions)

    def _get_condensed_action_plan(self, actions: list[Action]) -> list[Action]:
        condensed_actions = []

        for i, action in enumerate(actions):
            if i == 0:
                self._init_current_action(action=action)
                continue

            if action == self.cur_action:
                self.repeat_count += action.n
            else:
                condensed_action = self._get_condensed_action()
                condensed_actions.append(condensed_action)

                self._init_current_action(action=action)

        condensed_action = self._get_condensed_action()
        condensed_actions.append(condensed_action)

        return condensed_actions

    def _get_condensed_action(self) -> Action:
        condensed_action = replace(self.cur_action)
        condensed_action.n = self.repeat_count
        return condensed_action

    def _get_primitive_actions(self, actions: list[Action]) -> list[Action]:
        primitive_actions = []

        for action in actions:
            primitive = action.n * [self._get_primitive_action(action)]
            primitive_actions += primitive

        return primitive_actions

    def _get_primitive_action(self, action: Action) -> Action:
        primitive_action = replace(action)
        primitive_action.n = 1
        return primitive_action

    def get_power_used(self, board: Board) -> float:
        cur_c = self.unit.c
        total_power = self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

        for action in self:
            power_action = action.get_power_change(unit=self.unit, start_c=cur_c, board=board)
            power_used = max(power_action, 0)
            total_power += power_used
            cur_c = action.get_final_pos(start_c=cur_c)

        return total_power

    def _init_current_action(self, action: Action) -> None:
        self.cur_action: Action = action
        self.repeat_count: int = action.n

    def to_action_arrays(self) -> list[np.array]:
        return [action.to_array() for action in self.actions]

    def __lt__(self, other: "ActionPlan") -> bool:
        self.value < other.value

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def unit_can_carry_out_plan(self, game_state: GameState) -> bool:
        return self.is_valid_size and self.unit_has_enough_power(game_state=game_state)

    @property
    def is_valid_size(self) -> bool:
        return len(self) <= 20

    def unit_has_enough_power(self, game_state: GameState) -> bool:
        self._init_start()
        self._update_action_queue()

        for t, action in enumerate(self.primitive_actions):
            self._simul_action(action=action, board=game_state.board)

            if self.cur_power < 0:
                return False

            self._simul_charge(game_state=game_state, t=t)

        if not self._can_update_action_queue():
            return False

        return True

    def _init_start(self) -> None:
        self.cur_power = self.unit.power
        self.cur_c = self.unit.c

    def _update_action_queue(self) -> None:
        self.cur_power -= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    def _simul_action(self, action: Action, board: Board) -> None:
        power_change = action.get_power_change(unit=self.unit, start_c=self.cur_c, board=board)
        self.cur_power += power_change
        self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)
        self.cur_c = action.get_final_pos(start_c=self.cur_c)

    def _simul_charge(self, game_state: GameState, t: int) -> None:
        if game_state.is_day(t):
            self.cur_power += self.unit.unit_cfg.CHARGE

    def _can_update_action_queue(self) -> bool:
        return self.cur_power >= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    def unit_can_still_reach_factory_with_new_plan(self, game_state: GameState, graph: Graph) -> bool:
        self._init_start()
        self._update_action_queue()

        for t, action in enumerate(self.primitive_actions):
            self._simul_action(action=action, board=game_state.board)
            self._simul_charge(game_state=game_state, t=t)

        closest_factory_c = game_state.get_closest_factory_tile(c=self.cur_c)
        actions_back = get_actions_a_to_b(graph, start=self.cur_c, end=closest_factory_c)

        self._update_action_queue()

        for t, action in enumerate(actions_back, start=t + 1):
            self._simul_action(action=action, board=game_state.board)

            if self.cur_power < 0:
                return False

            self._simul_charge(game_state=game_state, t=t)

        if not self._can_update_action_queue():
            return False

        return True
