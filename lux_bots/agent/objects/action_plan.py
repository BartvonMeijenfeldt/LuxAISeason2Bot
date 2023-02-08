from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from dataclasses import dataclass, replace, field
from collections.abc import Iterator
from search import get_actions_a_to_b

if TYPE_CHECKING:
    from search import Graph
    from objects.unit import Unit
    from objects.action import Action, MoveAction
    from objects.board import Board
    from objects.coordinate import Coordinate
    from objects.game_state import GameState


@dataclass(kw_only=True)
class ActionPlan:
    original_actions: list[Action] = field(default_factory=list)
    unit: Unit

    def __post_init__(self):
        self._actions: Optional[list[Action]] = None
        self._primitive_actions: Optional[list[Action]] = None
        self._value: Optional[int] = None
        self._final_pos: Optional[Coordinate] = None

    def __iadd__(self, other: list[Action]) -> None:
        other = list(other)
        self.original_actions += other
        self.__post_init__()

    def __add__(self, other) -> ActionPlan:
        other = list(other)
        new_actions = self.original_actions + other
        return replace(self, original_actions=new_actions)

    def append(self, action: Action) -> None:
        self.original_actions.append(action)
        self.__post_init__()

    def extend(self, actions: Sequence[Action]) -> None:
        self.original_actions.extend(actions)
        self.__post_init__()

    @property
    def actions(self) -> list[Action]:
        if self._actions is None:
            self._actions = self._get_condensed_action_plan()

        return self._actions

    def _get_condensed_action_plan(self) -> list[Action]:
        return ActionPlanCondenser(original_actions=self.original_actions).condense()

    @property
    def primitive_actions(self) -> list[Action]:
        if self._primitive_actions is None:
            self._primitive_actions = self._get_primitive_actions()

        return self._primitive_actions

    def _get_primitive_actions(self) -> list[Action]:
        return ActionPlanPrimitiveMaker(original_actions=self.original_actions).make_primitive()

    @property
    def nr_primitive_actions(self) -> int:
        return len(self.primitive_actions)

    @property
    def final_c(self) -> Coordinate:
        if self._final_pos is None:
            self._final_pos = ActionPlanSimulator(self, unit=self.unit).get_final_c()

        return self._final_pos

    def get_power_used(self, board: Board) -> float:
        cur_c = self.unit.c
        total_power = self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

        for action in self:
            power_action = action.get_power_change(unit=self.unit, start_c=cur_c, board=board)
            power_used = max(power_action, 0)
            total_power += power_used
            cur_c = action.get_final_pos(start_c=cur_c)

        return total_power

    def unit_can_carry_out_plan(self, game_state: GameState) -> bool:
        return self.is_valid_size and self.unit_has_enough_power(game_state=game_state)

    @property
    def is_valid_size(self) -> bool:
        return len(self) <= 20

    def unit_has_enough_power(self, game_state: GameState) -> bool:
        simulator = ActionPlanSimulator(action_plan=self, unit=self.unit)

        try:
            simulator.simulate_action_plan(game_state=game_state)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def unit_can_add_reach_factory_to_plan(self, game_state: GameState, graph: Graph) -> bool:
        new_action_plan = self._get_action_plan_with_go_to_closest_factory(game_state=game_state, graph=graph)
        simulator = ActionPlanSimulator(action_plan=new_action_plan, unit=self.unit)

        try:
            simulator.simulate_action_plan(game_state=game_state)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def _get_action_plan_with_go_to_closest_factory(self, game_state: GameState, graph: Graph) -> ActionPlan:
        actions_to_factory_c = self.get_actions_go_to_closest_factory_c_after_plan(game_state=game_state, graph=graph)
        return self + actions_to_factory_c

    def get_actions_go_to_closest_factory_c_after_plan(self, game_state: GameState, graph: Graph) -> list[MoveAction]:
        closest_factory_c = game_state.get_closest_factory_c(c=self.final_c)
        return get_actions_a_to_b(graph=graph, start=self.final_c, end=closest_factory_c)

    def unit_can_reach_factory_after_action_plan(self, game_state: GameState, graph: Graph) -> bool:
        simulator = ActionPlanSimulator(action_plan=self, unit=self.unit)

        try:
            simulator.simulate_action_plan(game_state=game_state)
            simulator.simulate_action_plan_go_to_closest_factory(game_state=game_state, graph=graph)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def to_action_arrays(self) -> list[np.ndarray]:
        return [action.to_array() for action in self.actions]

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class ActionPlanCondenser:
    original_actions: list[Action]

    def condense(self) -> list[Action]:
        if not self.original_actions:
            return []

        self.condensed_actions = []

        for i, action in enumerate(self.original_actions):
            if i == 0:
                self._set_current_action(action=action)
                continue

            if action == self.cur_action:
                self.repeat_count += action.n
            else:
                self._add_condensed_action()
                self._set_current_action(action=action)

        self._add_condensed_action()

        return self.condensed_actions

    def _set_current_action(self, action: Action) -> None:
        self.cur_action: Action = action
        self.repeat_count: int = action.n

    def _add_condensed_action(self) -> None:
        condensed_action = self._get_condensed_action()
        self.condensed_actions.append(condensed_action)

    def _get_condensed_action(self) -> Action:
        return replace(self.cur_action, n=self.repeat_count)


@dataclass
class ActionPlanPrimitiveMaker:
    original_actions: list[Action]

    def make_primitive(self) -> list[Action]:
        primitive_actions = []

        for action in self.original_actions:
            primitive = action.n * [self._get_primitive_action(action)]
            primitive_actions += primitive

        return primitive_actions

    def _get_primitive_action(self, action: Action) -> Action:
        return replace(action, n=1)


@dataclass
class ActionPlanSimulator:
    action_plan: ActionPlan
    unit: Unit

    def simulate_action_plan(self, game_state: GameState) -> None:
        self._init_start()
        self._update_action_queue()
        self._simulate_actions(actions=self.action_plan.primitive_actions, game_state=game_state)

    def _init_start(self) -> None:
        self.cur_power = self.unit.power
        self.cur_c = self.unit.c
        self.t = 0

    def _update_action_queue(self) -> None:
        self.cur_power -= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    def _simulate_actions(self, actions: Sequence[Action], game_state: GameState) -> None:
        for action in actions:
            self._carry_out_action(action=action, board=game_state.board)

            if self.cur_power < 0:
                self._raise_negative_power_error()

            self._simul_charge(game_state=game_state)
            self.t += 1

    def _raise_negative_power_error(self) -> ValueError:
        raise ValueError("Power is below 0")

    def _carry_out_action(self, action: Action, board: Board) -> None:
        self._update_power(action=action, board=board)
        self._update_pos(action=action)

    def _update_power(self, action: Action, board: Board) -> None:
        power_change = action.get_power_change(unit=self.unit, start_c=self.cur_c, board=board)
        self.cur_power += power_change
        self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def _update_pos(self, action: Action) -> None:
        self.cur_c = action.get_final_pos(start_c=self.cur_c)

    def _simul_charge(self, game_state: GameState) -> None:
        if game_state.is_day(self.t):
            self.cur_power += self.unit.unit_cfg.CHARGE
            self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def get_final_c(self) -> Coordinate:
        self._init_start()

        for action in self.action_plan.primitive_actions:
            self._update_pos(action=action)

        return self.cur_c

    def can_update_action_queue(self) -> bool:
        return self.cur_power >= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    def simulate_action_plan_go_to_closest_factory(self, game_state: GameState, graph: Graph) -> None:
        actions_to_factory = self._get_actions_to_closest_factory_c(game_state=game_state, graph=graph)
        self._update_action_queue()
        self._simulate_actions(actions=actions_to_factory, game_state=game_state)

    def _get_actions_to_closest_factory_c(self, game_state: GameState, graph: Graph) -> list[MoveAction]:
        closest_factory_c = game_state.get_closest_factory_c(c=self.cur_c)
        return get_actions_a_to_b(graph, start=self.cur_c, end=closest_factory_c)
