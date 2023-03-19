from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence

from dataclasses import dataclass, replace, field
from search.search import Search
from objects.coordinate import TimeCoordinate, PowerTimeCoordinate
from objects.direction import Direction
from search.search import MoveToGraph
from objects.actions.unit_action import DigAction
from objects.actions.action_plan import ActionPlan

if TYPE_CHECKING:
    from objects.actors.unit import Unit
    from objects.actions.unit_action import UnitAction
    from objects.board import Board
    from objects.game_state import GameState
    from logic.constraints import Constraints


@dataclass
class UnitActionPlan(ActionPlan):
    actor: Unit
    original_actions: list[UnitAction] = field(default_factory=list)
    is_set: bool = field(default=False)

    def __post_init__(self):
        self._actions: Optional[list[UnitAction]] = None
        self._primitive_actions: Optional[list[UnitAction]] = None
        self._value: Optional[int] = None
        self._final_tc: Optional[TimeCoordinate] = None
        self._final_ptc: Optional[PowerTimeCoordinate] = None

    def __iadd__(self, other: list[UnitAction]) -> None:
        other = list(other)
        self.original_actions += other
        self.__post_init__()

    def __add__(self, other) -> UnitActionPlan:
        other = list(other)
        new_actions = self.original_actions + other
        return replace(self, original_actions=new_actions)

    def append(self, action: UnitAction) -> None:
        self.original_actions.append(action)
        self.__post_init__()

    def extend(self, actions: Sequence[UnitAction]) -> None:
        self.original_actions.extend(actions)
        self.__post_init__()

    @property
    def actions(self) -> list[UnitAction]:
        if self._actions is None:
            self._actions = self._get_condensed_action_plan()

        return self._actions

    @property
    def nr_time_steps(self) -> int:
        return sum(action.n for action in self.original_actions)

    def _get_condensed_action_plan(self) -> list[UnitAction]:
        return ActionPlanCondenser(original_actions=self.original_actions).condense()

    @property
    def primitive_actions(self) -> list[UnitAction]:
        if self._primitive_actions is None:
            self._primitive_actions = self._get_primitive_actions()

        return self._primitive_actions

    def _get_primitive_actions(self) -> list[UnitAction]:
        return ActionPlanPrimitiveMaker(original_actions=self.original_actions).make_primitive()

    @property
    def nr_primitive_actions(self) -> int:
        return len(self.primitive_actions)

    @property
    def nr_digs(self) -> int:
        return sum(dig_action.n for dig_action in self.actions if isinstance(dig_action, DigAction))

    @property
    def final_tc(self) -> TimeCoordinate:
        if self._final_tc is None:
            self._final_tc = ActionPlanSimulator(self, unit=self.actor).get_final_tc()

        return self._final_tc

    def get_final_ptc(self, game_state: GameState) -> PowerTimeCoordinate:
        if self._final_ptc is None:
            self._final_ptc = ActionPlanSimulator(self, unit=self.actor).get_final_ptc(game_state)

        return self._final_ptc

    @property
    def time_coordinates(self) -> set[TimeCoordinate]:
        if len(self.actions) == 0:
            return {self.actor.tc + Direction.CENTER}

        simulator = ActionPlanSimulator(self, unit=self.actor)
        return simulator.get_time_coordinates()

    def get_power_used(self, board: Board) -> float:
        cur_c = self.actor.tc
        total_power = self.actor.unit_cfg.ACTION_QUEUE_POWER_COST

        for action in self.actions:
            power_action = action.get_power_change(unit_cfg=self.actor.unit_cfg, start_c=cur_c, board=board)
            power_used = max(power_action, 0)
            total_power += power_used
            cur_c = action.get_final_c(start_c=cur_c)

        return total_power

    def actor_can_carry_out_plan(self, game_state: GameState) -> bool:
        return self.is_valid_size and self.unit_has_enough_power(game_state=game_state)

    @property
    def is_valid_size(self) -> bool:
        return len(self) <= 20

    def get_nr_valid_primitive_actions(self, game_state: GameState):
        if len(self.actions) == 0:
            return 0

        simulator = self._init_simulator()
        return simulator.get_nr_valid_primitive_actions(game_state)

    def _init_simulator(self) -> ActionPlanSimulator:
        return ActionPlanSimulator(action_plan=self, unit=self.actor)

    def unit_has_enough_power(self, game_state: GameState) -> bool:
        if len(self.actions) == 0:
            return True

        simulator = self._init_simulator()

        try:
            simulator.simulate_action_plan(game_state=game_state)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def unit_can_add_reach_factory_to_plan(self, game_state: GameState, constraints: Constraints) -> bool:
        new_action_plan = self._get_action_plan_with_go_to_closest_factory(
            game_state=game_state, constraints=constraints
        )
        simulator = ActionPlanSimulator(action_plan=new_action_plan, unit=self.actor)

        try:
            simulator.simulate_action_plan(game_state=game_state)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def _get_action_plan_with_go_to_closest_factory(
        self, game_state: GameState, constraints: Constraints
    ) -> UnitActionPlan:
        actions_to_factory_c = self.get_actions_go_to_closest_factory_c_after_plan(
            game_state=game_state, constraints=constraints
        )
        return self + actions_to_factory_c

    def get_actions_go_to_closest_factory_c_after_plan(
        self, game_state: GameState, constraints: Constraints
    ) -> list[UnitAction]:
        closest_factory_c = game_state.get_closest_factory_c(self.final_tc)
        graph = MoveToGraph(
            board=game_state.board,
            time_to_power_cost=self.actor.time_to_power_cost,
            unit_cfg=self.actor.unit_cfg,
            goal=closest_factory_c,
            constraints=constraints,
        )
        search = Search(graph=graph)
        actions = search.get_actions_to_complete_goal(start=self.final_tc)

        return actions

    def unit_can_reach_factory_after_action_plan(self, game_state: GameState, constraints: Constraints) -> bool:
        simulator = ActionPlanSimulator(action_plan=self, unit=self.actor)

        try:
            simulator.simulate_action_plan(game_state=game_state)
            simulator.simulate_action_plan_go_to_closest_factory(game_state=game_state, constraints=constraints)
        except ValueError:
            return False

        return simulator.can_update_action_queue()

    def to_lux_output(self):
        return [action.to_lux_output() for action in self.actions]


@dataclass
class ActionPlanCondenser:
    original_actions: list[UnitAction]

    def condense(self) -> list[UnitAction]:
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

    def _set_current_action(self, action: UnitAction) -> None:
        self.cur_action: UnitAction = action
        self.repeat_count: int = action.n

    def _add_condensed_action(self) -> None:
        condensed_action = self._get_condensed_action()
        self.condensed_actions.append(condensed_action)

    def _get_condensed_action(self) -> UnitAction:
        return replace(self.cur_action, n=self.repeat_count)


@dataclass
class ActionPlanPrimitiveMaker:
    original_actions: list[UnitAction]

    def make_primitive(self) -> list[UnitAction]:
        primitive_actions = []

        for action in self.original_actions:
            if action.n == 1:
                primitive_actions.append(action)
            else:
                primitive = action.n * [self._get_primitive_action(action)]
                primitive_actions.extend(primitive)

        return primitive_actions

    def _get_primitive_action(self, action: UnitAction) -> UnitAction:
        return replace(action, n=1)


@dataclass
class ActionPlanSimulator:
    action_plan: UnitActionPlan
    unit: Unit

    def simulate_action_plan(self, game_state: GameState) -> None:
        self._init_start()
        self._optional_update_action_queue()
        self._simulate_primitive_actions(actions=self.action_plan.primitive_actions, game_state=game_state)

    def get_time_coordinates(self) -> set[TimeCoordinate]:
        self._init_start()
        self._simulate_actions_for_tc(actions=self.action_plan.primitive_actions)
        return self.time_coordinates

    def _init_start(self) -> None:
        self.cur_power = self.unit.power
        self.t = self.unit.tc.t
        self.cur_tc = TimeCoordinate(x=self.unit.tc.x, y=self.unit.tc.y, t=self.t)
        self.time_coordinates = set()

    def _optional_update_action_queue(self) -> None:
        if not self.action_plan.is_set and self.action_plan.original_actions:
            self._update_action_queue()

    def _update_action_queue(self) -> None:
        self.cur_power -= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST
        self._check_power()

    def _check_power(self):
        if self.cur_power < 0:
            raise ValueError("Power is below 0")

    def get_nr_valid_primitive_actions(self, game_state: GameState) -> int:
        self._init_start()

        try:
            self._optional_update_action_queue()
        except ValueError:
            return 0

        for i, action in enumerate(self.action_plan.primitive_actions):
            try:
                self._simulate_primitive_action(action, game_state)
            except ValueError:
                return i

        return len(self.action_plan.primitive_actions)

    def _simulate_primitive_actions(self, actions: Sequence[UnitAction], game_state: GameState) -> None:
        for action in actions:
            self._simulate_primitive_action(action, game_state)

    def _simulate_primitive_action(self, action: UnitAction, game_state: GameState) -> None:
        self._update_power_due_to_action(action=action, board=game_state.board)
        self._check_power()
        self._simul_charge(game_state=game_state)
        self._increase_time_count()
        self._update_tc(action=action)

    def _simulate_actions_for_tc(self, actions: Sequence[UnitAction]) -> None:
        for action in actions:
            self._increase_time_count()
            self._update_tc(action=action)

    def _update_power_due_to_action(self, action: UnitAction, board: Board) -> None:
        power_change = action.get_power_change(unit_cfg=self.unit.unit_cfg, start_c=self.cur_tc, board=board)
        self.cur_power += power_change
        self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def _update_tc(self, action: UnitAction) -> None:
        self.cur_tc = action.get_final_c(start_c=self.cur_tc)
        self.time_coordinates.add(self.cur_tc)

    def _simul_charge(self, game_state: GameState) -> None:
        if game_state.is_day(self.t):
            self.cur_power += self.unit.unit_cfg.CHARGE
            self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def _increase_time_count(self) -> None:
        self.t += 1

    def get_final_tc(self) -> TimeCoordinate:
        self._init_start()
        self._simulate_actions_for_tc(actions=self.action_plan.primitive_actions)
        return self.cur_tc

    def get_final_ptc(self, game_state: GameState) -> PowerTimeCoordinate:
        self.simulate_action_plan(game_state)
        return PowerTimeCoordinate(*self.cur_tc, p=self.cur_power)

    def can_update_action_queue(self) -> bool:
        return self.cur_power >= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST

    def simulate_action_plan_go_to_closest_factory(self, game_state: GameState, constraints: Constraints) -> None:
        actions_to_factory = self._get_actions_to_closest_factory_c(game_state=game_state, constraints=constraints)
        self._update_action_queue()
        self._simulate_primitive_actions(actions=actions_to_factory, game_state=game_state)

    def _get_actions_to_closest_factory_c(self, game_state: GameState, constraints: Constraints) -> list[UnitAction]:
        closest_factory_c = game_state.get_closest_factory_c(self.action_plan.final_tc)
        graph = MoveToGraph(
            board=game_state.board,
            time_to_power_cost=self.unit.time_to_power_cost,
            unit_cfg=self.unit.unit_cfg,
            goal=closest_factory_c,
            constraints=constraints,
        )
        search = Search(graph=graph)
        actions = search.get_actions_to_complete_goal(start=self.cur_tc)
        return actions
