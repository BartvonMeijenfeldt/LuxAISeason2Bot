from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from lux.config import EnvConfig
from objects.actions.action_plan import ActionPlan, PowerRequest
from objects.actions.unit_action import DigAction, MoveAction
from objects.coordinate import PowerTimeCoordinate, TimeCoordinate
from objects.direction import Direction
from utils.utils import is_day

if TYPE_CHECKING:
    from objects.actions.unit_action import UnitAction
    from objects.actors.unit import Unit
    from objects.board import Board
    from objects.game_state import GameState


@dataclass
class UnitActionPlan(ActionPlan):
    actor: Unit
    original_actions: list[UnitAction] = field(default_factory=list)
    is_set: bool = field(default=False)

    def __post_init__(self):
        self._actions: Optional[list[UnitAction]] = None
        self._primitive_actions: Optional[list[UnitAction]] = None
        self._final_tc: Optional[TimeCoordinate] = None
        self._final_ptc: Optional[PowerTimeCoordinate] = None

    def __iadd__(self, other: list[UnitAction]) -> None:
        other = list(other)
        self.original_actions += other
        self.__post_init__()

    def __bool__(self) -> bool:
        if self.original_actions:
            return True

        return False

    def __add__(self, other) -> UnitActionPlan:
        other = list(other)
        new_actions = self.original_actions + other
        new_action_plan = replace(self, original_actions=new_actions)
        new_action_plan.__post_init__()
        return new_action_plan

    def step(self) -> None:
        """Update the action plan to remove the action of the last time step"""
        self.original_actions = self.primitive_actions[1:]
        self.__post_init__()

    def append(self, action: UnitAction) -> None:
        self.original_actions.append(action)
        self.__post_init__()

    def extend(self, actions: Sequence[UnitAction]) -> None:
        self.original_actions.extend(actions)
        self.__post_init__()

    def set_actions(self, actions: List[UnitAction]) -> None:
        """Set the actions in the action plan with the given actions

        Args:
            actions: Actions to set.
        """
        self.original_actions = actions
        self.__post_init__()

    @property
    def actions(self) -> list[UnitAction]:
        """Condensed version of actions.

        Repeated succesive actions are condensed into a single action."""
        if self._actions is None:
            self._actions = self._get_condensed_action_plan()

        return self._actions

    @property
    def nr_time_steps(self) -> int:
        if self._primitive_actions:
            return len(self._primitive_actions)

        return sum(action.n for action in self.original_actions)

    def _get_condensed_action_plan(self) -> list[UnitAction]:
        return ActionPlanCondenser(original_actions=self.original_actions).condense()

    @property
    def primitive_actions(self) -> list[UnitAction]:
        """Primitive version of actions.

        Any repeated actions are split into single repeated succesive actions."""
        if self._primitive_actions is None:
            self._primitive_actions = get_primitive_actions_from_list(self.original_actions)

        return self._primitive_actions

    @property
    def is_first_action_stationary(self) -> bool:
        if len(self.actions) == 0:
            return False

        return self.actions[0].is_stationary

    def is_first_action_move_center(self) -> bool:
        if len(self.actions) == 0:
            return False

        return self.primitive_actions[0] == MoveAction(Direction.CENTER)

    def get_power_requests(self, game_state: GameState) -> List[PowerRequest]:
        return [
            self._create_power_request(action, tc, game_state)
            for action, tc in zip(self.primitive_actions, [self.actor.tc] + self.get_time_coordinates(game_state))
            if action.requested_power
        ]

    @staticmethod
    def _create_power_request(action: UnitAction, tc: TimeCoordinate, game_state: GameState) -> PowerRequest:
        factory = game_state.get_closest_player_factory(tc)
        return PowerRequest(factory=factory, t=tc.t, p=action.requested_power)

    @property
    def nr_digs(self) -> int:
        return sum(dig_action.n for dig_action in self.actions if isinstance(dig_action, DigAction))

    @property
    def final_tc(self) -> TimeCoordinate:
        """The final time coordinate of the unit, after completing the actions in the plan."""
        if self._final_tc is None:
            self._final_tc = ActionPlanSimulator(self, unit=self.actor).get_final_tc()

        return self._final_tc

    def get_final_ptc(self, game_state: GameState) -> PowerTimeCoordinate:
        """Get the final power time coordinate of the unit, after completing the actions in the plan.

        Args:
            game_state: The current game state.

        Returns:
            The final power time coordinate, after complting the actions in the plan.
        """
        if self._final_ptc is None:
            self._final_ptc = ActionPlanSimulator(self, unit=self.actor).get_final_ptc(game_state)

        return self._final_ptc

    def get_final_p(self, game_state: GameState) -> int:
        return self.get_final_ptc(game_state).p

    @property
    def next_tc(self) -> TimeCoordinate:
        if self.is_empty():
            return self.actor.tc + Direction.CENTER

        first_action = self.primitive_actions[0]
        return self.actor.tc.add_action(first_action)

    def get_time_coordinates(self, game_state: GameState) -> List[TimeCoordinate]:
        """Get the time coordinates of the unit after each action required to complete its plan.

        Args:
            game_state: Current game state.
        Returns:
            Time coordinates.
        """

        if self.is_empty():
            return [self.actor.tc + Direction.CENTER]

        simulator = ActionPlanSimulator(self, unit=self.actor)
        time_coordinates = simulator.get_time_coordinates(game_state)
        return time_coordinates

    def get_power_time_coordinates(self, game_state: GameState) -> List[PowerTimeCoordinate]:
        """Get the power time coordinates of the unit after each action required to complete its plan.

        Args:
            game_state: Current game state.
        Returns:
            Power time coordinates.
        """
        if self.is_empty():
            ptc = PowerTimeCoordinate(*self.actor.tc.xyt, self.actor.power, self.actor.unit_cfg, game_state)
            return [ptc + Direction.CENTER]

        simulator = ActionPlanSimulator(self, unit=self.actor)
        power_time_coordinates = simulator.get_power_time_coordinates(game_state)
        return power_time_coordinates

    def get_power_used(self, board: Board) -> float:
        """Get the total power used to complete the action plan.

        Args:
            board: Current board.

        Returns:
            Total power to use.
        """
        cur_c = self.actor.tc
        total_power = self.actor.unit_cfg.ACTION_QUEUE_POWER_COST if not self.is_set else 0

        for action in self.actions:
            power_action = action.get_power_change(unit_cfg=self.actor.unit_cfg, start_c=cur_c, board=board)
            power_used = max(power_action, 0)
            total_power += power_used
            cur_c = action.get_final_c(start_c=cur_c)

        return total_power

    def has_enough_power_to_add_actions(
        self, actions: List[UnitAction], game_state: GameState, min_power_end: int = 0
    ) -> bool:
        """Calculates whether the unit has enough power to carry out the current plan and the actions to add.

        Args:
            actions: Actions to add.
            game_state: Current game state.
            min_power_end: Minimum power required at end of action plan.. Defaults to 0.

        Returns:
            Boolean indicating whether the unit has enough power.
        """
        new_action_plan = self + actions
        return new_action_plan.unit_has_enough_power(game_state, min_power_end=min_power_end)

    def is_empty(self) -> bool:
        """Whether the current plan is an empty plan, with no actions."""
        return not self.original_actions

    def get_actions_valid_to_add(self, actions: list[UnitAction], game_state: GameState) -> list[UnitAction]:
        """Get the valid primitive actions out of a list of actions that are valid to add, where the unit has enough
        power for to complete.

        Args:
            actions: Potential actions to add.
            game_state: Current game state.

        Returns:
            List of primitive actions that are valid to add.
        """
        new_action_plan = self + actions
        nr_valid_primitive_actions = new_action_plan._get_nr_valid_primitive_actions(game_state)
        nr_actions_to_add = nr_valid_primitive_actions - self.nr_time_steps
        return actions[:nr_actions_to_add]

    def _get_nr_valid_primitive_actions(self, game_state: GameState):
        if self.is_empty():
            return 0

        simulator = self._init_simulator()
        return simulator.get_nr_valid_primitive_actions(game_state)

    def _init_simulator(self) -> ActionPlanSimulator:
        return ActionPlanSimulator(action_plan=self, unit=self.actor)

    def unit_has_enough_power(self, game_state: GameState, min_power_end: int = 0) -> bool:
        """Does the unit have enough power to carry out its current plan.

        Args:
            game_state: Current game state.
            min_power_end: Min power required at end of action plan. Defaults to 0.

        Returns:
            Whether the unit has enough power to carry out its current plan.
        """
        if self.is_empty():
            return True

        simulator = self._init_simulator()

        try:
            simulator.simulate_action_plan(game_state=game_state)
        except ValueError:
            return False

        return simulator.cur_power >= min_power_end

    def to_lux_output(self):
        return [action.to_lux_output() for action in self.actions[: EnvConfig.UNIT_ACTION_QUEUE_SIZE]]


@dataclass
class ActionPlanCondenser:
    """Class to condense the actions into a condensed form. Where single succesive repeated actions get grouped into
    a single action with repeats.

    Args:
        original_actions: the current list of actions
    """

    # TODO, consider converting to functions, seems a bit of an artificial class.
    original_actions: list[UnitAction]

    def condense(self) -> list[UnitAction]:
        """Condense the actions."""
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


def get_primitive_actions_from_list(actions: Iterable[UnitAction]) -> list[UnitAction]:
    return [primitive_action for action in actions for primitive_action in get_primitive_actions_from_action(action)]


def get_primitive_actions_from_action(action: UnitAction) -> list[UnitAction]:
    if action.n == 1:
        return [action]

    primitive_action = replace(action, n=1)
    return action.n * [primitive_action]


@dataclass
class ActionPlanSimulator:
    """Class to simulate actions forward.

    Args:
        action_plan: Unit's action plan.
        unit: The unit to carry out the action plan.
    """

    # TODO, unit is already given for UnitActinPlan, consider removing unit as argument here.
    action_plan: UnitActionPlan
    unit: Unit

    def simulate_action_plan(self, game_state: GameState) -> None:
        """Simulate action plan.

        Args:
            game_state: Current game state.

        Raises:
            ValueError: If unit has too little power to carry out the action plan.
        """
        self.confirm_power_levels = True
        self._simulate_action_plan(game_state)

    def _simulate_action_plan(self, game_state: GameState) -> None:
        self._init_start()
        self._simulate_actions(actions=self.action_plan.actions, game_state=game_state)

    def get_time_coordinates(self, game_state: GameState) -> List[TimeCoordinate]:
        """Get the time coordinates for the unit to carry out the action plan. Adds an extra time coordinate at the
        current coordinate one time step later, if the unit has too little power to carry out the first action

        Args:
            game_state: Current game state.

        Returns:
            List of time coordinates to carry out the action plan
        """
        self._init_start()
        self._add_center_action_if_too_little_power(game_state)
        self._simulate_actions_for_tc(actions=self.action_plan.primitive_actions)
        return self.time_coordinates

    def _add_center_action_if_too_little_power(self, game_state: GameState) -> None:
        if self._unit_has_not_enough_power_first_action(game_state):
            self._add_center_move_action()

    def _unit_has_not_enough_power_first_action(self, game_state: GameState):
        if len(self.action_plan.actions) == 0:
            return False

        first_primitive_action = self.action_plan.primitive_actions[0]
        if self._requires_queue_update_due_to_new_primitive_action(first_primitive_action):
            self._update_action_queue()

        if self.cur_power < 0:
            return True

        power_cost_action = first_primitive_action.get_power_change(
            self.action_plan.actor.unit_cfg, self.unit.tc, game_state.board
        )
        return self.cur_power < power_cost_action

    def _add_center_move_action(self) -> None:
        new_actions = [MoveAction(Direction.CENTER)] + self.action_plan.original_actions
        self.action_plan = UnitActionPlan(
            self.action_plan.actor, original_actions=new_actions, is_set=self.action_plan.is_set
        )

    def get_power_time_coordinates(self, game_state: GameState) -> List[PowerTimeCoordinate]:
        self.confirm_power_levels = False
        self._simulate_action_plan(game_state=game_state)
        return self.power_time_coordinates

    def _init_start(self) -> None:
        self.cur_power = self.unit.power
        self.cur_tc = self.unit.tc

        self.action_nr = 0
        self.primitive_action_nr = 0
        self.last_updated_action_nr = None

        self.time_coordinates: List[TimeCoordinate] = []
        self.power_time_coordinates: List[PowerTimeCoordinate] = []

    def _update_action_queue(self) -> None:
        self.cur_power -= self.unit.unit_cfg.ACTION_QUEUE_POWER_COST
        self.last_updated_action_nr = self.action_nr

    def _confirm_power_level_is_valid(self):
        if self.cur_power < 0:
            raise ValueError("Power is below 0")

    def get_nr_valid_primitive_actions(self, game_state: GameState) -> int:
        """Get nr valid primitive actions, the number of actions for which the unit has enough power to carry out.

        Args:
            game_state: Current game state.

        Returns:
            Nr of primitive actions for which the unit has enough power to carry out.
        """
        if len(self.action_plan) == 0:
            return 0

        try:
            self.simulate_action_plan(game_state)
        except ValueError:
            pass

        return self.primitive_action_nr

    def _simulate_actions(self, actions: Sequence[UnitAction], game_state: GameState) -> None:
        for action in actions:
            if self._requires_update_due_to_action():
                self._update_action_queue()

            for primitive_action in get_primitive_actions_from_action(action):
                self._simulate_primitive_action(primitive_action, game_state)

            self._increment_action_nr()

    def _requires_update_due_to_action(self) -> bool:
        return (
            self.last_updated_action_nr is not None
            and self.last_updated_action_nr + EnvConfig.UNIT_ACTION_QUEUE_SIZE == self.action_nr
        ) or (self.last_updated_action_nr is None and not self.unit.primitive_actions_in_queue)

    def _simulate_primitive_action(self, action: UnitAction, game_state: GameState) -> None:
        if self._requires_queue_update_due_to_new_primitive_action(action):
            self._update_action_queue()
            if self.confirm_power_levels:
                self._confirm_power_level_is_valid()

        self._update_power_due_to_primitive_action(action=action, board=game_state.board)
        if self.confirm_power_levels:
            self._confirm_power_level_is_valid()
        self._simul_charge()
        self._update_tc(action=action)
        self._add_ptc(game_state)
        self._increment_primitive_action_nr()

    def _requires_queue_update_due_to_new_primitive_action(self, action: UnitAction) -> bool:
        primitive_actions_in_queue = self.unit.primitive_actions_in_queue

        return self.last_updated_action_nr is None and (
            self.primitive_action_nr == len(primitive_actions_in_queue)
            or action != primitive_actions_in_queue[self.primitive_action_nr]
        )

    def _simulate_actions_for_tc(self, actions: Sequence[UnitAction]) -> None:
        for action in actions:
            self._update_tc(action=action)
            self._add_tc()

    def _update_power_due_to_primitive_action(self, action: UnitAction, board: Board) -> None:
        power_change = action.get_power_change(unit_cfg=self.unit.unit_cfg, start_c=self.cur_tc, board=board)
        self.cur_power += power_change
        self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def _update_tc(self, action: UnitAction) -> None:
        self.cur_tc = action.get_final_c(start_c=self.cur_tc)

    def _add_tc(self) -> None:
        self.time_coordinates.append(self.cur_tc)

    def _add_ptc(self, game_state: GameState) -> None:
        ptc = PowerTimeCoordinate(*self.cur_tc.xyt, self.cur_power, self.unit.unit_cfg, game_state)
        self.power_time_coordinates.append(ptc)

    def _simul_charge(self) -> None:
        if is_day(self.cur_tc.t):
            self.cur_power += self.unit.unit_cfg.CHARGE
            self.cur_power = min(self.cur_power, self.unit.unit_cfg.BATTERY_CAPACITY)

    def get_final_tc(self) -> TimeCoordinate:
        self._init_start()
        self._simulate_actions_for_tc(actions=self.action_plan.primitive_actions)
        return self.cur_tc

    def get_final_ptc(self, game_state: GameState) -> PowerTimeCoordinate:
        """Get the final power time coordinate of the unit after carrying out the action plan.

        Args:
            game_state: Current game state.

        Raises:
            ValueError: If unit has too little power to carry out the action plan.

        Returns:
            Final power time coordinate of the unit after carrying out the action plan.
        """
        self.simulate_action_plan(game_state)
        return PowerTimeCoordinate(
            self.cur_tc.x, self.cur_tc.y, self.cur_tc.t, self.cur_power, self.unit.unit_cfg, game_state
        )

    def _increment_action_nr(self) -> None:
        self.action_nr += 1

    def _increment_primitive_action_nr(self) -> None:
        self.primitive_action_nr += 1
