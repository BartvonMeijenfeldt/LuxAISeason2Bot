from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple

from config import CONFIG
from logic.constraints import Constraints
from lux.config import HEAVY_CONFIG
from objects.actions.unit_action import (
    DigAction,
    MoveAction,
    PickupAction,
    TransferAction,
    UnitAction,
)
from objects.coordinate import (
    Coordinate,
    DigCoordinate,
    DigTimeCoordinate,
    ResourcePowerTimeCoordinate,
    TimeCoordinate,
)
from objects.direction import Direction
from objects.resource import Resource

if TYPE_CHECKING:
    from logic.goal_resolution.power_tracker import PowerTracker
    from lux.config import UnitConfig
    from objects.actors.factory import Factory
    from objects.board import Board


@dataclass
class Graph(metaclass=ABCMeta):
    board: Board
    time_to_power_cost: float
    unit_cfg: UnitConfig
    unit_type: str
    constraints: Constraints

    def get_valid_action_nodes(self, tc: TimeCoordinate) -> Generator[Tuple[UnitAction, TimeCoordinate], None, None]:
        """For the current TimeCoordinate, generates all Action and corresponding next TimeCoordinate pairs that are
        valid.

        Args:
            c: TimeCoordinate

        Yields:
            Actions and corresponding next TimeCoordinate tuples
        """
        for action in self._get_potential_actions(tc=tc):
            to_c = tc.add_action(action)
            if self._is_valid_action_node(action, to_c):
                yield action, to_c

    @abstractmethod
    def _get_potential_actions(self, tc: TimeCoordinate) -> Generator[UnitAction, None, None]:
        """Generates actions that lead to potentially valid action next TimeCoordinate pairs by considering the current
        TimeCoordinate."""
        ...

    def _is_valid_action_node(self, action: UnitAction, to_tc: TimeCoordinate) -> bool:
        """Confirms whether action node pairs are valid based on the action and the next TimeCoordinate."""
        return (
            not self.constraints.tc_violates_constraint(to_tc)
            and self.board.is_valid_c_for_player(c=to_tc)
            and not self.constraints.get_danger_cost(to_tc, action.is_stationary)
        )

    def get_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        """Get the cost of the action based on the action itself and the next TimeCoordinate

        Args:
            action: Action to perform
            to_c: Next TimeCoordinate
        Returns:
            Cost of performing action
        """
        action_power_cost = self._get_power_cost(action=action, to_c=to_c)
        on_resource_next_to_base_cost = self._get_penalty_on_resource_next_to_base(action=action, to_c=to_c)
        danger_cost = self._get_danger_cost(action=action, to_c=to_c)
        return action_power_cost + self.time_to_power_cost + on_resource_next_to_base_cost + danger_cost

    def _get_power_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        power_change = action.get_power_change_by_end_c(unit_cfg=self.unit_cfg, end_c=to_c, board=self.board)
        power_cost = max(0, -power_change)
        return power_cost

    def _get_penalty_on_resource_next_to_base(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        """Adds a penalty for coordinates next to the base, to discourage pathing on resources."""
        if self.board.is_resource_c(to_c) and self.board.get_min_distance_to_any_player_factory(to_c) == 1:
            return 1
        else:
            return 0

    def _get_danger_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        # TODO, figure out if this is duplication, there is also a danger cost in the constraints
        if self.unit_type == "HEAVY":
            return 0

        distance_to_opp_heavy = self.board.get_min_dis_to_opp_heavy(c=to_c)
        if distance_to_opp_heavy == 0:
            return 50
        elif distance_to_opp_heavy == 1:
            return 5
        else:
            return 0

    @abstractmethod
    def get_heuristic(self, tc: TimeCoordinate) -> float:
        """Heuristic of cost to omplete the goal. Must be a minimum of the total cost for the A* algorithm to guarantee
        an optimal solution.

        Args:
            tc: Current TimeCoordinate

        Returns:
            Minimum cost for current TimeCoordinate to reach its goal
        """
        ...

    @abstractmethod
    def completes_goal(self, tc: TimeCoordinate) -> bool:
        """Whether the current TimeCoordinate completes the goal.

        Args:
            tc: TimeCoordinate

        Returns:
            Boolean, does current TimeCoordinate complete the goal?
        """
        ...


@dataclass
class GoalGraph(Graph):
    goal: Coordinate

    def __post_init__(self) -> None:
        self.last_action_cost = self.time_to_power_cost + MoveAction.get_move_onto_cost(
            self.unit_cfg, self.goal, self.board
        )

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        return self._get_distance_heuristic(tc=tc)

    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        min_nr_steps = tc.distance_to(self.goal)
        if min_nr_steps == 0:
            return 0

        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = (min_nr_steps - 1) * min_cost_per_step + self.last_action_cost
        return min_distance_cost

    def completes_goal(self, tc: TimeCoordinate) -> bool:
        return self.goal == tc


@dataclass
class FleeGraph(Graph):
    _potential_actions = [MoveAction(direction) for direction in Direction]

    def _is_valid_action_node(self, action: UnitAction, to_c: TimeCoordinate) -> bool:
        return not self.constraints.tc_violates_constraint(to_c) and self.board.is_valid_c_for_player(c=to_c)

    def _get_potential_actions(self, tc: TimeCoordinate) -> Generator[UnitAction, None, None]:
        for action in self._potential_actions:
            yield action

    def _get_danger_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        base_danger_cost = super()._get_danger_cost(action, to_c)
        constraints_danger_cost = self.constraints.get_danger_cost(to_c, action.is_stationary)
        return base_danger_cost + constraints_danger_cost


class FleeTowardsAnyFactoryGraph(FleeGraph):
    def completes_goal(self, tc: TimeCoordinate) -> bool:
        return self.board.get_min_distance_to_any_player_factory(tc) == 0

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        min_nr_steps = self.board.get_min_distance_to_any_player_factory(tc)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps * min_cost_per_step
        return min_distance_cost


@dataclass
class FleeDistanceGraph(FleeGraph):
    start_tc: TimeCoordinate
    distance: int
    _potential_actions = [MoveAction(direction) for direction in Direction if direction != Direction.CENTER]

    def completes_goal(self, tc: TimeCoordinate) -> bool:
        return tc.distance_to(self.start_tc) >= self.distance

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        min_nr_steps = self.distance - tc.distance_to(self.start_tc)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps * min_cost_per_step
        return min_distance_cost


@dataclass
class TilesToClearGraph(GoalGraph):
    time_to_power_cost: int = field(init=False, default=CONFIG.OPTIMAL_PATH_TIME_TO_POWER_COST)
    unit_cfg: UnitConfig = field(init=False, default=HEAVY_CONFIG)
    unit_type: str = field(init=False, default="HEAVY")
    constraints: Constraints = field(init=False, default_factory=Constraints)
    goal: Coordinate
    _potential_actions = [MoveAction(direction) for direction in Direction if direction != direction.CENTER]

    def _is_valid_action_node(self, action: UnitAction, to_c: TimeCoordinate) -> bool:
        return self.board.is_valid_c_for_player(c=to_c)

    def get_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        action_power_cost = self._get_power_cost(action=action, to_c=to_c)
        resource_cost = 100 if self.board.is_resource_c(to_c) else 0
        total_cost = action_power_cost + self.time_to_power_cost + resource_cost
        return total_cost

    # TODO, consider is_valid_action node to exclude resource tiles Or at least a big extra cost
    def _get_potential_actions(self, tc: TimeCoordinate) -> List[MoveAction]:
        return self._potential_actions

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        return self._get_distance_heuristic(tc=tc)


@dataclass
class MoveToGraph(GoalGraph):
    _potential_actions = [MoveAction(direction) for direction in Direction]

    def __post_init__(self):
        super().__post_init__()
        if not self.constraints:
            self._potential_actions = [MoveAction(dir) for dir in Direction if dir != Direction.CENTER]

    def _get_potential_actions(self, tc: TimeCoordinate) -> List[MoveAction]:
        return self._potential_actions

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        return self._get_distance_heuristic(tc=tc)


@dataclass
class MoveNearCoordinateGraph(GoalGraph):
    distance: int
    _potential_actions = [MoveAction(direction) for direction in Direction]

    def __post_init__(self):
        if not self.constraints:
            self._potential_actions = [MoveAction(dir) for dir in Direction if dir != Direction.CENTER]

    def _get_potential_actions(self, tc: TimeCoordinate) -> Generator[UnitAction, None, None]:
        for action in self._potential_actions:
            yield action

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        return self._get_distance_heuristic(tc=tc)

    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        min_nr_steps = self._get_distance_near_goal(tc)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps * min_cost_per_step
        return min_distance_cost

    def _get_distance_near_goal(self, to_tc: TimeCoordinate) -> int:
        distance_to_goal = self.goal.distance_to(to_tc)
        difference_required_distance = abs(distance_to_goal - self.distance)
        return difference_required_distance

    def completes_goal(self, tc: Coordinate) -> bool:
        return self.goal.distance_to(tc) == self.distance


@dataclass
class MoveRecklessNearCoordinateGraph(MoveNearCoordinateGraph):
    def _is_valid_action_node(self, action: UnitAction, to_c: TimeCoordinate) -> bool:
        return not self.constraints.tc_violates_constraint(to_c) and self.board.is_valid_c_for_player(c=to_c)


class EvadeConstraintsGraph(Graph):
    _potential_actions = [MoveAction(direction) for direction in Direction]
    _move_center_action = MoveAction(Direction.CENTER)

    def _is_valid_action_node(self, action: UnitAction, to_c: TimeCoordinate) -> bool:
        return not self.constraints.tc_violates_constraint(to_c) and self.board.is_valid_c_for_player(c=to_c)

    def _get_danger_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        base_danger_cost = super()._get_danger_cost(action, to_c)
        constraints_danger_cost = self.constraints.get_danger_cost(to_c, action.is_stationary)
        return base_danger_cost + constraints_danger_cost

    def _get_potential_actions(self, tc: TimeCoordinate) -> List[MoveAction]:
        return self._potential_actions

    def get_heuristic(self, tc: TimeCoordinate) -> float:
        return -tc.t

    def completes_goal(self, tc: TimeCoordinate) -> bool:
        to_c = tc.add_action(self._move_center_action)
        return not self.constraints.tc_violates_constraint(to_c)


@dataclass
class PickupPowerGraph(Graph):
    power_tracker: PowerTracker
    later_pickup: bool
    next_goal_c: Optional[Coordinate] = field(default=None)
    _potential_move_actions = [MoveAction(direction) for direction in Direction]

    def _get_potential_actions(self, tc: ResourcePowerTimeCoordinate) -> Generator[UnitAction, None, None]:
        if self.board.is_player_factory_tile(c=tc):
            factory = self.board.get_closest_player_factory(c=tc)
            power_available_in_factory = self.power_tracker.get_power_available(factory, tc.t)
            if power_available_in_factory:
                battery_space_left = self.unit_cfg.BATTERY_CAPACITY - tc.p - self.unit_cfg.CHARGE
                power_pickup_amount = min(battery_space_left, power_available_in_factory, 3000)
                power_pickup_amount = max(0, power_pickup_amount)

                potential_pickup_action = PickupAction(amount=power_pickup_amount, resource=Resource.POWER)
                yield potential_pickup_action

        for action in self._potential_move_actions:
            yield action

    def get_cost(self, action: UnitAction, to_c: TimeCoordinate) -> float:
        move_cost = super().get_cost(action, to_c)
        if self.next_goal_c is None or not isinstance(action, PickupAction):
            return move_cost

        distance_to_goal = to_c.distance_to(self.next_goal_c)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = distance_to_goal * min_cost_per_step
        # prefering picking up earlier to reduce changes of unit not being able to make it to the
        if self.later_pickup:
            heuristic_preference_pickup = -1 * to_c.t / 100
        else:
            heuristic_preference_pickup = to_c.t / 100
        return move_cost + min_distance_cost + heuristic_preference_pickup

    def get_heuristic(self, tc: ResourcePowerTimeCoordinate) -> float:
        if self.completes_goal(tc):
            return 0

        min_distance_cost = self._get_distance_heuristic(tc=tc)
        min_time_recharge_cost = self._get_time_supply_heuristic(tc=tc)
        return min_distance_cost + min_time_recharge_cost

    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        closest_factory_tile = self.board.get_closest_player_factory_tile(tc)
        distance_to_closest_factory_tiles = tc.distance_to(closest_factory_tile)

        if self.next_goal_c:
            # TODO, now it calculates from closest_factory_tile the heuristic, it could be that a tile at a different
            # factory will have the min distance if you take into account the next goal
            min_distance_factory_to_next_goal = self.next_goal_c.distance_to(closest_factory_tile)
            total_distance = distance_to_closest_factory_tiles + min_distance_factory_to_next_goal
        else:
            total_distance = distance_to_closest_factory_tiles

        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = total_distance * min_cost_per_step

        return min_distance_cost

    def _get_time_supply_heuristic(self, tc: ResourcePowerTimeCoordinate) -> float:
        if self.completes_goal(tc=tc):
            return 0
        else:
            return self.time_to_power_cost

    def completes_goal(self, tc: ResourcePowerTimeCoordinate) -> bool:
        return tc.q > 0


@dataclass
class TransferResourceGraph(Graph):
    _potential_move_actions = [MoveAction(direction) for direction in Direction]
    resource: Resource
    q: int

    def _get_potential_actions(self, tc: ResourcePowerTimeCoordinate) -> Generator[UnitAction, None, None]:
        if self._can_transfer(tc):
            receiving_tile = self._get_receiving_tile(tc)
            dir = tc.direction_to(receiving_tile)
            transfer_action = TransferAction(
                direction=dir,
                amount=self.q,
                resource=self.resource,
            )
            yield transfer_action

        for action in self._potential_move_actions:
            yield action

    @abstractmethod
    def _can_transfer(self, tc: TimeCoordinate) -> bool:
        ...

    @abstractmethod
    def _get_receiving_tile(self, tc: TimeCoordinate) -> TimeCoordinate:
        ...

    def get_heuristic(self, tc: ResourcePowerTimeCoordinate) -> float:
        if self.completes_goal(tc):
            return 0

        min_distance_cost = self._get_distance_heuristic(tc=tc)
        min_transfer_cost = self.time_to_power_cost
        resource_cost = self.q if self.resource == Resource.POWER else 0
        return min_distance_cost + min_transfer_cost + resource_cost

    @abstractmethod
    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        ...

    def completes_goal(self, tc: ResourcePowerTimeCoordinate) -> bool:
        return tc.q < 0


@dataclass
class TransferPowerToUnitResourceGraph(TransferResourceGraph):
    receiving_unit_c: Coordinate

    def _can_transfer(self, tc: TimeCoordinate) -> bool:
        return tc.distance_to(self.receiving_unit_c) == 1

    def _get_receiving_tile(self, tc: TimeCoordinate) -> Coordinate:
        return self.receiving_unit_c

    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        distance_to_unit = tc.distance_to(self.receiving_unit_c)
        min_nr_steps_next_to_unit = max(distance_to_unit - 1, 0)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps_next_to_unit * min_cost_per_step
        return min_distance_cost


@dataclass
class TransferToFactoryResourceGraph(TransferResourceGraph):
    factory: Optional[Factory] = field(default=None)

    def get_cost(self, action: UnitAction, to_c: ResourcePowerTimeCoordinate) -> float:
        cost = super().get_cost(action, to_c)

        # To prefer returning on factory tile instead of next to it when under threath
        if self.completes_goal(to_c) and not self.board.is_player_factory_tile(to_c):
            cost += 1

        return cost

    def _can_transfer(self, tc: TimeCoordinate) -> bool:
        if (not self.factory and self.board.get_min_distance_to_any_player_factory(c=tc) <= 1) or (
            self.factory and self.board.get_min_distance_to_player_factory(tc, self.factory.strain_id) <= 1
        ):
            return True

        return False

    def _get_receiving_tile(self, tc: TimeCoordinate) -> Coordinate:
        return self.board.get_closest_player_factory_tile(c=tc)

    def _get_distance_heuristic(self, tc: TimeCoordinate) -> float:
        if not self.factory:
            min_nr_steps_to_factory = self.board.get_min_distance_to_any_player_factory(c=tc)
        else:
            min_nr_steps_to_factory = self.board.get_min_distance_to_player_factory(tc, self.factory.strain_id)

        min_nr_steps_next_to_factory = max(min_nr_steps_to_factory - 1, 0)
        min_cost_per_step = self.time_to_power_cost + self.unit_cfg.MOVE_COST
        min_distance_cost = min_nr_steps_next_to_factory * min_cost_per_step
        return min_distance_cost


@dataclass
class DigAtGraph(GoalGraph):
    goal: DigCoordinate
    _potential_move_actions = [MoveAction(direction) for direction in Direction]
    _potential_dig_action = DigAction()

    def __post_init__(self):
        super().__post_init__()
        if not self.constraints:
            self._potential_move_actions = [
                MoveAction(direction) for direction in Direction if direction != direction.CENTER
            ]

    def _get_potential_actions(self, tc: TimeCoordinate) -> Generator[UnitAction, None, None]:
        if self.goal.x == tc.x and self.goal.y == tc.y:
            yield self._potential_dig_action

        for action in self._potential_move_actions:
            yield action

    def get_heuristic(self, tc: DigTimeCoordinate) -> float:
        distance_min_cost = self._get_distance_heuristic(tc=tc)
        digs_min_cost = self._get_digs_min_cost(tc=tc)

        return distance_min_cost + digs_min_cost

    def _get_digs_min_cost(self, tc: DigTimeCoordinate) -> float:
        nr_digs_required = self.goal.d - tc.d
        cost_per_dig = self.unit_cfg.DIG_COST + self.time_to_power_cost
        min_cost = nr_digs_required * cost_per_dig
        return min_cost

    def completes_goal(self, tc: DigTimeCoordinate) -> bool:
        return self.goal == tc
