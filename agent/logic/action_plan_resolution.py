from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

from search import PriorityQueue
from objects.unit import Unit
from objects.action import PickupAction
from objects.coordinate import TimeCoordinate
from objects.resource import Resource
from objects.action_plan import ActionPlan
from objects.game_state import GameState
from logic.constraints import Constraints
from logic.goal import Goal


@dataclass
class PotentialSolution:
    unit_constraints: dict[Unit, Constraints]
    solution: dict[Unit, ActionPlan]
    value: float

    def __lt__(self, other: PotentialSolution) -> bool:
        return self.value < other.value


@dataclass
class PowerCollision:
    unit: Unit
    max_power: int


@dataclass
class UnitCollision:
    unit: Unit
    time_coordinate: TimeCoordinate


class ActionPlanResolver:
    def __init__(self, unit_goals: dict[Unit, Goal], game_state: GameState) -> None:
        self.solutions = PriorityQueue()
        self.game_state = game_state
        self.unit_goals = unit_goals
        self._add_root_to_queue()

    def resolve(self) -> dict[str, ActionPlan]:
        while not self.solutions.empty():
            best_potential_solution: PotentialSolution = self.solutions.get()
            power_collision = get_power_collision(best_potential_solution.solution, self.game_state)

            if power_collision:
                self._optional_add_power_node(best_potential_solution, power_collision)
                continue

            unit_collision = get_unit_collision(best_potential_solution.solution, self.game_state)

            if unit_collision:
                self._optional_add_positive_tc_node(best_potential_solution, unit_collision)
                self._optional_add_negative_tc_node(best_potential_solution, unit_collision)

                continue

            break

        best_action_plan = dict()
        for unit, action_plan in best_potential_solution.solution.items():
            if action_plan:
                best_action_plan[unit.unit_id] = action_plan

        return best_action_plan

    def _add_root_to_queue(self) -> None:
        root = self._init_root()
        self._add_node_to_queue(node=root)

    def _add_node_to_queue(self, node: PotentialSolution) -> None:
        self.solutions.put(item=node, priority=node.value)

    def _init_root(self) -> PotentialSolution:
        unit_constraints = {unit: Constraints() for unit in self.unit_goals.keys()}
        unit_action_plans = {}

        sum_value = 0

        for unit, goal in self.unit_goals.items():
            if goal.has_set_action_plan:
                action_plan = goal.action_plan
            else:
                action_plan = goal.generate_action_plan(self.game_state)

            unit_action_plans[unit] = action_plan

            value = goal.get_value_action_plan(action_plan, self.game_state)
            sum_value += value

        return PotentialSolution(unit_constraints, unit_action_plans, sum_value)

    def _optional_add_power_node(self, parent_solution: PotentialSolution, power_collision: PowerCollision) -> None:

        unit_constraints = self._get_new_power_constraints(parent_solution.unit_constraints, power_collision)

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_constraints=unit_constraints,
            unit_with_new_constraint=power_collision.unit,
        )

    def _get_new_power_constraints(
        self, unit_constraints: dict[Unit, Constraints], power_collision: PowerCollision
    ) -> dict[Unit, Constraints]:
        unit_constraints = deepcopy(unit_constraints)
        unit = power_collision.unit
        unit_constraint = unit_constraints[unit]
        unit_constraint.max_power_request = power_collision.max_power
        return unit_constraints

    def _optional_add_positive_tc_node(self, parent_solution: PotentialSolution, unit_collision: UnitCollision) -> None:

        unit_constraints = self._get_new_positive_tc_constraints(parent_solution.unit_constraints, unit_collision)
        if not unit_constraints:
            return

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_constraints=unit_constraints,
            unit_with_new_constraint=unit_collision.unit,
        )

    def _get_new_positive_tc_constraints(
        self, unit_constraints: dict[Unit, Constraints], collision: UnitCollision
    ) -> Optional[dict[Unit, Constraints]]:
        unit_constraints = deepcopy(unit_constraints)
        unit = collision.unit
        unit_constraint = unit_constraints[unit]

        time_coordinate = collision.time_coordinate
        t = time_coordinate.t
        if t in unit_constraint.positive:
            return None

        unit_constraint.positive[time_coordinate.t] = time_coordinate
        return unit_constraints

    def _optional_add_negative_tc_node(
        self,
        parent_solution: PotentialSolution,
        unit_collision: UnitCollision,
    ) -> None:

        unit_constraints = self._get_new_negative_tc_constraints(parent_solution.unit_constraints, unit_collision)
        if not unit_constraints:
            return

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_constraints=unit_constraints,
            unit_with_new_constraint=unit_collision.unit,
        )

    def _get_new_negative_tc_constraints(
        self, unit_constraints: dict[Unit, Constraints], collision: UnitCollision
    ) -> Optional[dict[Unit, Constraints]]:
        unit_constraints = deepcopy(unit_constraints)
        unit = collision.unit
        unit_constraint = unit_constraints[unit]

        time_coordinate = collision.time_coordinate
        t = time_coordinate.t
        if (t in unit_constraint.negative and time_coordinate in unit_constraint.negative[t]) or (
            t in unit_constraint.positive and time_coordinate == unit_constraint.positive[t]
        ):
            return None

        unit_constraint.negative[time_coordinate.t].append(time_coordinate)
        return unit_constraints

    def _optional_add_node(
        self,
        parent_solution: PotentialSolution,
        unit_constraints: dict[Unit, Constraints],
        unit_with_new_constraint: Unit,
    ) -> None:
        new_action_plan = self._get_new_action_plan(unit=unit_with_new_constraint, unit_constraints=unit_constraints)

        if not new_action_plan.unit_can_carry_out_plan(game_state=self.game_state):
            return

        new_joint_action_plans = self._get_new_joint_action_plans(
            parent_solution, new_action_plan, unit=unit_with_new_constraint
        )
        new_solution_value = self._get_new_solution_value(parent_solution, new_action_plan, unit_with_new_constraint)

        new_solution = PotentialSolution(unit_constraints, new_joint_action_plans, new_solution_value)
        self._add_node_to_queue(node=new_solution)

    def _get_new_action_plan(self, unit: Unit, unit_constraints: dict[Unit, Constraints]) -> ActionPlan:
        unit_goal = self.unit_goals[unit]
        unit_constraint = unit_constraints[unit]
        new_action_plan = unit_goal.generate_action_plan(self.game_state, unit_constraint)
        return new_action_plan

    def _get_new_joint_action_plans(
        self, parent_solution: PotentialSolution, new_action_plan: ActionPlan, unit: Unit
    ) -> dict[Unit, ActionPlan]:

        unit_action_plans = deepcopy(parent_solution.solution)
        unit_action_plans[unit] = new_action_plan
        return unit_action_plans

    def _get_new_solution_value(
        self,
        parent_solution: PotentialSolution,
        new_action_plan: ActionPlan,
        unit: Unit,
    ) -> float:

        unit_action_plans = parent_solution.solution
        unit_goal = self.unit_goals[unit]

        old_action_plan_value = unit_goal.get_value_action_plan(unit_action_plans[unit], self.game_state)
        new_value = unit_goal.get_value_action_plan(new_action_plan, self.game_state)
        old_solution_value = parent_solution.value

        new_solution_value = old_action_plan_value - old_solution_value + new_value
        return new_solution_value


def get_power_collision(unit_action_plans: dict[Unit, ActionPlan], game_state: GameState):
    factories_power_requested = {factory: 0 for factory in game_state.board.player_factories}
    factories_power_available = {factory: factory.power for factory in game_state.board.player_factories}

    for unit, action_plan in unit_action_plans.items():
        if action_plan.primitive_actions:
            first_action = action_plan.primitive_actions[0]
            if isinstance(first_action, PickupAction) and first_action.resource == Resource.Power:
                unit_power_requested = first_action.amount
                factory = game_state.get_closest_factory(unit.tc)

                power_available = factories_power_available[factory] - factories_power_requested[factory]

                if unit_power_requested > power_available:
                    return PowerCollision(unit=unit, max_power=power_available)

                factories_power_requested[factory] += unit_power_requested

    return None


def get_unit_collision(unit_action_plans: dict[Unit, ActionPlan], game_state: GameState) -> Optional[UnitCollision]:
    all_time_coordinates = set()

    for unit, action_plan in unit_action_plans.items():
        time_coordinates = action_plan.get_time_coordinates(game_state=game_state)
        if collisions := all_time_coordinates & time_coordinates:
            collision_time_coordinate = next(iter(collisions))
            return UnitCollision(time_coordinate=collision_time_coordinate, unit=unit)

        all_time_coordinates.update(time_coordinates)

    return None
