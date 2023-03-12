from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
from collections import defaultdict
from copy import copy
from itertools import count

from search import PriorityQueue
from objects.actors.actor import Actor
from objects.actors.unit import Unit
from objects.coordinate import TimeCoordinate
from objects.actions.action_plan import ActionPlan
from objects.actions.unit_action_plan import UnitActionPlan
from objects.game_state import GameState
from logic.constraints import Constraints
from logic.goals.goal import GoalCollection
from logic.goals.goal import Goal
from logic.goal_resolution import resolve_goal_conflicts


@dataclass
class Solution:
    constraints: dict[Actor, Constraints]
    goals: dict[Actor, Goal]
    action_plans: dict[Actor, ActionPlan]
    value: float

    @property
    def joint_action_plan(self) -> dict[Actor, ActionPlan]:
        return {unit: action_plan for unit, action_plan in self.action_plans.items()}

    def __lt__(self, other: Solution) -> bool:
        return self.value < other.value


@dataclass
class PowerCollision:
    actors: Sequence[Actor]
    power_deficit: int


@dataclass
class ActorCollision:
    actors: list[Actor]
    tc: TimeCoordinate

    @property
    def constraint_actor(self) -> Actor:
        return self.actors[0]

    @property
    def non_constraint_actors(self) -> list[Actor]:
        return self.actors[1:]


class ActionPlanResolver:
    def __init__(self, actor_goal_collections: dict[Actor, GoalCollection], game_state: GameState) -> None:
        self.solutions = PriorityQueue()
        self.game_state = game_state
        self.actor_goal_collections = actor_goal_collections
        self._add_root_to_queue()

    def resolve(self) -> tuple[dict[Actor, Goal], dict[Actor, ActionPlan]]:
        best_solution = self._get_best_solution()
        return best_solution.goals, best_solution.joint_action_plan

    def _get_best_solution(self) -> Solution:
        for i in count():
            if i > 10 or self.solutions.empty():
                break

            best_potential_solution: Solution = self.solutions.get()
            power_collision = get_power_collision(best_potential_solution.action_plans, self.game_state)

            if power_collision:
                for actor in power_collision.actors:
                    self._if_valid_add_power_node(best_potential_solution, power_collision.power_deficit, actor)
                continue

            unit_collision = get_unit_collision(best_potential_solution.action_plans)

            if unit_collision:
                self._if_valid_add_positive_tc_node(best_potential_solution, unit_collision)
                self._if_valid_add_negative_tc_node(best_potential_solution, unit_collision)
                continue

            return best_potential_solution

        # TODO make this penalize solutions with collisions instead of just taking the last one
        return best_potential_solution
        # raise RuntimeError("No best solution found")

    def _add_root_to_queue(self) -> None:
        root = self._init_root()
        self._add_node_to_queue(node=root)

    def _add_node_to_queue(self, node: Solution) -> None:
        self.solutions.put(item=node, priority=node.value)

    def _init_root(self) -> Solution:
        actor_constraints = {actor: Constraints() for actor in self.actor_goal_collections.keys()}
        actor_goals = resolve_goal_conflicts(self.actor_goal_collections, self.game_state, actor_constraints)

        actor_action_plans = {}

        sum_value = 0

        for actor, goal in actor_goals.items():
            action_plan = goal.generate_action_plan(self.game_state, constraints=Constraints())
            actor_action_plans[actor] = action_plan
            value = goal.get_value_action_plan(action_plan, self.game_state)
            sum_value += value

        return Solution(actor_constraints, actor_goals, actor_action_plans, sum_value)

    def _if_valid_add_power_node(self, parent_solution: Solution, power_deficit: int, actor: Actor) -> None:

        unit_constraints = self._get_new_power_constraints(parent_solution, power_deficit, actor)
        if not unit_constraints:
            return

        unit_goals = resolve_goal_conflicts(self.actor_goal_collections, self.game_state, unit_constraints)

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_goals=unit_goals,
            unit_constraints=unit_constraints,
            units_to_adjust_action_plan=[actor],
        )

    def _get_new_power_constraints(
        self, parent_solution: Solution, power_deficit: int, unit: Actor
    ) -> Optional[dict[Actor, Constraints]]:
        unit_constraints = copy(parent_solution.constraints)
        unit_constraint = copy(unit_constraints[unit])
        unit_power_requested = parent_solution.action_plans[unit].actions[0].requested_power

        if unit_constraint.max_power_request == 0:
            return None

        unit_constraint.parent = unit_constraint.key
        unit_constraint.max_power_request = max(unit_power_requested - power_deficit, 0)
        unit_constraints[unit] = unit_constraint

        return unit_constraints

    def _if_valid_add_positive_tc_node(self, parent_solution: Solution, unit_collision: ActorCollision) -> None:

        unit_constraints = self._get_new_positive_tc_constraints(parent_solution.constraints, unit_collision)
        if not unit_constraints:
            return

        unit_goals = resolve_goal_conflicts(self.actor_goal_collections, self.game_state, unit_constraints)

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_goals=unit_goals,
            unit_constraints=unit_constraints,
            units_to_adjust_action_plan=unit_collision.non_constraint_actors,
        )

    def _get_new_positive_tc_constraints(
        self, parent_constraints: dict[Actor, Constraints], collision: ActorCollision
    ) -> Optional[dict[Actor, Constraints]]:
        unit_constraints = copy(parent_constraints)
        collsion_unit = collision.constraint_actor
        time_coordinate = collision.tc

        for unit, constraints in unit_constraints.items():
            new_constraints = copy(constraints)
            new_constraints.parent = constraints.key

            if collsion_unit == unit:
                if constraints.t_in_positive_constraints(time_coordinate.t):
                    return None

                new_constraints.add_positive_constraint(time_coordinate)
            else:
                new_constraints.add_negative_constraint(time_coordinate)

            unit_constraints[unit] = new_constraints

        return unit_constraints

    def _if_valid_add_negative_tc_node(self, parent_solution: Solution, unit_collision: ActorCollision,) -> None:

        unit_constraints = self._get_new_negative_tc_constraints(parent_solution.constraints, unit_collision)
        if not unit_constraints:
            return

        unit_goals = resolve_goal_conflicts(self.actor_goal_collections, self.game_state, unit_constraints)

        self._optional_add_node(
            parent_solution=parent_solution,
            unit_goals=unit_goals,
            unit_constraints=unit_constraints,
            units_to_adjust_action_plan=[unit_collision.constraint_actor],
        )

    def _get_new_negative_tc_constraints(
        self, parent_constraints: dict[Actor, Constraints], collision: ActorCollision
    ) -> Optional[dict[Actor, Constraints]]:
        unit_constraints = copy(parent_constraints)
        unit = collision.constraint_actor
        unit_constraint = parent_constraints[unit]

        if unit_constraint.tc_in_constraints(collision.tc):
            return None

        new_constraints = copy(unit_constraint)
        new_constraints.parent = unit_constraint.key
        new_constraints.add_negative_constraint(collision.tc)
        unit_constraints[unit] = new_constraints

        return unit_constraints

    def _optional_add_node(
        self,
        parent_solution: Solution,
        unit_goals: dict[Actor, Goal],
        unit_constraints: dict[Actor, Constraints],
        units_to_adjust_action_plan: list[Actor],
    ) -> None:

        solution = parent_solution

        for unit in units_to_adjust_action_plan:
            new_action_plan = self._get_new_action_plan(
                unit=unit, unit_goals=unit_goals, unit_constraints=unit_constraints
            )
            if not new_action_plan.actor_can_carry_out_plan(game_state=self.game_state):
                return

            new_joint_action_plans = self._get_new_joint_action_plans(solution, new_action_plan, unit=unit)
            new_solution_value = self._get_new_solution_value(solution, unit_goals, new_action_plan, unit)

            solution = Solution(unit_constraints, unit_goals, new_joint_action_plans, new_solution_value)
        self._add_node_to_queue(node=solution)

    def _get_new_action_plan(
        self, unit: Actor, unit_goals: dict[Actor, Goal], unit_constraints: dict[Actor, Constraints]
    ) -> ActionPlan:

        unit_goal = unit_goals[unit]
        unit_constraint = unit_constraints[unit]
        new_action_plan = unit_goal.generate_action_plan(self.game_state, unit_constraint)
        return new_action_plan

    def _get_new_joint_action_plans(
        self, parent_solution: Solution, new_action_plan: ActionPlan, unit: Actor
    ) -> dict[Actor, ActionPlan]:

        unit_action_plans = copy(parent_solution.action_plans)
        unit_action_plans[unit] = copy(new_action_plan)
        return unit_action_plans

    def _get_new_solution_value(
        self, parent_solution: Solution, unit_goals: dict[Actor, Goal], new_action_plan: ActionPlan, unit: Actor,
    ) -> float:

        actor_action_plans = parent_solution.action_plans
        unit_goal = unit_goals[unit]

        old_action_plan_value = unit_goal.get_value_action_plan(actor_action_plans[unit], self.game_state)
        new_value = unit_goal.get_value_action_plan(new_action_plan, self.game_state)
        old_solution_value = parent_solution.value

        new_solution_value = old_action_plan_value - old_solution_value + new_value
        return new_solution_value


def get_power_collision(actor_action_plans: dict[Actor, ActionPlan], game_state: GameState):
    factories_power_available = {factory: factory.power for factory in game_state.board.player_factories}
    factories_requested_by_units = defaultdict(list)
    unit_action_plans: list[tuple[Unit, UnitActionPlan]] = [
        (unit, action_plan)
        for unit, action_plan in actor_action_plans.items()
        if isinstance(unit, Unit) and isinstance(action_plan, UnitActionPlan)
    ]

    for unit, action_plan in unit_action_plans:
        if not action_plan.primitive_actions:
            continue

        first_action = action_plan.primitive_actions[0]
        unit_power_requested = first_action.requested_power

        if not unit_power_requested:
            continue

        factory = game_state.get_closest_factory(unit.tc)
        factories_requested_by_units[factory].append(unit)

        if unit_power_requested > factories_power_available[factory]:
            power_deficit = unit_power_requested - factories_power_available[factory]
            return PowerCollision(actors=factories_requested_by_units[factory], power_deficit=power_deficit)

        factories_power_available[factory] -= unit_power_requested

    return None


def get_unit_collision(unit_action_plans: dict[Actor, ActionPlan]) -> Optional[ActorCollision]:
    all_time_coordinates = set()

    for action_plan in unit_action_plans.values():
        time_coordinates = action_plan.time_coordinates

        collisions = all_time_coordinates & time_coordinates
        if collisions:
            collision_tc = next(iter(collisions))
            return _get_unit_collision(unit_action_plans=unit_action_plans, collision_tc=collision_tc)

        all_time_coordinates.update(time_coordinates)

    return None


def _get_unit_collision(unit_action_plans: dict[Actor, ActionPlan], collision_tc: TimeCoordinate) -> ActorCollision:
    collision_units = []
    for unit, action_plan in unit_action_plans.items():
        time_coordinates = action_plan.time_coordinates
        if collision_tc in time_coordinates:
            collision_units.append(unit)

    assert len(collision_units) >= 2

    return ActorCollision(actors=collision_units, tc=collision_tc)
