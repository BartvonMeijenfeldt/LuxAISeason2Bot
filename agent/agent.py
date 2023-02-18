import numpy as np
import pandas as pd

from typing import Optional, Literal
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import is_my_turn_to_place_factory
from objects.game_state import GameState
from objects.unit import Unit
from objects.action_plan import ActionPlan
from objects.coordinate import TimeCoordinate
from logic.early_setup import get_factory_spawn_loc
from logic.goal import Goal, GoalCollection
from logic.constraints import Constraints
from search import PriorityQueue


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg
        self.prev_steps_goals: dict[str, Goal] = {}

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)

        if step == 0:
            return dict(faction="AlphaStrike", bid=0)
        else:
            if is_my_turn_to_place_factory(game_state, step):
                spawn_loc = get_factory_spawn_loc(obs)
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """
        game_state = obs_to_game_state(step, self.env_cfg, obs, self.player, self.opp_player)
        factory_actions = self.get_factory_actions(game_state)
        unit_actions = self.get_unit_actions(game_state)

        actions = factory_actions | unit_actions
        return actions

    def get_factory_actions(self, game_state: GameState) -> dict[str, list[np.ndarray]]:
        actions = dict()
        for factory in game_state.player_factories:
            action = factory.act(game_state=game_state)
            if isinstance(action, int):
                actions[factory.unit_id] = action

        return actions

    def get_unit_actions(self, game_state: GameState) -> dict[str, list[np.ndarray]]:
        unit_goal_collections = []

        for unit in game_state.player_units:
            if unit.has_actions_in_queue:
                last_step_goal = self.prev_steps_goals[unit.unit_id]
                last_step_goal.unit = unit
                last_step_goal.action_plan = ActionPlan(original_actions=unit.action_queue, unit=unit)
                last_step_goal._value = 1_000_000
                unit_goal_collection = (unit, GoalCollection([last_step_goal]))
            else:
                goal_collection = unit.generate_goals(game_state=game_state)
                goal_collection.generate_and_evaluate_action_plans(game_state=game_state)
                unit_goal_collection = (unit, goal_collection)
                if unit.unit_id in self.prev_steps_goals:
                    del self.prev_steps_goals[unit.unit_id]

            unit_goal_collections.append(unit_goal_collection)

        unit_goals = resolve_goal_conflicts(unit_goal_collections)
        best_action_plans = pick_best_collective_action_plan(unit_goals, game_state=game_state)
        unit_actions = {
            unit_id: plan.to_action_arrays()
            for unit_id, plan in best_action_plans.items()
            if unit_id not in self.prev_steps_goals and plan.actions
        }

        self._update_prev_step_goals(unit_goals)

        return unit_actions

    def _update_prev_step_goals(self, unit_goal_collections: dict[Unit, Goal]) -> None:
        self.prev_steps_goals = {unit.unit_id: goal for unit, goal in unit_goal_collections.items()}


def resolve_goal_conflicts(unit_goal_collections: list[tuple[Unit, GoalCollection]]) -> dict[Unit, Goal]:
    if not unit_goal_collections:
        return {}

    cost_matrix = _create_cost_matrix(unit_goal_collections)
    goal_keys = _solve_sum_assigment_problem(cost_matrix)
    unit_goals = _get_unit_goals(unit_goal_collections=unit_goal_collections, goal_keys=goal_keys)
    unit_goals = {unit: goal for unit, goal in unit_goals if goal}

    return unit_goals


def _create_cost_matrix(unit_goal_collections: list[tuple[Unit, GoalCollection]]) -> pd.DataFrame:
    entries = [goal_collection.get_key_values() for _, goal_collection in unit_goal_collections]
    value_matrix = pd.DataFrame(entries)
    cost_matrix = -1 * value_matrix
    cost_matrix = cost_matrix.fillna(np.inf)

    return cost_matrix


def _solve_sum_assigment_problem(cost_matrix: pd.DataFrame) -> list[str]:
    rows, cols = linear_sum_assignment(cost_matrix)
    goal_keys = []
    for i in range(len(cost_matrix)):
        if i not in rows:
            goal_keys.append(None)
        else:
            index_ = np.argmax(rows == 4)
            c = cols[index_]
            goal_keys.append(cost_matrix.columns[c])

    goal_keys = [cost_matrix.columns[c] for c in cols]
    return goal_keys


def _get_unit_goals(
    unit_goal_collections: list[tuple[Unit, GoalCollection]], goal_keys: list[str]
) -> list[tuple[Unit, Goal]]:
    return [
        (unit, goal_collection.get_goal(goal))
        for goal, (unit, goal_collection) in zip(goal_keys, unit_goal_collections)
    ]


@dataclass
class PotentialSolution:
    unit_constraints: dict[Unit, Constraints]
    solution: dict[Unit, ActionPlan]
    value: float

    def __lt__(self, other) -> bool:
        return True


def pick_best_collective_action_plan(unit_goals: dict[Unit, Goal], game_state: GameState) -> dict[str, ActionPlan]:
    if not unit_goals:
        return {}

    root = _init_root(unit_goals, game_state)
    solutions = PriorityQueue()
    solutions.put(item=root, priority=root.value)

    while not solutions.empty():
        best_potential_solution: PotentialSolution = solutions.get()
        break # Temp for debugging
        collision = get_collision(best_potential_solution.solution, game_state=game_state)

        if not collision:
            break

        unit_goal = unit_goals[collision.unit]

        node_positive = get_new_node(collision, best_potential_solution, game_state, unit_goal, "positive")
        node_negative = get_new_node(collision, best_potential_solution, game_state, unit_goal, "negative")

        if node_positive:
            solutions.put(item=node_positive, priority=node_positive.value)

        if node_negative:
            solutions.put(item=node_negative, priority=node_negative.value)

        # append time

    best_action_plan = dict()
    for unit, action_plan in best_potential_solution.solution.items():
        if action_plan:
            best_action_plan[unit.unit_id] = action_plan

    return best_action_plan


def _init_root(unit_goals: dict[Unit, Goal], game_state: GameState) -> PotentialSolution:
    unit_constraints = {unit: Constraints() for unit in unit_goals.keys()}
    unit_action_plans = {}

    sum_value = 0

    for unit, goal in unit_goals.items():
        action_plan = goal.generate_action_plan(game_state)
        unit_action_plans[unit] = action_plan

        value = goal.get_value_action_plan(action_plan, game_state)
        sum_value += value

    return PotentialSolution(unit_constraints, unit_action_plans, sum_value)


@dataclass
class Collision:
    time_coordinate: TimeCoordinate
    unit: Unit


from copy import deepcopy


def get_new_node(
    collision: Collision,
    parent_solution: PotentialSolution,
    game_state: GameState,
    unit_goal: Goal,
    node_type: Literal["positive", "negative"],
) -> Optional[PotentialSolution]:
    unit_constraints = deepcopy(parent_solution.unit_constraints)
    unit_action_plans = deepcopy(parent_solution.solution)

    unit_constraint = unit_constraints[collision.unit]
    time_coordinate = collision.time_coordinate


    if node_type == "positive":
        t = time_coordinate.t
        if t in unit_constraint.positive or time_coordinate in unit_constraint.negative[t]:
            return None

        unit_constraint.positive[time_coordinate.t] = time_coordinate

    else:
        t = time_coordinate.t
        if t in unit_constraint.positive and time_coordinate == unit_constraint.positive[t]:
            return None

        unit_constraint.negative[time_coordinate.t].append(time_coordinate)

    old_solution_value = parent_solution.value
    old_action_plan_value = unit_goal.get_value_action_plan(unit_action_plans[collision.unit], game_state)

    new_action_plan = unit_goal.generate_action_plan(game_state, unit_constraint)
    new_value = unit_goal.get_value_action_plan(new_action_plan, game_state)
    # assert new_value <= old_solution_value, "Constrained solution can not be better"

    new_solution_value = old_action_plan_value - old_solution_value + new_value

    unit_action_plans[collision.unit] = new_action_plan
    return PotentialSolution(unit_constraints, unit_action_plans, new_solution_value)


def get_collision(unit_action_plans: dict[Unit, ActionPlan], game_state: GameState) -> Optional[Collision]:
    all_time_coordinates = set()

    for unit, action_plan in unit_action_plans.items():
        time_coordinates = action_plan.get_time_coordinates(game_state=game_state)
        if collisions := all_time_coordinates & time_coordinates:
            collision_time_coordinate = next(iter(collisions))
            return Collision(time_coordinate=collision_time_coordinate, unit=unit)

        all_time_coordinates.update(time_coordinates)

    return None
