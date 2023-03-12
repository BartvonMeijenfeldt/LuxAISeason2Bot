import pandas as pd
import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Dict
from collections import defaultdict

from objects.actors.actor import Actor
from objects.game_state import GameState
from logic.goals.goal import Goal, GoalCollection
from logic.constraints import Constraints


def resolve_goal_conflicts(
    goal_collections: Dict[Actor, GoalCollection],
    game_state: GameState,
    constraints: Dict[Actor, Constraints],
) -> Dict[Actor, Goal]:
    if not goal_collections:
        return {}

    if not constraints:
        constraints = defaultdict(lambda: Constraints())

    cost_matrix = _create_cost_matrix(goal_collections, constraints, game_state)
    actor_keys_goals_keys = _solve_sum_assigment_problem(cost_matrix)
    actor_goals = _get_actor_goals(actor_goal_collections=goal_collections, actor_keys_goals_keys=actor_keys_goals_keys)

    return actor_goals


def _create_cost_matrix(
    goal_collections: Dict[Actor, GoalCollection], actor_constraints: Dict[Actor, Constraints], game_state: GameState
) -> pd.DataFrame:
    entries = [
        goal_collection.get_key_values(game_state=game_state, constraints=actor_constraints[actor])
        for actor, goal_collection in goal_collections.items()
    ]

    actor_ids = [actor.unit_id for actor in goal_collections]
    value_matrix = pd.DataFrame(entries, index=actor_ids)
    cost_matrix = -1 * value_matrix
    cost_matrix = cost_matrix.fillna(np.inf)

    return cost_matrix


def _solve_sum_assigment_problem(cost_matrix: pd.DataFrame) -> Dict[str, str]:
    rows, cols = linear_sum_assignment(cost_matrix)
    unit_keys_goals_keys = {cost_matrix.index[r]: cost_matrix.columns[c] for r, c in zip(rows, cols)}
    return unit_keys_goals_keys


def _get_actor_goals(
    actor_goal_collections: Dict[Actor, GoalCollection], actor_keys_goals_keys: Dict[str, str]
) -> Dict[Actor, Goal]:
    return {
        actor: goal_collection.get_goal(goal)
        for goal, (actor, goal_collection) in zip(actor_keys_goals_keys.values(), actor_goal_collections.items())
    }
