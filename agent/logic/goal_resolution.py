import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

from objects.unit import Unit
from logic.goal import Goal, GoalCollection


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
