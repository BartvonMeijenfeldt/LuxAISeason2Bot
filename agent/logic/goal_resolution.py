import pandas as pd
import numpy as np

from scipy.optimize import linear_sum_assignment
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

from objects.unit import Unit
from objects.game_state import GameState
from logic.goal import Goal, GoalCollection
from logic.constraints import Constraints


def resolve_goal_conflicts(
    goal_collections: Dict[Unit, GoalCollection],
    game_state: GameState,
    unit_constraints: Optional[Dict[Unit, Constraints]] = None,
) -> Dict[Unit, Goal]:
    if not goal_collections:
        return {}

    if not unit_constraints:
        unit_constraints = defaultdict(lambda: Constraints())

    cost_matrix = _create_cost_matrix(goal_collections, unit_constraints, game_state)
    goal_keys = _solve_sum_assigment_problem(cost_matrix)
    unit_goals = _get_unit_goals(unit_goal_collections=goal_collections, goal_keys=goal_keys)
    unit_goals = {unit: goal for unit, goal in unit_goals if goal}

    return unit_goals


def _create_cost_matrix(
    goal_collections: Dict[Unit, GoalCollection], unit_constraints: Dict[Unit, Constraints], game_state: GameState
) -> pd.DataFrame:
    entries = [
        goal_collection.get_key_values(game_state=game_state, constraints=unit_constraints[unit])
        for unit, goal_collection in goal_collections.items()
    ]

    value_matrix = pd.DataFrame(entries)
    cost_matrix = -1 * value_matrix
    cost_matrix = cost_matrix.fillna(np.inf)

    return cost_matrix


def _solve_sum_assigment_problem(cost_matrix: pd.DataFrame) -> List[str]:
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
    assert len(goal_keys) == len(set(goal_keys))
    return goal_keys


def _get_unit_goals(unit_goal_collections: Dict[Unit, GoalCollection], goal_keys: List[str]) -> List[Tuple[Unit, Goal]]:
    return [
        (unit, goal_collection.get_goal(goal))
        for goal, (unit, goal_collection) in zip(goal_keys, unit_goal_collections.items())
    ]
