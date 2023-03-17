import random

from typing import List, Optional
from logic.constraints import Constraints
from objects.coordinate import TimeCoordinate as TC


def init_constraints(
    positive_constraints: Optional[List[TC]] = None,
    negative_constraints: Optional[List[TC]] = None,
    max_power_constraint: Optional[int] = None,
) -> Constraints:
    constraints = Constraints()

    if not positive_constraints:
        positive_constraints = []

    if not negative_constraints:
        negative_constraints = []

    # Shuffle constraints to make sure the order of adding does not matter
    random.shuffle(positive_constraints)
    random.shuffle(negative_constraints)

    for tc in positive_constraints:
        constraints = constraints.add_positive_constraint(tc)

    for tc in negative_constraints:
        constraints = constraints.add_negative_constraint(tc)

    constraints.max_power_request = max_power_constraint

    return constraints
