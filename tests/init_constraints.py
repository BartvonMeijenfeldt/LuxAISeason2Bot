import random
from typing import List, Optional

from logic.constraints import Constraints
from objects.coordinate import TimeCoordinate as TC


def init_constraints(negative_constraints: Optional[List[TC]] = None) -> Constraints:
    constraints = Constraints()
    if not negative_constraints:
        negative_constraints = []

    # Shuffle constraints to make sure the order of adding does not matter
    random.shuffle(negative_constraints)

    if negative_constraints:
        constraints.add_negative_constraints(negative_constraints)

    return constraints
