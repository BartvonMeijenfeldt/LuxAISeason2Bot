class InvalidGoalError(Exception):
    def __str__(self) -> str:
        return "Invalid Goal"


class NoValidGoalFoundError(Exception):
    def __str__(self) -> str:
        return "No valid goal found"


class NoSolutionSearchError(Exception):
    def __str__(self) -> str:
        return "No solution to search"


class SolutionNotFoundWithinBudgetError(Exception):
    def __str__(self) -> str:
        return "Solution not found within budget"
