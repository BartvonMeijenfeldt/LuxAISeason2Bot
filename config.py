class CONFIG:
    LIGHT_TIME_TO_POWER_COST = 5
    HEAVY_TIME_TO_POWER_COST = 10
    # TODO potential adaptation: start lower, and each timestep that passes increase the optimal path time to power cost
    OPTIMAL_PATH_TIME_TO_POWER_COST = 50

    RUBBLE_VALUE_CLEAR_FOR_RESOURCE: float = 10.0
    RUBBLE_VALUE_CLEAR_FOR_WATER_BASE: float = 10.0
    RUBBLE_CLEAR_FOR_LICHEN_MAX_DISTANCE: int = 6
    RUBBLE_CLEAR_FOR_LICHEN_MULTIPLIER_CLEARING: int = 3
