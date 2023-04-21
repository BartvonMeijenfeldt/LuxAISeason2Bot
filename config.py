class CONFIG:
    # Setup
    # --------------------------------------

    # ICE
    BASE_SCORE_ICE: int = 40
    PENALTY_DISTANCE_CLOSEST_ICE: int = 10_000

    # ORE
    BASE_SCORE_ORE: int = 20
    BONUS_CLOSEST_NEIGHBOR_ORE: int = 80

    # RUBBLE
    VALUE_CONNECTED_TILE: float = 2
    BEST_N_RUBBLE_TILES: int = 50

    # Gameplay
    # --------------------------------------

    LIGHT_TIME_TO_POWER_COST = 5
    HEAVY_TIME_TO_POWER_COST = 10
    # TODO potential adaptation: start lower, and each timestep that passes increase the optimal path time to power cost
    OPTIMAL_PATH_TIME_TO_POWER_COST = 50

    START_STEP_DESTROYING_LICHEN = 500

    RUBBLE_VALUE_CLEAR_FOR_RESOURCE: float = 10.0
    RUBBLE_VALUE_CLEAR_FOR_LICHEN_BASE: float = 10.0
    RUBBLE_VALUE_CLEAR_FOR_LICHEN_DISTANCE_PENALTY: float = 1.0
    RUBBLE_CLEAR_FOR_LICHEN_MAX_DISTANCE: int = 3
    RUBBLE_CLEAR_FOR_LICHEN_BONUS_CLEARING: int = 50

    # 1 ice -> 0.25 water -> 5 power (best case, alternating water)
    ICE_TO_POWER: float = 5
    # Randomly set
    ORE_TO_POWER: float = 10
    BENEFIT_ORE_REDUCTION_PER_T: float = 0.01

    BENEFIT_FLEEING: float = 0
    COST_POTENTIALLY_LOSING_UNIT: float = 10_000

    ENOUGH_WATER_COLLECTION_NR_TURNS: int = 20

    DESTROY_LICHEN_BASE_VALUE: int = 80
    DESTROY_LICHEN_VALUE_PER_LICHEN: int = 1
    START_FOCUSSING_ON_DESTROYING_LICHEN: int = 875
    FOCUS_ON_DESTROY_LICHEN_VALUE_MULTIPLIER: int = 3

    SUPPLY_POWER_VALUE: int = 10_000

    HUNT_VALUE: int = 10_000

    TURN_1_NR_DIGS_HEAVY: int = 37
    LAST_STEP_SCHEDULE_ORE_MINING: int = 900
    FIRST_STEP_HEAVY_ALLOWED_TO_DIG_RUBBLE: int = 300
    FIRST_STEP_HEAVY_ALLOWED_TO_DESTROY_LICHEN: int = 300

    # Factory Scheduling
    # --------------------------------------
    WATER_COLLECTION_VERSUS_USAGE_MIN_TARGET: float = 1.2

    # Power Usage
    EXPECTED_POWER_CONSUMPTION_HEAVY_PER_TURN: float = 40
    EXPECTED_POWER_CONSUMPTION_LIGHT_PER_TURN: float = 3.5

    # PowerUnitSignal
    POWER_UNIT_RATIO_NR_STEPS: int = 50

    # UnitImportanceSignal
    START_UNIT_IMPORTANCE_SIGNAL: float = 2.0
    LAST_TURN_UNIT_IMPORTANCE: int = 900
