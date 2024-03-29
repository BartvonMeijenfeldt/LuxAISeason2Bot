class CONFIG:
    # Setup
    # --------------------------------------

    # ICE
    BASE_SCORE_ICE: int = 40
    PENALTY_DISTANCE_CLOSEST_ICE: int = 10_000

    # ORE
    BASE_SCORE_ORE: int = 50
    BONUS_CLOSEST_NEIGHBOR_ORE: int = 350

    # RUBBLE
    VALUE_CONNECTED_TILE: float = 2
    BEST_N_RUBBLE_TILES: int = 50

    # At Border
    BORDER_PENALTY: int = 15

    # Gameplay
    # --------------------------------------
    MAX_DISTANCE_DESTROY_LICHEN: int = 100

    SEARCH_BUDGET_HEAVY: int = 300
    SEARCH_BUDGET_LIGHT: int = 150

    MIN_OWNERSHIP_REQUIRED_FOR_MINING: float = 0.34

    LIGHT_TIME_TO_POWER_COST = 5
    HEAVY_TIME_TO_POWER_COST = 30
    OPTIMAL_PATH_TIME_TO_POWER_COST = 50

    RUBBLE_CLEAR_FOR_LICHEN_BONUS_CLEARING: int = 50

    # Supply Power
    LOW_ECO_FACTORY_THRESHOLD: int = 4000
    MINIMUM_POWER_RECEIVING_UNIT_LOW_ECO: int = 180

    # 1 ice -> 0.25 water -> 5 power (best case, alternating water)
    MIN_SIGNAL_ICE: float = 0.2
    ICE_TO_POWER: float = 5
    # Based on power generation untill end of game
    ORE_TO_POWER: float = 12
    BENEFIT_ORE_REDUCTION_PER_T: float = 0.012

    BENEFIT_FLEEING: float = 0
    COST_POTENTIALLY_LOSING_UNIT: float = 10_000

    ENOUGH_WATER_COLLECTION_NR_TURNS: int = 20

    DESTROY_LICHEN_BASE_VALUE: int = 80
    DESTROY_LICHEN_VALUE_PER_LICHEN: int = 1

    SUPPLY_POWER_VALUE: int = 10_000

    TURN_1_NR_DIGS_HEAVY: int = 37
    LAST_STEP_SCHEDULE_ORE_MINING: int = 900
    FIRST_STEP_HEAVY_ALLOWED_TO_DIG_RUBBLE: int = 900
    FIRST_STEP_HEAVY_ALLOWED_TO_DESTROY_LICHEN: int = 300

    # Factory Scheduling
    # --------------------------------------
    MAX_SCHEDULING_FAILURES_ALLOWED_FACTORY: int = 100
    MAX_DISTANCE_FOR_RESOURCE_CLEARING: int = 15

    ICE_MUST_COME_IN_BEFORE_LEVEL: int = 3
    TOO_LITTLE_WATER_DISTRESS_LEVEL: int = 40
    DISTRESS_SIGNAL: int = 100

    WATER_COLLECTION_VERSUS_USAGE_MIN_TARGET: float = 1.2

    # Too Little Lichen
    MAX_SIGNAL_TOO_LITTE_LICHEN: int = 3
    MIN_TILES_GROWTH_TARGET: int = 100

    # Power Usage
    EXPECTED_POWER_CONSUMPTION_HEAVY_PER_TURN: float = 40
    EXPECTED_POWER_CONSUMPTION_LIGHT_PER_TURN: float = 3.5

    # PowerUnitSignal
    POWER_UNIT_RATIO_NR_STEPS: int = 100

    # UnitImportanceSignal
    START_UNIT_IMPORTANCE_SIGNAL: float = 2.0
    LAST_TURN_UNIT_IMPORTANCE: int = 750
    UNIT_IMPORTANCE_MIN_LEVEL_POWER_UNIT: float = 1.0

    # Start attack en masse
    ATTACK_EN_MASSE_START_STEP: int = 850
    ATTACK_EN_MASSE_SIGNAL: float = 2.5

    # Clear Rubble Around Base
    CLEAR_RUBBLE_MAX_SIGNAL: float = 0.6
    SLOPE_CLEAR_RUBBLE_SIGNAL: float = 0.001

    # Schedule
    OUT_OF_TIME_MAIN_SCHEDULING: float = 2.5
    OUT_OF_TIME_UNASSIGNED_SCHEDULING: float = 2.8

    # Watering
    MIN_RATIO_WATER_WATER_COST_ALWAYS_GROW_LICHEN: float = 25.0
    MIN_RATIO_WATER_WATER_COST_MAINTAIN_LICHEN: float = 10.0
    WATER_LICHEN_SIZE_FRACTION: float = 0.90

    MIN_WATER_SAFETY_LEVEL: int = 50
    WATER_SAFETY_SLOPE_PER_STEP: float = 0.02
    MAX_WATER_SAFETY_LEVEL: int = 100

    # Units Building
    LAST_STEP_UNIT_BUILDING: int = 994

    # INVADER
    DISTANCE_TO_FACTORY_CONSIDERED_INVADER: int = 4
    INVADER_HEAVY_PRIORITY: int = 100

    # Units
    # --------------------------------------
    MAIN_CARGO_THRESHOLD_LIGHT = 20
    MAIN_CARGO_THRESHOLD_HEAVY = 400

    MAX_DISTANCE_COLLECTING: int = 25

    MIN_VALUE_PER_STEP_LIGHT: float = 0.2
    MIN_VALUE_PER_STEP_HEAVY: float = 2

    STANDARD_FLEE_DISTANCE: int = 5
