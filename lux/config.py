# Competition file supplied by Competition hosts

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UnitConfig:
    METAL_COST: int = 100
    POWER_COST: int = 500
    CARGO_SPACE: int = 1000
    BATTERY_CAPACITY: int = 1500
    CHARGE: int = 1
    INIT_POWER: int = 50
    MOVE_COST: int = 1
    RUBBLE_MOVEMENT_COST: float = 1
    DIG_COST: int = 5
    DIG_RUBBLE_REMOVED: int = 1
    DIG_RESOURCE_GAIN: int = 2
    DIG_LICHEN_REMOVED: int = 10
    SELF_DESTRUCT_COST: int = 10
    RUBBLE_AFTER_DESTRUCTION: int = 1
    ACTION_QUEUE_POWER_COST: int = 1

    def __hash__(self) -> int:
        return self.DIG_COST

    def __eq__(self, __value: UnitConfig) -> bool:
        return self.DIG_COST == __value.DIG_COST


LIGHT_CONFIG = UnitConfig(
    METAL_COST=10,
    POWER_COST=50,
    INIT_POWER=50,
    CARGO_SPACE=100,
    BATTERY_CAPACITY=150,
    CHARGE=1,
    MOVE_COST=1,
    RUBBLE_MOVEMENT_COST=0.05,
    DIG_COST=5,
    SELF_DESTRUCT_COST=5,
    DIG_RUBBLE_REMOVED=2,
    DIG_RESOURCE_GAIN=2,
    DIG_LICHEN_REMOVED=10,
    RUBBLE_AFTER_DESTRUCTION=1,
    ACTION_QUEUE_POWER_COST=1,
)

HEAVY_CONFIG = UnitConfig(
    METAL_COST=100,
    POWER_COST=500,
    INIT_POWER=500,
    CARGO_SPACE=1000,
    BATTERY_CAPACITY=3000,
    CHARGE=10,
    MOVE_COST=20,
    RUBBLE_MOVEMENT_COST=1,
    DIG_COST=60,
    SELF_DESTRUCT_COST=100,
    DIG_RUBBLE_REMOVED=20,
    DIG_RESOURCE_GAIN=20,
    DIG_LICHEN_REMOVED=100,
    RUBBLE_AFTER_DESTRUCTION=10,
    ACTION_QUEUE_POWER_COST=10,
)


@dataclass
class EnvConfig:
    # various options that can be configured if needed

    # Variable parameters that don't affect game logic much ###
    max_episode_length: int = 1000
    map_size: int = 48
    verbose: int = 1

    # this can be disabled to improve env FPS but assume your actions are well formatted
    # During online competition this is set to True
    validate_action_space: bool = True

    # Constants
    # you can only ever transfer in/out 1000 as this is the max cargo space.
    max_transfer_amount: int = 10000
    MIN_FACTORIES: int = 2
    MAX_FACTORIES: int = 5
    CYCLE_LENGTH: int = 50
    DAY_LENGTH: int = 30
    UNIT_ACTION_QUEUE_SIZE: int = 20  # when set to 1, then no action queue is used

    MAX_RUBBLE: int = 100
    FACTORY_RUBBLE_AFTER_DESTRUCTION: int = 50
    INIT_WATER_METAL_PER_FACTORY: int = 150  # amount of water and metal units given to each factory
    INIT_POWER_PER_FACTORY: int = 1000

    # LICHEN
    MIN_LICHEN_TO_SPREAD: int = 20
    LICHEN_LOST_WITHOUT_WATER: int = 1
    LICHEN_GAINED_WITH_WATER: int = 1
    MAX_LICHEN_PER_TILE: int = 100
    POWER_PER_CONNECTED_LICHEN_TILE: int = 1

    # cost of watering with a factory is `ceil(# of connected lichen tiles) / (this factor) + 1`
    LICHEN_WATERING_COST_FACTOR: int = 10

    # Bidding System
    BIDDING_SYSTEM: bool = True

    # Factories
    FACTORY_PROCESSING_RATE_WATER: int = 100
    ICE_WATER_RATIO: int = 4
    FACTORY_PROCESSING_RATE_METAL: int = 50
    ORE_METAL_RATIO: int = 5
    # game design note: Factories close to resource cluster = more resources are refined per turn
    # Then the high ice:water and ore:metal ratios encourages transfer of refined resources between
    # factories dedicated to mining particular clusters which is more possible as it is more compact

    FACTORY_CHARGE: int = 50
    FACTORY_WATER_CONSUMPTION: int = 1
    # game design note: with a positve water consumption, game becomes quite hard for new competitors.
    # so we set it to 0

    # Collision Mechanics
    POWER_LOSS_FACTOR: float = 0.5

    LIGHT_ROBOT: UnitConfig = field(default=LIGHT_CONFIG)

    HEAVY_ROBOT: UnitConfig = field(default=HEAVY_CONFIG)

    @classmethod
    def get_unit_config(cls, unit_type: str) -> UnitConfig:
        if unit_type == "LIGHT":
            return cls.LIGHT_ROBOT
        else:
            return cls.HEAVY_ROBOT

    @classmethod
    def from_dict(cls, data):
        light_robot = UnitConfig(**data["ROBOTS"]["LIGHT"])
        heavy_robot = UnitConfig(**data["ROBOTS"]["HEAVY"])
        return cls(LIGHT_ROBOT=light_robot, HEAVY_ROBOT=heavy_robot)
