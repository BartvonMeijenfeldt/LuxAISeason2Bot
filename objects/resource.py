from enum import Enum


class Resource(Enum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4


NON_POWER_RESOURCES = [resource for resource in Resource if resource != Resource.POWER]
