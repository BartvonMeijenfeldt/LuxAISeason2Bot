from dataclasses import dataclass
from operator import itemgetter
from typing import Optional

from objects.resource import NON_POWER_RESOURCES, Resource


@dataclass
class Cargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    def get_resource(self, resource: Resource) -> int:
        if resource == Resource.ICE:
            return self.ice
        elif resource == Resource.ORE:
            return self.ore
        elif resource == Resource.WATER:
            return self.water
        elif resource == Resource.METAL:
            return self.metal
        else:
            raise ValueError("Unexpexcted resoruce")

    @property
    def main_resource(self) -> Optional[Resource]:
        resources_and_quantities = [(resource, self.get_resource(resource)) for resource in NON_POWER_RESOURCES]
        max_resource_and_quantity = max(resources_and_quantities, key=itemgetter(1))
        if max_resource_and_quantity[1]:
            return max_resource_and_quantity[0]

        return None

    @property
    def total(self) -> int:
        return self.ice + self.ore + self.water + self.metal
