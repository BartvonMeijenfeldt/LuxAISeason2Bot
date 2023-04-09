from dataclasses import dataclass

from objects.resource import Resource


@dataclass
class Cargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    def get_resource(self, resource: Resource) -> int:
        if resource.name == "ICE":
            return self.ice
        elif resource.name == "ORE":
            return self.ore
        elif resource.name == "WATER":
            return self.water
        elif resource.name == "METAL":
            return self.metal
        else:
            raise ValueError("Unexpexcted resoruce")

    @property
    def total(self) -> int:
        return self.ice + self.ore + self.water + self.metal
