from dataclasses import dataclass


@dataclass
class UnitCargo:
    ice: int = 0
    ore: int = 0
    water: int = 0
    metal: int = 0

    @property
    def total(self) -> int:
        return self.ice + self.ore + self.water + self.metal
