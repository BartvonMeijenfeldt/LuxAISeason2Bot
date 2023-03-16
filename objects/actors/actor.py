from dataclasses import dataclass

from objects.cargo import UnitCargo


@dataclass
class Actor:
    team_id: int
    unit_id: str
    power: int
    cargo: UnitCargo
