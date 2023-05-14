from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class FactionInfo:
    color: str = "none"
    alt_color: str = "red"
    faction_id: int = -1


class FactionTypes(Enum):
    Null = FactionInfo(color="gray", faction_id=0)
    AlphaStrike = FactionInfo(color="yellow", faction_id=1)
    MotherMars = FactionInfo(color="green", faction_id=2)
    TheBuilders = FactionInfo(color="blue", faction_id=3)
    FirstMars = FactionInfo(color="red", faction_id=4)


class Team:
    def __init__(
        self,
        team_id: int,
        agent: str,
        faction: Optional[FactionTypes] = None,
        water=0,
        metal=0,
        factories_to_place=0,
        factory_strains=[],
        place_first=False,
        bid=0,
    ) -> None:
        self.faction = faction
        self.team_id = team_id
        # the key used to differentiate ownership of things in state
        self.agent = agent

        self.water = water
        self.metal = metal
        self.factories_to_place = factories_to_place
        self.factory_strains = factory_strains
        # whether this team gets to place factories down first or not. The bid winner has this set to True.
        # If tied, player_0's team has this True
        self.place_first = place_first
