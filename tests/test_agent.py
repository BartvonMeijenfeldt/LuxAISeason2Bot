import unittest

from agent import Agent
from objects.actors.factory import Factory
from objects.actors.unit import Unit
from objects.actions.unit_action import DigAction
from objects.cargo import UnitCargo
from objects.coordinate import TimeCoordinate as TC
from lux.config import EnvConfig


ENV_CFG = EnvConfig()
LIGHT_CFG = ENV_CFG.ROBOTS["LIGHT"]
HEAVY_CFG = ENV_CFG.ROBOTS["HEAVY"]


class TestSortedActors(unittest.TestCase):
    def test_base_5_actors(self):
        agent = Agent("player_0", ENV_CFG)
        factory = Factory(
            team_id=0, unit_id="0", power=0, cargo=UnitCargo(), strain_id=0, center_tc=TC(5, 5, 0), env_cfg=ENV_CFG
        )
        light_unit = Unit(
            team_id=0,
            unit_id="0",
            power=500,
            cargo=UnitCargo(),
            unit_type="LIGHT",
            tc=TC(5, 5, 0),
            unit_cfg=LIGHT_CFG,
            action_queue=[],
        )

        light_unit_with_action_queue = Unit(
            team_id=0,
            unit_id="0",
            power=500,
            cargo=UnitCargo(),
            unit_type="LIGHT",
            tc=TC(5, 5, 0),
            unit_cfg=LIGHT_CFG,
            action_queue=[DigAction()],
        )

        heavy_unit = Unit(
            team_id=0,
            unit_id="0",
            power=500,
            cargo=UnitCargo(),
            unit_type="HEAVY",
            tc=TC(5, 5, 0),
            unit_cfg=HEAVY_CFG,
            action_queue=[],
        )

        heavy_unit_with_action_queue = Unit(
            team_id=0,
            unit_id="0",
            power=500,
            cargo=UnitCargo(),
            unit_type="HEAVY",
            tc=TC(5, 5, 0),
            unit_cfg=HEAVY_CFG,
            action_queue=[DigAction()],
        )

        unsorted_actors = [light_unit, factory, light_unit_with_action_queue, heavy_unit, heavy_unit_with_action_queue]
        sorted_actors = agent._get_sorted_actors(unsorted_actors)
        expected_sorted_actors = [
            factory,
            heavy_unit_with_action_queue,
            heavy_unit,
            light_unit_with_action_queue,
            light_unit,
        ]

        self.assertEqual(sorted_actors, expected_sorted_actors)


if __name__ == "__main__":
    unittest.main()
