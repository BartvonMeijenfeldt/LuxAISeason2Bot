import unittest

from typing import List
from objects.actors.factory import Factory
from objects.actions.action_plan import PowerRequest as PR
from logic.goal_resolution.power_availabilty_tracker import PowerAvailabilityTracker as FPAT
from lux.config import EnvConfig
from tests.generate_game_state import get_state, FactoryPositions, FactoryPos


ENV_CFG = EnvConfig()
DAILY_CHARGE = ENV_CFG.FACTORY_CHARGE


class TestPowerAvailabilityTracker(unittest.TestCase):
    @staticmethod
    def _get_factories(factory_positions: FactoryPositions, t: int, board_width: int = 9) -> List[Factory]:

        state = get_state(factory_positions=factory_positions, real_env_steps=t, board_width=board_width)
        return state.board.player_factories

    def test_current_power(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)

        test_t = t
        expected_available_power = init_power

        power_availability_tracker = FPAT(factories)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_next_power(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)

        power_availability_tracker = FPAT(factories)

        test_t = t + 1
        expected_available_power = init_power + DAILY_CHARGE

        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)
        self.assertEqual(expected_available_power, available_power)

    def test_far_away_power(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)

        power_availability_tracker = FPAT(factories)

        nr_steps_ahead = 25
        test_t = t + nr_steps_ahead
        expected_available_power = init_power + nr_steps_ahead * DAILY_CHARGE

        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)
        self.assertEqual(expected_available_power, available_power)

    def test_next_power_current_step_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        power_requests = [PR(factories[0], t=t, p=5 * DAILY_CHARGE)]

        test_t = t + 1
        expected_available_power = init_power - 5 * DAILY_CHARGE + DAILY_CHARGE

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_current_step_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        power_requests = [PR(factories[0], t=t, p=5 * DAILY_CHARGE)]

        test_t = t
        expected_available_power = init_power - 5 * DAILY_CHARGE

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_next_step_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        power_requests = [PR(factories[0], t=t + 1, p=init_power)]

        test_t = t
        expected_available_power = DAILY_CHARGE
        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_future_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        power_requests = [PR(factories[0], t=t + 5, p=init_power)]

        test_t = t
        expected_available_power = 5 * DAILY_CHARGE

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_multiple_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        daily_power_taken = DAILY_CHARGE + 10
        nr_days_power_taken = 5
        power_requests = [
            PR(factories[0], t=t_take, p=daily_power_taken) for t_take in range(t + 1, t + 1 + nr_days_power_taken)
        ]

        test_t = t
        expected_available_power = init_power - (daily_power_taken - DAILY_CHARGE) * 5

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_power_taken_far_away(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        nr_days_in_future = 200
        all_power_available = nr_days_in_future * DAILY_CHARGE + init_power
        power_requests = [PR(factories[0], t=t + nr_days_in_future, p=all_power_available)]

        test_t = t
        expected_available_power = 0

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_current_power_too_much_power_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        nr_days = 2
        daily_power_taken = DAILY_CHARGE + 8 * DAILY_CHARGE
        power_requests = [PR(factories[0], t=t_take, p=daily_power_taken) for t_take in range(t + 1, t + 1 + nr_days)]

        test_t = t
        expected_available_power = init_power - 2 * 8 * DAILY_CHARGE

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)

    def test_next_power_before_and_after_taken(self):
        t = 1
        init_power = 10 * DAILY_CHARGE
        factory_positions = FactoryPositions(player=[FactoryPos(x=3, y=3, id=0, p=init_power)])
        factories = self._get_factories(factory_positions, t)
        daily_power_taken = 5 * DAILY_CHARGE
        power_requests = [PR(factories[0], t=t, p=daily_power_taken), PR(factories[0], t=t + 2, p=daily_power_taken)]

        test_t = t + 1
        expected_available_power = init_power - 2 * daily_power_taken + 2 * DAILY_CHARGE

        power_availability_tracker = FPAT(factories)
        power_availability_tracker.update_power_available(power_requests)
        available_power = power_availability_tracker.get_power_available(factories[0], t=test_t)

        self.assertEqual(expected_available_power, available_power)


if __name__ == "__main__":
    unittest.main()
