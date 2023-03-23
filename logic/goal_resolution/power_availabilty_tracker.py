import numpy as np

from dataclasses import dataclass
from typing import List, Dict

from objects.actions.action_plan import PowerRequest
from objects.actors.factory import Factory


@dataclass
class PowerAvailabilityTracker:
    factories: List[Factory]

    def __post_init__(self) -> None:
        self.t = self.factories[0].center_tc.t
        self.power_available = self._get_init_power_available()

    def _get_init_power_available(self) -> Dict[Factory, np.ndarray]:
        start_size_power_available = 20

        return {
            factory: self._get_init_power_available_factory(factory, n=start_size_power_available)
            for factory in self.factories
        }

    @staticmethod
    def _get_init_power_available_factory(factory: Factory, n: int) -> np.ndarray:
        # TODO add the expected effect of Lichen
        expected_increase_power = np.arange(n) * factory.daily_charge
        return expected_increase_power + factory.power

    def get_power_available(self, factory: Factory, t: int) -> int:
        array_index = self._get_array_index(t)
        if array_index >= len(self.power_available[factory]):
            self._extend_size_power_available(factory, new_size=array_index + 1)

        onwards_power_available = self.power_available[factory][array_index:]
        return min(onwards_power_available)

    def _extend_size_power_available(self, factory: Factory, new_size: int) -> None:
        power_available = self.power_available[factory]

        nr_required_extension_steps = new_size - len(power_available)
        expected_increase_power = np.arange(1, nr_required_extension_steps + 1) * factory.daily_charge
        last_power_available = power_available[-1]
        available_power_to_add = expected_increase_power + last_power_available

        new_power_available = np.append(power_available, available_power_to_add)
        self.power_available[factory] = new_power_available

    def update_power_available(self, power_requests: List[PowerRequest]) -> None:
        for power_request in power_requests:
            array_index = self._get_array_index(power_request.t)
            factory = power_request.factory
            if array_index >= len(self.power_available[factory]):
                self._extend_size_power_available(factory, new_size=array_index + 1)

            self.power_available[factory][array_index:] -= power_request.p

    def _get_array_index(self, t) -> int:
        return t - self.t
