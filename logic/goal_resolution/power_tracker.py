from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from objects.actions.action_plan import PowerRequest
from objects.actors.factory import Factory


class PowerTracker:
    """Keeps track of the current and future power levels of all factories. This is used to plan ahead whether units can
    request power at specific time steps. If a unit is scheduled to receive power in a future time step, a different
    unit can not request this power.
    """

    def __init__(self, factories: Iterable[Factory]) -> None:

        self.factories = list(factories)
        self.t = self.factories[0].center_tc.t
        self.power_available = self._get_init_power_available()

    def __copy__(self) -> PowerTracker:
        new = PowerTracker(self.factories)
        new.power_available = {
            factory: power_available.copy() for factory, power_available in self.power_available.items()
        }
        return new

    def _get_init_power_available(self) -> Dict[Factory, np.ndarray]:
        start_size_power_available = 20

        return {
            factory: self._get_init_power_available_factory(factory, n=start_size_power_available)
            for factory in self.factories
        }

    @staticmethod
    def _get_init_power_available_factory(factory: Factory, n: int) -> np.ndarray:
        # TODO add the effect of expected Lichen
        expected_increase_power = np.arange(n) * factory.expected_power_gain
        return expected_increase_power + factory.power

    def get_power_available(self, factory: Factory, t: int) -> int:
        """Get the power of the factory that will be available at time step t for request (not promised to other units).

        Args:
            factory: The Factory from which to request power
            t: Time step to request power.

        Returns:
            The available power.
        """
        array_index = self._get_array_index(t)
        if array_index >= len(self.power_available[factory]):
            self._extend_size_power_available(factory, new_size=array_index + 1)

        onwards_power_available = self.power_available[factory][array_index:]
        return min(onwards_power_available)

    def _extend_size_power_available(self, factory: Factory, new_size: int) -> None:
        power_available = self.power_available[factory]

        nr_required_extension_steps = new_size - len(power_available)
        expected_increase_power = np.arange(1, nr_required_extension_steps + 1) * factory.expected_power_gain
        last_power_available = power_available[-1]
        available_power_to_add = expected_increase_power + last_power_available

        new_power_available = np.append(power_available, available_power_to_add)
        self.power_available[factory] = new_power_available

    def add_power_requests(self, power_requests: Iterable[PowerRequest]) -> None:
        """Update power available given the new supplied power_requests.

        Args:
            power_requests: Iterable of PowerRequests to add.
        """
        for power_request in power_requests:
            array_index = self._get_array_index(power_request.t)
            factory = power_request.factory
            if array_index >= len(self.power_available[factory]):
                self._extend_size_power_available(factory, new_size=array_index + 1)

            self.power_available[factory][array_index:] -= power_request.p

    def remove_power_requests(self, power_requests: Iterable[PowerRequest]) -> None:
        """Update power available by retracting previous power requests.

        Args:
            power_requests: Iterable of PowerRequests to remove.
        """
        for power_request in power_requests:
            array_index = self._get_array_index(power_request.t)
            factory = power_request.factory
            self.power_available[factory][array_index:] += power_request.p

    def _get_array_index(self, t) -> int:
        return t - self.t
