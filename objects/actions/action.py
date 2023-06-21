from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class Action(metaclass=ABCMeta):
    """An action of actor that can be communicated to the environment. In the case of a unit this can represent multiple
    atomic actions, since units are allowed to repeat actions to efficiently use the action queue."""

    @property
    @abstractmethod
    def requested_power(self) -> int:
        """Requested power from factory to carry out the action."""
        ...

    @abstractmethod
    def to_lux_output(self):
        """Convert the action to the output the Lux environment expects."""
        ...
