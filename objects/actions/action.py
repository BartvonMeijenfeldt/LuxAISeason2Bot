from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass
class Action(metaclass=ABCMeta):
    @property
    @abstractmethod
    def requested_power(self) -> int:
        ...

    @abstractmethod
    def to_lux_output(self):
        ...
