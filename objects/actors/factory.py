from __future__ import annotations

import numpy as np

from typing import Tuple, TYPE_CHECKING
from itertools import product
from dataclasses import dataclass
from collections import Counter

from objects.cargo import Cargo
from objects.actions.factory_action import WaterAction
from objects.actors.actor import Actor
from objects.coordinate import TimeCoordinate, Coordinate, CoordinateList
from logic.goals.goal import GoalCollection
from logic.goals.factory_goal import BuildHeavyGoal, BuildLightGoal, WaterGoal, FactoryNoGoal, FactoryGoal
from distances import (
    get_min_distance_between_positions,
    get_min_distances_between_positions,
    get_closest_pos_between_pos_and_positions,
    get_positions_on_optimal_path_between_pos_and_pos,
)
from image_processing import get_islands
from positions import init_empty_positions, get_neighboring_positions
from lux.config import EnvConfig, LIGHT_CONFIG, HEAVY_CONFIG
from config import CONFIG

if TYPE_CHECKING:
    from objects.game_state import GameState
    from objects.board import Board


@dataclass
class Factory(Actor):
    strain_id: int
    center_tc: TimeCoordinate
    env_cfg: EnvConfig
    radius = 1

    def __post_init__(self) -> None:
        self._set_unit_state_variables()

    def _set_unit_state_variables(self) -> None:
        self.x = self.center_tc.x
        self.y = self.center_tc.y
        self.positions = np.array([[self.x + x, self.y + y] for x, y in product(range(-1, 2), range(-1, 2))])

    def update_state(self, center_tc: TimeCoordinate, power: int, cargo: Cargo) -> None:
        self.center_tc = center_tc
        self.power = power
        self.cargo = cargo
        self._set_unit_state_variables()

    def set_positions(self, board: Board) -> None:
        self.lichen_positions = np.argwhere(board.lichen_strains == self.strain_id)
        self.connected_lichen_positions = self._get_connected_lichen_positions(board)
        self.spreadable_lichen_positions = self._get_spreadable_lichen_positions(board, self.connected_lichen_positions)
        self.can_spread_positions = np.append(self.positions, self.spreadable_lichen_positions, axis=0)
        self.can_spread_to_positions = self._get_can_spread_to_positions(board, self.can_spread_positions)
        self.max_nr_tiles_to_water = len(self.lichen_positions) + len(self.can_spread_to_positions)
        self.connected_positions = self._get_empty_or_lichen_connected_positions(board)
        self.nr_connected_positions = len(self.connected_positions)
        self.rubble_positions_pathing = self._get_rubble_positions_to_clear_for_resources(board)
        self.rubble_positions_values_for_lichen = self._get_rubble_positions_to_clear_for_lichen(board)

    def _get_connected_lichen_positions(self, board: Board) -> np.ndarray:
        own_lichen = board.lichen_strains == self.strain_id
        return self._get_connected_positions(own_lichen)

    def _get_spreadable_lichen_positions(self, board: Board, connected_lichen_positions: np.ndarray) -> np.ndarray:
        if not connected_lichen_positions.shape[0]:
            return init_empty_positions()

        lichen_available = board.lichen[connected_lichen_positions[:, 0], connected_lichen_positions[:, 1]]
        can_spread_mask = lichen_available >= EnvConfig.MIN_LICHEN_TO_SPREAD
        return connected_lichen_positions[can_spread_mask]

    def _get_empty_or_lichen_connected_positions(self, board: Board) -> np.ndarray:
        is_empty_or_own_lichen = (board.lichen_strains == self.strain_id) | (board.is_empty_array)
        return self._get_connected_positions(is_empty_or_own_lichen)

    def _get_connected_positions(self, array: np.ndarray) -> np.ndarray:
        islands = get_islands(array)

        connected_positions = init_empty_positions()
        for single_island in islands:
            if get_min_distance_between_positions(self.positions, single_island) == 1:
                connected_positions = np.append(connected_positions, single_island, axis=0)

        return connected_positions

    def _get_can_spread_to_positions(self, board: Board, can_spread_positions: np.ndarray) -> np.ndarray:
        neighboring_positions = get_neighboring_positions(can_spread_positions)
        is_empty_mask = board.are_empty_postions(neighboring_positions)
        return neighboring_positions[is_empty_mask]

    def _get_rubble_positions_to_clear_for_resources(self, board: Board) -> Counter[Tuple[int, int]]:
        closest_2_ice_positions = board.get_n_closest_ice_positions_to_factory(self, n=2)
        closest_2_ore_positions = board.get_n_closest_ore_positions_to_factory(self, n=2)
        closest_resource_positions = np.append(closest_2_ice_positions, closest_2_ore_positions, axis=0)

        positions = init_empty_positions()

        for pos in closest_resource_positions:
            closest_factory_pos = get_closest_pos_between_pos_and_positions(pos, self.positions)
            optimal_positions = get_positions_on_optimal_path_between_pos_and_pos(pos, closest_factory_pos, board)
            positions = np.append(positions, optimal_positions, axis=0)

        rubble_mask = board.are_rubble_positions(positions)
        rubble_positions = positions[rubble_mask]

        rubble_value_dict = Counter({tuple(pos): CONFIG.RUBBLE_VALUE_CLEAR_FOR_RESOURCE for pos in rubble_positions})

        return rubble_value_dict

    def _get_rubble_positions_to_clear_for_lichen(self, board: Board) -> Counter[Tuple[int, int]]:
        rubble_positions, distances = self._get_rubble_positions_and_distances_within_max_distance(board)
        values = self._get_rubbe_positions_to_clear_for_lichen_score(distances)
        rubble_value_dict = Counter({tuple(pos): value for pos, value in zip(rubble_positions, values)})

        return rubble_value_dict

    def _get_rubbe_positions_to_clear_for_lichen_score(self, distances: np.ndarray) -> np.ndarray:
        base_score = CONFIG.RUBBLE_VALUE_CLEAR_FOR_LICHEN_BASE
        distance_penalty = CONFIG.RUBBLE_VALUE_CLEAR_FOR_LICHEN_DISTANCE_PENALTY
        return base_score - distance_penalty * distances

    def _get_rubble_positions_and_distances_within_max_distance(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        distances = get_min_distances_between_positions(board.rubble_positions, self.can_spread_positions)
        valid_distance_mask = distances < CONFIG.RUBBLE_CLEAR_FOR_LICHEN_MAX_DISTANCE
        rubble_postions_within_max_distance = board.rubble_positions[valid_distance_mask]
        distances_within_max_distance = distances[valid_distance_mask]
        return rubble_postions_within_max_distance, distances_within_max_distance

    def __hash__(self) -> int:
        return hash(self.unit_id)

    def __eq__(self, __o: Factory) -> bool:
        return self.unit_id == __o.unit_id

    def min_distance_to_connected_positions(self, positions: np.ndarray) -> int:
        rel_positions = np.append(self.positions, self.connected_positions, axis=0)
        return get_min_distance_between_positions(rel_positions, positions)

    def generate_goals(self, game_state: GameState, can_build: bool = True) -> GoalCollection:
        water_cost = self.water_cost()
        goals = []

        if can_build and self.can_build_heavy and game_state.player_nr_heavies / game_state.player_nr_factories < 1:
            goals.append(BuildHeavyGoal(self))

        elif (
            can_build
            and self.can_build_light
            and (game_state.player_nr_lights / game_state.player_nr_factories <= 10 or self.power > 1000)
        ):
            goals.append(BuildLightGoal(self))

        elif self.cargo.water - water_cost > 50 and water_cost < 5:
            goals.append(WaterGoal(self))

        elif game_state.env_steps > 750 and self.can_water() and self.cargo.water - water_cost > game_state.steps_left:
            goals.append(WaterGoal(self))

        goals += self._get_dummy_goals()
        return GoalCollection(goals)

    def _get_dummy_goals(self) -> list[FactoryGoal]:
        return [FactoryNoGoal(self)]

    @property
    def daily_charge(self) -> int:
        return self.env_cfg.FACTORY_CHARGE

    @property
    def can_build_heavy(self) -> bool:
        return self.power >= HEAVY_CONFIG.POWER_COST and self.cargo.metal >= HEAVY_CONFIG.METAL_COST

    @property
    def can_build_light(self) -> bool:
        return self.power >= LIGHT_CONFIG.POWER_COST and self.cargo.metal >= LIGHT_CONFIG.METAL_COST

    def water_cost(self) -> int:
        return WaterAction.get_water_cost(self)

    def can_water(self):
        return self.cargo.water >= self.water_cost()

    @property
    def pos_slice(self) -> Tuple[slice, slice]:
        return self.x_slice, self.y_slice

    @property
    def x_slice(self) -> slice:
        return slice(self.center_tc.x - self.radius, self.center_tc.x + self.radius + 1)

    @property
    def y_slice(self) -> slice:
        return slice(self.center_tc.y - self.radius, self.center_tc.y + self.radius + 1)

    @property
    def pos_x_range(self):
        return range(self.center_tc.x - self.radius, self.center_tc.x + self.radius + 1)

    @property
    def pos_y_range(self):
        return range(self.center_tc.y - self.radius, self.center_tc.y + self.radius + 1)

    @property
    def coordinates(self) -> CoordinateList:
        return CoordinateList([Coordinate(x, y) for x in self.pos_x_range for y in self.pos_y_range])

    def __repr__(self) -> str:
        return f"Factory[id={self.unit_id}, center={self.center_tc.xy}]"
