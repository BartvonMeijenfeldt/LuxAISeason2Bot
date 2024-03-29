from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from math import floor
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Tuple

import numpy as np

from config import CONFIG
from exceptions import (
    ActorFoundNoValidGoalError,
    FactorySchedulerNoValidGoalFoundError,
    NoValidGoalFoundForStrategyError,
)
from logic.goals.factory_goal import (
    BuildHeavyGoal,
    BuildLightGoal,
    FactoryGoal,
    FactoryNoGoal,
    WaterGoal,
)
from logic.goals.unit_goal import (
    ClearRubbleGoal,
    CollectGoal,
    CollectIceGoal,
    CollectOreGoal,
    DestroyLichenGoal,
    DigGoal,
    SupplyPowerGoal,
    TransferIceGoal,
    TransferOreGoal,
    UnitGoal,
)
from lux.config import HEAVY_CONFIG, LIGHT_CONFIG, EnvConfig
from objects.actions.factory_action import WaterAction
from objects.actions.factory_action_plan import FactoryActionPlan
from objects.actions.unit_action import TransferAction
from objects.actions.unit_action_plan import UnitActionPlan
from objects.actors.actor import Actor
from objects.cargo import Cargo
from objects.coordinate import Coordinate, CoordinateList, TimeCoordinate
from objects.resource import Resource
from utils.distances import (
    get_closest_pos_between_pos_and_positions,
    get_min_distance_between_pos_and_positions,
    get_min_distance_between_positions,
    get_min_distances_between_positions,
    get_positions_on_optimal_path_between_pos_and_pos,
)
from utils.image_processing import get_islands
from utils.positions import (
    append_positions,
    get_neighboring_positions,
    init_empty_positions,
    positions_to_set,
)

if TYPE_CHECKING:
    from logic.goal_resolution.scheduler import ScheduleInfo
    from objects.actors.unit import Unit
    from objects.board import Board
    from objects.game_state import GameState


class Strategy(Enum):
    IMMEDIATELY_RETURN_ICE = auto()
    INCREASE_LICHEN_TILES = auto()
    INCREASE_LICHEN = auto()
    COLLECT_ICE = auto()
    COLLECT_ORE = auto()
    ATTACK_OPPONENT = auto()
    CLEAR_RUBBLE_AROUND_BASE = auto()
    DEFEND_LICHEN_TILE = auto()


logger = logging.getLogger(__name__)


@dataclass(eq=False)
class Factory(Actor):
    """Factory building that can spawn robots and water lichen. In my implementation I also made the factory decide the
    strategical focus of the units belonging to the factory

    Args:
        strain_id: Unique identifier of the factory, the id of the lichen strain it produces.
        center_tc: The center time coordinate of the 3x3 blocks the factory takes up.
        env_cfg: Config of the environment.
        radius: Radius of the factory.
        units: The units belonging to the factory
        goal: The goal of the factory
        private_action_plan: The action plan of the factory.
        positions_set: Whether the calculations of positions that must only be calculated once a game have been made.
    """

    # TODO, split Factory Actor logic and Factory object
    strain_id: int
    center_tc: TimeCoordinate
    env_cfg: EnvConfig
    radius = 1
    units: set[Unit] = field(init=False, default_factory=set)
    goal: Optional[FactoryGoal] = field(init=False, default=None)
    private_action_plan: FactoryActionPlan = field(init=False)
    positions_set: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.id = int(self.unit_id[8:])
        self.x = self.center_tc.x
        self.y = self.center_tc.y
        self.positions = np.array([[self.x + x, self.y + y] for x, y in product(range(-1, 2), range(-1, 2))])
        self.private_action_plan = FactoryActionPlan(self, [])
        self._set_unit_state_variables()

    def _set_unit_state_variables(self) -> None:
        self.is_scheduled = False
        self.distress_signal_can_not_be_handled = False
        self.nr_schedule_failures_this_step = 0
        self._set_resource_state_variables()

    def _set_resource_state_variables(self) -> None:
        self.water = self.cargo.water
        self.ice = self.cargo.ice
        self.ore = self.cargo.ore
        self.metal = self.cargo.metal
        self.water_after_processing = self._get_water_after_processing()

    def _get_water_after_processing(self) -> int:
        return self.water + floor(min(self.ice, EnvConfig.FACTORY_PROCESSING_RATE_WATER) / EnvConfig.ICE_WATER_RATIO)

    def update_state(self, center_tc: TimeCoordinate, power: int, cargo: Cargo) -> None:
        """Update the state of the Factory

        Args:
            center_tc: The center time coordinate of the 3x3 blocks the factory takes up.
            power: The power at the factory.
            cargo: The cargo at the factory.
        """
        self.center_tc = center_tc
        self.power = power
        self.cargo = cargo
        self._set_unit_state_variables()

    def remove_units_not_in_obs(self, obs_units: set[Unit]) -> None:
        """Remove the units assigned to the factory that are not in the observation any more (because they perished last
        time step).

        Args:
            obs_units: All the units in the observation.
        """
        units_to_remove = self.units.difference(obs_units)
        for unit in units_to_remove:
            unit.remove_goal_and_private_action_plan()

        self.units.intersection_update(obs_units)

    def set_positions(self, board: Board) -> None:
        if not self.positions_set:
            self._set_positions_once(board)
            self.positions_set = True

        if self in board.player_factories and board.opp_factories:
            self._rubble_positions_to_clear_for_ore = self._get_positions_to_clear_for_resource_pathing(
                board, self.ore_positions_distance_sorted[:5]
            )
            self._rubble_positions_to_clear_for_ice = self._get_positions_to_clear_for_resource_pathing(
                board, self.ice_positions_distance_sorted[:5]
            )

        self.lichen_positions = np.argwhere(board.lichen_strains == self.strain_id)
        self.lichen_positions_set = positions_to_set(self.lichen_positions)

        self.sorted_threaths_invaders = [
            unit
            for unit in board.opp_units
            if unit.tc.xy in self.lichen_positions_set
            or self.min_distance_to_c(unit.tc) <= CONFIG.DISTANCE_TO_FACTORY_CONSIDERED_INVADER
        ]
        self.sorted_threaths_invaders.sort(
            key=lambda unit: self.min_distance_to_c(unit.tc) - CONFIG.INVADER_HEAVY_PRIORITY * unit.is_heavy
        )

        self.connected_lichen_positions = self._get_connected_lichen_positions(board)
        self.spreadable_lichen_positions = self._get_spreadable_lichen_positions(board)
        self.non_spreadable_connected_lichen_positions = self._get_not_spreadable_connected_lichen_positions(board)
        self.can_spread_positions = append_positions(self.positions, self.spreadable_lichen_positions)

        self.can_spread_to_positions = self._get_can_spread_to_positions(board, self.can_spread_positions)
        self.connected_positions = self._get_empty_or_lichen_connected_positions(board)

        self.rubble_positions_next_to_can_spread_pos = self._get_rubble_positions_next_to_can_spread_pos(board)
        self.rubble_positions_next_to_can_not_spread_lichen = self._get_rubble_next_to_can_not_spread_lichen(board)
        self.closest_rubble_positions_within_4_distance_set = self._get_rubble_within_4_distance_to_base(board)
        self.rubble_positions_next_to_connected_or_self = self._get_rubble_next_to_connected_or_self(board)
        self.rubble_positions_next_to_connected_or_self_set = positions_to_set(
            self.rubble_positions_next_to_connected_or_self
        )

        self.nr_connected_lichen_tiles = len(self.connected_lichen_positions)
        self.min_lichen_in_connected_lichens = self._get_min_lichen_in_connected_lichens(board)
        self.nr_connected_lichen_tiles_after_not_watering = self._get_nr_connected_lichen_after_not_watering(board)
        self.nr_connected_positions = len(self.connected_positions)
        self.max_nr_tiles_to_water = len(self.connected_lichen_positions) + len(self.can_spread_to_positions)

    def _get_min_lichen_in_connected_lichens(self, board: Board) -> int:
        if not self.lichen_positions_set:
            return 0

        return min(board.get_lichen_at_pos(lichen_pos) for lichen_pos in self.lichen_positions_set)

    def _get_nr_connected_lichen_after_not_watering(self, board: Board) -> int:
        return sum(1 for lichen_pos in self.lichen_positions_set if board.get_lichen_at_pos(lichen_pos) > 1)

    def _set_positions_once(self, board: Board) -> None:
        self.ice_positions_distance_sorted = self._get_positions_distance_sorted(board.ice_positions)
        self.ore_positions_distance_sorted = self._get_positions_distance_sorted(board.ore_positions)
        self.connected_ice_coordinates = self._get_connected_ice_coordinates(board)

    def _get_positions_distance_sorted(self, positions: np.ndarray) -> np.ndarray:
        distances = get_min_distances_between_positions(positions, self.positions)
        sorted_indexes = np.argsort(distances)
        sorted_distance_positions = positions[sorted_indexes]
        return sorted_distance_positions

    def _get_connected_ice_coordinates(self, board: Board) -> list[Coordinate]:
        distances = get_min_distances_between_positions(board.ice_positions, self.positions)
        connected_mask = distances == 1
        connected_ice_positions = board.ice_positions[connected_mask]
        return [Coordinate(*pos) for pos in connected_ice_positions]

    def get_rubble_positions_to_clear_for_ice(self, game_state: GameState) -> Set[Tuple]:
        """Get the rubble positions to clear to improve ice mining.

        Args:
            game_state: Current game state.

        Returns:
            The rubble positions to clear to improve ice mining.
        """
        rubble_positions_free = self._rubble_positions_to_clear_for_ice - game_state.positions_in_dig_goals
        return rubble_positions_free

    def get_rubble_positions_to_clear_for_ore(self, game_state: GameState) -> Set[Tuple]:
        """Get the rubble positions to clear to improve ore mining.

        Args:
            game_state: Current game state.

        Returns:
            The rubble positions to clear to improve ore mining.
        """
        rubble_positions_free = self._rubble_positions_to_clear_for_ore - game_state.positions_in_dig_goals
        return rubble_positions_free

    def nr_tiles_needed_to_grow_to_lichen_target(self, game_state: GameState) -> int:
        """Get the number of tiles needed to grow to the lichen target. The lichen target is based on the current water
        collection.

        Args:
            game_state: Current game state.

        Returns:
            The number of tiles needed to grow to the lichen target.
        """
        lichen_size_target = self.get_lichen_size_target_for_current_water_collection()
        lichen_size_target = max(CONFIG.MIN_TILES_GROWTH_TARGET, lichen_size_target)
        nr_connected_positions = self._get_nr_connected_positions_including_being_cleared(game_state)
        nr_tiles_needed = lichen_size_target - nr_connected_positions
        nr_tiles_needed = max(0, nr_tiles_needed)
        return nr_tiles_needed

    def _get_nr_connected_positions_including_being_cleared(self, game_state: GameState) -> int:
        # TODO, take into account if positions being cleared connects to other positions
        nr_positions_being_cleared = self._get_nr_positions_being_cleared_next_to_connected(game_state)
        return self.nr_connected_positions + nr_positions_being_cleared

    def _get_nr_positions_being_cleared_next_to_connected(self, game_state: GameState) -> int:
        return sum(
            1
            for unit in game_state.player_units
            if isinstance(unit.goal, ClearRubbleGoal)
            and unit.goal.dig_c.xy in self.rubble_positions_next_to_connected_or_self_set
        )

    def _get_connected_lichen_positions(self, board: Board) -> np.ndarray:
        own_lichen = board.lichen_strains == self.strain_id
        return self._get_connected_positions(own_lichen)

    def _get_spreadable_lichen_positions(self, board: Board) -> np.ndarray:
        if not self.connected_lichen_positions.shape[0]:
            return init_empty_positions()

        lichen_available = board.lichen[self.connected_lichen_positions[:, 0], self.connected_lichen_positions[:, 1]]
        can_spread_mask = lichen_available >= EnvConfig.MIN_LICHEN_TO_SPREAD
        return self.connected_lichen_positions[can_spread_mask]

    def _get_not_spreadable_connected_lichen_positions(self, board: Board) -> np.ndarray:
        if not self.connected_lichen_positions.shape[0]:
            return init_empty_positions()

        lichen_available = board.lichen[self.connected_lichen_positions[:, 0], self.connected_lichen_positions[:, 1]]
        can_not_spread_mask = lichen_available < EnvConfig.MIN_LICHEN_TO_SPREAD
        return self.connected_lichen_positions[can_not_spread_mask]

    def _get_empty_or_lichen_connected_positions(self, board: Board) -> np.ndarray:
        is_empty_or_own_lichen = (board.lichen_strains == self.strain_id) | (board.is_empty_array)
        return self._get_connected_positions(is_empty_or_own_lichen)

    def _get_connected_positions(self, array: np.ndarray) -> np.ndarray:
        islands = get_islands(array)

        connected_positions = init_empty_positions()
        for single_island in islands:
            if get_min_distance_between_positions(self.positions, single_island) == 1:
                connected_positions = append_positions(connected_positions, single_island)

        return connected_positions

    def _get_can_spread_to_positions(self, board: Board, can_spread_positions: np.ndarray) -> np.ndarray:
        neighboring_positions = get_neighboring_positions(can_spread_positions)
        is_empty_mask = board.are_empty_postions(neighboring_positions)
        return neighboring_positions[is_empty_mask]

    def _get_positions_to_clear_for_resource_pathing(self, board: Board, positions: np.ndarray) -> Set[Tuple]:
        for i, pos in enumerate(positions[:5]):
            if hasattr(board, "minable_positions_set") and tuple(pos) not in board.minable_positions_set:
                continue

            closest_factory_pos = get_closest_pos_between_pos_and_positions(pos, self.positions)

            if (
                i
                and Coordinate(*pos).distance_to(Coordinate(*closest_factory_pos))
                > CONFIG.MAX_DISTANCE_FOR_RESOURCE_CLEARING
            ):
                continue

            # TODO, make sure the path prefers to stay away from the opponents factory so as to not
            # Clear a path for him as well
            positions = get_positions_on_optimal_path_between_pos_and_pos(closest_factory_pos, pos, board)
            rubble_mask = board.are_rubble_positions(positions)
            rubble_positions = positions[rubble_mask]
            if rubble_positions.shape[0]:
                break
        else:
            rubble_positions = init_empty_positions()

        return positions_to_set(rubble_positions)

    def _get_rubble_positions_next_to_can_spread_pos(self, board: Board) -> np.ndarray:
        distances = get_min_distances_between_positions(board.rubble_positions, self.can_spread_positions)
        valid_distance_mask = distances == 1
        rubble_positions_next_to_can_spread_pos = board.rubble_positions[valid_distance_mask]
        return rubble_positions_next_to_can_spread_pos

    def _get_rubble_next_to_can_not_spread_lichen(self, board: Board) -> np.ndarray:
        if not board.rubble_positions.shape[0] or not self.non_spreadable_connected_lichen_positions.shape[0]:
            return init_empty_positions()

        distances_valid = get_min_distances_between_positions(
            board.rubble_positions, self.non_spreadable_connected_lichen_positions
        )
        valid_distance_mask = distances_valid == 1
        rubble_positions = board.rubble_positions[valid_distance_mask]
        return rubble_positions

    def _get_rubble_within_4_distance_to_base(self, board: Board) -> Set[Tuple]:
        if not board.rubble_positions.shape[0]:
            return set()

        distances = get_min_distances_between_positions(board.rubble_positions, self.positions)
        if min(distances) > 4:
            return set()

        valid_distance_mask = distances == distances.min()
        rubble_positions = board.rubble_positions[valid_distance_mask]
        return positions_to_set(rubble_positions)

    def _get_rubble_next_to_connected_or_self(self, board: Board) -> np.ndarray:
        if not board.rubble_positions.shape[0]:
            return init_empty_positions()

        connected_or_self_positions = append_positions(self.positions, self.connected_positions)

        distances = get_min_distances_between_positions(board.rubble_positions, connected_or_self_positions)
        valid_distance_mask = distances == 1
        rubble_positions = board.rubble_positions[valid_distance_mask]
        return rubble_positions

    def add_unit(self, unit: Unit) -> None:
        """Add unit to the units of the factory.

        Args:
            unit: Unit to add.
        """
        self.units.add(unit)

    def schedule_build_or_no_goal(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        """Schedule a build or no goal goal.

        Args:
            schedule_info: Schedule info.

        Returns:
            An action plan corresponding to the scheduled goal.
        """
        goal = self._get_build_or_no_goal(schedule_info.game_state)
        return self._generate_and_schedule_action_plan(goal, schedule_info)

    def _generate_and_schedule_action_plan(self, goal: FactoryGoal, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        action_plan = goal.generate_action_plan(schedule_info)
        self.goal = goal
        self.private_action_plan = action_plan
        return action_plan

    def _get_build_or_no_goal(self, game_state: GameState) -> FactoryGoal:
        if game_state.real_env_steps > CONFIG.LAST_STEP_UNIT_BUILDING:
            return FactoryNoGoal(self)

        if self.can_build_heavy:
            return BuildHeavyGoal(self)

        elif self.can_build_light and (
            self.nr_light_units < 15
            or (self.nr_light_units < 20 and self.nr_heavy_units > 1)
            or (self.nr_light_units < 30 and self.nr_heavy_units > 2)
        ):
            return BuildLightGoal(self)
        elif self.can_build_light and game_state.real_env_steps > 900:
            return BuildLightGoal(self)

        return FactoryNoGoal(self)

    def schedule_water_or_no_goal(self, schedule_info: ScheduleInfo) -> FactoryActionPlan:
        """Schedule a water or no goal goal

        Args:
            schedule_info: Schedule info.

        Returns:
            An action plan corresponding to the scheduled goal.
        """
        goal = self._get_water_or_no_goal(schedule_info.game_state)
        return self._generate_and_schedule_action_plan(goal, schedule_info)

    def _get_water_or_no_goal(self, game_state: GameState) -> FactoryGoal:
        return WaterGoal(self) if self._wants_to_add_water_goal(game_state) else FactoryNoGoal(self)

    def _wants_to_add_water_goal(self, game_state: GameState) -> bool:
        if self.water_after_processing - self.water_cost < EnvConfig.FACTORY_WATER_CONSUMPTION:
            return False

        if self.water_after_processing / self.water_cost > game_state.steps_left:
            return True

        water_available = self.water_after_processing - self.water_safety_level - self.water_cost
        if (
            game_state.real_env_steps > 200
            and water_available / self.water_cost > CONFIG.MIN_RATIO_WATER_WATER_COST_ALWAYS_GROW_LICHEN
        ):
            return True

        if water_available <= 0:
            return False

        target_lichen_size = self.get_lichen_size_target_for_current_water_collection()
        target_lichen_size = target_lichen_size * CONFIG.WATER_LICHEN_SIZE_FRACTION

        if self.watering_grows_lichen:
            return self.max_nr_tiles_to_water <= target_lichen_size

        if self.not_watering_shrinks_lichen and self.nr_connected_lichen_tiles_after_not_watering < target_lichen_size:
            return True

        return (
            self.min_lichen_in_connected_lichens < EnvConfig.MIN_LICHEN_TO_SPREAD
            and water_available / self.water_cost > CONFIG.MIN_RATIO_WATER_WATER_COST_MAINTAIN_LICHEN
        )

    @property
    def water_safety_level(self) -> int:
        """Minimum water level, this water is never used for watering."""
        safety_level = floor(CONFIG.MIN_WATER_SAFETY_LEVEL + CONFIG.WATER_SAFETY_SLOPE_PER_STEP * self.center_tc.t)
        safety_level = min(CONFIG.MAX_WATER_SAFETY_LEVEL, safety_level)
        return safety_level

    @property
    def watering_grows_lichen(self) -> bool:
        """Whether watering would result into growing the number of lichen tiles."""
        return self.max_nr_tiles_to_water > self.nr_connected_lichen_tiles

    @property
    def not_watering_shrinks_lichen(self) -> bool:
        """whether not watering would result into shrinking the number of lichen tiles."""
        return self.nr_connected_lichen_tiles_after_not_watering < self.nr_connected_lichen_tiles

    @property
    def water_cost(self):
        """The cost to water."""
        return WaterAction.get_water_cost(self)

    @property
    def light_units(self) -> list[Unit]:
        """Light units assigned to the factory."""
        return [unit for unit in self.units if unit.is_light]

    @property
    def heavy_units(self) -> list[Unit]:
        """Heavy units assigned to the factory."""
        return [unit for unit in self.units if unit.is_heavy]

    @property
    def nr_light_units(self) -> int:
        return len(self.light_units)

    @property
    def nr_heavy_units(self) -> int:
        return len(self.heavy_units)

    @property
    def can_build_light(self) -> bool:
        """Whether factory has enough power and metal to build a light."""
        return self.power >= LIGHT_CONFIG.POWER_COST and self.cargo.metal >= LIGHT_CONFIG.METAL_COST

    @property
    def can_build_heavy(self) -> bool:
        """Whether factory has enough power and metal to build a heavy."""
        return self.power >= HEAVY_CONFIG.POWER_COST and self.cargo.metal >= HEAVY_CONFIG.METAL_COST

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

    @property
    def power_including_units(self) -> float:
        """Sum of power of factory and all its units."""
        units_power = sum(unit.power for unit in self.units)
        power = self.power + units_power
        return power

    @property
    def expected_power_gain(self) -> int:
        """Expected power gain for next turns."""
        return self.calculate_power_gain(self.nr_connected_lichen_tiles)

    def get_expected_power_generation(self, game_state: GameState) -> int:
        expected_lichen_size = self.get_expected_lichen_size(game_state)
        return self.calculate_power_gain(expected_lichen_size)

    @staticmethod
    def calculate_power_gain(lichen_size: int) -> int:
        return EnvConfig.FACTORY_CHARGE + lichen_size * EnvConfig.POWER_PER_CONNECTED_LICHEN_TILE

    def get_incoming_ice_before_no_water(self, game_state: GameState) -> int:
        """Get the amount of incoming ice before the water runs out. Assumes the factory will not water in the mean time
        and the current units action plans and goals.

        Args:
            game_state: Current game state.

        Returns:
            The amount of incoming ice before the water runs out.
        """
        incoming_ice = 0
        for unit in self.scheduled_units:
            if not isinstance(unit.goal, CollectIceGoal) or isinstance(unit.goal, TransferIceGoal):
                continue

            for action in unit.private_action_plan.primitive_actions[: self.water]:
                if isinstance(action, TransferAction):
                    incoming_ice += unit.goal.quantity_ice_to_transfer(game_state)
                    break

        return incoming_ice

    def get_expected_lichen_size(self, game_state: GameState) -> int:
        """Get the expected lichen size. Assumes watering based on the current units goals and action plans, and that
        if the target is below the nr connected lichen tiles, that the connected lichen tiles will be maintained.

        Args:
            game_state: Current game state.

        Returns:
            The expected lichen size
        """
        lichen_size_target = self.get_lichen_size_target_for_current_water_collection()

        if lichen_size_target < self.nr_connected_lichen_tiles:
            return self.nr_connected_lichen_tiles

        nr_expected_connected_positions = self._get_nr_connected_positions_including_being_cleared(game_state)
        return min(nr_expected_connected_positions, lichen_size_target)

    def get_lichen_size_target_for_current_water_collection(self) -> int:
        """Get the expected lichen size target, given the current water collection.

        Returns:
            The expected lichen size target
        """
        water_collection_per_step = self.get_water_collection_per_step()
        tiles_target = floor(water_collection_per_step * EnvConfig.LICHEN_WATERING_COST_FACTOR) * 2  # Alternating water
        return tiles_target

    def get_expected_power_consumption(self) -> float:
        """Get the expected power consumptions based on the lights and heavies on the board, and the expected units to
        spawn (based on the expected metal in the short future).

        Returns:
            _description_
        """
        metal_in_factory = self.metal + self.ore / EnvConfig.ORE_METAL_RATIO
        metal_collection = self.get_metal_collection_per_step()
        metal_expected = metal_in_factory + metal_collection

        # Assume next unit just as likely to be light as heavy
        expected_nr_lights = self.nr_light_units + metal_expected / LIGHT_CONFIG.METAL_COST / 2
        expected_nr_heavies = self.nr_heavy_units + metal_expected / HEAVY_CONFIG.METAL_COST / 2

        lights_power_consumption = expected_nr_lights * CONFIG.EXPECTED_POWER_CONSUMPTION_LIGHT_PER_TURN
        heavies_power_consumption = expected_nr_heavies * CONFIG.EXPECTED_POWER_CONSUMPTION_HEAVY_PER_TURN

        expected_power_usage_per_step = lights_power_consumption + heavies_power_consumption
        return expected_power_usage_per_step

    def get_water_collection_per_step(self) -> float:
        """Get water collection per step, given the units current goals and action plans.

        Returns:
            The water colection per step.
        """
        ice_collection_per_step = self.get_ice_collection_per_step()
        water_collection_per_step = ice_collection_per_step / EnvConfig.ICE_WATER_RATIO
        return water_collection_per_step

    def get_ice_collection_per_step(self) -> float:
        """Get ice collection per step, given the units current goals and action plans.

        Returns:
            The water colection per step.
        """
        return sum(
            self._resource_collection_per_step(unit.goal)
            for unit in self.scheduled_units
            if isinstance(unit.goal, CollectIceGoal)
        )

    def get_metal_collection_per_step(self) -> float:
        ore_collection_per_step = self.get_ore_collection_per_step()
        metal_collection_per_step = ore_collection_per_step / EnvConfig.ORE_METAL_RATIO
        return metal_collection_per_step

    def get_ore_collection_per_step(self) -> float:
        return sum(
            self._resource_collection_per_step(unit.goal)
            for unit in self.scheduled_units
            if isinstance(unit.goal, CollectOreGoal)
        )

    def _resource_collection_per_step(self, goal: CollectGoal) -> float:
        # Assumes full battery

        unit = goal.unit

        if goal.is_supplied:
            nr_steps_moving = 0
            nr_steps_power_pickup = 0
            nr_steps_digging = unit.nr_digs_empty_to_full_cargo
        else:

            distance_resource_to_factory = self.min_distance_to_c(goal.dig_c)
            nr_steps_moving = 2 * distance_resource_to_factory
            nr_steps_power_pickup = 1

            power_available_for_digging = unit.battery_capacity - nr_steps_moving * unit.move_power_cost
            ratio_day_night = EnvConfig.DAY_LENGTH / EnvConfig.CYCLE_LENGTH
            average_power_charge = ratio_day_night * unit.recharge_power
            net_power_change_per_dig = unit.dig_power_cost - average_power_charge
            # This is an approximate value which will be different for day and night
            max_nr_digs = power_available_for_digging / net_power_change_per_dig
            nr_steps_digging = min(unit.nr_digs_empty_to_full_cargo, max_nr_digs)

        resource_collection = nr_steps_digging * unit.resources_gained_per_dig

        nr_steps_transfer = 1
        nr_steps = nr_steps_digging + nr_steps_moving + nr_steps_power_pickup + nr_steps_transfer

        resource_collection_per_step = resource_collection / nr_steps
        return resource_collection_per_step

    @property
    def has_unsupplied_heavy_collecting_next_to_factory_free_supply_c(self) -> bool:
        return any(True for _ in self.heavy_units_unsupplied_collecting_next_to_factory_free_supply_c)

    @property
    def heavy_units_unsupplied_collecting_next_to_factory_free_supply_c(self) -> List[Unit]:
        return [
            heavy
            for heavy in self.heavy_units
            if isinstance(heavy.goal, CollectGoal)
            and self.min_distance_to_c(heavy.goal.dig_c) == 1
            and not heavy.supplied_by
            and not any(self.min_distance_to_c(c) == 1 for c in self.coordinates_in_supply_c_goals)
        ]

    @property
    def coordinates_in_supply_c_goals(self) -> list[Coordinate]:
        return [unit.goal.supply_c for unit in self.units if isinstance(unit.goal, SupplyPowerGoal)]

    def min_distance_to_c(self, c: Coordinate) -> int:
        pos = np.array(c.xy)
        return get_min_distance_between_pos_and_positions(pos, self.positions)

    @property
    def has_unit_available(self) -> bool:
        return any(True for _ in self.available_units)

    @property
    def has_heavy_unit_available(self) -> bool:
        return any(True for _ in self.heavy_available_units)

    @property
    def has_light_unit_available(self) -> bool:
        return any(True for _ in self.light_available_units)

    @property
    def available_units(self) -> List[Unit]:
        return [
            unit
            for unit in self.units
            if unit.can_update_action_queue and not unit.private_action_plan and unit.can_be_assigned
        ]

    @property
    def heavy_available_units(self) -> List[Unit]:
        return [unit for unit in self.available_units if unit.is_heavy]

    @property
    def light_available_units(self) -> List[Unit]:
        return [unit for unit in self.available_units if unit.is_light]

    @property
    def scheduled_units(self) -> List[Unit]:
        return [unit for unit in self.units if unit.is_scheduled]

    @property
    def unscheduled_units(self) -> List[Unit]:
        return [unit for unit in self.units if not unit.is_scheduled]

    @property
    def units_with_ice(self) -> List[Unit]:
        return [unit for unit in self.units if unit.ice]

    @property
    def heavies_with_main_ore(self) -> List[Unit]:
        return [unit for unit in self.units if unit.is_heavy and unit.main_cargo == Resource.ORE]

    @property
    def heavies_not_having_ice_goal(self) -> List[Unit]:
        return [
            unit
            for unit in self.units
            if unit.is_heavy and not (isinstance(unit.goal, CollectIceGoal) or isinstance(unit.goal, TransferIceGoal))
        ]

    @property
    def lights_not_having_ice_goal(self) -> List[Unit]:
        return [
            unit
            for unit in self.units
            if unit.is_light and not (isinstance(unit.goal, CollectIceGoal) or isinstance(unit.goal, TransferIceGoal))
        ]

    @property
    def units_collecting_ice(self) -> List[Unit]:
        return [unit for unit in self.units if isinstance(unit.goal, CollectIceGoal)]

    def schedule_units(self, strategy: Strategy, schedule_info: ScheduleInfo) -> List[UnitGoal]:
        # TODO, make this a strategy/change structure
        if self.has_unsupplied_heavy_collecting_next_to_factory_free_supply_c and self.has_light_unit_available:
            try:
                return self._schedule_supply_goal_and_reschedule_receiving_unit(schedule_info)
            except ActorFoundNoValidGoalError:
                logger.debug("No schedule supply goal and reschedule receiving unit goal found")

        try:
            return [self._schedule_unit_on_strategy(strategy, schedule_info)]
        except Exception as e:
            logger.debug(e)
            self.nr_schedule_failures_this_step += 1
            raise NoValidGoalFoundForStrategyError(self, strategy)

    def schedule_defend_lichen_tile(self, schedule_info: ScheduleInfo) -> UnitGoal:
        # TODO, sort the threaths by distance to the factory
        while self.sorted_threaths_invaders:
            invader = self.sorted_threaths_invaders.pop()
            if schedule_info.game_state.c_is_defended(invader.tc):
                continue

            try:
                return self.schedule_defend_lichen_tile_from_invader(invader, schedule_info)
            except Exception as e:
                logging.debug(e)
                continue

        raise FactorySchedulerNoValidGoalFoundError(self, sub_strategy="defend lichen tile")

    def schedule_defend_lichen_tile_from_invader(self, invader: Unit, schedule_info: ScheduleInfo) -> UnitGoal:
        if invader.is_light:
            return self.schedule_defend_lichen_tile_from_invader_with_units(
                invader, self.light_available_units, schedule_info
            )
        else:
            return self.schedule_defend_lichen_tile_from_invader_with_units(
                invader, self.heavy_available_units, schedule_info
            )

    def schedule_defend_lichen_tile_from_invader_with_units(
        self, invader: Unit, units: Iterable[Unit], schedule_info: ScheduleInfo
    ) -> UnitGoal:

        potential_assignments = [
            (unit, goal) for unit in units for goal in unit.get_defend_lichen_goals(schedule_info.game_state, invader)
        ]
        return self.get_best_assignment(potential_assignments, schedule_info)  # type: ignore

    def _schedule_unit_on_strategy(self, strategy: Strategy, schedule_info: ScheduleInfo) -> UnitGoal:
        if strategy == Strategy.INCREASE_LICHEN_TILES:
            goal = self.schedule_strategy_increase_lichen_tiles(schedule_info)
        elif strategy == Strategy.INCREASE_LICHEN:
            goal = self.schedule_strategy_increase_lichen(schedule_info)
        elif strategy == Strategy.COLLECT_ORE:
            goal = self.schedule_strategy_collect_ore(schedule_info)
        elif strategy == Strategy.COLLECT_ICE:
            goal = self.schedule_strategy_collect_ice(schedule_info)
        elif strategy == Strategy.ATTACK_OPPONENT:
            goal = self.schedule_strategy_attack_opponent(schedule_info)
        elif strategy == Strategy.IMMEDIATELY_RETURN_ICE:
            goal = self.schedule_strategy_immediately_return_ice(schedule_info)
        elif strategy == Strategy.CLEAR_RUBBLE_AROUND_BASE:
            goal = self.schedule_strategy_clear_rubble_around_base(schedule_info)
        elif strategy == Strategy.DEFEND_LICHEN_TILE:
            goal = self.schedule_defend_lichen_tile(schedule_info)
        else:
            raise ValueError("Strategy is not a known strategy")

        return goal

    def schedule_strategy_increase_lichen(self, schedule_info: ScheduleInfo) -> UnitGoal:
        if not self.enough_water_collection_for_next_turns():
            return self.schedule_strategy_collect_ice(schedule_info)
        else:
            return self.schedule_strategy_increase_lichen_tiles(schedule_info)

    def schedule_strategy_increase_lichen_tiles(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state

        rubble_positions = self._get_suitable_dig_positions_for_lichen(game_state)
        if not rubble_positions:
            raise FactorySchedulerNoValidGoalFoundError(
                self, sub_strategy="increase lichen tiles", reason="no rubble positions"
            )

        if game_state.real_env_steps < CONFIG.FIRST_STEP_HEAVY_ALLOWED_TO_DIG_RUBBLE:
            units = self.light_available_units
        else:
            units = self.available_units

        return self._schedule_unit_on_rubble_pos(rubble_positions, units, schedule_info)

    def schedule_strategy_clear_rubble_around_base(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state

        rubble_positions = self._get_suitable_dig_positions_for_clearing_base(game_state)
        if not rubble_positions:
            raise FactorySchedulerNoValidGoalFoundError(
                self, sub_strategy="clear rubble around base", reason="no rubble positions"
            )

        if game_state.real_env_steps < CONFIG.FIRST_STEP_HEAVY_ALLOWED_TO_DIG_RUBBLE:
            units = self.light_available_units
        else:
            units = self.available_units

        return self._schedule_unit_on_rubble_pos(rubble_positions, units, schedule_info)

    def _schedule_unit_on_rubble_pos(
        self, rubble_positions: Iterable[Tuple], units: Iterable[Unit], schedule_info: ScheduleInfo
    ) -> UnitGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in rubble_positions
            for goal in unit.get_clear_rubble_goals(schedule_info.game_state, Coordinate(*pos))
        ]

        return self.get_best_assignment(potential_assignments, schedule_info)  # type: ignore

    def get_best_assignment(
        self, potential_assignments: List[Tuple[Unit, UnitGoal]], schedule_info: ScheduleInfo
    ) -> UnitGoal:

        potential_assignments = [
            (unit, goal) for unit, goal in potential_assignments if unit.is_feasible_assignment(goal)
        ]

        if not potential_assignments:
            # TODO, consider whether this sub_strategy get best assignment is valid or whether different error is needed
            raise FactorySchedulerNoValidGoalFoundError(
                self, sub_strategy="get best assignment", reason="no potential assignments"
            )

        unit, goal = max(
            potential_assignments, key=lambda x: x[1].get_best_case_value_per_step(schedule_info.game_state)
        )
        if isinstance(goal, DigGoal):
            schedule_info = schedule_info.copy_without_units_on_dig_c(goal.dig_c)

        schedule_info = schedule_info.copy_without_unit_scheduled_actions(unit)

        try:
            goal = unit.get_best_version_goal(goal, schedule_info)
        except Exception as e:
            logger.debug(e)
            raise e

        return goal

    def _get_suitable_dig_positions_for_lichen(self, game_state: GameState) -> Set[Tuple]:
        rubble_positions_favorite = positions_to_set(self.rubble_positions_next_to_can_spread_pos)
        valid_rubble_positions = rubble_positions_favorite - game_state.positions_in_dig_goals
        if valid_rubble_positions:
            return valid_rubble_positions

        rubble_positions_second_favorite = positions_to_set(self.rubble_positions_next_to_can_not_spread_lichen)
        valid_rubble_positions = rubble_positions_second_favorite - game_state.positions_in_dig_goals

        if valid_rubble_positions:
            return valid_rubble_positions

        return set()

    def _get_suitable_dig_positions_for_clearing_base(self, game_state: GameState) -> Set[Tuple]:
        valid_rubble_positions = self.closest_rubble_positions_within_4_distance_set - game_state.positions_in_dig_goals
        return valid_rubble_positions

    def enough_water_collection_for_next_turns(self) -> bool:
        water_collection = self.get_water_collection_per_step()
        water_available_next_n_turns = self.water + CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * water_collection
        water_cost_next_n_turns = CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * self.water_cost
        return water_available_next_n_turns > water_cost_next_n_turns

    def _schedule_supply_goal_and_reschedule_receiving_unit(self, schedule_info: ScheduleInfo) -> List[UnitGoal]:
        game_state = schedule_info.game_state

        supplying_unit, receiving_unit, receiving_c = self._assign_supplying_unit_and_receiving_unit(game_state)
        schedule_info = schedule_info.copy_without_unit_scheduled_actions(receiving_unit)

        goal_receiving_unit = self._reschedule_receiving_collect_goal(
            schedule_info=schedule_info,
            unit=receiving_unit,
            receiving_c=receiving_c,
        )

        tcs = goal_receiving_unit.action_plan.get_time_coordinates(game_state)
        schedule_info.constraints.add_negative_constraints(tcs)
        schedule_info.power_tracker.add_power_requests(goal_receiving_unit.action_plan.get_power_requests(game_state))

        goal_supplying_unit = self._schedule_supply_goal(
            schedule_info=schedule_info,
            supplying_unit=supplying_unit,
            receiving_action_plan=goal_receiving_unit.action_plan,
            receiving_unit=receiving_unit,
            receiving_c=receiving_c,
        )

        return [goal_receiving_unit, goal_supplying_unit]

    def _assign_supplying_unit_and_receiving_unit(self, game_state: GameState) -> Tuple[Unit, Unit, Coordinate]:
        potential_assignments = [
            (supply_unit, goal)
            for supply_unit in self.light_available_units
            for receiving_unit in self.heavy_units_unsupplied_collecting_next_to_factory_free_supply_c
            for goal in supply_unit.get_supply_power_goals(
                game_state,
                receiving_unit,
                receiving_unit.private_action_plan,
                receiving_unit.goal.dig_c,  # type: ignore
                supply_c=game_state.get_closest_player_factory_c(receiving_unit.goal.dig_c),  # type: ignore
            )
        ]

        if not potential_assignments:
            raise FactorySchedulerNoValidGoalFoundError(
                self, "assign supply and receiving unit", reason="no potential assignments"
            )

        supplying_unit, goal = max(potential_assignments, key=lambda x: x[1].get_best_case_value_per_step(game_state))
        receiving_unit = goal.receiving_unit
        receiving_c = goal.receiving_c

        return (supplying_unit, receiving_unit, receiving_c)

    def _reschedule_receiving_collect_goal(
        self, unit: Unit, receiving_c: Coordinate, schedule_info: ScheduleInfo
    ) -> CollectGoal:

        if isinstance(unit.goal, CollectOreGoal):
            return unit.generate_collect_ore_goal(
                schedule_info=schedule_info,
                c=receiving_c,
                is_supplied=True,
                factory=self,
                quantity=unit.goal.quantity,
            )
        elif isinstance(unit.goal, CollectIceGoal):
            return unit.generate_collect_ice_goal(
                schedule_info=schedule_info, c=receiving_c, is_supplied=True, factory=self, quantity=unit.goal.quantity
            )

        raise ValueError("Goal is not a Collect Ice or Collect Ore Goal")

    def _schedule_supply_goal(
        self,
        schedule_info: ScheduleInfo,
        supplying_unit: Unit,
        receiving_unit: Unit,
        receiving_action_plan: UnitActionPlan,
        receiving_c: Coordinate,
    ) -> SupplyPowerGoal:

        goal = supplying_unit.generate_supply_power_goal(
            schedule_info, receiving_unit, receiving_action_plan, receiving_c
        )
        return goal

    def schedule_strategy_collect_ore(self, schedule_info: ScheduleInfo) -> UnitGoal:
        if self.has_heavy_unit_available:
            return self._schedule_heavy_on_ore(schedule_info)
        else:
            return self._schedule_light_on_ore_task(schedule_info)

    def _schedule_heavy_on_ore(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ore_positions_set = game_state.board.minable_ore_positions_set - game_state.positions_in_heavy_dig_goals
        goal = self._schedule_unit_on_ore_pos(valid_ore_positions_set, self.heavy_available_units, schedule_info)
        return goal

    def _schedule_light_on_ore_task(self, schedule_info: ScheduleInfo) -> UnitGoal:
        try:
            return self._schedule_light_on_ore(schedule_info)
        except Exception:
            rubble_positions = self.get_rubble_positions_to_clear_for_ore(schedule_info.game_state)
            if not rubble_positions:
                raise FactorySchedulerNoValidGoalFoundError(
                    self, "schedule light on clear rubble for ore", "no rubble positions"
                )

            return self._schedule_unit_on_rubble_pos(
                rubble_positions,
                self.light_available_units,
                schedule_info,
            )

    def _schedule_light_on_ore(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ore_positions_set = game_state.board.minable_ore_positions_set - game_state.positions_in_dig_goals
        return self._schedule_unit_on_ore_pos(valid_ore_positions_set, self.light_available_units, schedule_info)

    def _schedule_unit_on_ore_pos(
        self, ore_positions: Iterable[Tuple], units: Iterable[Unit], schedule_info: ScheduleInfo
    ) -> CollectOreGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in ore_positions
            for goal in unit.get_collect_ore_goals(
                Coordinate(*pos), schedule_info.game_state, factory=self, is_supplied=False
            )
        ]

        return self.get_best_assignment(potential_assignments, schedule_info)  # type: ignore

    def schedule_strategy_collect_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        if self.has_heavy_unit_available:
            return self._schedule_heavy_on_ice(schedule_info)
        else:
            return self._schedule_light_on_ice_task(schedule_info)

    def _schedule_heavy_on_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ice_positions_set = game_state.board.minable_ice_positions_set - game_state.positions_in_heavy_dig_goals
        return self._schedule_unit_on_ice_pos(valid_ice_positions_set, self.heavy_available_units, schedule_info)

    def _schedule_light_on_ice_task(self, schedule_info: ScheduleInfo) -> UnitGoal:
        try:
            return self._schedule_light_on_ice(schedule_info)
        except Exception:
            rubble_positions = self.get_rubble_positions_to_clear_for_ice(schedule_info.game_state)
            if not rubble_positions:
                raise FactorySchedulerNoValidGoalFoundError(
                    self, "schedule light on clear rubble for ice", "no rubble positions"
                )

            return self._schedule_unit_on_rubble_pos(rubble_positions, self.light_available_units, schedule_info)

    def _schedule_light_on_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ice_positions_set = game_state.board.minable_ice_positions_set - game_state.positions_in_dig_goals
        return self._schedule_unit_on_ice_pos(valid_ice_positions_set, self.light_available_units, schedule_info)

    def _schedule_unit_on_ice_pos(
        self, ice_positions: Iterable[Tuple], units: Iterable[Unit], schedule_info: ScheduleInfo
    ) -> CollectIceGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in ice_positions
            for goal in unit.get_ice_goals(Coordinate(*pos), schedule_info.game_state, factory=self, is_supplied=False)
        ]

        return self.get_best_assignment(potential_assignments, schedule_info)  # type: ignore

    def schedule_strategy_attack_opponent(self, schedule_info: ScheduleInfo) -> UnitGoal:
        return self._schedule_unit_destroy_lichen(schedule_info)

    def _schedule_unit_destroy_lichen(self, schedule_info: ScheduleInfo) -> DestroyLichenGoal:
        game_state = schedule_info.game_state
        dig_pos_set = {c.xy for c in game_state.opp_lichen_tiles}
        valid_pos = dig_pos_set - game_state.positions_in_dig_goals

        if game_state.real_env_steps >= CONFIG.FIRST_STEP_HEAVY_ALLOWED_TO_DESTROY_LICHEN:
            units = self.available_units
        else:
            units = self.light_available_units

        potential_assignments = [
            (unit, goal)
            for pos in valid_pos
            for unit in units
            for goal in unit.get_destroy_lichen_goals(Coordinate(*pos), game_state)
        ]

        return self.get_best_assignment(potential_assignments, schedule_info)  # type: ignore

    def schedule_strategy_immediately_return_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        if self.distress_signal_can_not_be_handled:
            raise FactorySchedulerNoValidGoalFoundError(
                self, "schedule strategy immediately return ice", "distress signal can not be handled"
            )

        nr_steps_to_go = self.water - CONFIG.ICE_MUST_COME_IN_BEFORE_LEVEL

        for unit in self.units_collecting_ice:
            try:
                return self._attempt_get_shortened_collect_ice_goal(schedule_info, unit, nr_steps_to_go)
            except Exception as e:
                logger.debug(e)
                continue

        for unit in self.units_with_ice:
            if isinstance(unit.goal, TransferIceGoal):
                continue

            try:
                return unit.generate_transfer_ice_goal(schedule_info, self)
            except Exception as e:
                logger.debug(e)
                continue

        try:
            return self._schedule_any_heavy_on_ice(schedule_info)
        except Exception as e:
            logger.debug(e)

        try:
            return self._schedule_any_light_on_ice(schedule_info)
        except Exception as e:
            logger.debug(e)

        for unit in self.heavies_with_main_ore:
            if isinstance(unit.goal, TransferOreGoal):
                continue

            try:
                return unit.generate_transfer_ore_goal(schedule_info, self)
            except Exception as e:
                logger.debug(e)
                continue

        self.distress_signal_can_not_be_handled = True

        raise FactorySchedulerNoValidGoalFoundError(self, "strategy immediately return ice")

    def _schedule_any_heavy_on_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ice_positions_set = game_state.board.minable_ice_positions_set - game_state.positions_in_heavy_dig_goals
        return self._schedule_unit_on_ice_pos(valid_ice_positions_set, self.heavies_not_having_ice_goal, schedule_info)

    def _schedule_any_light_on_ice(self, schedule_info: ScheduleInfo) -> UnitGoal:
        game_state = schedule_info.game_state
        valid_ice_positions_set = game_state.board.minable_ice_positions_set - game_state.positions_in_dig_goals
        return self._schedule_unit_on_ice_pos(valid_ice_positions_set, self.lights_not_having_ice_goal, schedule_info)

    def _attempt_get_shortened_collect_ice_goal(
        self, schedule_info: ScheduleInfo, unit: Unit, nr_steps_to_go: int
    ) -> UnitGoal:
        goal: CollectIceGoal = unit.goal  # type: ignore
        step_ice_incoming = unit.private_action_plan.nr_time_steps
        if step_ice_incoming <= nr_steps_to_go:
            raise FactorySchedulerNoValidGoalFoundError(
                self, "get shortened collect ice goal", reason="ice already on time"
            )

        nr_steps_to_reduce = step_ice_incoming - nr_steps_to_go
        ice_to_transfer = goal.quantity_ice_to_transfer(schedule_info.game_state)
        new_quantity = ice_to_transfer - nr_steps_to_reduce * unit.resources_gained_per_dig
        if new_quantity <= 0:
            raise FactorySchedulerNoValidGoalFoundError(
                self, "get shortened collect ice goal", "can not reduce this number of steps"
            )

        schedule_info_without_unit = schedule_info.copy_without_unit_scheduled_actions(unit)
        return unit.generate_collect_ice_goal(
            schedule_info_without_unit, goal.dig_c, goal.is_supplied, self, new_quantity
        )

    def has_connected_safe_or_defended_ice_coordinate(self, game_state: GameState) -> bool:
        return any(
            True
            for ice_c in self.connected_ice_coordinates
            if game_state.get_min_distance_to_any_player_factory(ice_c) > 1 or game_state.c_is_undefended(ice_c)
        )
