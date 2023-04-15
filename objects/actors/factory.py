from __future__ import annotations

import numpy as np

from typing import Tuple, TYPE_CHECKING, Optional, Iterable, Set, Generator, List
from itertools import product
from dataclasses import dataclass, field
from collections import Counter
from copy import copy
from enum import Enum, auto
import logging

from objects.cargo import Cargo
from objects.actions.factory_action import WaterAction
from objects.actors.actor import Actor
from exceptions import NoValidGoalFound
from objects.coordinate import TimeCoordinate, Coordinate, CoordinateList
from logic.constraints import Constraints
from logic.goals.unit_goal import DigGoal, CollectGoal
from objects.actions.factory_action_plan import FactoryActionPlan
from logic.goals.unit_goal import UnitGoal, ClearRubbleGoal, CollectOreGoal, CollectIceGoal, SupplyPowerGoal
from logic.goals.factory_goal import BuildHeavyGoal, BuildLightGoal, WaterGoal, FactoryNoGoal, FactoryGoal
from distances import (
    get_min_distance_between_positions,
    get_min_distance_between_pos_and_positions,
    get_min_distances_between_positions,
    get_closest_pos_between_pos_and_positions,
    get_positions_on_optimal_path_between_pos_and_pos,
)
from image_processing import get_islands
from positions import init_empty_positions, get_neighboring_positions, append_positions, positions_to_set
from lux.config import EnvConfig, LIGHT_CONFIG, HEAVY_CONFIG
from config import CONFIG

if TYPE_CHECKING:
    from logic.goal_resolution.power_availabilty_tracker import PowerTracker
    from objects.game_state import GameState
    from objects.board import Board
    from objects.actors.unit import Unit


class Strategy(Enum):
    INCREASE_LICHEN_TILES = auto()
    INCREASE_LICHEN = auto()
    COLLECT_ICE = auto()
    INCREASE_UNITS = auto()


@dataclass(eq=False)
class Factory(Actor):
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

        self.private_action_plan = FactoryActionPlan(self, [])
        self._set_unit_state_variables()

    def _set_unit_state_variables(self) -> None:
        self.is_scheduled = False
        self.positions = np.array([[self.x + x, self.y + y] for x, y in product(range(-1, 2), range(-1, 2))])

    def update_state(self, center_tc: TimeCoordinate, power: int, cargo: Cargo) -> None:
        self.center_tc = center_tc
        self.power = power
        self.cargo = cargo
        self._set_unit_state_variables()

    def remove_units_not_in_obs(self, obs_units: set[Unit]) -> None:
        self.units.intersection_update(obs_units)

    def set_positions(self, board: Board) -> None:
        if not self.positions_set:
            self._set_positions_once(board)
            self.positions_set = True

        if self in board.player_factories:
            self._rubble_positions_to_clear_for_ore = self._get_rubble_positions_to_clear_for_pathing(
                board, self.ore_positions_distance_sorted[:5]
            )
            self._rubble_positions_to_clear_for_ice = self._get_rubble_positions_to_clear_for_pathing(
                board, self.ice_positions_distance_sorted[:5]
            )

            # TODO add other rubble positions
            self._rubble_positions_to_clear = (
                self._rubble_positions_to_clear_for_ore | self._rubble_positions_to_clear_for_ice
            )

        self.lichen_positions = np.argwhere(board.lichen_strains == self.strain_id)
        self.nr_lichen_tiles = len(self.lichen_positions)
        self.connected_lichen_positions = self._get_connected_lichen_positions(board)
        self.spreadable_lichen_positions = self._get_spreadable_lichen_positions(board)
        self.non_spreadable_connected_lichen_positions = self._get_not_spreadable_connected_lichen_positions(board)
        self.can_spread_positions = append_positions(self.positions, self.spreadable_lichen_positions)

        self.can_spread_to_positions = self._get_can_spread_to_positions(board, self.can_spread_positions)
        self.connected_positions = self._get_empty_or_lichen_connected_positions(board)

        # self.rubble_positions_pathing = self._get_rubble_positions_to_clear_for_resources(board)
        # self.rubble_positions_values_for_lichen = self._get_rubble_positions_to_clear_for_lichen(board)
        self.rubble_positions_next_to_can_spread_pos = self._get_rubble_positions_next_to_can_spread_pos(board)
        self.rubble_positions_next_to_can_not_spread_lichen = self._get_rubble_next_to_can_not_spread_lichen(board)
        self.rubble_positions_next_to_connected_or_self = self._get_rubble_next_to_connected_or_self(board)
        self.rubble_positions_next_to_connected_or_self_set = positions_to_set(
            self.rubble_positions_next_to_connected_or_self
        )

        self.nr_connected_lichen_tiles = len(self.connected_lichen_positions)
        self.nr_can_spread_positions = len(self.can_spread_positions)
        self.nr_connected_positions = len(self.connected_positions)
        self.nr_connected_positions_non_lichen_connected = self.nr_connected_positions - self.nr_connected_lichen_tiles
        self.max_nr_tiles_to_water = len(self.connected_lichen_positions) + len(self.can_spread_to_positions)

    def _set_positions_once(self, board: Board) -> None:
        self.ice_positions_distance_sorted = self._get_positions_distance_sorted(board.ice_positions)
        self.ore_positions_distance_sorted = self._get_positions_distance_sorted(board.ore_positions)

    def _get_positions_distance_sorted(self, positions: np.ndarray) -> np.ndarray:
        distances = get_min_distances_between_positions(positions, self.positions)
        sorted_indexes = np.argsort(distances)
        sorted_distance_positions = positions[sorted_indexes]
        return sorted_distance_positions

    def get_rubble_positions_to_clear(self, game_state: GameState) -> Set[Tuple]:
        rubble_positions_free = self._rubble_positions_to_clear - game_state.positions_in_dig_goals
        return rubble_positions_free

    def get_rubble_positions_to_clear_for_ice(self, game_state: GameState) -> Set[Tuple]:
        rubble_positions_free = self._rubble_positions_to_clear_for_ice - game_state.positions_in_dig_goals
        return rubble_positions_free

    def get_rubble_positions_to_clear_for_ore(self, game_state: GameState) -> Set[Tuple]:
        rubble_positions_free = self._rubble_positions_to_clear_for_ore - game_state.positions_in_dig_goals
        return rubble_positions_free

    def has_enough_space_to_increase_lichen(self, game_state: GameState) -> bool:
        nr_connected_positions = self.get_nr_connected_positions_including_being_cleared(game_state)
        space_free_positions = nr_connected_positions - self.nr_connected_lichen_tiles
        return space_free_positions > 2

    def get_nr_connected_positions_including_being_cleared(self, game_state: GameState) -> int:
        # TODO, take into account if positions being cleared connects to other positions
        nr_positions_being_cleared = self.get_nr_positions_being_cleared_next_to_connected(game_state)
        return self.nr_connected_positions + nr_positions_being_cleared

    def get_nr_positions_being_cleared_next_to_connected(self, game_state: GameState) -> int:
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

    def _get_rubble_positions_to_clear_for_pathing(self, board: Board, positions: np.ndarray) -> Set[Tuple]:
        for ore_pos in positions[:5]:
            closest_factory_pos = get_closest_pos_between_pos_and_positions(ore_pos, self.positions)
            positions = get_positions_on_optimal_path_between_pos_and_pos(ore_pos, closest_factory_pos, board)
            rubble_mask = board.are_rubble_positions(positions)
            rubble_positions = positions[rubble_mask]
            if rubble_positions.shape[0]:
                break
        else:
            rubble_positions = init_empty_positions()

        return positions_to_set(rubble_positions)

    # def _get_rubble_positions_to_clear_for_resources(self, board: Board) -> Counter[Tuple[int, int]]:
    #     closest_2_ice_positions = board.get_n_closest_ice_positions_to_factory(self, n=2)
    #     closest_2_ore_positions = board.get_n_closest_ore_positions_to_factory(self, n=2)
    #     closest_resource_positions = append_positions(closest_2_ice_positions, closest_2_ore_positions)

    #     positions = init_empty_positions()

    #     for pos in closest_resource_positions:
    #         closest_factory_pos = get_closest_pos_between_pos_and_positions(pos, self.positions)
    #         optimal_positions = get_positions_on_optimal_path_between_pos_and_pos(pos, closest_factory_pos, board)
    #         positions = append_positions(positions, optimal_positions)

    #     rubble_mask = board.are_rubble_positions(positions)
    #     rubble_positions = positions[rubble_mask]

    #     rubble_value_dict = Counter({tuple(pos): CONFIG.RUBBLE_VALUE_CLEAR_FOR_RESOURCE for pos in rubble_positions})

    #     return rubble_value_dict

    def _get_rubble_positions_to_clear_for_lichen(self, board: Board) -> Counter[Tuple[int, int]]:
        rubble_positions, distances = self._get_rubble_positions_and_distances_within_max_distance(board)
        values = self._get_rubble_positions_to_clear_for_lichen_score(distances)
        rubble_value_dict = Counter({tuple(pos): value for pos, value in zip(rubble_positions, values)})

        return rubble_value_dict

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

    def _get_rubble_next_to_connected_or_self(self, board: Board) -> np.ndarray:
        if not board.rubble_positions.shape[0]:
            return init_empty_positions()

        connected_or_self_positions = append_positions(self.positions, self.connected_positions)

        distances = get_min_distances_between_positions(board.rubble_positions, connected_or_self_positions)
        valid_distance_mask = distances == 1
        rubble_positions = board.rubble_positions[valid_distance_mask]
        return rubble_positions

    def _get_rubble_positions_to_clear_for_lichen_score(self, distances: np.ndarray) -> np.ndarray:
        base_score = CONFIG.RUBBLE_VALUE_CLEAR_FOR_LICHEN_BASE
        distance_penalty = CONFIG.RUBBLE_VALUE_CLEAR_FOR_LICHEN_DISTANCE_PENALTY
        return base_score - distance_penalty * distances

    def _get_rubble_positions_and_distances_within_max_distance(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        distances = get_min_distances_between_positions(board.rubble_positions, self.can_spread_positions)
        valid_distance_mask = distances < CONFIG.RUBBLE_CLEAR_FOR_LICHEN_MAX_DISTANCE
        rubble_postions_within_max_distance = board.rubble_positions[valid_distance_mask]
        distances_within_max_distance = distances[valid_distance_mask]
        return rubble_postions_within_max_distance, distances_within_max_distance

    def add_unit(self, unit: Unit) -> None:
        self.units.add(unit)

    def min_distance_to_connected_positions(self, positions: np.ndarray) -> int:
        rel_positions = np.append(self.positions, self.connected_positions, axis=0)
        return get_min_distance_between_positions(rel_positions, positions)

    # TODO Can build should be put into the constraints
    def schedule_goal(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker, can_build: bool = True
    ) -> FactoryActionPlan:
        goal = self.get_goal(game_state, can_build)
        action_plan = goal.generate_and_evaluate_action_plan(game_state, constraints, power_tracker)
        self.goal = goal
        self.private_action_plan = action_plan
        return action_plan

    def get_goal(self, game_state: GameState, can_build: bool = True) -> FactoryGoal:
        water_cost = self.water_cost
        if can_build and self.can_build_heavy:
            return BuildHeavyGoal(self)

        elif can_build and self.can_build_light and self.nr_light_units < 15:
            return BuildLightGoal(self)

        elif self.cargo.water - water_cost > 50 and (water_cost < 5 or self.water > 150):
            return WaterGoal(self)

        elif game_state.env_steps > 750 and self.can_water() and self.cargo.water - water_cost > game_state.steps_left:
            return WaterGoal(self)

        return FactoryNoGoal(self)

    @property
    def light_units(self) -> Generator[Unit, None, None]:
        return (unit for unit in self.units if unit.is_light)

    @property
    def heavy_units(self) -> Generator[Unit, None, None]:
        return (unit for unit in self.units if unit.is_heavy)

    @property
    def nr_light_units(self) -> int:
        return sum(1 for _ in self.light_units)

    @property
    def nr_heavy_units(self) -> int:
        return sum(1 for _ in self.heavy_units)

    @property
    def daily_charge(self) -> int:
        return self.env_cfg.FACTORY_CHARGE

    @property
    def expected_power_gain(self) -> int:
        return self.env_cfg.FACTORY_CHARGE + self.nr_connected_lichen_tiles

    @property
    def can_build_heavy(self) -> bool:
        return self.power >= HEAVY_CONFIG.POWER_COST and self.cargo.metal >= HEAVY_CONFIG.METAL_COST

    @property
    def can_build_light(self) -> bool:
        return self.power >= LIGHT_CONFIG.POWER_COST and self.cargo.metal >= LIGHT_CONFIG.METAL_COST

    @property
    def water_cost(self) -> int:
        return WaterAction.get_water_cost(self)

    def can_water(self):
        return self.cargo.water >= self.water_cost

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

    @property
    def water(self) -> int:
        return self.cargo.water

    @property
    def ice(self) -> int:
        return self.cargo.ice

    @property
    def ore(self) -> int:
        return self.cargo.ore

    @property
    def metal(self) -> int:
        return self.cargo.metal

    def __repr__(self) -> str:
        return f"Factory[id={self.unit_id}, center={self.center_tc.xy}]"

    def get_ice_collection_per_step(self, game_state: GameState) -> float:
        ice_collection_per_step = 0

        for unit in self.units:
            goal = unit.goal

            if goal:
                quantity_ice = goal.quantity_ice_to_transfer(game_state)
                nr_steps = max(goal.action_plan.nr_primitive_actions, 1)
                ice_collection_per_step_unit = quantity_ice / nr_steps
                ice_collection_per_step += ice_collection_per_step_unit

        return ice_collection_per_step

    def get_water_collection_per_step(self, game_state: GameState) -> float:
        ice_collection_per_step = self.get_ice_collection_per_step(game_state)
        water_collection_per_step = ice_collection_per_step / EnvConfig.ICE_WATER_RATIO
        return water_collection_per_step

    def get_ore_collection_per_step(self, game_state: GameState) -> float:
        ore_collection_per_step = 0

        for unit in self.units:
            goal = unit.goal

            if goal:
                quantity_ore = goal.quantity_ore_to_transfer(game_state)
                nr_steps = max(goal.action_plan.nr_primitive_actions, 1)
                ore_collection_per_step_unit = quantity_ore / nr_steps
                ore_collection_per_step += ore_collection_per_step_unit

        return ore_collection_per_step

    def get_metal_collection_per_step(self, game_state: GameState) -> float:
        ore_collection_per_step = self.get_ore_collection_per_step(game_state)
        metal_collection_per_step = ore_collection_per_step / EnvConfig.ORE_METAL_RATIO
        return metal_collection_per_step

    @property
    def nr_can_spread_to_positions_being_cleared(self) -> int:
        nr_positions = 0

        for unit in self.units:
            goal = unit.goal
            if isinstance(goal, ClearRubbleGoal):
                dig_pos = np.array(goal.dig_c.xy)  # type: ignore
                dis = get_min_distance_between_pos_and_positions(dig_pos, self.can_spread_positions)
                if dis == 1:
                    nr_positions += 1

        return nr_positions

    @property
    def has_heavy_unsupplied_collecting_next_to_factory(self) -> bool:
        return any(True for _ in self.heavy_units_unsupplied_collecting_next_to_factory)

    @property
    def heavy_units_unsupplied_collecting_next_to_factory(self) -> Generator[Unit, None, None]:
        return (
            heavy
            for heavy in self.heavy_units
            if isinstance(heavy.goal, CollectGoal)
            and self.min_distance_to_c(heavy.goal.dig_c) == 1
            and not heavy.is_supplied_by
        )

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
    def available_units(self) -> Generator[Unit, None, None]:
        # TODO some checks to see if there is enough power or some other mechanic to set units as unavailable
        return (
            unit
            for unit in self.units
            if unit.can_update_action_queue and not unit.private_action_plan and unit.can_be_assigned
        )

    @property
    def heavy_available_units(self) -> Generator[Unit, None, None]:
        return (unit for unit in self.available_units if unit.is_heavy)

    @property
    def light_available_units(self) -> Generator[Unit, None, None]:
        return (unit for unit in self.available_units if unit.is_light)

    @property
    def unassigned_units(self) -> Generator[Unit, None, None]:
        # TODO some checks to see if there is enough power or some other mechanic to set units as unavailable
        return (unit for unit in self.units if not unit.private_action_plan)

    @property
    def has_unassigned_units(self) -> bool:
        return any(not unit.private_action_plan for unit in self.units)

    def schedule_units(
        self,
        strategies: Iterable[Strategy],
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> List[UnitGoal]:
        if self.has_heavy_unsupplied_collecting_next_to_factory and self.has_light_unit_available:
            try:
                return self._schedule_supply_goal_and_reschedule_receiving_unit(game_state, constraints, power_tracker)
            except NoValidGoalFound:
                pass

        for strategy in strategies:
            try:
                return [self._schedule_unit_on_strategy(strategy, game_state, constraints, power_tracker)]
            except NoValidGoalFound:
                continue

        return [self._schedule_first_unit_by_own_preference(game_state, constraints, power_tracker)]

    def _schedule_unit_on_strategy(
        self, strategy: Strategy, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:

        if strategy == Strategy.INCREASE_LICHEN_TILES:
            goal = self.schedule_strategy_increase_lichen_tiles(game_state, constraints, power_tracker)
        elif strategy == Strategy.INCREASE_LICHEN:
            goal = self.schedule_strategy_increase_lichen(game_state, constraints, power_tracker)
        elif strategy == Strategy.INCREASE_UNITS:
            goal = self.schedule_strategy_increase_units(game_state, constraints, power_tracker)
        elif strategy == Strategy.COLLECT_ICE:
            goal = self.schedule_strategy_collect_ice(game_state, constraints, power_tracker)
        else:
            raise ValueError("Strategy is not a known strategy")

        return goal

    def _schedule_first_unit_by_own_preference(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        logging.info(
            f"{game_state.real_env_steps}: player {game_state.player_team.team_id} scheduled unit by own preference"
        )

        unit = next(self.available_units)
        all_goals = unit.generate_goals(game_state, self)
        best_goal = max(all_goals, key=lambda g: g.get_best_value_per_step(game_state))
        dummy_goals = unit._get_dummy_goals(game_state)
        goals = [best_goal] + dummy_goals
        goal = unit.get_best_goal(goals, game_state, constraints, power_tracker)
        return goal

    def schedule_strategy_increase_lichen(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        if not self.enough_water_collection_for_next_turns(game_state):
            return self.schedule_strategy_collect_ice(game_state, constraints, power_tracker)
        else:
            return self.schedule_strategy_increase_lichen_tiles(game_state, constraints, power_tracker)

    def schedule_strategy_increase_lichen_tiles(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        rubble_positions = self._get_suitable_dig_positions_for_lichen(game_state)
        if not rubble_positions:
            raise NoValidGoalFound

        return self._schedule_unit_on_rubble_pos(
            rubble_positions, self.available_units, game_state, constraints, power_tracker
        )

    def _schedule_unit_on_rubble_pos(
        self,
        rubble_positions: Iterable[Tuple],
        units: Iterable[Unit],
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> UnitGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in rubble_positions
            for goal in unit._get_clear_rubble_goals(Coordinate(*pos))
        ]

        unit, goal = max(potential_assignments, key=lambda x: x[1].get_best_value_per_step(game_state))
        constraints, power_tracker = self._get_constraints_and_power_without_units_on_dig_c(
            goal.dig_c, game_state, constraints, power_tracker
        )
        goal = unit.generate_clear_rubble_goal(game_state, goal.dig_c, constraints, power_tracker)
        # self._schedule_unit_on_goal(unit, goal)
        return goal

    def _get_closest_available_unit_to_pos(self, pos: np.ndarray) -> Unit:
        c = Coordinate(*pos)
        unit = min(self.available_units, key=lambda u: c.distance_to(u.tc))
        return unit

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

    def enough_water_collection_for_next_turns(self, game_state: GameState) -> bool:
        water_collection = self.get_water_collection_per_step(game_state)
        water_available_next_n_turns = self.water + CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * water_collection
        water_cost_next_n_turns = CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * self.water_cost
        return water_available_next_n_turns > water_cost_next_n_turns

    def _schedule_supply_goal_and_reschedule_receiving_unit(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> List[UnitGoal]:
        supplying_unit, receiving_unit, receiving_c = self._assign_supplying_unit_and_receiving_unit(game_state)
        constraints = copy(constraints)
        constraints.remove_negative_constraints(receiving_unit.private_action_plan.time_coordinates)
        power_tracker = copy(power_tracker)
        power_tracker.remove_power_requests(receiving_unit.private_action_plan.get_power_requests(game_state))

        goal_receiving_unit = self._reschedule_receiving_collect_goal(
            unit=receiving_unit,
            receiving_c=receiving_c,
            game_state=game_state,
            constraints=constraints,
            power_tracker=power_tracker,
        )

        constraints.add_negative_constraints(goal_receiving_unit.action_plan.time_coordinates)
        power_tracker.add_power_requests(goal_receiving_unit.action_plan.get_power_requests(game_state))

        goal_supplying_unit = self._schedule_supply_goal(
            game_state=game_state,
            supplying_unit=supplying_unit,
            receiving_unit=receiving_unit,
            receiving_c=receiving_c,
            constraints=constraints,
            power_tracker=power_tracker,
        )

        # TODO add check here whether the receiving unit will get power in time
        return [goal_receiving_unit, goal_supplying_unit]

    def _assign_supplying_unit_and_receiving_unit(self, game_state: GameState) -> Tuple[Unit, Unit, Coordinate]:
        potential_assignments = [
            (supply_unit, goal)
            for supply_unit in self.light_available_units
            for receiving_unit in self.heavy_units_unsupplied_collecting_next_to_factory
            for goal in supply_unit._get_supply_power_goal(receiving_unit, receiving_unit.goal.dig_c)  # type: ignore
        ]

        suppling_unit, goal = max(potential_assignments, key=lambda x: x[1].get_best_value_per_step(game_state))
        receiving_unit = goal.receiving_unit
        receiving_c = goal.receiving_c

        return (suppling_unit, receiving_unit, receiving_c)

    def _reschedule_receiving_collect_goal(
        self,
        unit: Unit,
        receiving_c: Coordinate,
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> CollectGoal:

        if isinstance(unit.goal, CollectOreGoal):
            return unit.generate_collect_ore_goal(
                game_state=game_state,
                c=receiving_c,
                is_supplied=True,
                constraints=constraints,
                power_tracker=power_tracker,
                factory=self,
            )
        elif isinstance(unit.goal, CollectIceGoal):
            return unit.generate_collect_ice_goal(
                game_state=game_state,
                c=receiving_c,
                is_supplied=True,
                constraints=constraints,
                power_tracker=power_tracker,
                factory=self,
            )

        raise RuntimeError("Not supposed to happen")

    def _schedule_supply_goal(
        self,
        game_state: GameState,
        supplying_unit: Unit,
        receiving_unit: Unit,
        receiving_c: Coordinate,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> SupplyPowerGoal:

        goal = supplying_unit.generate_supply_power_goal(
            game_state, receiving_unit, receiving_c, constraints, power_tracker
        )
        return goal

    def schedule_strategy_increase_units(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        # Collect Ore / Clear Path to Ore / Supply Power to heavy on Ore
        if self.has_heavy_unit_available:
            return self._schedule_heavy_on_ore(game_state, constraints, power_tracker)
        else:
            return self._schedule_light_on_ore_task(game_state, constraints, power_tracker)

    def _schedule_heavy_on_ore(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        valid_ore_positions_set = game_state.board.ore_positions_set - game_state.positions_in_heavy_dig_goals
        goal = self._schedule_unit_on_ore_pos(
            valid_ore_positions_set, self.heavy_available_units, game_state, constraints, power_tracker
        )
        return goal

    def _schedule_light_on_ore_task(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        try:
            return self._schedule_light_on_ore(game_state, constraints, power_tracker)
        except Exception:
            rubble_positions = self.get_rubble_positions_to_clear_for_ore(game_state)
            if not rubble_positions:
                raise NoValidGoalFound

            return self._schedule_unit_on_rubble_pos(
                rubble_positions,
                self.light_available_units,
                game_state,
                constraints,
                power_tracker,
            )

    def _schedule_light_on_ore(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        valid_ore_positions_set = game_state.board.ore_positions_set - game_state.positions_in_dig_goals
        return self._schedule_unit_on_ore_pos(
            valid_ore_positions_set, self.light_available_units, game_state, constraints, power_tracker
        )

    def _schedule_unit_on_ore_pos(
        self,
        ore_positions: Iterable[Tuple],
        units: Iterable[Unit],
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> CollectOreGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in ore_positions
            for goal in unit._get_collect_ore_goals(Coordinate(*pos), factory=self, is_supplied=False)
        ]

        unit, goal = max(potential_assignments, key=lambda x: x[1].get_best_value_per_step(game_state))
        constraints, power_tracker = self._get_constraints_and_power_without_units_on_dig_c(
            goal.dig_c, game_state, constraints, power_tracker
        )
        goal = unit.generate_collect_ore_goal(
            game_state=game_state,
            c=goal.dig_c,
            is_supplied=False,
            constraints=constraints,
            power_tracker=power_tracker,
            factory=self,
        )
        # self._schedule_unit_on_goal(unit, goal)
        return goal

    def schedule_strategy_collect_ice(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        # Collect Ice / Clear Path to Ice / Supply Power to heavy on Ice
        if self.has_heavy_unit_available:
            return self._schedule_heavy_on_ice(game_state, constraints, power_tracker)
        else:
            return self._schedule_light_on_ice_task(game_state, constraints, power_tracker)

    def _schedule_heavy_on_ice(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        valid_ice_positions_set = game_state.board.ice_positions_set - game_state.positions_in_heavy_dig_goals
        return self._schedule_unit_on_ice_pos(
            valid_ice_positions_set, self.heavy_available_units, game_state, constraints, power_tracker
        )

    def _schedule_light_on_ice_task(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        try:
            return self._schedule_light_on_ice(game_state, constraints, power_tracker)
        except Exception:
            rubble_positions = self.get_rubble_positions_to_clear_for_ice(game_state)
            if not rubble_positions:
                raise NoValidGoalFound

            return self._schedule_unit_on_rubble_pos(
                rubble_positions,
                self.light_available_units,
                game_state,
                constraints,
                power_tracker,
            )

    def _schedule_light_on_ice(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> UnitGoal:
        valid_ice_positions_set = game_state.board.ice_positions_set - game_state.positions_in_dig_goals
        return self._schedule_unit_on_ice_pos(
            valid_ice_positions_set, self.light_available_units, game_state, constraints, power_tracker
        )

    def _schedule_unit_on_ice_pos(
        self,
        ice_positions: Iterable[Tuple],
        units: Iterable[Unit],
        game_state: GameState,
        constraints: Constraints,
        power_tracker: PowerTracker,
    ) -> CollectIceGoal:

        potential_assignments = [
            (unit, goal)
            for unit in units
            for pos in ice_positions
            for goal in unit._get_collect_ice_goals(Coordinate(*pos), factory=self, is_supplied=False)
        ]

        unit, goal = max(potential_assignments, key=lambda x: x[1].get_best_value_per_step(game_state))
        constraints, power_tracker = self._get_constraints_and_power_without_units_on_dig_c(
            goal.dig_c, game_state, constraints, power_tracker
        )
        goal = unit.generate_collect_ice_goal(
            game_state=game_state,
            c=goal.dig_c,
            is_supplied=False,
            constraints=constraints,
            power_tracker=power_tracker,
            factory=self,
        )
        return goal

    def _get_constraints_and_power_without_units_on_dig_c(
        self, c: Coordinate, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> Tuple[Constraints, PowerTracker]:
        constraints = copy(constraints)
        power_tracker = copy(power_tracker)

        for unit in game_state.units:
            if isinstance(unit.goal, DigGoal) and unit.goal.dig_c == c:
                if unit.private_action_plan:
                    constraints.remove_negative_constraints(unit.private_action_plan.time_coordinates)
                    power_tracker.remove_power_requests(unit.private_action_plan.get_power_requests(game_state))

        return constraints, power_tracker
