from __future__ import annotations

import numpy as np

from typing import Tuple, TYPE_CHECKING, Optional
from itertools import product
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum, auto

from objects.cargo import Cargo
from objects.actions.factory_action import WaterAction
from objects.actors.actor import Actor
from objects.coordinate import TimeCoordinate, Coordinate, CoordinateList
from logic.constraints import Constraints
from objects.actions.action_plan import ActionPlan
from objects.actions.factory_action_plan import FactoryActionPlan
from logic.goals.unit_goal import ClearRubbleGoal
from logic.goals.factory_goal import BuildHeavyGoal, BuildLightGoal, WaterGoal, FactoryNoGoal, FactoryGoal
from distances import (
    get_min_distance_between_positions,
    get_min_distance_between_pos_and_positions,
    get_min_distances_between_positions,
    get_closest_pos_between_pos_and_positions,
    get_positions_on_optimal_path_between_pos_and_pos,
    get_closest_pos_and_pos_between_positions,
    get_closest_positions_between_positions,
)
from image_processing import get_islands
from positions import init_empty_positions, get_neighboring_positions
from lux.config import EnvConfig, LIGHT_CONFIG, HEAVY_CONFIG
from config import CONFIG

if TYPE_CHECKING:
    from logic.goal_resolution.power_availabilty_tracker import PowerTracker
    from objects.game_state import GameState
    from objects.board import Board
    from objects.actors.unit import Unit


class Strategy(Enum):
    INCREASE_LICHEN = auto()
    COLLECT_ICE = auto()
    INCREASE_UNITS = auto()


@dataclass
class Factory(Actor):
    strain_id: int
    center_tc: TimeCoordinate
    env_cfg: EnvConfig
    radius = 1
    units: set[Unit] = field(init=False, default_factory=set)
    goal: Optional[FactoryGoal] = field(init=False, default=None)
    private_action_plan: Optional[FactoryActionPlan] = field(init=False, default=None)

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
        closest_ice_positions = get_closest_positions_between_positions(board.ice_positions, self.positions)
        self.closest_ice_positions_set = {tuple(pos) for pos in closest_ice_positions}
        closest_ore_positions = get_closest_positions_between_positions(board.ore_positions, self.positions)
        self.closest_ore_positions = {tuple(pos) for pos in closest_ore_positions}

        self.lichen_positions = np.argwhere(board.lichen_strains == self.strain_id)
        self.nr_lichen_tiles = len(self.lichen_positions)
        self.connected_lichen_positions = self._get_connected_lichen_positions(board)
        self.spreadable_lichen_positions = self._get_spreadable_lichen_positions(board)
        self.non_spreadable_connected_lichen_positions = self._get_not_spreadable_connected_lichen_positions(board)
        self.can_spread_positions = np.append(self.positions, self.spreadable_lichen_positions, axis=0)

        self.can_spread_to_positions = self._get_can_spread_to_positions(board, self.can_spread_positions)
        self.connected_positions = self._get_empty_or_lichen_connected_positions(board)

        self.rubble_positions_pathing = self._get_rubble_positions_to_clear_for_resources(board)
        self.rubble_positions_values_for_lichen = self._get_rubble_positions_to_clear_for_lichen(board)
        self.rubble_positions_next_to_can_spread_pos = self._get_rubble_positions_next_to_can_spread_pos(board)
        self.rubble_positions_next_to_can_not_spread_lichen = self._get_rubble_next_to_can_not_spread_lichen(board)

        self.nr_connected_lichen_tiles = len(self.lichen_positions)
        self.nr_can_spread_positions = len(self.can_spread_positions)
        self.nr_connected_positions = len(self.connected_positions)
        self.max_nr_tiles_to_water = len(self.connected_lichen_positions) + len(self.can_spread_to_positions)

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

    def add_unit(self, unit: Unit) -> None:
        self.units.add(unit)

    def min_distance_to_connected_positions(self, positions: np.ndarray) -> int:
        rel_positions = np.append(self.positions, self.connected_positions, axis=0)
        return get_min_distance_between_positions(rel_positions, positions)

    # TODO Can build should be put into the constraints
    def set_goal(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker, can_build: bool = True
    ) -> FactoryActionPlan:
        goal = self.get_goal(game_state, can_build)
        action_plan = goal.generate_and_evaluate_action_plan(game_state, constraints, power_tracker)
        self.goal = goal
        self.private_action_plan = action_plan
        return action_plan

    def get_goal(self, game_state: GameState, can_build: bool = True) -> FactoryGoal:
        water_cost = self.water_cost
        if can_build and self.can_build_heavy and game_state.player_nr_heavies / game_state.player_nr_factories < 1:
            return BuildHeavyGoal(self)

        elif (
            can_build
            and self.can_build_light
            and (game_state.player_nr_lights / game_state.player_nr_factories <= 10 or self.power > 1000)
        ):
            return BuildLightGoal(self)

        elif self.cargo.water - water_cost > 50 and water_cost < 5:
            return WaterGoal(self)

        elif game_state.env_steps > 750 and self.can_water() and self.cargo.water - water_cost > game_state.steps_left:
            return WaterGoal(self)

        return FactoryNoGoal(self)

    @property
    def daily_charge(self) -> int:
        return self.env_cfg.FACTORY_CHARGE

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
    def available_units(self) -> list[Unit]:
        # TODO some checks to see if there is enough power or some other mechanic to set units as unavailable
        return [unit for unit in self.units if not unit.private_action_plan and unit.can_be_assigned]

    @property
    def heavy_available_units(self) -> list[Unit]:
        return [unit for unit in self.available_units if unit.is_heavy]

    @property
    def light_available_units(self) -> list[Unit]:
        return [unit for unit in self.available_units if unit.is_light]

    @property
    def unassigned_units(self) -> list[Unit]:
        # TODO some checks to see if there is enough power or some other mechanic to set units as unavailable
        return [unit for unit in self.units if not unit.private_action_plan]

    @property
    def has_unassigned_units(self) -> bool:
        return any(not unit.private_action_plan for unit in self.units)

    def schedule_unit(
        self, strategy: Strategy, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        if strategy == Strategy.INCREASE_LICHEN:
            action_plan = self.schedule_strategy_increase_lichen(game_state, constraints, power_tracker)
        elif strategy == Strategy.INCREASE_UNITS:
            action_plan = self.schedule_strategy_increase_units(game_state, constraints, power_tracker)
        elif strategy == Strategy.COLLECT_ICE:
            action_plan = self.schedule_strategy_collect_ice(game_state, constraints, power_tracker)
        else:
            raise ValueError("Strategy is not a known strategy")

        return action_plan

    def schedule_strategy_increase_lichen(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        if not self.enough_water_collection_for_next_turns(game_state):
            return self.schedule_strategy_collect_ice(game_state, constraints, power_tracker)
        else:
            return self._schedule_dig_rubble_for_lichen(game_state, constraints, power_tracker)

    def _schedule_dig_rubble_for_lichen(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        rubble_pos = self._get_most_suitable_dig_pos_for_lichen(game_state)
        unit = self._get_closest_available_unit_to_pos(rubble_pos)
        # TODO, check if the unit can do this
        # try:
        goal = unit.generate_clear_rubble_goal(game_state, Coordinate(*rubble_pos), constraints, power_tracker)
        # except:
        # ...
        # Something about the unit being unassignable?

        unit.set_goal(goal)
        unit.set_private_action_plan(goal.action_plan)
        return goal.action_plan

    def _get_closest_available_unit_to_pos(self, pos: np.ndarray) -> Unit:
        c = Coordinate(*pos)
        unit = min(self.available_units, key=lambda u: c.distance_to(u.tc))
        return unit

    def _get_most_suitable_dig_pos_for_lichen(self, game_state: GameState) -> np.ndarray:
        rubble_positions_favorite = {tuple(pos) for pos in self.rubble_positions_next_to_can_spread_pos}
        valid_rubble_positions = rubble_positions_favorite - game_state.positions_in_dig_goals
        if valid_rubble_positions:
            # TODO Pick best e.g. lowest rubble or optimally given units closeby
            pos = np.array(list(valid_rubble_positions)[0])
            return pos

        rubble_positions_second_favorite = {tuple(pos) for pos in self.rubble_positions_next_to_can_not_spread_lichen}
        valid_rubble_positions = rubble_positions_second_favorite - game_state.positions_in_dig_goals

        if valid_rubble_positions:
            pos = np.array(list(valid_rubble_positions)[0])
            return pos

        # TODO, handle this either catch it or find another pos
        raise ValueError("No suitable pos")

    def enough_water_collection_for_next_turns(self, game_state: GameState) -> bool:
        water_collection = self.get_water_collection_per_step(game_state)
        water_available_next_n_turns = self.water + CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * water_collection
        water_cost_next_n_turns = CONFIG.ENOUGH_WATER_COLLECTION_NR_TURNS * self.water_cost
        return water_available_next_n_turns > water_cost_next_n_turns

    def schedule_strategy_increase_units(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        # Collect Ore / Clear Path to Ore / Supply Power to heavy on Ore
        ore_positions = {tuple(pos) for pos in game_state.board.ore_positions}
        valid_ore_positions = ore_positions - game_state.positions_in_dig_goals
        valid_ore_positions = np.array([np.array(pos) for pos in valid_ore_positions])
        ore_pos, _ = get_closest_pos_and_pos_between_positions(valid_ore_positions, self.positions)
        unit = self._get_closest_available_unit_to_pos(ore_pos)

        goal = unit.generate_collect_ore_goal(game_state, Coordinate(*ore_pos), constraints, power_tracker, self)
        unit.set_goal(goal)
        unit.set_private_action_plan(goal.action_plan)
        return goal.action_plan

    def schedule_strategy_collect_ice(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        # Collect Ice / Clear Path to Ice / Supply Power to heavy on Ice
        # If heavy available:
        #    Find closest position next to factory with no heavy on it
        #         Put heavy on it, potentially remove light on it
        if self.heavy_available_units:
            self._schedule_heavy_on_ice(self)
        else:
            ice_positions = {tuple(pos) for pos in game_state.board.ice_positions}
            valid_ice_positions = ice_positions - game_state.positions_in_dig_goals
            valid_ice_positions = np.array([np.array(pos) for pos in valid_ice_positions])
            ice_pos, _ = get_closest_pos_and_pos_between_positions(valid_ice_positions, self.positions)
            unit = self._get_closest_available_unit_to_pos(ice_pos)

            goal = unit.generate_collect_ice_goal(game_state, Coordinate(*ice_pos), constraints, power_tracker, self)
            unit.set_goal(goal)
            unit.set_private_action_plan(goal.action_plan)
            return goal.action_plan

    def _schedule_heavy_on_ice(
        self, game_state: GameState, constraints: Constraints, power_tracker: PowerTracker
    ) -> ActionPlan:
        valid_ice_positions = self.closest_ice_positions_set - game_state.positions_in_heavy_dig_goals

        heavy_available_positions = np.array([tuple(heavy.tc.xy) for heavy in self.heavy_available_units])
        heavy_pos, ice_pos = get_closest_pos_and_pos_between_positions(heavy_available_positions, valid_ice_positions)
        heavy_unit = game_state.get_player_unit_on_c(heavy_pos)
        goal = heavy_unit.generate_collect_ice_goal(game_state, Coordinate(*ice_pos), constraints, power_tracker, self)
        heavy_unit.set_goal(goal)
        heavy_unit.set_private_action_plan(goal.action_plan)
        return goal.action_plan
