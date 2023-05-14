import argparse
import csv
from typing import List

import pandas as pd


def strip_non_alpha_numeric(str_: str) -> str:
    return "".join(ch for ch in str_ if ch.isalnum())


def is_nr_seconds_row(row: List[str]) -> bool:
    return len(row) == 1


def is_init_row(row: List[str]) -> bool:
    return row[0].startswith("player_")


def is_unit_collision_row(row: List[str]) -> bool:
    return "collided" in row and "surviving" not in row


def is_unit_surviving_collision_row(row: List[str]) -> bool:
    return "collided" in row and "surviving" in row


def is_invalid_dig_action_row(row: List[str]) -> bool:
    return row[1] == "Invalid" and row[2] == "Dig"


def is_invalid_movement_action_row(row: List[str]) -> bool:
    return row[1] == "Invalid" and row[2] == "movement"


def is_invalid_update_action_queue(row: List[str]) -> bool:
    return row[2] == "Tried"


def is_episode_final_row(row: List[str]) -> bool:
    return row[0] == "Episode:"


def parse_collision_row(row: List[str]) -> dict:
    time_step = int(strip_non_alpha_numeric(row[0]))
    nr_units_collided = int(row[1]) + 1
    collision_spot = row[-1]
    collision_spot_x, collision_spot_y = [int(c) for c in collision_spot.split(",")]
    lost_units = [strip_non_alpha_numeric(c) for c in row[3:-4]]

    return dict(
        time_step=time_step,
        nr_units_collided=nr_units_collided,
        collision_spot_x=collision_spot_x,
        collision_spot_y=collision_spot_y,
        lost_unit=lost_units,
    )


def parse_surviving_collision_row(row: List[str]) -> dict:
    time_step = int(strip_non_alpha_numeric(row[0]))
    nr_units_collided = int(row[1]) + 1
    collision_spot = row[-12]
    collision_spot_x, collision_spot_y = [int(c) for c in collision_spot.split(",")]
    lost_units = [strip_non_alpha_numeric(c) for c in row[3:-14]]
    surviving_unit = row[-9]
    surviving_power = int(row[-2])

    return dict(
        time_step=time_step,
        nr_units_collided=nr_units_collided,
        collision_spot_x=collision_spot_x,
        collision_spot_y=collision_spot_y,
        surviving_unit=surviving_unit,
        lost_unit=lost_units,
        surviving_power=surviving_power,
    )


def parse_invalid_action_row(row: List[str]) -> dict:
    time_step = int(strip_non_alpha_numeric(row[0][:-1]))
    action_type = row[2]
    unit_id = row[7]
    unit_type = row[8]
    unit_at_x = strip_non_alpha_numeric(row[10])
    unit_at_y = strip_non_alpha_numeric(row[11])

    if len(row) == 22:
        unit_to_x = strip_non_alpha_numeric(row[17])
        unit_to_y = strip_non_alpha_numeric(row[18])
    else:
        unit_to_x = None
        unit_to_y = None

    required_power = row[-2]

    return dict(
        time_step=time_step,
        action_type=action_type,
        unit_id=unit_id,
        unit_type=unit_type,
        unit_at_x=unit_at_x,
        unit_at_y=unit_at_y,
        unit_to_x=unit_to_x,
        unit_to_y=unit_to_y,
        required_power=required_power,
    )


def parse_invalid_update_action_queue(row: List[str]) -> dict:
    time_step = int(strip_non_alpha_numeric(row[0][:-1]))
    action_type = "Action Queue Update"
    unit_id = row[9]
    unit_type = row[10]
    unit_at_x = strip_non_alpha_numeric(row[12])
    unit_at_y = strip_non_alpha_numeric(row[13])
    required_power = row[-2]

    return dict(
        time_step=time_step,
        action_type=action_type,
        unit_id=unit_id,
        unit_type=unit_type,
        unit_at_x=unit_at_x,
        unit_at_y=unit_at_y,
        unit_to_x=None,
        unit_to_y=None,
        required_power=required_power,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli_output_path", type=str)
    args = parser.parse_args()

    cli_output_path: str = args.cli_output_path

    with open(cli_output_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=" ")

        collision_rows = []
        invalid_action_rows = []
        for row in reader:
            if is_nr_seconds_row(row) or is_init_row(row) or is_episode_final_row(row):
                continue
            elif is_unit_collision_row(row):
                collision_rows.append(parse_collision_row(row))
            elif is_unit_surviving_collision_row(row):
                collision_rows.append(parse_surviving_collision_row(row))
                continue
            elif is_invalid_dig_action_row(row):
                invalid_action_rows.append(parse_invalid_action_row(row))
            elif is_invalid_movement_action_row(row):
                invalid_action_rows.append(parse_invalid_action_row(row))
            elif is_invalid_update_action_queue(row):
                invalid_action_rows.append(parse_invalid_update_action_queue(row))
            else:
                raise ValueError(f"Unknown type of row: \n {row}")

    cli_output_name = cli_output_path.split(".")[0].split(r"/")[1]

    pd.DataFrame(collision_rows).explode("lost_unit").to_csv(f"data/{cli_output_name}_collisions.csv", index=False)
    pd.DataFrame(invalid_action_rows).to_csv(f"data/{cli_output_name}_invalid_actions.csv", index=False)
