from agent.lux.kit import GameState


def is_my_turn_to_place_factory(game_state: GameState, step: int) -> bool:
    has_factories_to_place = game_state.player_team.factories_to_place > 0
    my_turn = _is_my_turn(game_state.player_team.place_first, step=step)

    return has_factories_to_place and my_turn


def _is_my_turn(place_first: bool, step: int) -> bool:
    if place_first:
        return step % 2 == 1
    else:
        return step % 2 == 0
