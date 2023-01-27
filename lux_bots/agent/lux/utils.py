def is_my_turn_to_place_factory(player: str, game_state, step: int) -> bool:
    factories_to_place = game_state.teams[player].factories_to_place
    my_turn = _is_my_turn(game_state.teams[player].place_first, step=step)

    return factories_to_place and my_turn


def _is_my_turn(place_first: bool, step: int) -> bool:
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1