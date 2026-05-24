from __future__ import annotations

from co_scientist.memory import update_elo


def test_even_match_updates_by_sixteen_points() -> None:
    assert update_elo(1200, 1200) == (1216, 1184)


def test_underdog_win_moves_more_than_favorite_win() -> None:
    underdog_win = update_elo(1000, 1400)
    favorite_win = update_elo(1400, 1000)

    assert underdog_win[0] - 1000 > favorite_win[0] - 1400
    assert underdog_win[1] < 1400
    assert favorite_win[1] < 1000
