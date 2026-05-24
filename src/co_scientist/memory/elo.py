from __future__ import annotations


def update_elo(
    winner_rating: int,
    loser_rating: int,
    *,
    k: int = 32,
) -> tuple[int, int]:
    """Return updated Elo ratings after the first player beats the second."""
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
    new_winner = round(winner_rating + k * (1 - expected_winner))
    new_loser = round(loser_rating + k * (0 - expected_loser))
    return new_winner, new_loser
