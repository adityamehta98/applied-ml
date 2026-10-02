import pytest

from src.sql_features import (PER_USER_TOTALS, ROLLING_7DAY, SECOND_HIGHEST,
                              connect, run)

SEED = [
    # event_id, user_id, event_date, amount
    (1, 1, "2024-01-01", 10.0),
    (2, 1, "2024-01-03", 20.0),
    (3, 1, "2024-01-05", 5.0),
    (4, 1, "2024-01-09", 100.0),   # 8 days after 01-01, so 01-01 drops out of its 7-day window
    (5, 2, "2024-01-02", 7.0),
    (6, 2, "2024-01-02", 3.0),     # same day as event 5
    (7, 2, "2024-01-20", 50.0),
]


@pytest.fixture()
def conn():
    c = connect()
    c.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", SEED)
    yield c
    c.close()


def test_per_user_totals(conn):
    rows = run(conn, PER_USER_TOTALS)
    assert rows[0] == {"user_id": 1, "n_events": 4, "total_amount": 135.0, "avg_amount": 33.75}
    assert rows[1]["user_id"] == 2 and rows[1]["total_amount"] == 60.0


def test_rolling_7day_window(conn):
    rows = run(conn, ROLLING_7DAY)
    by = {(r["user_id"], r["event_date"]): r["rolling_7d"] for r in rows}
    assert by[(1, "2024-01-05")] == 35.0    # 10 + 20 + 5, all within 7 days
    assert by[(1, "2024-01-09")] == 125.0   # 20 + 5 + 100; the 01-01 event has aged out
    assert by[(2, "2024-01-20")] == 50.0    # isolated event, only itself


def test_second_highest_per_user(conn):
    rows = run(conn, SECOND_HIGHEST)
    result = {r["user_id"]: r["amount"] for r in rows}
    assert result == {1: 20.0, 2: 7.0}      # user 1: 100 then 20; user 2: 50 then 7


def test_second_highest_handles_a_tie_at_the_top(conn):
    conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)",
                     [(8, 3, "2024-02-01", 90.0), (9, 3, "2024-02-02", 90.0), (10, 3, "2024-02-03", 40.0)])
    result = {r["user_id"]: r["amount"] for r in run(conn, SECOND_HIGHEST)}
    assert result[3] == 40.0                # DENSE_RANK: the tied 90s share rank 1, so 40 is rank 2
