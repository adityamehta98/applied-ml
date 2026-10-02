"""SQL feature queries against SQLite: per-user aggregates, a rolling window, a per-group rank.

The queries every analyst writes for features live here as named statements, so a test can
run each one and check the numbers. SQLite ships with Python, so this needs no database server.
"""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    event_date TEXT    NOT NULL,   -- ISO date 'YYYY-MM-DD'
    amount     REAL    NOT NULL
);
"""

# Per-user aggregates: the bread-and-butter GROUP BY feature.
PER_USER_TOTALS = """
SELECT user_id,
       COUNT(*)    AS n_events,
       SUM(amount) AS total_amount,
       AVG(amount) AS avg_amount
FROM events
GROUP BY user_id
ORDER BY user_id;
"""

# 7-day rolling sum per user. julianday() turns the date into a number of days, so a RANGE
# frame can look back exactly 6 days (today plus the 6 days before it).
ROLLING_7DAY = """
SELECT user_id,
       event_date,
       SUM(amount) OVER (
           PARTITION BY user_id
           ORDER BY julianday(event_date)
           RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7d
FROM events
ORDER BY user_id, event_date;
"""

# Second-highest amount per user. DENSE_RANK gives tied amounts the same rank, so rnk = 2 is
# the true runner-up value even when the top value is tied.
SECOND_HIGHEST = """
SELECT user_id, amount
FROM (
    SELECT user_id,
           amount,
           DENSE_RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS rnk
    FROM events
)
WHERE rnk = 2
ORDER BY user_id;
"""


def connect():
    """A fresh in-memory database with the schema loaded and dict-style rows."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def run(conn, query):
    """Execute a query and return its rows as a list of dicts (column name -> value)."""
    return [dict(r) for r in conn.execute(query)]
