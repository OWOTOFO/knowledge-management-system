import pytest

from km.retrieval.sql import SQLGuardError, SQLRetriever, guard_sql

ALLOWED = {"products", "complaints", "branches", "monthly_loan_volume"}


@pytest.mark.parametrize("sql", [
    "DROP TABLE products",
    "UPDATE products SET annual_fee = 0",
    "DELETE FROM complaints",
    "INSERT INTO branches VALUES (9, 'x', 'y', 'z', '2020-01-01')",
    "SELECT * FROM products; DROP TABLE products",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM salaries",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA writable_schema = 1",
    "SELECT load_extension('evil')",
])
def test_guard_rejects_unsafe(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql, ALLOWED, 50)


def test_guard_adds_limit_and_allows_ctes():
    out = guard_sql("WITH t AS (SELECT type FROM products) SELECT type, COUNT(*) FROM t GROUP BY type", ALLOWED, 50)
    assert out.endswith("LIMIT 50")
    # the cap applies even when the query brings its own, larger LIMIT
    assert guard_sql("SELECT name FROM products LIMIT 100000", ALLOWED, 50).endswith("LIMIT 50")


def test_connection_is_read_only(settings):
    import sqlite3

    retriever = SQLRetriever(settings, llm=None)
    from contextlib import closing

    with closing(retriever._connect()) as con, pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM products")
