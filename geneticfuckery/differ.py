#!/usr/bin/env python3

import sqlite3
import pathlib
import difflib
import itertools
import datetime

from loguru import logger
from .gf import GeneticFuckery

import multiprocessing

multiprocessing.set_start_method("fork")

from collections import defaultdict

here = pathlib.Path(__file__).parent


def queryResults(poolargs):
    """More efficient parallel query runner for the initial big result sets"""
    query, args, db = poolargs
    cur = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    logger.info("[{}] Executing maximal query...", db)
    return (db, list(cur.execute(query, args).fetchall()))


@logger.catch
def differ(modulecmd: str, *dbs: str, output: str = "gf-diffed", maxCompare: int = 10):
    """Diff gf parameter DBs using `query` and saving HTML output to `output`."""

    logger.info("Comparing: {}", dbs)

    opened = []
    for db in dbs:
        opened.append(
            (
                db,
                sqlite3.connect(db, isolation_level=None, check_same_thread=False),
            )
        )

    # query for BEST algos in entire DB
    maximal = (here / "maximal.sql").read_text()
    pidsPerDB = {}
    with multiprocessing.Pool() as pool:
        for db, result in pool.map(
            queryResults, itertools.product([maximal], [(maxCompare,)], dbs)
        ):
            # these lists are: pid, dataset, category, score (profit), weight (profit / trades)
            # Note: 'category' here is ALGO NAME
            pidsPerDB[db] = result

    # Copy original result set of each pid in each DB into a new DB!
    now = str(round(datetime.datetime.now().timestamp() * 1000))

    newdb = f"differ-{now}.db"
    logger.info("Creating new parameter DB: {}", newdb)
    gf = GeneticFuckery(dbfile=newdb)

    c = gf.db.cursor()
    c.execute("BEGIN")

    for db, cur in opened:
        for pid, dataset, algo, *rest in pidsPerDB[db]:
            # Read from EXISTING DB
            got = cur.execute("SELECT * FROM params WHERE pid = ?", (pid,)).fetchall()

            # Insert into NEW DB
            c.execute(
                "INSERT INTO param_meta (generated, parents) VALUES (?, ?)",
                (
                    datetime.datetime.now(),
                    f"from {db} :: {pid} (because {algo} on {dataset}",
                ),
            )
            newpid = c.lastrowid

            # logger.info("Inserting {}", [(newpid, *rest[1:]) for rest in got])

            # Insert all parameters (removing original pid and replacing with new pid)
            c.executemany(
                "INSERT INTO params (pid, category, name, value) VALUES (?, ?, ?, ?)",
                [(newpid, *rest[1:]) for rest in got],
            )

    c.execute("COMMIT")
    c.close()

    # query for BEST algos BY CATEGORY in entire DB
    # categories = (here / "categories.sql").read_text()

    # query to read all settings for a given parameter group
    # Note: 'category' here is PARAMETER NAME
    q = """SELECT category, name, value FROM params WHERE pid = ? ORDER BY category, name"""

    results = defaultdict(lambda: None)
    categoryCompares = set()
    resultPerCategory = defaultdict(set)

    for db, cur in opened:
        # top N per DB
        for key in pidsPerDB[db]:
            pid, dataset, algo, score, weight = key
            rs = []
            for category, name, value in cur.execute(q, (pid,)).fetchall():
                categoryCompares.add(category)
                c = str(category)
                n = str(name)
                v = str(value)
                rowstr = f"{c:<50} | {n:<35} | {v:<25}"
                rs.append(rowstr)

            logger.info("Adding result: {}", (db, key))

            dbkey = (db,) + key
            results[dbkey] = rs
            resultPerCategory[algo].add(dbkey)

    d = difflib.HtmlDiff(tabsize=4)

    # generate a diff against all combinations of input requests

    # we want to generate diffs only for same category/algo compares:

    # report algo counts
    logger.info("------========== ALGO REPORT ==========------")
    with open("data/static-logger.txt", "w") as sl:
        for algo, row in sorted(
            resultPerCategory.items(), key=lambda x: len(x[1]), reverse=True
        ):
            logger.info(
                "[{}]: {:,} members (avg weight: {:,.2f})",
                algo,
                len(row),
                sum([x[-1] for x in row]) / len(row),
            )

            for db, pid, dataset, algo, score, weight in row:
                sl.write(
                    f"time GF_DB={db} GF_PARAM_GROUP={pid} poetry run {modulecmd.replace('{}', dataset)}\n"
                )

    logger.info("Wrote graph helper to data/static-logger.txt")
    logger.info("------======== END ALGO REPORT =========------")

    categoryCombos = {}
    for algo, members in resultPerCategory.items():
        combos = list(itertools.combinations(members, 2))
        if combos:
            categoryCombos[algo] = combos
        else:
            logger.info("NO COMBOS FOR {}: {}", algo, members)
            # else, NO combination meaning 'members' is only one element, but we
            # still want it in the output, so make it a (self, None) combo so we can
            # at least print the full output.
            categoryCombos[algo] = [(list(members)[0], None)]

    for algo, combos in categoryCombos.items():
        for idxCombo, (key1, key2) in enumerate(combos):
            a = results[key1]
            b = results[key2] or []  # protect against same-compare-empty-None thing

            db1, pid1, ds1, cat1, score1, weight1 = key1

            # helper for when we're just printing with no actual compare diff
            if key2:
                db2, pid2, ds2, cat2, score2, weight2 = key2
            else:
                db2, pid2, ds2, cat2, score2, weight2 = key1
                score2 = 0
                weight2 = 0

            assert algo == cat1 == cat2

            got = d.make_file(
                a,
                b,
                f"{db1} for pid {pid1} ({ds1} :: {cat1} => {score1:,.2f} ({weight1:,.2f}))",
                f"{db2} for pid {pid2} ({ds2} :: {cat2} => {score2:,.2f} ({weight2:,.2f}))",
                context=True,
            )

            f = output + f"-{algo}-{round(score1 + score2)}-{idxCombo}.html"
            logger.info("[{}, {}] Writing file: {}", pid1, pid2, f)
            fixed = got.replace("Courier", "Monaco").replace(
                "td ", "td style='padding: 0 1em'"
            )
            pathlib.Path(f).open("w").write(fixed)


def cmd():
    import fire

    fire.Fire(differ)


if __name__ == "__main__":
    cmd()
