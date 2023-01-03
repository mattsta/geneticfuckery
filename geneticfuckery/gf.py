from loguru import logger

from dataclasses import dataclass, field

from typing import Any, Optional, Iterator, Collection, Iterable

from .config import settings

import math
import sqlite3
import datetime
import orjson
import pathlib
import numpy as np
import pandas as pd

import os
import sys
import functools
import itertools
import contextlib
import subprocess
import multiprocessing

from collections import defaultdict, Counter
import random  # good enough

import pprint as pp

isLinux = sys.platform == "linux"


def metricQuery(poolargs):
    db, dataset, topN = poolargs
    cur = sqlite3.connect(db, isolation_level=None, check_same_thread=False)
    logger.info("[{}] Querying dataset: {}", db, dataset)
    fs = []

    # this should be in an external thing somewhere, but it's a good example of creating a complex metric
    # by combining ranking and filtering of multiple result output values.
    # this is also kinda duplicated in mergePerformers() as `BEST_PERFORMERS_QUERY` with different assumptions.
    BIGQ = """
WITH
-- find trades with reasonable values
-- (each "trade" here would actually be two market transactions to open+close)
goodwins AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS win
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'win')
    ),

goodtrades AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS trades
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'trades' AND value > 0 AND value <= 150 AND dataset = ?)
    ),

-- find the best profit for each dataset from the trades group
goodprofit AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS profit
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'profit' AND run IN (SELECT run FROM goodtrades))
    )

-- return combined result
SELECT run,
       pid,
       dataset,
       family,
       algo,
       max(profit) AS profit,
       trades,
       round(profit / trades, 2) AS ppt -- "profit per trade" including all trades
FROM goodprofit
JOIN goodtrades USING (run, pid, dataset, family, algo)
JOIN goodwins USING (run, pid, dataset, family, algo)
-- WHERE substr(dataset, 1, 3) = 'SPY'
-- WHERE pid >= 400
-- ORDER BY profit DESC;
-- GROUP BY dataset, algo
GROUP BY algo
ORDER BY ppt DESC
LIMIT ?
"""

    for (run, pid, dataset, family, algo, profit, trades, ppt) in cur.execute(
        BIGQ,
        (dataset, topN),
    ).fetchall():
        # TODO: replcae family-detection with just pid merging.
        # The "family" field should be a debug column, not for data processing.
        fs.append(family)

    cur.close()
    return fs


def buildbestOfResultSet(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    """Promote "name" => name; "value" => value pairs in result set
    to top-level 'name' => 'value' pairs per row."""

    cur.row_factory = sqlite3.Row

    results = []
    for row in cur.fetchall():
        result = dict(row)

        # add helper for direct top level name => value access
        result.update({row["name"]: row["value"]})
        results.append(result)

    return results


def loadIfNeeded(v):
    # load as JSON if looks like a string-thing
    # (NB: any saved string values need to be quoted so the JSON
    #  deocder keeps them as strings...)
    return orjson.loads(v) if type(v) in {str, bytes, memoryview} else v


def runner(args, env):
    logger.info("[{}] with env {}", " ".join(args), env)
    subprocess.run(args, shell=False, check=True, env=env)


@dataclass
class SubSelector:
    """Helper class for returning a nested attribte accessor."""

    memory: dict[str, Any]

    @logger.catch
    def __getattr__(self, param: str) -> Any:
        # logger.opt(depth=2).info("Requesting {} from memory: {}", param, self.memory)
        param = param.lower()
        got = self.memory[param]

        # if got a default empty value, it didn't exist...
        if isinstance(got, dict) and not got:
            return None
            raise KeyError(f"Not present: {param}")

        if param.endswith("_int"):
            return int(got)

        if param.endswith("_bool"):
            return bool(got)

        if param.endswith("_choice"):
            assert isinstance(got, list)
            return got[0]

        if isinstance(got, dict):
            return SubSelector(got)

        # default to real/float
        return float(got)


@dataclass
class GeneticFuckery:
    """A system for searching parameter spaces for optimal performance.

    Data layout looks like:
        METADATA :: PK iteration id
            - iteration id / run id
            - timestamp of iteration start
            - timestamp of iteration stop
            - heritage
            - FAMILY (dataset run under; same for entire run ID)

        PARAMETERS :: PK (iteration id, param, name)
            - iteration id / run id
            - param (think of: class/method/function scope for parameter k/v)
            - param name
            - param value

        RESULTS :: PK (iteration id, algo, name)
            - iteration id / run id
            - algo (think of: algo for this k/v result)
            - result name
            - result value

    Using this data layout, we can manage an arbitrary number of
    parameters in a 1-depth hierarchy where parameters are namespaced
    so multiple instances of the same parameter set can be used
    in each individual iteration.

    This system also provides two interfaces for accessing the
    store parameters for future optimization runs or long-term
    preferred iteration repeating:
        - direct interactive instance attribute access
        - exporting a full environment configuration for .env loading
    """

    # Note: there's a "GF_" prefix prepended to these settings, so
    #       'settings.DB. is actually env var 'GF_DB' etc.
    dbfile: str = field(default=str(settings.DB) if "DB" in settings else "gf.db")

    # if this is a restore, the run ID can be provided for accessing
    # previously stored parameters and auto-selecting run ID for result populating.
    pid: Optional[int] = field(
        default_factory=lambda: int(settings.PARAM_GROUP)
        if "PARAM_GROUP" in settings
        else None
    )

    # default family for parameter lookup
    family: Optional[str] = field(
        default=str(settings.FAMILY) if "FAMILY" in settings else None
    )

    # current dataset under evaluation
    dataset: Optional[str] = field(
        default=str(settings.DATA) if "DATA" in settings else None
    )

    # holder for getattr lookup cache
    # cache is {FAMILY => {PARAM => {NAME => VALUE}}}
    memory: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def __post_init__(self) -> None:
        """Getting started..."""

        logger.opt(depth=2).info(
            "[{} :: {}] Loading parameters...", self.dbfile, self.pid
        )

        self.db = sqlite3.connect(
            self.dbfile, isolation_level=None, check_same_thread=False
        )

        logger.opt(depth=2).info("[{} :: {}] Loaded parameters!", self.dbfile, self.pid)

        # runtime bug prevention during read-only macos fork sessions
        if False and (not isLinux) and (multiprocessing.get_start_method() == "fork"):
            logger.warning(
                "Not running DB table creations because running in forked configuration."
            )
            return

        self.db.execute("PRAGMA journal_mode = WAL")

        # off to the races
        v = self.db

        # TODO: we could replace pid and run ids with ulids if we want easier merging of databases
        #       across multiple runs: ulid.new()
        v.execute(
            f"CREATE TABLE IF NOT EXISTS param_meta (pid INTEGER PRIMARY KEY AUTOINCREMENT, generated, parents)"
        )
        v.execute(
            f"CREATE TABLE IF NOT EXISTS run_meta (run INTEGER PRIMARY KEY AUTOINCREMENT, pid INTEGER, start, stop, family, dataset)"
        )
        v.execute(
            f"CREATE TABLE IF NOT EXISTS params (pid, param, name, value, PRIMARY KEY(pid, param, name))"
        )
        v.execute(
            f"CREATE TABLE IF NOT EXISTS results (run, algo, name, value, PRIMARY KEY(run, algo, name))"
        )

        # livin la index loca
        v.execute(f"CREATE INDEX IF NOT EXISTS results_cat ON results (algo)")
        v.execute(f"CREATE INDEX IF NOT EXISTS results_fcn ON results (algo, name)")
        v.execute(f"CREATE INDEX IF NOT EXISTS results_val ON results (name, value)")
        v.execute(f"CREATE INDEX IF NOT EXISTS params_runcat ON params (pid, param)")
        v.execute(f"CREATE INDEX IF NOT EXISTS run_meta_family ON run_meta (family)")
        v.execute(f"CREATE INDEX IF NOT EXISTS run_meta_dataset ON run_meta (dataset)")

    def pids(self) -> Iterable[int]:
        return [
            x
            for x, in self.db.execute("SELECT DISTINCT pid FROM param_meta").fetchall()
        ]

    def populate(self, filenames: list[str]) -> int:
        """Populate DB entries from a bootstrap source file.

        Bootstrap source file format is lines of hierarchical-like key value pairs like:
        param__name = value (where 'param' can also have nested __ attributes).

        Returns new run ID for this new parameter group."""

        populate: dict[str, dict[str, Any]] = defaultdict(dict)
        for filename in filenames:
            with open(filename, "r") as f:
                for line in f:
                    # Lines look like:
                    # partA__partB__partC__name_int = value
                    # So we want to deconstruct then re-assemble a dict entry of:
                    # populate[partA__partB__partC][name_int] = value
                    line = line.strip()

                    # skip empty lines or comments
                    if not line or line.startswith("#"):
                        continue

                    name, val = line.split("=")

                    name = name.strip()
                    val = val.strip()

                    nameparts = name.split("__")
                    param = "__".join(nameparts[:-1])
                    name = nameparts[-1]

                    logger.info("Adding {} :: {} = {}", param, name, val)
                    populate[param][name] = orjson.loads(val)

        newId = self.metaParamsNew(dict(origin=f"bootstrapping via {filename}"))

        self.parametersNew(populate)

        logger.info("Created parameters under new run id: {}", newId)
        return newId

    def metaParamsNew(
        self,
        parents: dict[str, Any],
    ) -> int:
        """Creates new meta entry and returns the new unique run id."""
        c = self.db.cursor()

        c.execute(
            "INSERT INTO param_meta (generated, parents) VALUES (?, ?)",
            (datetime.datetime.now().timestamp(), orjson.dumps(parents)),
        )

        made = c.lastrowid

        c.close()

        # new family/id is now the run id for this instance
        self.useParameterId(made)

        return made

    def useParameterId(self, pid: int) -> None:
        """Set input parameter id as global state for this instance."""
        self.pid = pid

    def parametersNew(self, params: dict[str, dict[str, Any]]) -> None:
        """Insert new parameters for current 'runId'.

        'params' format is:
            param => {name => value}

        (family and runid are used from instance globals)
        """

        c = self.db.cursor()
        c.execute("BEGIN")

        paramId = self.pid

        # record all used parameters for this run entry
        pInsert = []
        for k, ps in params.items():
            for pname, pval in ps.items():
                pInsert.append(
                    (
                        paramId,
                        k.lower(),
                        pname.lower(),
                        orjson.dumps(pval) if type(pval) in {list, bool} else pval,
                    )
                )

        # use INSERT OR REPLACE to allow our input format to have redundant keys where the
        # later key-value pairs overwrite earlier key-value pairs (good for setting new
        # defaults, etc)
        c.executemany(
            "INSERT OR REPLACE INTO params (pid, param, name, value) VALUES (?, ?, ?, ?)",
            tuple(pInsert),
        )

        if False:
            logger.info("Inserting: {}", pp.pformat(pInsert))
            logger.info(
                "ALL:\n{}", pp.pformat(c.execute("SELECT * FROM params").fetchall())
            )
            logger.info(
                "ALL_M:\n{}",
                pp.pformat(c.execute("SELECT * FROM param_meta").fetchall()),
            )

        c.execute("COMMIT")

    @contextlib.contextmanager
    def report(
        self, family: Optional[str] = None
    ) -> Iterator[dict[str, dict[str, Any]]]:
        """Context manager to auto-handle start/stop times and provide a dict for reporting."""
        assert self.pid

        start = datetime.datetime.now().timestamp()
        grower: dict[str, dict[str, Any]] = defaultdict(dict)
        yield grower
        stop = datetime.datetime.now().timestamp()
        self.resultsAdd(start, stop, grower, family)
        logger.info(
            "[{}] Logging results for: {}", ", ".join(sorted(grower)), self.dbfile
        )
        logger.info(
            "[reporter] Added Results over Duration {:,.3f} seconds", stop - start
        )

    def resultsAdd(
        self,
        start: float,
        stop: float,
        results: dict[str, dict[str, Any]],
        family: Optional[str] = None,
    ) -> None:
        """Record a result for current runId.

        TODO: it would be nice to also record which result algos depend
        on which parameters because we have things like
        'result ATR-30 ALGO' only depending on the 12 ATR-30 params, not
        the other 40 algo params; so we should be able to merge the top-performing
        ATR-30 params with the top performing ATR-15, ATR-90, LRSI-{15,30,90} etc
        params as determined by their result algos.

        So we need a map between result algos and the set of parameter
        params and names they heavily depend on (param prefix params should
        be fine because params are like:
            d15__VolStopExternalATR__Fast_True__name = val

        So we would want to match-merge on 'd15__VolStopExternalATR' with top performers
        from things like 'd90__VolStopExternalATR...' into one new solid run group,
        eventually, when they are all performing nicely independently."""

        if not (family or self.family):
            logger.warning("Not logging results because no family provided...")
            return

        c = self.db.cursor()
        c.execute("BEGIN")

        # Create new run entry...
        c.execute(
            "INSERT INTO run_meta (pid, start, stop, family, dataset) VALUES (?, ?, ?, ?, ?)",
            (self.pid, start, stop, (family or self.family).lower(), self.dataset),
        )

        runid = c.lastrowid

        # record all results for this run entry
        rInsert = []
        for k, rs in results.items():
            for rname, rval in rs.items():
                # don't allow conflicts with column names themselves since
                # we reduce key-value entries to top-level peers of column
                # names later in some results.
                assert rname.lower() not in {
                    "run",
                    "family",
                    "algo",
                    "name",
                    "value",
                }

                rInsert.append((runid, k.lower(), rname.lower(), rval))

        # allow overwrite of previous results instead of being
        # outright rejected for violating PK uniqueness.
        c.executemany(
            "INSERT OR REPLACE INTO results (run, algo, name, value) VALUES (?, ?, ?, ?)",
            tuple(rInsert),
        )

        c.execute("COMMIT")

    def highest(
        self, family: str, algos: Collection[str], name: str, howmany: int = 10
    ) -> list[dict[str, Any]]:
        """Returns column dicts with highest N results for individual features/algos."""

        logger.info(
            "Checking highest for: FAM {} :: CATS {} :: NAME {} :: COUNT {}",
            family,
            algos,
            name,
            howmany,
        )

        assert isinstance(algos, Collection)

        # ugly garbage because the sqlite3 driver needs one "?" for each IN check parameter
        # sorts HIGH to LOW
        cur = self.db.execute(
            f"""SELECT results.run, algo, name, value FROM results
                JOIN run_meta USING (run) WHERE run_meta.family = ? AND
                algo IN ({','.join(['?' for _ in range(len(algos))])})
                AND name = ? ORDER BY value DESC LIMIT ?""",
            (family.lower(), *[c.lower() for c in algos], name.lower(), howmany),
        )

        # convert theoretical row mappings into concrete dict mappings.
        # result is list of rows with first row being best result, etc.
        # as a usability helper, we promote the "name" field itself into
        # a top level key->value pair.
        return buildbestOfResultSet(cur)

    def lowest(
        self, family: str, algos: Collection[str], name: str, howmany: int = 10
    ) -> list[dict[str, Any]]:
        """Returns column dicts with lowest N algo results."""

        assert isinstance(algos, Collection)

        # sorts LOW to HIGH
        cur = self.db.execute(
            f"""SELECT results.run, algo, name, value FROM results
                JOIN run_meta USING (run) WHERE run_meta.family = ?
                AND algo IN ({','.join(['?' for _ in range(len(algos))])})
                AND name = ? ORDER BY value ASC LIMIT ?""",
            (family.lower(), *[c.lower() for c in algos], name.lower(), howmany),
        )

        return buildbestOfResultSet(cur)

    def namespace(self, src, *args, **kwargs) -> Any:
        """Self-describing wrapper for multi-level namespaced parameters."""

        srcName = src.__class__.__name__
        kwfmt = []
        for k, v in kwargs.items():
            kwfmt.extend("_".join([k, v]))

        category = "__".join([srcName, *args, *kwfmt])
        raise NotImplementedError  # TODO: finish the idea

    def __getattr__(self, param: str) -> Any:
        """Allow two level dot access based on param name.

        All attributes and names are case insensitive.

        Usage:
            gf = GF()
            depth = gf.algo_30.depth
            height = gf.algo_30.height

            a30 = gf.aglo_30
            depth = a30.depth
            height = a30.height
        """

        param = param.lower()

        # Only do lookup once because parameters should be immutable,
        # so once we read them we never have to check them again.
        if param not in self.memory:
            # we lowercase all params for case insentive usage

            # note: we don't need "family" here since we are using the run id
            # for matching and all parameters of a run id have the same family.
            got = self.db.execute(
                "SELECT param, name, value FROM params WHERE pid = ?", (self.pid,)
            )

            currentDict = self.memory

            # The tier hack below allows us to store params as linear strings
            # like:
            # - "d15__IndicatorState__init__length_int"
            # but allow dotted access like:
            # - gf.d15.IndicatorState.init.length_int
            for fcat, k, v in got.fetchall():
                # logger.info("pooopulating for {} => {}", k, v)
                subparts = fcat.split("__")
                for part in subparts:
                    # each param part is its own subdict
                    pname = part.lower()
                    if pname not in currentDict:
                        s = defaultdict(dict)
                        currentDict[part.lower()] = s
                        currentDict = s
                    else:
                        currentDict = currentDict[pname]
                else:
                    # final dict gets the actual key->value mappings
                    currentDict[k.lower()] = loadIfNeeded(v)

                # reset search dict to the primary holder again for next loop
                currentDict = self.memory

        # logger.info("Memory: {}", pp.pformat(self.memory))
        return SubSelector(self.memory[param])

    def breedRun(
        self,
        families: list[str],
        name: str,
        cmd: str,
        highest: bool = True,
        maxCompare: int = 6,
        maxSelfLooping: int = 1,
        generationBoost: int = 1,
    ):
        """Breed the families provided then exec args with proper environment for new run ids."""

        # families are all lower case...
        families = [x.lower() for x in families]

        logger.info("Running breed against families: {}", families)
        logger.info("Targeting result name: {}", name)
        logger.info("Targeting command: {}", cmd)
        logger.info("Iterating for future runs: {}", maxSelfLooping)

        # get all datasets in families for launch replacement
        datasets = set()

        # get all datasets used by all families for re-running the evaluations automatically...
        # (double loop instead of using IN (?, ?, ?) because sqlite3 driver doesn't support it nicely)

        # Convert all merged families into underlying familes for more dataset pulls
        allfams = set(families)
        for f in families:
            for innerfam in [
                x.strip() for x in f.replace("{", "").replace("}", "").split(",")
            ]:
                allfams.add(innerfam)

        logger.info(
            "Searching fams for underlying dataset comparisons: {}",
            list(sorted(allfams)),
        )

        for family in allfams:
            for (dataset,) in self.db.execute(
                "SELECT DISTINCT dataset FROM results JOIN run_meta USING (run) WHERE family = ?",
                (family,),
            ).fetchall():
                datasets.add(dataset)

        logger.info("Running against data: {}", list(sorted(datasets)))

        futurecmds = []
        for dataset in datasets:
            futurecmds.append((dataset, cmd.replace("{}", dataset).split()))

        assert futurecmds
        assert families
        assert name

        # when breeding generates new families, record them here so we can re-breed
        # the families for (hopefully) stronger and stronger outcomes (and if this is
        # against a large enough family set with enough datasets underneath, hopefully
        # overfitting won't be an excessive problem...)
        newfams = set(families)
        for gb in range(generationBoost):
            logger.info("Breeding fams: {}", newfams)

            for i in range(maxSelfLooping):
                # GENERATE NEW PARAMS - REGULAR
                generated = self.breed(newfams, name, highest, maxCompare)

                # new fams is all cross-breeding fams the breed generated
                newfams = set([x[0] for x in generated])
                logger.info("Generated fams: {}", newfams)

                # TODO: calculate the highest performing families from newfams then
                #       pariwise combine them.

                # Now for all input families datasets, we want to query:
                #  - For each dataset
                #       - Find top-K performing families (by name and sort order)
                #       - Add those top families to the next newfams set
                #       - Iterate on the top performers for breeding up to generationBoost count

                # RUN NEW PARAMETERS THROUGH METRIC GENERATION APPARATUS
                tasks = []
                for dataset, args in futurecmds:
                    for newfamily, newparamid in generated:
                        logger.info(
                            "[{} :: {}] [{} :: {}] {}",
                            gb,
                            i,
                            newparamid,
                            newfamily,
                            " ".join(args),
                        )
                        # https://docs.python.org/3/library/subprocess.html
                        env = {
                            k: str(v)
                            for k, v in dict(
                                GF_DB=self.dbfile,
                                GF_PARAM_GROUP=newparamid,
                                GF_FAMILY=newfamily,
                                GF_DATA=dataset,
                            ).items()
                        }

                        tasks.append((args, env))

                with multiprocessing.Pool() as pool:
                    logger.info("Running: {}", pp.pformat(tasks))
                    # 'args' replaces '{}' with the DATA / DATASET under test for the current family thing.

                    # DATA / DATASET = {SYMBOL}
                    # FAMILY = {SYMBOL}-{DATE}
                    pool.starmap(runner, tasks)
            else:
                newfams = set()

    def autoBreeder(
        self,
        cmd: str,
        metricName: str = "profit",
        topN: int = 3,
        loop: int = 1,
        highest: bool = True,
        maxCompare: int = 6,
        limitDatasetsDeny: list[str] = None,
        limitDatasetsAllow: list[str] = None,
    ):
        """Using current DB metrics, breed the top N families for each dataset K times."""

        datasets = set()

        # First get all datasets
        # (NOTE: you should probably have all datasets locally when running this test, or your evaluation
        #        process should safely handle non-existing datasets applied to it if needed)
        for (dataset,) in self.db.execute(
            "SELECT DISTINCT dataset FROM run_meta"
        ).fetchall():
            datasets.add(dataset)

        # if we want to remove some datasets, filter here
        if limitDatasetsDeny:
            remove = set()
            for d in datasets:
                for l in limitDatasetsDeny:
                    if l in d:
                        remove.add(d)

            datasets -= remove

        # if we are only allowing certain datasets, restrict it here
        if limitDatasetsAllow:
            keep = set()
            for d in datasets:
                for l in limitDatasetsAllow:
                    if l in d:
                        keep.add(d)

            datasets = keep

        # generate per-dataset commands using the 'cmd' template
        futurecmds = [
            (dataset, cmd.replace("{}", dataset).split()) for dataset in datasets
        ]

        # mix up symbols for a more even running distribution in case we want to
        # cancel a training run early
        random.shuffle(futurecmds)

        assert futurecmds
        assert datasets
        assert metricName

        # hardcoded special condition to use the big query from mergePerformers(). should be more externally extensible.
        breedmetricname = (
            "profit" if metricName == "max profit min trades" else metricName
        )

        for _i in range(loop):
            # breeders is a list because if we have a dominant strategy, we want it represented
            # multiple times to hopefully grow a longer successful lineage.
            breeders = []

            # Calculate best breed candidates
            with multiprocessing.Pool() as pool:
                if metricName == "max profit min trades":
                    # run this query in parallel because it takes 10-30 seconds per dataset fetch
                    for families in pool.map(
                        metricQuery, itertools.product([self.dbfile], datasets, [topN])
                    ):
                        breeders.extend(families)
                        logger.info("breeders length: {}", len(breeders))
                else:
                    # For each dataset, find the top N families for breeding:
                    # (TODO: we may want to limit this to only "growing successful" values instead of all datasets?)
                    for (family,) in self.db.execute(
                        "SELECT family FROM results JOIN run_meta USING (run) where dataset = ? AND name = ? order by value desc limit ?",
                        (dataset, metricName, topN),
                    ).fetchall():
                        breeders.append(family)

            assert breeders

            # re-order breeders randomly so if this is a long run we get a good mix up front
            # in case we want to stop it early.
            random.shuffle(breeders)

            # GENERATE NEW PARAMS - AUTO
            generated = self.breed(breeders, breedmetricname, highest, maxCompare)

            # new fams is all cross-breeding fams the breed generated
            newfams = set([x[0] for x in generated])
            logger.info("Generated fams: {}", newfams)

            # RUN NEW PARAMETERS THROUGH METRIC GENERATION APPARATUS
            tasks = []
            for dataset, args in futurecmds:
                for newfamily, newparamid in generated:
                    # https://docs.python.org/3/library/subprocess.html
                    env = {
                        k: str(v)
                        for k, v in dict(
                            GF_DB=self.dbfile,
                            GF_PARAM_GROUP=newparamid,
                            GF_FAMILY=newfamily,
                            GF_DATA=dataset,
                        ).items()
                    }

                    tasks.append((args, env))

            with multiprocessing.Pool() as pool:
                tasks = list(sorted(tasks, key=lambda x: int(x[1]["GF_PARAM_GROUP"])))

                logger.info("Running ({} tasks!): {}", len(tasks), pp.pformat(tasks))
                # 'args' replaces '{}' with the DATA / DATASET under test for the current family thing.

                # DATA / DATASET = {SYMBOL}
                # FAMILY = {SYMBOL}-{DATE}
                pool.starmap(runner, tasks)

    def breedRuns(
        self, runs: Iterable[int], name, highest: bool = True, maxCompare: int = 6
    ):
        """Instead of combining by families as breed(), combine by run+algo results.

        We should be only breeding, for example, trade-30 against other trade-30 which has nothing
        to do with the breed family, but is rather a property of breed run IDs.

        So, for a list of run IDs, get the highest performing N algos, then breed all those N algos
        together combinatorally. If an algo has no other peers in the run group, breed it against itself
        for future results.

        Then the new parameters can be applied to any given dataset for future metric
        determination for scoring and future looping for future results."""

    def exploreRun(self, runs: Iterable[int], name, highest: bool = True):
        ...

    def breed(
        self,
        families: Iterable[str],
        name: str,
        highest: bool = True,
        maxCompare: int = 6,
    ) -> list[int]:
        """Find results across 'families' for feature name 'name' in each algo then generate new parameters.

        If only one family, generates all combinations of highest or lowest 'maxCompare' results.
        If more than one family, generates nCr(maxCompare * len(families), 2) results.

        If 'highest' = True, score using highest values.
        If 'highest' = False, score using the smallest values.

        TODO: break up this method into more readable sub-components.
        """

        logger.info("[{}] Breeding families: {}", name, families)

        # TODO: need an adaptable learning rate?
        def findExpandFloppy(a: float, b: float) -> float:
            if a > b:
                # always make 'a' the smallest for our [low, high] range grabber
                a, b = b, a

            return random.uniform(a / 1.05, b * 1.05)

        def findExpandStiff(a: int, b: int) -> int:
            if a > b:
                a, b = b, a

            return round(findExpandFloppy(a / 1.25, b * 1.25))

        def findExapndFloppyRange(a: float, b: float, down: float, up: float) -> float:
            """Return random capped 5% between [a, b], but ALSO limit with min 'down' and max 'up'"""
            if a > b:
                a, b = b, a

            small = a / 1.05
            if small < down:
                small = down

            big = b * 1.05
            if big > up:
                big = up

            # if both are same, then random will just be the same value.
            if small == big:
                return small

            # random.uniform() doesn't guarantee limit is [small, big) so
            # if result is exactly 'big' by chance, try again.
            while (got := random.uniform(small, big)) == big:
                ...

            return got

        assert maxCompare > 0, "Trying to compare less than 1 high score?"
        COMBOBLAMBO = maxCompare

        processedRunIds: dict[tuple[int, int], tuple[str, int]] = {}

        # result of this entire algo selection will be a collection
        # of new run groups to try for result generation.

        algos: list[str] = list(
            str(c)
            for c, in self.db.execute("SELECT DISTINCT algo FROM results").fetchall()
        )

        logger.info("Potential algos are: {}", algos)

        # Generate new run parameter group for this parameter generation
        if highest:
            # probably trying to maxmize these.
            # .product() because we want to create a (originFamily, result) tuple
            # for EACH row in the result set.
            resultsByFamily = [
                itertools.product(
                    [family], self.highest(family, algos, name, COMBOBLAMBO)
                )
                for family in families
            ]
        else:
            # probably trying to minimize these
            resultsByFamily = [
                itertools.product(
                    [family], self.lowest(family, algos, name, COMBOBLAMBO)
                )
                for family in families
            ]

        # if more than one family, match results in alternating orders from:
        # [best a, best b, second best a, second best b, third best a, third best b, ...]
        # Also flatten the result set to a single list so the combo below selects all
        # 2-pair permutations available with the highest performers ranked for
        # higher comparing.
        resultsAll = list(itertools.chain(*zip(*resultsByFamily)))

        # If results is only one row, use it against itself for compares
        # more than 2 times so the choices don't break weirdly...
        if len(resultsAll) < 3:
            resultsAll *= 4

        logger.info("All results:\n{}", pp.pformat(resultsAll))

        # if we have fewer results than requested, use smaller range search
        COMBOBLAMBO = min(COMBOBLAMBO, len(resultsAll))

        # now we want to get all unordered permutation index pairs (nCr(6, 2) == 15); (nCr(12, 2) = 66)
        # itertools.combinations(collection, selection) is fastest way to do it.
        # equivalent to:
        # [(a, b) for a in range(COMBOBLAMBO) for b in range(COMBOBLAMBO) if a > b]
        # or:
        # [tuple(x) for x in {frozenset((a, b)) for a in range(6) for b in range(6)} if len(x) > 1]
        alldacombosyumyum: list[tuple[int, int]] = list(
            itertools.combinations(range(COMBOBLAMBO), 2)
        )

        # cache length of the combinations for compares and range extensions
        combolen = len(alldacombosyumyum)

        # verify reality still holds
        assert math.comb(COMBOBLAMBO, 2) == combolen

        # now get all algo-bound parameters for each result row
        # we are comparing here...
        # This a dictionary of comboId -> {parameter name: parameter value, ...}
        cAB = {}

        # note: don't use itertools.product() here because we want AT MOST
        # the full combo length number of results (i.e. the 'x' should never be duplicated
        # or else it will overwrite high perfoming results with lower performing
        # results if we loop past the max mixing sets).
        for x, (r, family) in zip(
            range(combolen),
            [(r["run"], family) for family, r in resultsAll],
        ):
            logger.info("Preparing against: {}, ({}, {})", x, r, family)

            # Iterate each unique param for this run group then collect all parameters
            # for the param for this rungroup.
            totalGot = {}
            for (param,) in self.db.execute(
                "SELECT DISTINCT param FROM params JOIN run_meta USING (pid) WHERE run_meta.run = ?",
                (r,),
            ).fetchall():
                # logger.info("Checking for param: {}", param)
                populate = dict(
                    self.db.execute(
                        "SELECT name, value FROM params JOIN run_meta USING (pid) WHERE run_meta.run = ? AND param = ?",
                        (r, param),
                    ).fetchall()
                )

                # restore types where necessary
                for k, v in populate.items():
                    populate[k] = loadIfNeeded(v)

                totalGot[param] = populate

            # attach total results to total results
            cAB[x] = dict(
                d=totalGot,
                f=family,
                r=r,
            )

        logger.info("Generated CAB:\n{}", pp.pformat(cAB))

        # for each unordered permutation calculated, BREED
        for a, b in alldacombosyumyum:
            logger.info("[{},{}] Processing COMBINATION", a, b)
            # unique param names pointing to values from each row.
            # we could have done a fancier thing by placing the actual rows inside
            # via itertools.combinations(resultsAll, 2), but this manual pair-wise
            # indexing gives us more direct control over the origin of the order-to-run
            # pairs being generated.
            # These two vars are the METADATA DICT ENTRIES
            cA = cAB[a]
            cB = cAB[b]

            # These two vars are the DATA ROWS
            cAd = cA["d"]
            cBd = cB["d"]

            # get cached run id for this (a, b) combination if already recorded
            found = processedRunIds.get((a, b))

            # if this is the first time we're encountring (a, b), generate new
            # run id and cache it for future runs (due to how these loops
            # are doing ALGOS (N) -> highest x highest nCr(6, 2) == 15 -> COLUMNS (K)
            if found:
                newFamily, newParamId = found
            else:
                familyA = cA["f"]
                familyB = cB["f"]

                # family is joined families unless same for both, then keep as singular.
                newFamily = ",".join(sorted({familyA, familyB}))
                newParamId = self.metaParamsNew(
                    dict(highest=highest, a=a, b=b, pA=cA["r"], pB=cB["r"]),
                )
                processedRunIds[(a, b)] = (f"{{{newFamily}}}", newParamId)

            # use this family/run id for all settings below
            # generate new parameter group from this family pairing...
            self.useParameterId(newParamId)

            # For each selected run iteration to breed, we need to parir-wise
            # match each name-value pair inside each param, so...
            # for param in params: for parameter in parameters: breed()
            sharedParams = cAd.keys() | cBd.keys()

            # standard layout: {param => {name => value}}
            insertParams: dict[str, dict[str, Any]] = defaultdict(dict)
            for param in sharedParams:
                # data inside each param for this combination selection...
                cAdi = cAd[param] if param in cAd else cBd[param]
                cBdi = cBd[param] if param in cBd else cAd[param]

                # use params from both (if one side key doesn't exist, do a same-copy)
                # (to accomidate different runs sometimes adding or removing parameter
                #  names as usage/refactoring/cleanup changes)
                sharedParams = cAdi.keys() | cBdi.keys()

                # 'col' here is the parameter name, which in a fully expanded
                # perhaps dataframe view, would be its own vertical column.
                for col in sharedParams:
                    # logger.info("Processing SP: {}", col)

                    # Hopefully the param name exists on both, but if not,
                    # we know it exists on _the other one_, so if doesn't
                    # exist, just use the other one for a same-same compare.
                    # Note: be sure to check for existence, and not just value,
                    # because 0 is still a valid value here.
                    aVal = cAdi[col] if col in cAdi else cBdi[col]
                    bVal = cBdi[col] if col in cBdi else cAdi[col]

                    # 50% chance we just swap values instead of mutating.
                    # TODO: instead of randomly swapping values, what about
                    #       only picking the currently best accomplished value?
                    # Also could not switch at 50/50 but weight outright switching
                    # to more or less severe based on populations of T/F in the list.
                    if col.endswith("_choice"):
                        # Choice fields are lists of possible choices, where the HEAD
                        # element in the list is the value we give to the user.
                        # So we pick the next "value" by permutating the list and changing
                        # which head element will be provided as the "value" (while maintaining
                        # the rest of the list because it will be needed for future breeding)
                        logger.info("Choicing with: {} :: {}", aVal, bVal)
                        if random.choice([True, False]):
                            # 50% chance we swap the current good values
                            val = random.choice([aVal[0], bVal[0]])
                            # make value head of a list to allow future permutations to still work
                            # (create set of origin values minus the head value we placed)
                            val = [val, *(set(aVal + bVal) - set([val]))]
                        else:
                            # else, pick a random combination to generate a new head element
                            # which will be used for the value provided for the argument.
                            # Use numpy because it allows returning ONE permutation instead of generating
                            # all of them then throwing them away.
                            val = list(np.random.permutation(list(set(aVal + bVal))))
                    else:
                        # else, values are singular elements we can act on without collections
                        if random.choice([True, False]):
                            val = random.choice([aVal, bVal])
                        else:
                            if col.endswith("_int"):
                                aVal = int(aVal)
                                bVal = int(bVal)
                                val = findExpandStiff(aVal, bVal)
                            elif col.endswith("_bool"):
                                # 50/50 chance a new random bool is picked,
                                # else pick a random existing value.
                                if random.choice([True, False]):
                                    val = random.choice([True, False])
                                else:
                                    val = random.choice([aVal, bVal])
                            elif col.endswith("_real_1"):
                                # limit range to [1, 2)
                                aVal = float(aVal)
                                bVal = float(bVal)
                                val = findExapndFloppyRange(aVal, bVal, 1, 2)
                            elif col.endswith("_real_0"):
                                # limit range to [0, 1)
                                aVal = float(aVal)
                                bVal = float(bVal)
                                val = findExapndFloppyRange(aVal, bVal, 0, 1)
                            else:  # col.endswith("_real"):
                                # default is assuming floats
                                aVal = float(aVal)
                                bVal = float(bVal)
                                val = findExpandFloppy(aVal, bVal)

                    # record new parameter name value pair for insert when done
                    insertParams[param][col] = val

            # insert new parameters for this single param on this run id
            assert insertParams
            logger.info("[breed] Generated new params:\n{}", pp.pformat(insertParams))
            self.parametersNew(insertParams)

        # x[0] is new family name, x[1] is new parameter id
        generated = [x[1] for x in processedRunIds.values()]
        logger.info("[breed] {} new parameter configs: {}!", combolen, generated)

        return processedRunIds.values()

    def paramReport(self, *pids: int) -> None:
        frames = []

        def report(pid):
            cur = self.db.cursor()
            cur.execute(
                "SELECT param, name, value FROM params WHERE pid = ? ORDER BY param, name",
                (pid,),
            )
            cur.row_factory = sqlite3.Row
            stuff = [dict(x) for x in cur.fetchall()]
            df = pd.DataFrame(stuff)
            frames.append(df)

            # also show highest performing dataset/algo/value for the parameter group
            cur.execute(
                "SELECT run_meta.run, dataset, algo, name, value FROM results JOIN run_meta USING (run)  WHERE name = 'profit' AND pid = ? ORDER BY value DESC LIMIT 5",
                (pid,),
            )
            highest = cur.fetchall()
            if highest:
                logger.info(
                    "[{}] Highest Performing Algos\n{}",
                    pid,
                    pd.DataFrame([dict(x) for x in highest]).to_string(),
                )
                logger.info("[{}] Report:\n{}", pid, df.to_string())
            else:
                logger.info("Report:\n{}", df.to_string())

        for pid in pids:
            logger.info("[{}] Params:", pid)
            report(pid)

        # due to the random nature of the combinations, almost every row will be
        # unique when compared across different parameter groups
        if False:
            logger.info(
                "Unique:\n{}", pd.concat(frames).drop_duplicates(keep=False).to_string()
            )


@logger.catch
def externalLoader() -> None:
    """Load external bootstrap file to start a new parameters database."""
    import sys

    # fake run ID because it generates a new RUN ID on every parameter import
    g = GeneticFuckery()
    g.populate(sys.argv[1:])


@logger.catch
def externalBreeder():
    import fire

    # Goal is to run this on ONE symbol to generate a 15-run complex
    # Then run the 15 complex to generate unique family run ids for the complex
    # THEN run multi-family breed against all the unique ids
    # THIS ALSO IMPLES the first run ID is "sacrificial" in that its result
    # isn't used for final breeding, the 15-complex is the future family results
    # TODO: also need to specify ACTUAL RUNNING DATA ARGUMENT so we can auto-loop
    #       the runner against differnt datasets (also so we can tell in the history
    #       which parent datasets were run on each data).
    # TODO: need to check combinations of inputs for combinatoral parameter growths because
    #       each new parameter group generates new pairs... ugh.
    def breedRun(
        families: list[str],
        name: str,
        cmd: str,
        highest: bool = True,
        maxCompare: int = 6,
        maxSelfLooping: int = 1,
        generationBoost: int = 1,
    ):
        g = GeneticFuckery()
        g.breedRun(
            families, name, cmd, highest, maxCompare, maxSelfLooping, generationBoost
        )

    fire.Fire(breedRun)


@logger.catch
def externalBreederAuto():
    """Run the breeding process using 100% DB details with no manual specifications."""
    import fire

    g = GeneticFuckery()
    fire.Fire(g.autoBreeder)


@logger.catch
def externalParamReport() -> None:
    import fire

    def epm(*pids: int) -> None:
        g = GeneticFuckery()
        g.paramReport(*pids)

    fire.Fire(epm)


@logger.catch
def externalHighest() -> None:
    import sys

    """Also note:
    Highest performing algo by dataset across all runs:
    select run_meta.run, run_meta.pid, dataset, family, algo, name, max(value) from results JOIN run_meta USING (run) where name = 'profit' group by dataset order by value;
    """

    family, algo, name, howmany = sys.argv[1 : 1 + 4]
    logger.info("Args: {}", sys.argv[1:])
    howmani = int(howmany)

    g = GeneticFuckery()
    got = g.highest(family, [algo], name, howmani)
    logger.info("Highest:\n{}", got)


@logger.catch
def generateRunTemplate():
    import fire

    def cmd(modulecmd: str, *dataset: str):
        """Generate command line ops for bootstrapping a new database with new datasets."""
        gf = GeneticFuckery()
        for pid in gf.pids():
            for d in dataset:
                print(
                    f"GF_DB={gf.dbfile} GF_PARAM_GROUP={pid} GF_DATA={d} GF_FAMILY={d} poetry run {modulecmd.replace('{}', d)}"
                )

    fire.Fire(cmd)


@logger.catch
def plotRuns():
    import fire

    def cmd(modulecmd: str, *runid: str):
        """Generate command line ops for only executing run IDs from DB."""
        gf = GeneticFuckery()
        for rid in set(runid):
            for pid, dataset in gf.db.execute(
                "SELECT pid, dataset FROM run_meta WHERE run = ?", (rid,)
            ).fetchall():
                print(
                    f"GF_DB={gf.dbfile} GF_PARAM_GROUP={pid} poetry run {modulecmd.replace('{}', dataset)}"
                )

    fire.Fire(cmd)


@logger.catch
def mergePerformers():
    import fire

    def cmd(algoParamMapFile: str, topNResults: int = 1, *dataset: str):
        """Merge multiple results per-algo from different results into a new combined parameter group.

        The new combined parameter group will have the highest performing parameters for each algo (hopefully)
        merged by either unique values or averaged between multiple high performing values."""

        gf = GeneticFuckery()
        amap = orjson.loads(pathlib.Path(algoParamMapFile).read_bytes())

        # Read input map of ALGO -> PARAMS and break it apart into ALGO -> [(param, name)]
        # for easier DB access of the dual part nested parameters.
        algoSplitParams = defaultdict(list)
        for algo, params in amap.items():
            for param in params:
                # Format here is text format param where the parameter hierarchy and its
                # implementation name are all in the same string, so break them apart
                # and re-assemble the two parts again.
                *pparams, name = param.split("__")
                pparamName = "__".join(pparams)

                # format is: WHOLE ("full param description"), PARAM ("prefix"), NAME ("final actual name")
                algoSplitParams[algo].append((param, pparamName, name))

        # logger.info("Working with: {}", pp.pformat(algoSplitParams))

        # REMAINING STEPS:
        #    - Save output as new input bootstrap file(s) for new DB.
        #    - Look up the top performing algos for dataset (if provided) to get
        #      their parameter group ids.
        #    - For each top performing (algo, pid) pair, look up the parameters
        #      the algo uses in the algoSplitParams dict
        #      - If looking up  more than one result, then average the top N results
        #        found for each parameter (while respecting int/float/choice where possible)
        #    - Now save those params into the new pid group

        # So, only look up datasets in 'dataset' IF GIVEN, else use all datasets in the DB.
        # Also only use top N results per algo (default 1, but could be more if we want more averages)

        # So x 2:
        #    - if more than one dataset, we have to average the top 1 parameters of each result.
        #    - if more than one dataset *and* more than one topNResults, then we need to average
        #      together (topNResults * len(dataset)) parameters for each algo in the final combined result.

        # this should be in an external thing somewhere, but it's a good example of creating a complex metric
        # by combining ranking and filtering of multiple result output values.
        BEST_PERFORMERS_QUERY = """
WITH
-- find trades with reasonable values
-- (each "trade" here would actually be two market transactions to open+close)
goodwins AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS win
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'win')
    ),

goodtrades AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS trades
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'trades' AND value > 5 AND value <= 155)
    ),

-- find the best profit for each dataset from the trades group
goodprofit AS
    (SELECT run,
            pid,
            dataset,
            family,
            algo,
            value AS profit
     FROM results
     JOIN run_meta USING (run)
     WHERE (name = 'profit' AND run IN (SELECT run FROM goodtrades))
    )

-- return combined result
SELECT run,
       pid,
       algo,
       dataset,
       max(profit) AS profit,
       round(profit / trades, 2) AS ppt, -- "profit per trade" including all trades
       trades,
       win,
       round((SELECT CAST (win AS FLOAT) / trades), 2) AS 'w%'
FROM goodprofit
JOIN goodtrades USING (run, pid, dataset, family, algo)
JOIN goodwins USING (run, pid, dataset, family, algo)
GROUP BY dataset, algo
HAVING ppt > 0 -- only profitable trades!
       -- AND profit > 2.01 -- only meaningful profit! no +$0.30 days
ORDER BY profit DESC -- CAST (win AS FLOAT) / trades DESC
"""

        valuesByParam = defaultdict(list)
        out = pathlib.Path(f"bootstrap-{int(datetime.datetime.now().timestamp())}.js")
        for run, pid, algo, dataset, profit, ppt, trades, win, wpct in gf.db.execute(
            BEST_PERFORMERS_QUERY
        ):
            logger.info(
                "ZAMBAMFOO: {}", (run, pid, algo, dataset, profit, ppt, trades, win)
            )

            # for now we're being lazy and selecting ALL parameters in the group, but then filtering
            # based on the target algo map.

            # map of param name to param values
            paramValues = defaultdict(
                None,
                gf.db.execute(
                    "SELECT param || '__' || name, value FROM params WHERE pid = ?",
                    (pid,),
                ).fetchall(),
            )

            for paramDesc in algoSplitParams[algo]:
                param = paramDesc[0]
                try:
                    # logger.info("[{}] {} = {}", algo, param, paramValues[param])
                    valuesByParam[param].append(paramValues[param])
                except:
                    logger.error("[{}] Not found: {}", algo, param)

        fullParams = []
        pdict = {}
        for param, values in sorted(valuesByParam.items(), key=lambda x: x):
            if not values:
                logger.info("{} = {}", param, None)
                continue

            try:
                logger.info(
                    "{} = {} (std: {}) (median: {}) (avg: {})",
                    param,
                    values,
                    np.std(values),
                    np.median(values),
                    np.mean(values),
                )

                useval = np.median(values)

                # don't allow fractional medians for integer parameters
                if param.endswith("_int"):
                    useval = round(useval)

                fullParams.append(f"{param} = {useval}")
            except:
                # if not a number-like thing, use proponderance of fields
                common = Counter(values)
                logger.info("{} = {} (counts: {})", param, sorted(values), common)

                useval = common.most_common()[0][0].decode()
                fullParams.append(f"{param} = {useval}")

            # TODO: after all created, evaulate parameters to make sure all "fast" values are
            #       less (shorter than) than "slow" values (longer than) then write out from
            #       the proper order in the dict instead of from 'fullParams' (which can be removed)
            pdict[param] = useval

        allpogies = "\n".join(fullParams)
        out.write_text(allpogies)
        logger.warning("[{}] Wrote parameters!", str(out))

    fire.Fire(cmd)
