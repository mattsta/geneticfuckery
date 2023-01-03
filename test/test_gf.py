import geneticfuckery.gf as gf
import pathlib

import contextlib
from tempfile import NamedTemporaryFile as ntf, TemporaryDirectory as td

from typing import Iterator

from loguru import logger
import pprint as pp

testdir = pathlib.Path(__file__).parent
testdb = testdir / "keep.db"


@contextlib.contextmanager
def tf() -> Iterator[str]:
    """Create a temporary directory with a temporary filename inside.

    We use temporary directory because we're using sqlite in WAL mode
    so it generates extra files the regular tempname cleanup doesn't
    delete when the context manager exits (but the temp directory context
    manager deletes the entire directory when done)"""
    with td(prefix="gf-testing", dir=testdir) as tdir:
        with ntf(dir=tdir) as tdb:
            yield tdb.name


def test_create() -> None:
    with tf() as tmpdb:
        g = gf.GeneticFuckery(dbfile=tmpdb)


@logger.catch
def test_populate() -> None:
    with tf() as tmpdb:
        g = gf.GeneticFuckery(dbfile=tmpdb)  # testdb)
        rid = g.metaParamsNew(dict(start="initial create"))
        g.parametersNew(
            dict(
                ATR_15=dict(lookback_int=16, depth_int=27, margin=1.0003),
                d15__IndicatorState=dict(length_int=15, fastLength_int=33),
            ),
        )

        with g.report("AAPL_2021_11_10") as resulter:
            resulter.update(
                dict(
                    CrossoverWatch=dict(profit=33.33, trades=66, wins=66),
                    CrossunderWatch=dict(
                        profit=2.22, trades=33, wins=1, losses=30, even=2
                    ),
                ),
            )

        results = g.highest("AAPL_2021_11_10", ["CrossoverWatch"], "profit")
        logger.info("Results is: {}", pp.pformat(results))
        r0 = results[0]
        assert r0["run"] == rid
        assert r0["profit"] == 33.33

        results = g.lowest("AAPL_2021_11_10", ["CrossoverWatch"], "trades")
        r0 = results[0]
        assert r0["run"] == rid
        assert r0["trades"] == 66

        if False:
            results = g.lowest("AAPL_2021_11_10", ["CrossunderWatch"], "trades")
            r0 = results[0]
            assert r0["run"] == rid
            assert r0["trades"] == 33

        assert g.ATR_15.lookback_int == 16
        assert g.AtR_15.dePth_INT == 27
        assert g.d15.IndicatorState.length_int == 15
        assert g.d15.IndicatorState.fastLength_int == 33

        g.breed(["AAPL_2021_11_10"], "profit")
