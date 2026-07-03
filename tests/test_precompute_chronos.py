import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import precompute_chronos as pc


def _fake_forecaster(context_arr, horizon):
    last = float(context_arr[-1])
    return {0.1: last * 0.98, 0.5: last * 1.01, 0.9: last * 1.04}


def _seed_prices(engine, table, n=80):
    df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=n).astype(str),
                       "Close": np.linspace(100, 120, n)})
    df.to_sql(table, engine, index=False)


def test_precompute_caches_and_is_incremental(tmp_path):
    eng = create_engine("sqlite:///" + str(tmp_path / "m.db"))
    _seed_prices(eng, "sp500")
    n1 = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64)
    assert n1 > 0
    cached = pd.read_sql("SELECT * FROM chronos_cache WHERE asset='sp500'", eng)
    assert set(["asset", "date", "chronos_ret", "chronos_spread", "chronos_dir"]).issubset(cached.columns)
    # re-run: nothing new (incremental skip)
    n2 = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64)
    assert n2 == 0
