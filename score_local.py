"""Local grader for the Leonida forecaster (not part of the entry).

Scores main.py on the two practice cases against the 04:30Z / 05:00Z
answering read, which is the live Exophase page as fetched 2026-11-19
11:35 UTC: its rates sum to 110.55, exactly 4276/3868*100, the totals
the competition notes give for the 04:30Z read, and the notes say the
05:00Z read was unchanged from it.  Both practice cases therefore share
this answer (see answers_practice.json).

Also runs the two house baselines on the same cases (flat persistence,
and the log-time trend fit) for comparison, a synthetic growth
backtest that exercises the data branch (the practice histories are
flat, so they only exercise the dilution prior), and a long-horizon
sanity check (p01's history, target pushed +4.5 days).

Pinball loss per the competition: 2*max(tau*d, (tau-1)*d) with
d = logit(y) - logit(q), rates in percent clamped to [0.01, 99.99].
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = Path.home() / "workspace/tools/makefun/gta6-trophy-rarity-forecast-01-kit"
sys.path.insert(0, str(HERE))

import main as entry  # noqa: E402  (the entry under test)

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]


def _clamp(pct):
    return min(99.99, max(0.01, pct))


def _logit(pct):
    p = _clamp(pct)
    return math.log(p / (100.0 - p))


def pinball(quantiles, actual):
    """Mean pinball loss over the five levels for one trophy."""
    la = _logit(actual)
    total = 0.0
    for tau, q in zip(LEVELS, quantiles):
        d = la - _logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def score_forecast(forecast, answers):
    """(mean pinball loss, mean |median - actual| in pp, 90% coverage)."""
    rows = {r["id"]: r["q"] for r in forecast["trophies"]}
    losses, misses, covered = [], [], 0
    for tid, actual in answers.items():
        q = rows[tid]
        losses.append(pinball(q, actual))
        misses.append(abs(q[2] - actual))
        if q[0] <= actual <= q[4]:
            covered += 1
    n = len(answers)
    return sum(losses) / n, sum(misses) / n, covered / n


def flat_baseline(case):
    """House persistence baseline: last rate, random-walk spread."""
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(days)
    rows = []
    for i, t in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = _logit(rate if rate is not None else 0)
        rows.append({"id": t["id"], "q": [
            entry._round4(entry._inv_logit_pct(lg + z * spread))
            for z in Z]})
    return {"trophies": rows}


def _ols_plain(xs, ys):
    """Plain OLS line (the house trend baseline's fit, kept separate
    from the entry's recency-weighted fit)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (my - slope * mx, slope)


def trend_baseline(case):
    """House curve baseline: OLS logit vs log(1+hours) over the last
    24h of reads, spread = random walk + fit residual; flat fallback."""
    release = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
    at = datetime.fromisoformat(case["at"].replace("Z", "+00:00"))
    target = datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
    days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max((target - release).total_seconds() / 3600.0, 0))
    rows = []
    for i, t in enumerate(case["trophies"]):
        pts = []
        for read in case["history"]:
            slot = datetime.fromisoformat(read["slot"].replace("Z", "+00:00"))
            if (at - slot).total_seconds() > 24 * 3600:
                continue
            if read["rates"][i] is None:
                continue
            pts.append((math.log1p(max(
                (slot - release).total_seconds() / 3600.0, 0)),
                _logit(read["rates"][i])))
        last_rate = case["history"][-1]["rates"][i]
        if len(pts) < 3:
            med, sig = _logit(last_rate), 0.5 * math.sqrt(days)
        else:
            fit = _ols_plain([p[0] for p in pts], [p[1] for p in pts])
            if fit is None:
                med, sig = _logit(last_rate), 0.5 * math.sqrt(days)
            else:
                a, b = fit
                med = a + b * x_target
                resid = sum((y - (a + b * x)) ** 2 for x, y in pts) / len(pts)
                sig = math.sqrt(0.25 * days + resid)
        rows.append({"id": t["id"], "q": [
            entry._round4(entry._inv_logit_pct(med + z * sig)) for z in Z]})
    return {"trophies": rows}


def load_practice():
    cases = {}
    for cid in ("p00", "p01"):
        cases[cid] = json.loads((KIT / "input" / (cid + ".in")).read_text())
    answers = json.loads((HERE / "answers_practice.json").read_text())
    return cases, {int(k): v for k, v in answers.items()}


def synthetic_case():
    """A fake 48h history with Hill-curve growth + 6-hourly refreshes,
    and a +4.5-day target, to exercise the data (trend) branch."""
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    case = json.loads((KIT / "input" / "p01.in").read_text())
    at = release + timedelta(hours=48)
    target = at + timedelta(hours=108)
    case["id"] = "synth"
    case["at"] = at.strftime("%Y-%m-%dT%H:%M:%SZ")
    case["target"] = target.strftime("%Y-%m-%dT%H:%M:%SZ")
    case["horizon_minutes"] = 108 * 60
    # Half-saturation time (h) and ceiling (%) per trophy class.
    half = {"platinum": 2000.0, "completion": 2000.0, "chapter1": 30.0,
            "chapter2": 60.0, "chapter3": 100.0, "chapter4": 150.0,
            "finale": 300.0, "story": 120.0, "early": 40.0, "easy": 60.0,
            "grind": 600.0, "economy": 200.0, "explore": 150.0,
            "side": 150.0, "misc": 150.0}

    def true_rate(trophy, hours):
        cls = entry._classify(trophy)
        ceil = entry.CLASS_PARAMS[cls][0]
        h = half[cls]
        return ceil * (hours ** 1.5) / (hours ** 1.5 + h ** 1.5)

    history = []
    players = 638
    refresh_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48]
    for rh in refresh_hours:
        slot = release + timedelta(hours=rh)
        players = int(638 * math.exp(0.055 * rh))
        rates = [round(true_rate(t, rh if rh > 0 else 0.5), 2)
                 for t in case["trophies"]]
        rates[1] = 100.0  # the 100% artefact stays put
        for half_hour in (0, 30):
            s = slot + timedelta(minutes=half_hour)
            if s >= at:
                break
            history.append({"slot": s.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "at": s.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                            "players_tracked": players,
                            "trophies_earned": sum(
                                int(r * players / 100) for r in rates),
                            "rates": list(rates)})
    case["history"] = history
    truth = {t["id"]: (100.0 if t["id"] == 2 else
                       round(true_rate(t, 156.0), 2))
             for t in case["trophies"]}
    return case, truth


def synthetic_case_dip():
    """Same growth as synthetic_case, but a dilution shock at 24h halves
    every rate (a big refresh: tracked players jump, earners do not),
    after which rates climb back toward the Hill curve.  Tests whether
    the forecaster adapts to a mid-history regime change instead of
    extrapolating the shock."""
    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    case, truth = synthetic_case()
    at = release + timedelta(hours=48)
    history = []
    for read in case["history"]:
        slot = datetime.fromisoformat(read["slot"].replace("Z", "+00:00"))
        hours = (slot - release).total_seconds() / 3600.0
        rates = list(read["rates"])
        if hours >= 24:
            dent = 0.55 + 0.45 * (1 - math.exp(-(hours - 24) / 24.0))
            rates = [100.0 if r == 100.0 else round(r * dent, 2)
                     for r in rates]
        history.append({**read, "rates": rates})
    case["history"] = history
    return case, truth


def main():
    cases, answers = load_practice()
    print("== practice (answers = 04:30Z/05:00Z read, sum-verified) ==")
    for cid, case in cases.items():
        mine = entry.forecast_case(case)
        fl = flat_baseline(case)
        tr = trend_baseline(case)
        for name, fc in (("mine ", mine), ("flat ", fl), ("trend", tr)):
            loss, miss, cov = score_forecast(fc, answers)
            print(f"{cid} {name}: pinball {loss:.4f}  "
                  f"typical miss {miss:.2f}pp  coverage90 {cov:.2f}")

    print("== synthetic growth backtest (48h history, +4.5d target) ==")
    synth, truth = synthetic_case()
    for name, fc in (("mine ", entry.forecast_case(synth)),
                     ("flat ", flat_baseline(synth)),
                     ("trend", trend_baseline(synth))):
        loss, miss, cov = score_forecast(fc, truth)
        print(f"synth {name}: pinball {loss:.4f}  "
              f"typical miss {miss:.2f}pp  coverage90 {cov:.2f}")

    print("== synthetic dilution-shock backtest (same, shock at 24h) ==")
    dip, truth2 = synthetic_case_dip()
    for name, fc in (("mine ", entry.forecast_case(dip)),
                     ("flat ", flat_baseline(dip)),
                     ("trend", trend_baseline(dip))):
        loss, miss, cov = score_forecast(fc, truth2)
        print(f"dip   {name}: pinball {loss:.4f}  "
              f"typical miss {miss:.2f}pp  coverage90 {cov:.2f}")

    print("== long-horizon sanity (p01 history, target +4.5 days) ==")
    long_case = json.loads((KIT / "input" / "p01.in").read_text())
    at = datetime.fromisoformat(long_case["at"].replace("Z", "+00:00"))
    target = at + timedelta(days=4.5)
    long_case["target"] = target.strftime("%Y-%m-%dT%H:%M:%SZ")
    long_case["horizon_minutes"] = int(4.5 * 1440)
    short = entry.forecast_case(
        json.loads((KIT / "input" / "p01.in").read_text()))
    longf = entry.forecast_case(long_case)
    srows = {r["id"]: r["q"] for r in short["trophies"]}
    lrows = {r["id"]: r["q"] for r in longf["trophies"]}
    bad = 0
    for tid in srows:
        qs, ql = srows[tid], lrows[tid]
        if not (all(0 <= v <= 100 for v in ql)
                and all(ql[j] <= ql[j + 1] for j in range(4))):
            bad += 1
            print("  INVALID row", tid, ql)
        spread_s = _logit(qs[4]) - _logit(qs[0])
        spread_l = _logit(ql[4]) - _logit(ql[0])
        if spread_l < spread_s - 1e-9:
            bad += 1
            print("  spread did NOT widen", tid, spread_s, spread_l)
    print(f"  rows checked: {len(srows)}, problems: {bad}")
    for tid in (1, 2, 3, 7, 14, 20, 21):
        print(f"  id {tid}: 90min q50={srows[tid][2]} "
              f"-> 4.5d q={[lrows[tid][0], lrows[tid][2], lrows[tid][4]]}")


if __name__ == "__main__":
    main()
