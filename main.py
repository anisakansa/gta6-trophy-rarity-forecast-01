"""Leonida week-one rarity forecaster.

Reads one GTA VI trophy case (Exophase earn rates sampled by the house
recorder) and forecasts, for each of the game's 55 trophies, five quantiles
of its earn rate at the case's target, on the logit scale.

How it thinks about the data
-----------------------------
Exophase only refreshes a game's counts every few hours, so a case's history
is a step function: long flat stretches, then a jump at a refresh. At a
refresh the tracked-player count jumps (launch night: 638 -> 3,868) while
each trophy's earner count grows only a little, so almost every rate FALLS
at a refresh -- the new players hold fewer trophies than the early ones.
That dilution is the dominant short-horizon effect, and the launch-night
practice answers prove it (every rate but the 100% one fell).

The forecaster therefore works in two regimes:

* Flat history (fewer than 3 distinct reads, e.g. the practice cases):
  no trend can be estimated, so it applies a refresh-dilution prior: the
  median logit drops by about one refresh's worth (~1.45 logit, modulated
  by trophy class -- trophies new players earn quickly, like Chapter 1 or
  the fishing challenge, dilute less; slow grind trophies dilute more),
  with a wide spread because a refresh may or may not land in the window.

* Real history (3+ distinct reads): per trophy, a recency-weighted
  least-squares line of logit(rate) on log(1 + hours since release),
  cross-trophy slope and a class prior slope, then read at the target.
  The median is damped toward a trophy-class ceiling when the trend
  points up (a week-one player base can only get so far: Chapter 1
  saturates far above the platinum) and toward a floor when it points
  down.  The spread combines the fit residual, a random-walk term that
  grows with the horizon, and the observed jumpiness between refreshes,
  so uncertainty honestly widens from a 90-minute cut to a 4.5-day one.

Trophy classes come from the trophy's own description and tier, so the
same code works on any case of this game.  Everything is deterministic:
no clock, no randomness, no network, standard library only.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at the five levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Spread calibration.  FLAT_SIGMA0 is the flat-history (dilution-prior)
# spread in logit units; it grows slowly with horizon.  WALK_SIGMA is
# the random-walk spread per sqrt(day) in the data branch.
FLAT_SIGMA0 = 0.72
FLAT_SIGMA_GROWTH = 0.10
FLAT_SIGMA_CAP = 1.15
WALK_SIGMA = 0.30

# Per-class week-one behaviour.  Fields: ceiling rate (%), floor rate (%),
# refresh-dilution factor (x the ~1.45-logit average drop at one refresh),
# prior slope of logit(rate) per unit of log(1 + hours since release).
CLASS_PARAMS = {
    "platinum":  (1.5, 0.02, 1.25, -0.15),
    "completion": (1.0, 0.02, 1.25, -0.15),
    "chapter1":  (60.0, 0.05, 0.65, 0.30),
    "chapter2":  (45.0, 0.05, 0.85, 0.15),
    "chapter3":  (35.0, 0.05, 0.90, 0.10),
    "chapter4":  (28.0, 0.05, 0.90, 0.05),
    "finale":    (15.0, 0.03, 1.00, -0.05),
    "story":     (20.0, 0.03, 1.05, 0.00),
    "early":     (35.0, 0.05, 0.50, 0.10),
    "easy":      (35.0, 0.05, 0.75, 0.10),
    "grind":     (6.0, 0.02, 1.20, -0.10),
    "economy":   (12.0, 0.03, 1.00, 0.00),
    "explore":   (25.0, 0.03, 0.70, 0.05),
    "side":      (18.0, 0.03, 1.00, 0.00),
    "misc":      (15.0, 0.03, 1.00, 0.00),
}

_EARLY_DESCS = ("fishing challenge", "arcade game", "discover 100 vehicles")
_GRIND_DESCS = ("250 stunts", "50 challenges", "100 locations",
                "collectible card sets", "20 different races",
                "every location", "shooting range", "8 mission strands",
                "try each type of activity")
_ECONOMY_DESCS = ("laundering", "spend $", "garage properties",
                  "register 10 vehicles")
_SIDE_DESCS = ("wanted level", "combat focus", "rhinos", "drive-by",
               "bank vaults", "armored truck", "jewelry store")


def _clamp_pct(pct):
    return min(99.99, max(0.01, pct))


def _logit(pct):
    p = _clamp_pct(pct)
    return math.log(p / (100.0 - p))


def _inv_logit_pct(lg):
    lg = min(9.21, max(-9.21, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(x):
    """Round to 4 decimals, halves away from zero; whole numbers as int."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if v == int(v) else v


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _classify(trophy):
    """Trophy class from its tier and description (no hardcoded ids)."""
    tier = (trophy.get("tier") or "").lower()
    desc = (trophy.get("description") or "").lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if "complete chapter 1" in desc:
        return "chapter1"
    if "complete chapter 2" in desc:
        return "chapter2"
    if "complete chapter 3" in desc:
        return "chapter3"
    if "complete chapter 4" in desc:
        return "chapter4"
    if "final mission" in desc and " for " not in desc:
        return "finale"
    if any(k in desc for k in _EARLY_DESCS):
        return "early"
    if any(k in desc for k in _GRIND_DESCS):
        return "grind"
    if any(k in desc for k in _ECONOMY_DESCS):
        return "economy"
    if "explore all of leonida" in desc:
        return "explore"
    if any(k in desc for k in _SIDE_DESCS):
        return "side"
    if any(k in desc for k in ("bowling", "cocktail", "meditate",
                               "arcade", "fishing", "activity once")):
        return "easy"
    if desc.startswith("complete ") and trophy.get("secret"):
        return "story"
    return "misc"


def _distinct_reads(history, release):
    """Collapse the recorder's step function to its distinct reads.

    Returns [(x, players, rates)] with x = log(1 + hours since release),
    keeping only reads whose (players, rates) differ from the previous
    kept read, oldest first.
    """
    out = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        if not any(r is not None for r in rates):
            continue
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        slot = _parse_time(read["slot"])
        hours = (slot - release).total_seconds() / 3600.0
        x = math.log(1.0 + max(hours, -0.9))
        out.append((x, read.get("players_tracked"), rates))
    return out


def _wols(xs, ys, ws=None):
    """Recency-weighted least-squares line; returns (intercept, slope).

    With no weights this is plain OLS.  Weights let the recent regime
    dominate the slope: Exophase's dilution regime (launch night) is not
    the regime that extrapolates to day seven, and a flat early stretch
    would otherwise drag every slope toward zero.
    """
    n = len(xs)
    if n < 2:
        return None
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    return (my - slope * mx, slope)


def _recency_weights(xs):
    """Weights decaying by one e-fold per unit of log-time behind the
    latest read: the last read weighs 1, a read one log-time unit older
    weighs ~0.37, the launch-night reads weigh almost nothing."""
    last = max(xs)
    return [math.exp(-(last - x)) for x in xs]


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _forecast_one(trophy, points, pooled_slope, jump_var, horizon_days,
                  x_target, last_rate):
    """Five quantiles (percent) for one trophy.

    points: [(x, logit_rate)] at the distinct reads; pooled_slope is the
    cross-trophy median slope; jump_var the pooled variance of
    refresh-to-refresh logit jumps; last_rate the latest observed rate.
    """
    cls = _classify(trophy)
    ceil_pct, floor_pct, drop_factor, prior_slope = CLASS_PARAMS[cls]

    # The 100% artefact: Exophase holds it at 100 through every refresh.
    if last_rate is not None and last_rate >= 99.5:
        return [99.9, 99.97, 99.99, 99.99, 99.99]

    if last_rate is None:
        last_rate = 1.0
    last_logit = _logit(last_rate)

    if len(points) < 3 or len({round(y, 9) for _, y in points}) < 2:
        # Flat history: no trend is estimable.  Expect one refresh's
        # dilution, a little more over a longer horizon, and stay unsure.
        base_drop = 1.45 + 0.25 * math.log1p(horizon_days)
        shift = min(2.8, base_drop * drop_factor)
        centre = last_logit - shift
        sigma = min(FLAT_SIGMA_CAP,
                    FLAT_SIGMA0 + FLAT_SIGMA_GROWTH * math.sqrt(horizon_days))
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ws = _recency_weights(xs)
        fit = _wols(xs, ys, ws)
        n = len(points)
        if fit is None:
            centre = last_logit
            sigma = 0.8
        else:
            _, slope_ols = fit
            w_data = n / (n + 3.0)
            blend_prior = 0.6 * pooled_slope + 0.4 * prior_slope
            slope = w_data * slope_ols + (1.0 - w_data) * blend_prior
            sw = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / sw
            my = sum(w * y for w, y in zip(ws, ys)) / sw
            intercept = my - slope * mx
            centre = intercept + slope * x_target
            resid = sum(w * (y - (intercept + slope * x)) ** 2
                        for w, x, y in zip(ws, xs, ys)) / sw
            sigma = math.sqrt(min(1.0, max(0.02, resid))
                              + WALK_SIGMA ** 2 * horizon_days + jump_var)
            sigma = min(1.5, max(0.30, sigma))

    # Damp the median toward the class ceiling / floor instead of letting
    # a steep fitted line run away over a multi-day horizon.
    ceil_logit = _logit(ceil_pct)
    floor_logit = _logit(floor_pct)
    if centre > ceil_logit:
        centre = ceil_logit + 0.3 * (centre - ceil_logit)
    if centre < floor_logit:
        centre = floor_logit + 0.3 * (centre - floor_logit)
    centre = min(9.21, max(-9.21, centre))

    qs = [_round4(_inv_logit_pct(centre + z * sigma)) for z in Z]
    # Rounding can only tie neighbours, never invert them; enforce it.
    for j in range(1, 5):
        if qs[j] < qs[j - 1]:
            qs[j] = qs[j - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    release_text = (case.get("game") or {}).get("release_date") or "2026-11-19"
    release = datetime.fromisoformat(release_text + "T00:00:00+00:00")
    target = _parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log(1.0 + max(
        (target - release).total_seconds() / 3600.0, -0.9))

    reads = _distinct_reads(history, release)

    # Pooled slope and refresh-jump variance across trophies, for shrinkage.
    slopes = []
    jumps = []
    for i in range(len(trophies)):
        pts = [(x, _logit(r[i])) for x, _, r in reads if r[i] is not None]
        if len(pts) >= 3:
            fit = _wols([p[0] for p in pts], [p[1] for p in pts],
                        _recency_weights([p[0] for p in pts]))
            if fit is not None:
                slopes.append(fit[1])
        for (_, y1), (_, y2) in zip(pts, pts[1:]):
            jumps.append(y2 - y1)
    pooled_slope = _median(slopes) if slopes else 0.0
    jump_var = 0.09
    if len(jumps) >= 4:
        mj = sum(jumps) / len(jumps)
        jump_var = min(0.6, sum((j - mj) ** 2 for j in jumps) / len(jumps))

    rows = []
    for i, trophy in enumerate(trophies):
        pts = [(x, _logit(r[i])) for x, _, r in reads if r[i] is not None]
        last_rate = None
        for _, _, r in reversed(reads):
            if r[i] is not None:
                last_rate = r[i]
                break
        rows.append({
            "id": trophy["id"],
            "q": _forecast_one(trophy, pts, pooled_slope, jump_var,
                               horizon_days, x_target, last_rate),
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
