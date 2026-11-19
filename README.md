# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/anisakansa/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/anisakansa/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each of GTA VI's 55 PlayStation trophies' Exophase earn rates a week after launch, as five quantiles per trophy. When the recorded history is flat it applies a refresh-dilution prior (a refresh floods the tracked-player count while earner counts barely move, so rates fall); when the history moves it fits a recency-weighted saturating log-time trend in logit space, shrunk toward the pooled cross-trophy slope and a trophy-class ceiling, with a spread that widens with the horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case in the input directory and writes `<id>.out` forecasts to the output directory. `score_local.py` grades it on the two practice cases against the 04:30Z/05:00Z answering read, compares it with the flat and trend baselines, and runs a synthetic growth backtest plus a long-horizon sanity check; `answers_practice.json` holds those practice answers (the live Exophase page as fetched 2026-11-19 11:35 UTC, whose rates sum to exactly the 04:30Z read's totals).

---

<sub>Entered on [make.fun](https://make.fun/?ref=2h55gjag&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
