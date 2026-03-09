# 🏏 IPL 2026 — AI Captain

An ML-powered cricket decision-support system that recommends the **next batter to send in** and the **next bowler to deploy**, based on live match situation, career statistics, head-to-head matchups, and playing-style analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Name Resolution System (Fuzzy Logic)](#name-resolution-system-fuzzy-logic)
- [Validation](#validation)
- [Streamlit App](#streamlit-app)
- [Key Constants](#key-constants)
- [Known Limitations](#known-limitations)

---

## Overview

The system answers two questions a T20 captain faces every few overs — who bats next after a wicket, and who bowls the next over. Rather than relying on gut feeling or simple averages, the model scores every eligible candidate using gradient-boosted trees trained on 278,205 balls of IPL ball-by-ball data, weighting features based on how relevant they are to the current match phase (powerplay, middle, or death).

---

## Project Structure

The repository contains a main Google Colab training notebook (`ipl_rl.ipynb`), a Streamlit inference app (`app.py`), a `data/` folder holding the ball-by-ball dataset and the player details CSV, and a `models/` folder containing all exported pickle files — the three batting sub-models, the three bowling sub-models, their corresponding feature column lists, and the pre-computed lookup tables used by the app.

---

## Data Pipeline

### Ball-by-Ball Dataset

The raw dataset contains 278,205 rows with one row per ball bowled across multiple IPL seasons. It records the match ID, innings, over, ball number, batter on strike, current bowler, runs scored, whether a wicket fell, and the target score for the second innings. From this, a batting candidates table of roughly 51,245 rows and a bowling candidates table of roughly 238,211 rows were constructed — each row representing a potential selection decision at a specific match state, labelled with whether the candidate was actually chosen.

### Player Details Dataset

`player_details_final.csv` maps every player to a canonical identity with four columns: a full display name, a unique dataset abbreviation that matches the tokens used in the ball-by-ball data, batting hand (right or left), and bowling type (fast, medium, spin, etc.).

**This dataset required substantial manual effort.** All 767 unique player name tokens were first extracted from the ball-by-ball data. An external dataset of player handedness and bowling type was used as the primary source, but it used different name formats — so fuzzy string matching was applied to align names across the two sources. Players who were absent from the external dataset (newer players, overseas players with limited coverage) had their bat hand and bowl type looked up and entered manually. The `unique_name` column was then carefully aligned to exactly match the tokens appearing in the ball-by-ball data, since any mismatch silently produces zero-stat rows at inference time and degrades recommendation quality.

### Player Handedness and Bowl Type

These two attributes are used to compute style matchup features. Batting hand enables a lookup of how economical a given bowler is against left-handed versus right-handed batters. Bowl type enables a lookup of how many runs per ball a given batter scores against pace versus spin. Together they provide a weaker but reliable fallback signal when direct head-to-head history between two players is sparse.

---

## Feature Engineering

All features are pre-computed at training time and stored in lookup tables for fast inference in the app. They fall into five groups.

### Career Stats

Career batting stats are computed as medians per batter across all innings — median runs, median balls faced, career strike rate, boundary rate, and batting average. Medians are preferred over means to reduce the influence of outlier innings. Career bowling stats are computed as career-level aggregates — total balls bowled, total wickets, economy rate, bowling average, and bowling strike rate. Bowlers with zero career wickets are assigned a sentinel value of 999 for average and strike rate to avoid division by zero, and a binary flag `career_no_wickets` is set to 1 for them.

### Phase Stats (Bayesian-Smoothed)

The match is divided into three phases: powerplay (overs 0–5), middle (overs 6–14), and death (overs 15–19). For every player-phase combination, stats are Bayesian-smoothed — blending the player's phase-specific record with the population prior, weighted by sample size. This prevents small samples from producing unreliable extreme values, for example a batter who has faced only 4 death-over balls getting an inflated or deflated phase strike rate that dominates their score.

### Innings-Split Stats

Separate stats are maintained for innings 1 (batting first, setting a target) versus innings 2 (chasing). A batter's strike rate and average when setting a total can differ substantially from when chasing, and similarly for a bowler's economy and strike rate depending on whether their team is defending or attacking. The innings is inferred from the `is_chasing` flag and never assumed from a hardcoded default.

### Head-to-Head Matchup Features

Direct batter versus bowler history is extracted from the ball-by-ball data for every pair. The core features are the number of balls faced, runs scored, wickets taken, and runs per ball in that specific matchup. Three derived features are also computed — a matchup confidence score (a sigmoid of the ball count, centred at 6 balls with a scale of 12), a binary flag indicating whether the matchup has at least 6 meaningful balls, and a log-scaled ball count. 71.8% of batter-bowler pairs have at least one recorded ball in the dataset, with a median of 6 balls per pair.

### Style Matchup Features

Rather than individual pair history, these features aggregate across player archetypes. For batting recommendations the model uses the batter's runs per ball against the current bowler's type (pace or spin), and the bowler's economy rate against the batter's hand (left or right). For bowling recommendations the same logic applies from the bowler's perspective. These features fill in when H2H data is sparse, providing a directional prior based on stylistic compatibility.

### Match Context Features

Computed live at inference time from the current match state. The key features are current score, wickets fallen, balls remaining, overs remaining, current run rate, runs needed (chasing only), required run rate (chasing only), and three one-hot phase dummies.

The **Pressure Index** is a composite score derived from these. When chasing, it is the ratio of required run rate to current run rate, multiplied by a wicket-weight factor of `wickets_fallen / 2 + 1`. When batting first, the required run rate is replaced by the par run rate of 8.0. Both formulas are multiplied by an early-over dampening factor — `min(overs_bowled / 3, 1)` — which ramps from zero to one over the first three overs to prevent noisy extreme pressure values at the very start of an innings. The result is clipped between 0 and 10. The `wickets / 2 + 1` formulation (rather than `wickets + 1`) was chosen to avoid overcounting wicket pressure relative to run-rate pressure.

---

## Models

### Architecture

Six gradient-boosted classifiers are trained in total — three for batting decisions and three for bowling. In each group, a matchup sub-model is trained on head-to-head features only, an extended sub-model is trained on H2H plus style and situation features, and a full model is trained on all features including career stats. The target label in all cases is `was_chosen` — whether the candidate was the player actually selected at that match state.

**Batting performance:** ROC-AUC 0.8719, top-1 accuracy 73.9%, MRR 0.8550.  
**Bowling performance:** ROC-AUC 0.7814, top-1 accuracy 48.1%, MRR 0.6828.  
5-fold cross-validation confirms stable metrics across folds for both models.

### Ensemble Weighting

Default ensemble weights are 0.50 for the matchup model, 0.30 for the extended model, and 0.20 for the full model — reflecting the priority order of direct history over style over career. These weights are dynamically adjusted at inference using **reliability scaling**: the matchup model's weight is multiplied by a reliability factor equal to `bb_balls / (bb_balls + 10)`, which approaches zero when there is no H2H history and approaches the full weight as sample size grows. Any weight removed from the matchup model is redistributed to the extended and full models proportionally. This means a completely uncharted matchup defaults entirely to situational and career signal, while a pair with 50+ balls of history gets the full H2H signal.

For batting recommendations only, an additional low-volume penalty is applied to the full model's probabilities — `bat_balls_faced / (bat_balls_faced + 10)` — so that tail-end batters with minimal career data are not ranked too highly based on match-situation features alone. The denominator of 10 (rather than a larger number) is intentional: it keeps the penalty soft enough that tail-enders are still meaningfully ranked against each other by their actual batting stats, so for example a genuine pinch-hitter tail-ender with a decent strike rate correctly ranks above a number-eleven specialist.

### Phase-Aware H2H Masking

A core design principle is that **head-to-head history should not influence death-over decisions**. In the death overs a batter or bowler must execute regardless of who they're facing — past matchup data introduces noise rather than signal in this phase. History matters most during the powerplay, where captain tactics and set batters make matchup data highly predictive, and is partially relevant in the middle overs when at least 6 balls of H2H data exist.

The implementation goes beyond simply setting the matchup model weight to zero in death overs. The extended sub-model also contains H2H columns in its feature set, and if those columns are left at their real values the gradient-boosted trees inside it will still use them during inference. The fix is to explicitly zero out all H2H and style columns in the extended model's feature matrix before calling predict — so the trees genuinely see no matchup signal, not just a downweighted one. In death overs the weights shift to 0.0 for the matchup model, 0.85 for the extended model (now matchup-blind), and 0.15 for the full model. In middle overs the H2H reliability threshold of 6 minimum balls is enforced — pairs with fewer balls than this are treated as having no meaningful matchup data. In powerplay, full H2H scaling applies.

---

## Name Resolution System (Fuzzy Logic)

Resolving player names is one of the trickiest parts of the system. The ball-by-ball dataset uses abbreviated names, but users type full names, partial names, or misspellings. A multi-stage pipeline resolves inputs to canonical dataset names.

**Stage 1** checks for an exact match against known dataset names — the fastest and most common case for the app's dropdown inputs.

**Stage 2** performs a case-insensitive exact match, handling inputs like "rg sharma" resolving to "RG Sharma".

**Stage 3** looks up the input in a full-name-to-unique-name dictionary built from `player_details_final.csv`, so "Rohit Gurunath Sharma" resolves directly to "RG Sharma".

**Stage 4** performs an all-word substring match — every word in the input is checked against every player's full name in the details file. If exactly one player matches all words, that player is selected. If multiple match, the closest one by difflib edit distance is chosen.

**Stage 5** applies fuzzy full-name matching using difflib with a cutoff of 0.65, catching misspellings like "Jasprit Bumra" resolving to the correct full name.

**Stage 6** checks a pre-built parts index — a dictionary mapping individual name tokens to the players whose full names contain that token. If a single-word input like "Bumrah" maps to exactly one player in the index, that player is returned.

**Stage 7** does the same but on the last word of a multi-word input, useful for inputs like "Jofra Archer" where the first name might not be in the index but the surname is unique.

**Stage 8** falls back to fuzzy matching against all raw dataset names with a cutoff of 0.55, the most lenient level of the pipeline.

Common surname tokens (singh, kumar, mohammed, mohammad, md) are excluded from the parts index to prevent false matches when a user types just a shared surname. Only tokens longer than two characters are indexed, to avoid single-letter initials causing incorrect resolutions. Any name that was auto-resolved through stages 2–8 is displayed in a collapsible expander in the UI so the user can verify the system understood them correctly. Players with no match at all trigger a warning banner explaining that their ranking will be based solely on match-situation features with zeroed personal stats.

---

## Validation

The training notebook includes a comprehensive validation framework covering four areas.

**Data quality** checks include missing value rates per column, IQR-based outlier detection on key batting and bowling features, and matchup sparsity analysis showing that 71.8% of batter-bowler pairs have at least one ball of history with a median coverage of 6 balls per pair.

**Model metrics** cover ROC-AUC overall and per phase, top-1 and top-3 accuracy measuring whether the model's highest-ranked candidate matches the actual selection, mean reciprocal rank, and 5-fold cross-validation to confirm stability across different data splits.

**Cricket logic checks** verify that bowler over limits are correctly enforced, phase boundaries are correct, innings-split features are consistent with the innings label, and the pressure index produces values in the expected range with proper early-over dampening behaviour.

**Feature importance analysis** provides per-feature importance scores from the GBM, grouped by category (H2H, style, career, and situation), with visual plots of the top 20 features for both batting and bowling models.

---

## Streamlit App

The app provides two preset IPL scenarios for immediate testing — MI chasing 185 vs CSK at the 16.2 over mark, and RCB batting first at 10 overs — as well as a fully custom scenario mode. All inputs use searchable multiselect dropdowns with full player display names, plus free-text fields to add players not present in the dropdown.

When a recommendation is requested, all inputs are passed through the fuzzy name resolution pipeline before inference. The interface is split by captain role — batting side or bowling side — and shows a ranked table and bar chart of the top five candidates with their confidence scores. A feature details tab provides a per-player breakdown of H2H stats, phase stats, style matchup, career stats, and innings-split stats for each candidate.

A phase mode banner informs the user of the current H2H policy — fully active in powerplay, gated to pairs with at least 6 balls of data in the middle overs, and suppressed entirely in death overs. An inline pressure index explainer shows the exact formula and current numbers so the displayed pressure value is transparent. Any auto-resolved names appear in a collapsible section, and any players with no historical data trigger a yellow warning.

---

## Key Constants

All constants are defined once at the top of both the notebook and the app and must remain in sync. The powerplay ends after over 5, middle overs end after over 14, and the innings runs to 20 overs with a maximum of 4 overs per bowler. The par run rate is 8.0 runs per over. Bowlers with no wickets use a sentinel value of 999 for average and strike rate. The matchup confidence sigmoid is centred at 6 balls with a scale of 12. Default ensemble weights are 0.50 / 0.30 / 0.20 for matchup, extended, and full models respectively.

---

## Known Limitations

The model is trained on historical IPL data and has no knowledge of current form, recent injuries, or squad changes from the IPL 2026 auction. A player who has moved to a new team will still be scored using their matchup history against bowlers from their previous team's opponents.

Venue and pitch effects are not captured — the model treats a spin-friendly Eden Gardens the same as a pace-friendly Wankhede Stadium.

28.2% of batter-bowler pairs have zero balls of shared history. For these pairs the model falls back to style and career features only, which is less precise than direct matchup data.

The bowling model's lower top-1 accuracy (48.1% vs 73.9% for batting) reflects the more tactical nature of bowling changes — over restrictions, match situation shifts mid-over, and captain intuition about batters' weaknesses are not fully captured by the available features.

The `player_details_final.csv` file is static. New players added to IPL 2026 squads after the dataset was created will not have batting hand or bowling type data, and will trigger the unknown-player warning in the app.
