"""
IPL 2026 AI Captain — Streamlit App
Recommends the best next batter and bowler based on match situation,
player career stats, head-to-head matchups, and playing-style features.

Fixes applied:
  1. compute_match_context: early_damp, cap at 10, wickets/2+1,
     innings derived from is_chasing
  2. matchup_confidence / has_meaningful_matchup / log_bb_balls added to
     ZERO_MATCHUP_BAT and get_matchup_features
  3. career_no_wickets added to ZERO_BOWLER and get_bowler_features
  4. phase_death explicitly set via _add_phase_dummies()
  5. Pressure index UI expander shows correct formula
  6. Tailender fix: low_vol_penalty denominator 30→10, H2H reliability
     requires MATCHUP_MIN_BALLS before getting any weight
  7. Phase-aware H2H masking:
       Death     → H2H + style zeroed in extended model, matchup model weight=0
       Middle    → H2H active only if bb_balls >= MATCHUP_MIN_BALLS
       Powerplay → H2H fully active
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from difflib import get_close_matches

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL 2026 AI Captain",
    page_icon="🏏",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ──────────────────────────────────────────────────────────────
# Global constants  (single source of truth — must match notebook)
# ──────────────────────────────────────────────────────────────
POWERPLAY_END     = 6
MIDDLE_END        = 15
TOTAL_OVERS       = 20
MAX_BOWLER_OVERS  = 4
TOTAL_BALLS       = TOTAL_OVERS * 6        # 120
PAR_RR            = 8.0
BOWL_SR_SENTINEL  = 999
MATCHUP_MIN_BALLS = 6
MATCHUP_SCALE     = 12
ENSEMBLE_W        = (0.50, 0.30, 0.20)    # matchup, extended, full

# H2H + style columns to zero out in death overs
H2H_COLS_BAT = [
    "bb_balls", "bb_runs", "bb_wickets", "bb_runs_per_ball",
    "matchup_confidence", "has_meaningful_matchup", "log_bb_balls",
    "bat_vs_type_rpb", "bowl_vs_hand_econ",
]
H2H_COLS_BOWL = [
    "bb_balls_bvb", "bb_runs_bvb", "bb_wkts_bvb", "bb_rpb_bvb",
    "bowl_vs_hand_econ_bvb", "bat_vs_type_rpb_bvb",
]


# ──────────────────────────────────────────────────────────────
# Load artifacts (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    batting_model = pickle.load(open(os.path.join(MODELS_DIR, "batting_model.pkl"),  "rb"))
    bowling_model = pickle.load(open(os.path.join(MODELS_DIR, "bowling_model.pkl"),  "rb"))
    bat_cols      = pickle.load(open(os.path.join(MODELS_DIR, "batting_columns.pkl"), "rb"))
    bowl_cols     = pickle.load(open(os.path.join(MODELS_DIR, "bowling_columns.pkl"), "rb"))

    def _load(name):
        p = os.path.join(MODELS_DIR, name)
        return pickle.load(open(p, "rb")) if os.path.exists(p) else None

    bat_model_matchup   = _load("batting_model_matchup.pkl")
    bat_model_extended  = _load("batting_model_extended.pkl")
    bowl_model_matchup  = _load("bowling_model_matchup.pkl")
    bowl_model_extended = _load("bowling_model_extended.pkl")
    bat_matchup_cols    = _load("bat_matchup_columns.pkl")
    bat_extended_cols   = _load("bat_extended_columns.pkl")
    bowl_matchup_cols   = _load("bowl_matchup_columns.pkl")
    bowl_extended_cols  = _load("bowl_extended_columns.pkl")

    return (
        batting_model, bowling_model, bat_cols, bowl_cols,
        bat_model_matchup, bat_model_extended, bat_matchup_cols, bat_extended_cols,
        bowl_model_matchup, bowl_model_extended, bowl_matchup_cols, bowl_extended_cols,
    )


@st.cache_data
def load_lookup_tables():
    return pickle.load(open(os.path.join(MODELS_DIR, "lookup_tables.pkl"), "rb"))


@st.cache_data
def load_player_details():
    return pd.read_csv(os.path.join(DATA_DIR, "player_details_final.csv"))


(batting_model, bowling_model, bat_cols, bowl_cols,
 bat_model_matchup, bat_model_extended, bat_matchup_cols, bat_extended_cols,
 bowl_model_matchup, bowl_model_extended, bowl_matchup_cols, bowl_extended_cols) = load_models()

lookup         = load_lookup_tables()
player_details = load_player_details()

ALL_DATASET_NAMES = set(lookup["all_dataset_names"])

# ── Display ↔ Dataset name mappings ──────────────────────────
_display_to_dataset = {}
_dataset_to_display = {}

for _, row in player_details.iterrows():
    uname = str(row["unique_name"]).strip()
    fname = str(row["name"]).strip()
    if uname in ALL_DATASET_NAMES:
        _display_to_dataset[fname]  = uname
        _dataset_to_display[uname] = fname

for dn in ALL_DATASET_NAMES:
    if dn not in _dataset_to_display:
        _dataset_to_display[dn] = dn
        _display_to_dataset[dn] = dn

ALL_DISPLAY_SORTED = sorted(_display_to_dataset.keys())


def display_to_ds(names):
    return [_display_to_dataset.get(n, n) for n in names]


def ds_to_display(names):
    return [_dataset_to_display.get(n, n) for n in names]


# ──────────────────────────────────────────────────────────────
# Name Resolver
# ──────────────────────────────────────────────────────────────
_lower_to_dataset   = {n.strip().lower(): n.strip() for n in ALL_DATASET_NAMES}
_fullname_to_unique = {}
_parts_index        = {}

for _, row in player_details.iterrows():
    uname = str(row["unique_name"]).strip()
    fname = str(row["name"]).strip()
    if uname in ALL_DATASET_NAMES:
        _fullname_to_unique[fname.lower()] = uname
        for word in fname.split():
            w = word.lower().strip()
            if len(w) > 2 and w not in ("singh", "kumar", "mohammed", "mohammad", "md"):
                _parts_index.setdefault(w, []).append(uname)


def cricket_overs_to_balls(overs_float):
    complete_overs = int(overs_float)
    partial_balls  = round((overs_float - complete_overs) * 10)
    partial_balls  = min(partial_balls, 5)
    return complete_overs * 6 + partial_balls


def balls_to_cricket_overs(balls):
    return f"{balls // 6}.{balls % 6}"


def resolve_name(input_name):
    name  = input_name.strip()
    if name in ALL_DATASET_NAMES:
        return name, False
    lower = name.lower()
    if lower in _lower_to_dataset:
        resolved = _lower_to_dataset[lower]
        return resolved, resolved != name
    if lower in _fullname_to_unique:
        return _fullname_to_unique[lower], True
    input_words = [w.lower() for w in name.split() if len(w) > 1]
    if len(input_words) >= 2:
        candidates = set()
        for _, row in player_details.iterrows():
            uname = str(row["unique_name"]).strip()
            fname = str(row["name"]).strip().lower()
            if uname in ALL_DATASET_NAMES and all(w in fname for w in input_words):
                candidates.add(uname)
        if len(candidates) == 1:
            return candidates.pop(), True
        elif len(candidates) > 1:
            best = get_close_matches(name, candidates, n=1, cutoff=0.3)
            if best:
                return best[0], True
    fuzzy_full = get_close_matches(lower, list(_fullname_to_unique.keys()), n=1, cutoff=0.65)
    if fuzzy_full:
        return _fullname_to_unique[fuzzy_full[0]], True
    if len(input_words) == 1:
        word = input_words[0]
        if word in _parts_index and len(_parts_index[word]) == 1:
            return _parts_index[word][0], True
    if len(input_words) >= 1:
        last = input_words[-1]
        if last in _parts_index and len(_parts_index[last]) == 1:
            return _parts_index[last][0], True
    suggestions = get_close_matches(name, ALL_DATASET_NAMES, n=1, cutoff=0.55)
    if suggestions:
        return suggestions[0], True
    return name, False


def resolve_names(names):
    resolved, changes = [], []
    for name in names:
        rname, changed = resolve_name(name)
        resolved.append(rname)
        if changed:
            changes.append(f"'{name}' → '{rname}'")
    return resolved, changes


# ──────────────────────────────────────────────────────────────
# Zero-feature sentinels
# ──────────────────────────────────────────────────────────────
ZERO_BATTER = {
    "bat_runs": 0, "bat_balls_faced": 0, "bat_strike_rate": 0,
    "bat_boundary_rate": 0, "bat_sr_innings": 0, "bat_avg_innings": 0,
    "bat_phase_sr": 0, "bat_phase_avg": 0, "bat_phase_balls": 0,
}

# Fix 3: career_no_wickets added; debutant sentinel = 1
ZERO_BOWLER = {
    "career_bowl_balls": 0, "career_bowl_runs": 0, "career_bowl_wickets": 0,
    "career_bowl_economy": 0, "career_bowl_avg": BOWL_SR_SENTINEL,
    "career_bowl_sr": BOWL_SR_SENTINEL, "career_no_wickets": 1,
    "bowler_overs_used": 0, "overs_left_for_bowler": MAX_BOWLER_OVERS,
    "bowl_econ_innings": 0, "bowl_sr_innings": BOWL_SR_SENTINEL,
    "bowl_phase_econ": 0, "bowl_phase_sr": BOWL_SR_SENTINEL, "bowl_phase_balls": 0,
}

# Fix 2: matchup confidence columns added
ZERO_MATCHUP_BAT = {
    "bb_balls": 0, "bb_runs": 0, "bb_wickets": 0, "bb_runs_per_ball": 0,
    "matchup_confidence":    float(1.0 / (1.0 + np.exp(MATCHUP_MIN_BALLS / MATCHUP_SCALE))),
    "has_meaningful_matchup": 0,
    "log_bb_balls":           0.0,
}

ZERO_MATCHUP_BOWL = {
    "bb_balls_bvb": 0, "bb_runs_bvb": 0, "bb_wkts_bvb": 0, "bb_rpb_bvb": 0,
}


# ──────────────────────────────────────────────────────────────
# Phase helper
# ──────────────────────────────────────────────────────────────
def get_phase(over):
    if over < POWERPLAY_END:
        return "powerplay"
    elif over < MIDDLE_END:
        return "middle"
    return "death"


# ──────────────────────────────────────────────────────────────
# Fix 1: compute_match_context
# ──────────────────────────────────────────────────────────────
def compute_match_context(mi):
    is_chasing   = mi.get("is_chasing", mi.get("target", 0) > 0)
    overs_bowled = TOTAL_OVERS - mi["overs_remaining"]
    current_rr   = mi["current_score"] / overs_bowled if overs_bowled > 0 else 0

    if is_chasing:
        runs_needed = mi["target"] - mi["current_score"]
        required_rr = runs_needed / mi["overs_remaining"] if mi["overs_remaining"] > 0 else 0
    else:
        runs_needed = 0
        required_rr = 0

    over_number = int(TOTAL_OVERS - mi["overs_remaining"])
    phase       = get_phase(over_number)
    innings     = 2 if is_chasing else 1

    early_damp = min(overs_bowled / 3.0, 1.0)
    wf_factor  = mi["wickets_fallen"] / 2 + 1

    if innings == 1:
        raw_pressure = (PAR_RR / max(current_rr, 1)) * wf_factor
    else:
        raw_pressure = (required_rr / max(current_rr, 1)) * wf_factor

    pressure_index = raw_pressure * early_damp
    if not np.isfinite(pressure_index):
        pressure_index = 0.0
    pressure_index = float(np.clip(pressure_index, 0.0, 10.0))

    return {
        "current_score":     mi["current_score"],
        "wickets_fallen":    mi["wickets_fallen"],
        "balls_remaining":   mi["balls_remaining"],
        "overs_remaining":   mi["overs_remaining"],
        "current_run_rate":  current_rr,
        "runs_needed":       runs_needed,
        "required_run_rate": required_rr,
        "phase":             phase,
        "pressure_index":    pressure_index,
        "innings":           innings,
    }


# ──────────────────────────────────────────────────────────────
# Feature builders
# ──────────────────────────────────────────────────────────────
def get_batter_features(batter_name, innings=None, phase=None):
    feats = dict(lookup["batter_career"].get(batter_name.strip().lower(), ZERO_BATTER))

    if innings is not None and "batter_innings_split" in lookup:
        inn = lookup["batter_innings_split"].get((batter_name.strip().lower(), innings), {})
        feats["bat_sr_innings"]  = inn.get("bat_sr_innings",  0)
        feats["bat_avg_innings"] = inn.get("bat_avg_innings", 0)
    else:
        feats.setdefault("bat_sr_innings",  0)
        feats.setdefault("bat_avg_innings", 0)

    if phase is not None and "batter_phase" in lookup:
        ph = lookup["batter_phase"].get((batter_name.strip().lower(), phase), {})
        feats["bat_phase_sr"]    = ph.get("bat_phase_sr",    0)
        feats["bat_phase_avg"]   = ph.get("bat_phase_avg",   0)
        feats["bat_phase_balls"] = ph.get("bat_phase_balls", 0)
    else:
        feats.setdefault("bat_phase_sr",    0)
        feats.setdefault("bat_phase_avg",   0)
        feats.setdefault("bat_phase_balls", 0)

    return feats


# Fix 3: career_no_wickets populated from lookup or derived on the fly
def get_bowler_features(bowler_name, innings=None, phase=None):
    feats = dict(lookup["bowler_career"].get(bowler_name.strip().lower(), ZERO_BOWLER))

    if "career_no_wickets" not in feats:
        feats["career_no_wickets"] = int(feats.get("career_bowl_wickets", 0) == 0)

    if innings is not None and "bowler_innings_split" in lookup:
        inn = lookup["bowler_innings_split"].get((bowler_name.strip().lower(), innings), {})
        feats["bowl_econ_innings"] = inn.get("bowl_econ_innings", 0)
        feats["bowl_sr_innings"]   = inn.get("bowl_sr_innings",   BOWL_SR_SENTINEL)
    else:
        feats.setdefault("bowl_econ_innings", 0)
        feats.setdefault("bowl_sr_innings",   BOWL_SR_SENTINEL)

    if phase is not None and "bowler_phase" in lookup:
        ph = lookup["bowler_phase"].get((bowler_name.strip().lower(), phase), {})
        feats["bowl_phase_econ"]  = ph.get("bowl_phase_econ",  0)
        feats["bowl_phase_sr"]    = ph.get("bowl_phase_sr",    BOWL_SR_SENTINEL)
        feats["bowl_phase_balls"] = ph.get("bowl_phase_balls", 0)
    else:
        feats.setdefault("bowl_phase_econ",  0)
        feats.setdefault("bowl_phase_sr",    BOWL_SR_SENTINEL)
        feats.setdefault("bowl_phase_balls", 0)

    return feats


# Fix 2: derive confidence columns on the fly if not pre-saved in lookup
def get_matchup_features(batter, bowler):
    key  = (batter.strip().lower(), bowler.strip().lower())
    base = dict(lookup["matchup_batting"].get(key, {}))
    if not base:
        return ZERO_MATCHUP_BAT.copy()
    balls = base.get("bb_balls", 0)
    base.setdefault(
        "matchup_confidence",
        float(1.0 / (1.0 + np.exp(-(balls - MATCHUP_MIN_BALLS) / MATCHUP_SCALE))),
    )
    base.setdefault("has_meaningful_matchup", int(balls >= MATCHUP_MIN_BALLS))
    base.setdefault("log_bb_balls",           float(np.log1p(balls)))
    return base


def get_bowler_matchup_features(bowler, batter):
    key = (bowler.strip().lower(), batter.strip().lower())
    return dict(lookup["matchup_bowling"].get(key, ZERO_MATCHUP_BOWL))


def get_style_features_batting(batter, bowler):
    batter_low = batter.strip().lower()
    bowler_low = bowler.strip().lower()
    bowl_type  = lookup["player_bowl_type"].get(bowler_low)
    bat_hand   = lookup["player_bat_hand"].get(batter_low)
    bat_vs_type_rpb   = lookup["batter_vs_type"].get((batter_low, bowl_type), 0) if bowl_type else 0
    bowl_vs_hand_econ = lookup["bowler_vs_hand"].get((bowler_low, bat_hand),  0) if bat_hand  else 0
    return {"bat_vs_type_rpb": bat_vs_type_rpb, "bowl_vs_hand_econ": bowl_vs_hand_econ}


def get_style_features_bowling(bowler, batter):
    bowler_low = bowler.strip().lower()
    batter_low = batter.strip().lower()
    bat_hand   = lookup["player_bat_hand"].get(batter_low)
    bowl_type  = lookup["player_bowl_type"].get(bowler_low)
    bowl_vs_hand_econ_bvb = lookup["bowler_vs_hand"].get((bowler_low, bat_hand),  0) if bat_hand  else 0
    bat_vs_type_rpb_bvb   = lookup["batter_vs_type"].get((batter_low, bowl_type), 0) if bowl_type else 0
    return {"bowl_vs_hand_econ_bvb": bowl_vs_hand_econ_bvb, "bat_vs_type_rpb_bvb": bat_vs_type_rpb_bvb}


# ──────────────────────────────────────────────────────────────
# Candidate builders
# ──────────────────────────────────────────────────────────────
def build_batting_candidates(mi):
    context     = compute_match_context(mi)
    current_set = set(mi["current_batters"])
    out_set     = set(mi.get("players_out", []))
    innings     = context["innings"]
    candidates  = []

    for batter in mi["batting_xi"]:
        if batter in current_set or batter in out_set:
            continue
        row = {}
        row.update(context)
        row.update(get_batter_features(batter, innings=innings, phase=context["phase"]))
        row.update(get_matchup_features(batter, mi["current_bowler"]))
        row.update(get_style_features_batting(batter, mi["current_bowler"]))
        row["candidate_batter_name"] = batter
        candidates.append(row)

    return pd.DataFrame(candidates)


def build_bowling_candidates(mi):
    context      = compute_match_context(mi)
    batting_set  = set(mi["batting_xi"])
    overs_bowled = mi.get("bowler_overs_bowled", {})
    on_strike    = mi.get("current_batters", [""])[0]
    innings      = context["innings"]
    candidates   = []

    for bowler in mi["bowling_xi"]:
        if bowler in batting_set or overs_bowled.get(bowler, 0) >= MAX_BOWLER_OVERS:
            continue
        row = {}
        row.update(context)
        row.update(get_bowler_features(bowler, innings=innings, phase=context["phase"]))

        used = overs_bowled.get(bowler, 0)
        row["bowler_overs_used"]     = used
        row["overs_left_for_bowler"] = MAX_BOWLER_OVERS - used

        if on_strike:
            row.update(get_bowler_matchup_features(bowler, on_strike))
            row.update(get_style_features_bowling(bowler, on_strike))
        else:
            row.update(ZERO_MATCHUP_BOWL)
            row.update({"bowl_vs_hand_econ_bvb": 0, "bat_vs_type_rpb_bvb": 0})

        row["candidate_bowler_name"] = bowler
        candidates.append(row)

    return pd.DataFrame(candidates)


# ──────────────────────────────────────────────────────────────
# Fix 4: phase dummies helper
# ──────────────────────────────────────────────────────────────
def _add_phase_dummies(X):
    X["phase_powerplay"] = (X["phase"] == "powerplay").astype(int)
    X["phase_middle"]    = (X["phase"] == "middle").astype(int)
    X["phase_death"]     = (X["phase"] == "death").astype(int)
    return X.drop(columns=["phase"])


# ──────────────────────────────────────────────────────────────
# Fix 6 + 7: phase-aware H2H reliability
# ──────────────────────────────────────────────────────────────
def _compute_reliability(bb_balls, phase_now):
    """
    Death     → always 0   (H2H irrelevant, history doesn't matter)
    Middle    → 0 unless >= MATCHUP_MIN_BALLS
    Powerplay → full scaling
    """
    if phase_now == "death":
        return np.zeros(len(bb_balls))
    elif phase_now == "powerplay":
        return bb_balls / (bb_balls + 10)
    else:  # middle
        return np.where(
            bb_balls >= MATCHUP_MIN_BALLS,
            bb_balls / (bb_balls + 10),
            0.0,
        )


# ──────────────────────────────────────────────────────────────
# Prediction — next batter
# ──────────────────────────────────────────────────────────────
def predict_next_batter(mi, top_k=5):
    candidates = build_batting_candidates(mi)
    if len(candidates) == 0:
        return pd.DataFrame()

    # Capture phase before dropping the column
    phase_now = candidates["phase"].iloc[0] if "phase" in candidates.columns else ""

    X = candidates.drop(columns=["candidate_batter_name"]).copy()
    X = _add_phase_dummies(X)                              # Fix 4

    for col in bat_cols:
        if col not in X.columns:
            X[col] = 0

    use_ensemble = (
        bat_model_matchup  is not None and bat_model_extended is not None
        and bat_matchup_cols is not None and bat_extended_cols is not None
    )

    if use_ensemble:
        X_m = X.reindex(columns=bat_matchup_cols,  fill_value=0)
        X_e = X.reindex(columns=bat_extended_cols, fill_value=0)
        X_f = X[bat_cols]

        # Fix 7: zero H2H in extended model for death overs
        if phase_now == "death":
            for col in H2H_COLS_BAT:
                if col in X_e.columns:
                    X_e[col] = 0.0
            w_m, w_e, w_f = 0.0, 0.85, 0.15
        else:
            w_m, w_e, w_f = ENSEMBLE_W

        p_m = bat_model_matchup.predict_proba(X_m)[:,  1]
        p_e = bat_model_extended.predict_proba(X_e)[:, 1]
        p_f = batting_model.predict_proba(X_f)[:,     1]

        # Fix 6: softer penalty — preserves differentiation among tailenders
        low_vol_penalty = candidates["bat_balls_faced"].values / (
            candidates["bat_balls_faced"].values + 10
        )
        p_f = p_f * low_vol_penalty

        bb_balls    = candidates["bb_balls"].values
        reliability = _compute_reliability(bb_balls, phase_now)   # Fix 6 + 7

        eff_wm = w_m * reliability
        redist  = w_m * (1 - reliability)
        denom   = (w_e + w_f) if (w_e + w_f) > 0 else 1
        eff_we  = w_e + redist * (w_e / denom)
        eff_wf  = w_f + redist * (w_f / denom)

        candidates["predicted_score"] = eff_wm * p_m + eff_we * p_e + eff_wf * p_f

    else:
        X_f = X[bat_cols]
        low_vol_penalty = candidates["bat_balls_faced"].values / (
            candidates["bat_balls_faced"].values + 10
        )
        candidates["predicted_score"] = batting_model.predict_proba(X_f)[:, 1] * low_vol_penalty

    return (
        candidates
        .sort_values("predicted_score", ascending=False)
        [["candidate_batter_name", "predicted_score"]]
        .head(top_k)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────
# Prediction — next bowler
# ──────────────────────────────────────────────────────────────
def predict_next_bowler(mi, top_k=5):
    candidates = build_bowling_candidates(mi)
    if len(candidates) == 0:
        return pd.DataFrame()

    phase_now = candidates["phase"].iloc[0] if "phase" in candidates.columns else ""

    X = candidates.drop(columns=["candidate_bowler_name"]).copy()
    X = _add_phase_dummies(X)                              # Fix 4

    for col in bowl_cols:
        if col not in X.columns:
            X[col] = 0

    use_ensemble = (
        bowl_model_matchup  is not None and bowl_model_extended is not None
        and bowl_matchup_cols is not None and bowl_extended_cols is not None
    )

    if use_ensemble:
        X_m = X.reindex(columns=bowl_matchup_cols,  fill_value=0)
        X_e = X.reindex(columns=bowl_extended_cols, fill_value=0)
        X_f = X[bowl_cols]

        # Fix 7: zero H2H in extended model for death overs
        if phase_now == "death":
            for col in H2H_COLS_BOWL:
                if col in X_e.columns:
                    X_e[col] = 0.0
            w_m, w_e, w_f = 0.0, 0.85, 0.15
        else:
            w_m, w_e, w_f = ENSEMBLE_W

        p_m = bowl_model_matchup.predict_proba(X_m)[:,  1]
        p_e = bowl_model_extended.predict_proba(X_e)[:, 1]
        p_f = bowling_model.predict_proba(X_f)[:,     1]

        bb_balls    = candidates["bb_balls_bvb"].values
        reliability = _compute_reliability(bb_balls, phase_now)   # Fix 6 + 7

        eff_wm = w_m * reliability
        redist  = w_m * (1 - reliability)
        denom   = (w_e + w_f) if (w_e + w_f) > 0 else 1
        eff_we  = w_e + redist * (w_e / denom)
        eff_wf  = w_f + redist * (w_f / denom)

        candidates["predicted_score"] = eff_wm * p_m + eff_we * p_e + eff_wf * p_f

    else:
        X_f = X[bowl_cols]
        candidates["predicted_score"] = bowling_model.predict_proba(X_f)[:, 1]

    return (
        candidates
        .sort_values("predicted_score", ascending=False)
        [["candidate_bowler_name", "predicted_score"]]
        .head(top_k)
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────
# Preset Scenarios
# ──────────────────────────────────────────────────────────────
PRESETS = {
    "MI chasing 185 vs CSK (145/5, 16.2 ov)": {
        "is_chasing": True,
        "batting_xi": [
            "Rohit Gurunath Sharma", "Ishan Pranav Kumar Pandey Kishan",
            "Suryakumar Ashok Yadav", "Namboori Thakur Tilak Varma",
            "Hardik Himanshu Pandya", "Timothy Hays David", "Nehal Wadhera",
            "Piyush Pramod Chawla", "Jasprit Jasbirsingh Bumrah",
            "Gerald Coetzee", "Akash Madhwal",
        ],
        "bowling_xi": [
            "Deepak Lokandersingh Chahar", "Morawakage Maheesh Theekshana",
            "Ravindrasinh Anirudhsinh Jadeja", "Matheesha Pathirana",
            "Tushar Uday Deshpande", "Moeen Munir Ali",
        ],
        "current_batters": ["Timothy Hays David", "Nehal Wadhera"],
        "players_out": [
            "Rohit Gurunath Sharma", "Ishan Pranav Kumar Pandey Kishan",
            "Suryakumar Ashok Yadav", "Namboori Thakur Tilak Varma",
            "Hardik Himanshu Pandya",
        ],
        "current_bowler": "Matheesha Pathirana",
        "current_score": 145,
        "wickets_fallen": 5,
        "balls_remaining": 22,
        "target": 185,
        "bowler_overs_bowled": {
            "Deepak Lokandersingh Chahar": 4,
            "Morawakage Maheesh Theekshana": 3,
            "Ravindrasinh Anirudhsinh Jadeja": 4,
            "Matheesha Pathirana": 3,
            "Tushar Uday Deshpande": 2,
            "Moeen Munir Ali": 0,
        },
    },
    "RCB batting first (82/1, 10 ov)": {
        "is_chasing": False,
        "batting_xi": [
            "Virat Kohli", "Francois du Plessis", "Rajat Manohar Patidar",
            "Glenn James Maxwell", "Krishnakumar Dinesh Karthik", "Shahbaz Ahmed",
            "Pinnaduwage Wanindu Hasaranga de Silva", "Harshal Vikram Patel",
            "Mohammed Siraj", "Josh Reginald Hazlewood", "Karn Vinod Sharma",
        ],
        "bowling_xi": [
            "Harshit Pradeep Rana", "Trent Alexander Boult", "Piyush Pramod Chawla",
            "Sunil Philip Narine", "Andre Dwayne Russell", "Varun Chakravarthy Vinod",
        ],
        "current_batters": ["Virat Kohli", "Rajat Manohar Patidar"],
        "players_out": ["Francois du Plessis"],
        "current_bowler": "Sunil Philip Narine",
        "current_score": 82,
        "wickets_fallen": 1,
        "balls_remaining": 60,
        "target": 0,
        "bowler_overs_bowled": {
            "Harshit Pradeep Rana": 3, "Trent Alexander Boult": 3,
            "Piyush Pramod Chawla": 1, "Sunil Philip Narine": 2,
            "Andre Dwayne Russell": 1, "Varun Chakravarthy Vinod": 0,
        },
    },
    "Custom scenario": None,
}


# ──────────────────────────────────────────────────────────────
# UI — Header
# ──────────────────────────────────────────────────────────────
st.title("🏏 IPL 2026 — AI Captain")
st.markdown(
    "ML-powered recommendations for the **next batter** and **next bowler** "
    "based on match situation, career stats, head-to-head matchups, "
    "and playing-style analysis."
)

# ──────────────────────────────────────────────────────────────
# Sidebar — Match Input
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Match Scenario")
    user_side   = st.radio("You are the captain of the", ["Batting Side", "Bowling Side"], horizontal=True)
    preset_name = st.selectbox("Load a preset", list(PRESETS.keys()))
    preset      = PRESETS[preset_name]

    if preset is not None:
        is_chasing     = st.checkbox("Chasing?", value=preset["is_chasing"])
        target         = st.number_input("Target", value=preset.get("target", 0), min_value=0, step=1) if is_chasing else 0
        current_score  = st.number_input("Current Score", value=int(preset["current_score"]),
                                         min_value=0, max_value=500, step=1, format="%d")
        wickets_fallen = st.number_input("Wickets Fallen", value=int(preset["wickets_fallen"]),
                                         min_value=0, max_value=10, step=1)

        _preset_overs   = (TOTAL_BALLS - preset["balls_remaining"]) / 6
        _complete       = int(_preset_overs)
        _partial        = round((_preset_overs - _complete) * 6)
        _preset_display = _complete + _partial / 10
        overs_bowled_ui = st.number_input(
            "Overs Bowled (e.g. 16.4 = 16 overs, 4 balls)",
            value=round(_preset_display, 1), min_value=0.0,
            max_value=float(TOTAL_OVERS), step=0.1, format="%.1f",
        )
        balls_bowled    = cricket_overs_to_balls(overs_bowled_ui)
        balls_remaining = TOTAL_BALLS - balls_bowled

        st.subheader("Batting XI")
        _bat_defaults      = [n for n in preset["batting_xi"] if n in _display_to_dataset]
        batting_xi_display = list(st.multiselect("Batting XI (search & pick)", ALL_DISPLAY_SORTED,
                                                  default=_bat_defaults, key=f"bat_xi_{preset_name}"))
        _extra_bat = st.text_input("+ Add unlisted batter(s)", key="extra_bat", placeholder="Name1, Name2, ...")
        if _extra_bat:
            batting_xi_display += [n.strip() for n in _extra_bat.split(",") if n.strip()]

        st.subheader("Bowling XI")
        _bowl_defaults     = [n for n in preset["bowling_xi"] if n in _display_to_dataset]
        bowling_xi_display = list(st.multiselect("Bowling XI (search & pick)", ALL_DISPLAY_SORTED,
                                                  default=_bowl_defaults, key=f"bowl_xi_{preset_name}"))
        _extra_bowl = st.text_input("+ Add unlisted bowler(s)", key="extra_bowl", placeholder="Name1, Name2, ...")
        if _extra_bowl:
            bowling_xi_display += [n.strip() for n in _extra_bowl.split(",") if n.strip()]

        st.subheader("Current State")
        _bat_choices            = batting_xi_display if batting_xi_display else ALL_DISPLAY_SORTED
        _cb_defaults            = [n for n in preset["current_batters"] if n in set(_bat_choices)]
        current_batters_display = st.multiselect("Batters at crease", _bat_choices,
                                                  default=_cb_defaults, key=f"crease_{preset_name}")
        _po_defaults            = [n for n in preset["players_out"] if n in set(_bat_choices)]
        players_out_display     = st.multiselect("Players dismissed", _bat_choices,
                                                  default=_po_defaults, key=f"dismissed_{preset_name}")
        _bowl_choices           = bowling_xi_display if bowling_xi_display else ALL_DISPLAY_SORTED
        _cb_idx                 = _bowl_choices.index(preset["current_bowler"]) if preset["current_bowler"] in _bowl_choices else 0
        current_bowler_display  = st.selectbox("Current bowler", _bowl_choices,
                                                index=_cb_idx, key=f"cur_bowler_{preset_name}")

        st.subheader("Bowler Overs")
        bowler_overs_display = {}
        for b in bowling_xi_display:
            default = preset.get("bowler_overs_bowled", {}).get(b, 0)
            bowler_overs_display[b] = st.number_input(
                f"{b}", value=default, min_value=0, max_value=MAX_BOWLER_OVERS,
                step=1, key=f"ov_{b}",
            )

    else:
        is_chasing     = st.checkbox("Chasing?", value=True)
        target         = st.number_input("Target", value=180, min_value=0, step=1) if is_chasing else 0
        current_score  = st.number_input("Current Score", value=100, min_value=0, max_value=500, step=1, format="%d")
        wickets_fallen = st.number_input("Wickets Fallen", value=3, min_value=0, max_value=10, step=1)
        overs_bowled_ui = st.number_input("Overs Bowled (e.g. 16.4 = 16 overs, 4 balls)",
                                          value=12.0, min_value=0.0, max_value=float(TOTAL_OVERS),
                                          step=0.1, format="%.1f")
        balls_bowled    = cricket_overs_to_balls(overs_bowled_ui)
        balls_remaining = TOTAL_BALLS - balls_bowled

        st.subheader("Batting XI")
        batting_xi_display = list(st.multiselect("Batting XI (search & pick)", ALL_DISPLAY_SORTED, key="bat_xi_c"))
        _extra_bat = st.text_input("+ Add unlisted batter(s)", key="extra_bat_c", placeholder="Name1, Name2, ...")
        if _extra_bat:
            batting_xi_display += [n.strip() for n in _extra_bat.split(",") if n.strip()]

        st.subheader("Bowling XI")
        bowling_xi_display = list(st.multiselect("Bowling XI (search & pick)", ALL_DISPLAY_SORTED, key="bowl_xi_c"))
        _extra_bowl = st.text_input("+ Add unlisted bowler(s)", key="extra_bowl_c", placeholder="Name1, Name2, ...")
        if _extra_bowl:
            bowling_xi_display += [n.strip() for n in _extra_bowl.split(",") if n.strip()]

        st.subheader("Current State")
        _bat_choices            = batting_xi_display if batting_xi_display else ALL_DISPLAY_SORTED
        current_batters_display = st.multiselect("Batters at crease", _bat_choices, key="crease_c")
        players_out_display     = st.multiselect("Players dismissed",  _bat_choices, key="dismissed_c")
        _bowl_choices           = bowling_xi_display if bowling_xi_display else ALL_DISPLAY_SORTED
        current_bowler_display  = st.selectbox("Current bowler", [""] + _bowl_choices, key="cur_bowler_c")

        st.subheader("Bowler Overs")
        bowler_overs_display = {}
        for b in bowling_xi_display:
            bowler_overs_display[b] = st.number_input(
                f"{b}", value=0, min_value=0, max_value=MAX_BOWLER_OVERS, step=1, key=f"ov_{b}",
            )

# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────
run_prediction = st.sidebar.button(
    "🏏 Get AI Captain Recommendation", type="primary", use_container_width=True
)

if run_prediction:
    if not batting_xi_display or not bowling_xi_display or not current_bowler_display:
        st.error("Please fill in Batting XI, Bowling XI, and Current Bowler.")
        st.stop()

    with st.spinner("Resolving player names…"):
        batting_xi_raw        = display_to_ds(batting_xi_display)
        bowling_xi_raw        = display_to_ds(bowling_xi_display)
        current_batters_raw   = display_to_ds(current_batters_display)
        players_out_raw       = display_to_ds(players_out_display)
        current_bowler_raw_ds = _display_to_dataset.get(current_bowler_display, current_bowler_display)

        batting_xi,      bat_changes  = resolve_names(batting_xi_raw)
        bowling_xi,      bowl_changes = resolve_names(bowling_xi_raw)
        current_batters, cb_changes   = resolve_names(current_batters_raw)
        players_out,     po_changes   = resolve_names(players_out_raw)
        current_bowler,  cb_changed   = resolve_name(current_bowler_raw_ds)

        bowler_overs = {}
        for b_disp, ov in bowler_overs_display.items():
            b_ds = _display_to_dataset.get(b_disp, b_disp)
            rb, _ = resolve_name(b_ds)
            bowler_overs[rb] = ov

    all_changes = bat_changes + bowl_changes + cb_changes + po_changes
    if cb_changed:
        all_changes.append(f"'{current_bowler_raw_ds}' → '{current_bowler}'")
    if all_changes:
        with st.expander("🔄 Auto-resolved names", expanded=False):
            for c in all_changes:
                st.write(f"✓ {c}")

    _all_lower   = {n.lower() for n in ALL_DATASET_NAMES}
    _unknown_all = [p for p in batting_xi + bowling_xi if p.lower() not in _all_lower]
    if current_bowler.lower() not in _all_lower and current_bowler not in _unknown_all:
        _unknown_all.append(current_bowler)
    if _unknown_all:
        st.warning(
            f"⚠️ **No historical stats for: {', '.join(_unknown_all)}**\n\n"
            "Rankings for these players will be based purely on match situation "
            "with zeroed-out personal stats — treat with caution."
        )

    match_input = {
        "is_chasing":          is_chasing,
        "batting_xi":          batting_xi,
        "bowling_xi":          bowling_xi,
        "current_batters":     current_batters,
        "players_out":         players_out,
        "current_bowler":      current_bowler,
        "current_score":       current_score,
        "wickets_fallen":      wickets_fallen,
        "balls_remaining":     balls_remaining,
        "overs_remaining":     balls_remaining / 6,
        "target":              target,
        "bowler_overs_bowled": bowler_overs,
    }

    # ── Match Summary ─────────────────────────────────────────
    ctx           = compute_match_context(match_input)
    innings_label = "CHASING" if is_chasing else "BATTING FIRST"
    overs_done    = TOTAL_OVERS - ctx["overs_remaining"]
    early_damp    = min(overs_done / 3.0, 1.0)
    phase_now     = ctx["phase"]

    col_summary, _ = st.columns([3, 1])
    with col_summary:
        st.subheader(f"📋 Match Situation — {innings_label}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score",      f"{current_score}/{wickets_fallen}")
        m2.metric("Overs Left", balls_to_cricket_overs(balls_remaining))
        m3.metric("Current RR", f"{ctx['current_run_rate']:.2f}")
        if is_chasing:
            m4.metric("Required RR", f"{ctx['required_run_rate']:.2f}")
        else:
            m4.metric("Phase", phase_now.title())

        st.caption(
            f"At crease: **{', '.join(ds_to_display(current_batters))}** · "
            f"Bowling: **{_dataset_to_display.get(current_bowler, current_bowler)}** · "
            f"Phase: **{phase_now.title()}** · "
            f"Pressure: {ctx['pressure_index']:.2f}"
        )

        # Fix 5: correct pressure formula in expander
        with st.expander("ℹ️ How is Pressure Index calculated?", expanded=False):
            if is_chasing:
                rr_ratio = f"{ctx['required_run_rate']:.2f} / {max(ctx['current_run_rate'], 1):.2f}"
                st.markdown(
                    f"**Chasing formula:** (Required RR ÷ Current RR) × (Wickets/2 + 1) × Early Damp\n\n"
                    f"= ({rr_ratio}) × ({wickets_fallen}/2 + 1) × {early_damp:.2f} "
                    f"= **{ctx['pressure_index']:.2f}**\n\n"
                    f"Early Damp = min(overs_done / 3, 1) = {early_damp:.2f} "
                    f"_(reduces noise in first 3 overs)_\n\n"
                    f"Higher value = more pressure on batting side."
                )
            else:
                rr_ratio = f"{PAR_RR:.2f} / {max(ctx['current_run_rate'], 1):.2f}"
                st.markdown(
                    f"**Batting first formula:** (Par RR ÷ Current RR) × (Wickets/2 + 1) × Early Damp\n\n"
                    f"Par RR = {PAR_RR}  ·  Early Damp = min(overs_done / 3, 1) = {early_damp:.2f}\n\n"
                    f"= ({rr_ratio}) × ({wickets_fallen}/2 + 1) × {early_damp:.2f} "
                    f"= **{ctx['pressure_index']:.2f}**\n\n"
                    f"If scoring below par or losing wickets, pressure rises."
                )

        # Phase mode banner
        if phase_now == "death":
            st.info(
                "🎯 **Death overs mode** — H2H matchup history and style matchup are "
                "ignored. Rankings are based purely on death-over skill, career stats, "
                "and match situation."
            )
        elif phase_now == "middle":
            st.info(
                "⚖️ **Middle overs mode** — H2H history counts only when "
                f"≥ {MATCHUP_MIN_BALLS} balls of data exist between the pair."
            )
        else:
            st.info("🏏 **Powerplay mode** — Full H2H matchup history is active.")

    st.divider()

    # ── Batting Side ──────────────────────────────────────────
    batter_result = pd.DataFrame()
    bowler_result = pd.DataFrame()

    if user_side == "Batting Side":
        tab_rec, tab_details = st.tabs(["🏏 Recommendation", "📊 Feature Details"])

        with tab_rec:
            st.subheader("🏏 Next Batter Recommendation")
            if wickets_fallen >= 10:
                st.info("All out — no batter recommendation needed.")
            else:
                batter_result = predict_next_batter(match_input)
                if len(batter_result) == 0:
                    st.warning("No candidates available (all batting or dismissed).")
                else:
                    batter_result.columns   = ["Batter", "Score"]
                    batter_result["Batter"] = batter_result["Batter"].map(
                        lambda n: _dataset_to_display.get(n, n)
                    )
                    batter_result.index = range(1, len(batter_result) + 1)
                    col_table, col_chart = st.columns(2)
                    with col_table:
                        st.dataframe(
                            batter_result.style.bar(subset=["Score"], color="#4CAF50", vmin=0, vmax=1),
                            use_container_width=True,
                        )
                    with col_chart:
                        st.bar_chart(batter_result.set_index("Batter")["Score"], color="#4CAF50")

        with tab_details:
            if wickets_fallen >= 10:
                st.info("All out — no feature details to show.")
            elif len(batter_result) == 0:
                st.info("No candidates available.")
            else:
                innings   = ctx["innings"]
                for rank, row in batter_result.iterrows():
                    batter_display = row["Batter"]
                    batter_ds      = _display_to_dataset.get(batter_display, batter_display)
                    with st.expander(f"**{batter_display}** — Score: {row['Score']:.4f}",
                                     expanded=(rank == 1)):
                        career    = get_batter_features(batter_ds, innings=innings, phase=phase_now)
                        matchup   = get_matchup_features(batter_ds, current_bowler)
                        style     = get_style_features_batting(batter_ds, current_bowler)
                        inn_label  = "Chasing" if is_chasing else "Setting"
                        h2h_label  = "H2H" if matchup["bb_balls"] > 0 else "H2H (no data)"
                        h2h_active = phase_now != "death"

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(
                                f"**vs {_dataset_to_display.get(current_bowler, current_bowler)}**"
                                f" — {h2h_label}"
                                + (" _(ignored — death overs)_" if not h2h_active else "")
                            )
                            st.metric("H2H Balls",    matchup["bb_balls"])
                            st.metric("H2H Runs",     matchup["bb_runs"])
                            st.metric("H2H Wickets",  matchup["bb_wickets"])
                            st.metric("H2H RPB",      f"{matchup['bb_runs_per_ball']:.3f}")
                            st.metric("Matchup Conf", f"{matchup.get('matchup_confidence', 0):.3f}")
                            if not h2h_active:
                                st.caption("_Death overs: H2H zeroed. Pure skill ranking._")
                            elif matchup["bb_balls"] == 0:
                                st.caption("_No H2H data — weight redistributed._")
                            st.markdown(f"**{phase_now.capitalize()} Stats** ")
                            st.metric("Phase SR",    f"{career.get('bat_phase_sr', 0):.1f}")
                            st.metric("Phase Avg",   f"{career.get('bat_phase_avg', 0):.1f}")
                            st.metric("Phase Balls", career.get("bat_phase_balls", 0))
                        with c2:
                            st.markdown(
                                "**Style Matchup**"
                                + (" _(ignored — death overs)_" if not h2h_active else "")
                            )
                            st.metric("Bat vs Bowl Type RPB",  f"{style['bat_vs_type_rpb']:.3f}")
                            st.metric("Bowl vs Bat Hand Econ", f"{style['bowl_vs_hand_econ']:.2f}")
                        with c3:
                            st.markdown("**Career Stats**")
                            st.metric("Runs (median)",  f"{career['bat_runs']:.1f}")
                            st.metric("Balls (median)", f"{career['bat_balls_faced']:.1f}")
                            st.metric("Strike Rate",    f"{career['bat_strike_rate']:.1f}")
                            st.metric("Boundary Rate",  f"{career['bat_boundary_rate']:.3f}")
                            st.markdown(f"**{inn_label} Stats**")
                            st.metric(f"SR ({inn_label})",  f"{career.get('bat_sr_innings', 0):.1f}")
                            st.metric(f"Avg ({inn_label})", f"{career.get('bat_avg_innings', 0):.1f}")

    # ── Bowling Side ──────────────────────────────────────────
    else:
        tab_rec, tab_details = st.tabs(["🎳 Recommendation", "📊 Feature Details"])

        with tab_rec:
            st.subheader("🎳 Next Bowler Recommendation")
            bowler_result = predict_next_bowler(match_input)
            if len(bowler_result) == 0:
                st.warning("No bowlers available (all maxed at 4 overs).")
            else:
                bowler_result.columns   = ["Bowler", "Score"]
                bowler_result["Bowler"] = bowler_result["Bowler"].map(
                    lambda n: _dataset_to_display.get(n, n)
                )
                bowler_result.index = range(1, len(bowler_result) + 1)
                col_table, col_chart = st.columns(2)
                with col_table:
                    st.dataframe(
                        bowler_result.style.bar(subset=["Score"], color="#2196F3", vmin=0, vmax=1),
                        use_container_width=True,
                    )
                with col_chart:
                    st.bar_chart(bowler_result.set_index("Bowler")["Score"], color="#2196F3")

        with tab_details:
            if len(bowler_result) == 0:
                st.info("No bowlers available.")
            else:
                innings   = ctx["innings"]
                on_strike = current_batters[0] if current_batters else ""
                for rank, row in bowler_result.iterrows():
                    bowler_display = row["Bowler"]
                    bowler_ds      = _display_to_dataset.get(bowler_display, bowler_display)
                    with st.expander(f"**{bowler_display}** — Score: {row['Score']:.4f}",
                                     expanded=(rank == 1)):
                        career    = get_bowler_features(bowler_ds, innings=innings, phase=phase_now)
                        matchup   = get_bowler_matchup_features(bowler_ds, on_strike) if on_strike else ZERO_MATCHUP_BOWL
                        style     = (get_style_features_bowling(bowler_ds, on_strike)
                                     if on_strike else {"bowl_vs_hand_econ_bvb": 0, "bat_vs_type_rpb_bvb": 0})
                        inn_label  = "Chasing" if is_chasing else "Setting"
                        h2h_label  = "H2H" if matchup["bb_balls_bvb"] > 0 else "H2H (no data)"
                        h2h_active = phase_now != "death"
                        on_strike_display = _dataset_to_display.get(on_strike, on_strike) if on_strike else ""

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(
                                (f"**vs {on_strike_display}**" if on_strike else "**vs Batter**")
                                + f" — {h2h_label}"
                                + (" _(ignored — death overs)_" if not h2h_active else "")
                            )
                            st.metric("H2H Balls",   matchup["bb_balls_bvb"])
                            st.metric("H2H Runs",    matchup["bb_runs_bvb"])
                            st.metric("H2H Wickets", matchup["bb_wkts_bvb"])
                            st.metric("H2H RPB",     f"{matchup['bb_rpb_bvb']:.3f}")
                            if not h2h_active:
                                st.caption("_Death overs: H2H zeroed. Pure skill ranking._")
                            elif matchup["bb_balls_bvb"] == 0:
                                st.caption("_No H2H data — weight redistributed._")
                            st.markdown(f"**{phase_now.capitalize()} Stats**")
                            st.metric("Phase Econ",  f"{career.get('bowl_phase_econ', 0):.2f}")
                            st.metric("Phase SR",    f"{career.get('bowl_phase_sr', BOWL_SR_SENTINEL):.1f}")
                            st.metric("Phase Balls", career.get("bowl_phase_balls", 0))
                        with c2:
                            st.markdown(
                                "**Style Matchup**"
                                + (" _(ignored — death overs)_" if not h2h_active else "")
                            )
                            st.metric("Bowl vs Bat Hand Econ", f"{style['bowl_vs_hand_econ_bvb']:.2f}")
                            st.metric("Bat vs Bowl Type RPB",  f"{style['bat_vs_type_rpb_bvb']:.3f}")
                        with c3:
                            st.markdown("**Career Stats**")
                            st.metric("Balls Bowled", career["career_bowl_balls"])
                            st.metric("Wickets",      career["career_bowl_wickets"])
                            st.metric("Economy",      f"{career['career_bowl_economy']:.2f}")
                            st.metric("Average",      f"{career['career_bowl_avg']:.1f}")
                            st.metric("Strike Rate",  f"{career['career_bowl_sr']:.1f}")
                            st.markdown(f"**{inn_label} Stats**")
                            st.metric(f"Economy ({inn_label})", f"{career.get('bowl_econ_innings', 0):.2f}")
                            st.metric(f"SR ({inn_label})",      f"{career.get('bowl_sr_innings', BOWL_SR_SENTINEL):.1f}")