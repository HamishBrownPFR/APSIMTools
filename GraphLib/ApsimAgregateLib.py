# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Style

# %%
# %%writefile C:\GitHubRepos\APSIMTools\GraphLib\apsim_tools\style.py

## Color set for graphs
Colors = {1:'#000000',
2:'#E69F00',
3:'#56B4E9',
4:'#009E73',
5:'#F0E442',
6:'#0072B2',
7:'#D55E00',
8:'#CC79A7',
9:'#1F77B4',
10:'#AEC7E8',
11:'#FF7F0E',
12:'#FFBB78',
13:'#2CA02C',
14:'#98DF8A',
15:'#D62728',
16:'#FF9896',
17:'#9467BD',
18:'#C5B0D5',
19:'#8C564B',
20:'#C49C94',
21:'#E377C2',
22:'#F7B6D2',
23:'#7F7F7F',
24:'#C7C7C7',
25:'#BCBD22',
26:'#DBDB8D',
27:'#17BECF',
28:'#9EDAE5'}

# marker set for graphs
Markers = {
 1: 'o',   # circle
 2: '^',   # triangle up
 3: 's',   # square
 4: 'D',   # diamond
 5: 'v',   # triangle down
 6: '<',   # triangle left
 7: '>',   # triangle right
 8: 'p',   # pentagon
 9: 'h',   # hexagon1 ✅ NEW
10: 'H',   # hexagon2 ✅ NEW
11: 'd',   # thin diamond
12: 'P',   # plus-filled (optional)
13: 'X',   # x-filled (optional)
}

#Line set for graphs
Lines = {1: '-',
 2: '--',
 3: '-.',
 4: ':',
 5: '-',
 6: '--',
 7: '-.',
 8: ':',
 9: '-',
 10: '--',
 11: '-.',
 12: ':',
 13: '-',
}

# %% [markdown]
# # Runner - helpers for setting up and running .apsim files

# %%
# %%writefile C:\GitHubRepos\APSIMTools\GraphLib\apsim_tools\runner.py

from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
import subprocess
import shutil
import warnings

from apsim_tools.style import Colors, Markers, Lines

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)


#__________________________________________________________
# Shift branchs, build and run simulations
#----------------------------------------------------------

def validate_run_branches(config):
    invalid = [b for b in config["run_branches"] if b not in config["branches"]]
    if invalid:
        raise ValueError(f"Invalid RUN_BRANCHES: {invalid}")

def checkout_branch(branch, config):
    subprocess.run(["git", "checkout", branch],
                   cwd=config["repo_path"],
                   check=True)

def build_apsim(config):
    subprocess.run(
        ["dotnet", "build", config["apsim_solution"], "-c", "Release"],
        cwd=config["repo_path"],
        check=True
    )
    return 
    
def run_apsim(sim_file, config, apply_fn):
    """
    High-level runner:
    - writes apply file
    - runs APSIM
    """
    print(f"▶ Running APSIM: {sim_file.name}")

    apply_file = apply_fn(sim_file)
    
    result = subprocess.run(
        [
            config["apsim_exe"],
            sim_file,
            "--apply",
            apply_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(result.stdout)

    return result
    
def db_path(sim_file):
    return sim_file.with_suffix(".db")

def branch_db_path(sim_file, branch):
    return sim_file.parent / f"{sim_file.stem}_{branch}.db"

def reset_repo(config):
    print("🔄 Resetting tracked files only")

    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=config["repo_path"],
        check=True
    )

def read_table(db_file, table):
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
  
    df.columns = [
        c.strip().replace('"', '').replace("'", '')
        for c in df.columns
    ]

    return df

def should_run(branch_name, config):
    return branch_name in config["run_branches"]


def load_all(config, apply_fn):
    return pd.concat([
        load_branch_data(name, git_branch, config, apply_fn)
        for name, git_branch in config["branches"].items()
    ], ignore_index=True)

#__________________________________________________________
# load data from .db files for specific branch
#----------------------------------------------------------
def load_branch_data(branch_name, git_branch, config, apply_fn):

    if should_run(branch_name, config):
        reset_repo(config)
        checkout_branch(git_branch, config)
        build_apsim(config)

    frames = []

    for sim in config["sim_files"]:

        db = branch_db_path(sim, branch_name)

        # --- Run APSIM ---
        if should_run(branch_name, config):
            print("")
            print(f"▶ Running APSIM [{branch_name}]: {sim.name}")
            run_apsim(sim, config, apply_fn)

            original_db = sim.with_suffix(".db")
            branch_db = branch_db_path(sim, branch_name)

            shutil.copyfile(original_db, branch_db)

        else:
            print(f"⏭ Skipping run [{branch_name}]: {sim.name}")

        if not db.exists():
            print(f"❌ DB not created: {db}")
            continue

        # ======================================================
        # ✅ READ DATABASE
        # ======================================================
        with sqlite3.connect(db) as conn:

            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';",
                conn
            )["name"].tolist()

            print(f"{sim.name} tables: {tables}")

            # ---------------------------------------------
            # ✅ Observed
            # ---------------------------------------------
            obs = None
            if "Observed" in tables:
                obs = pd.read_sql("SELECT * FROM [Observed]", conn)
                obs["type"] = "obs"

            # ---------------------------------------------
            # ✅ Predictions
            # ---------------------------------------------
            pred = None
            if "AnalysisReport" in tables:
                pred = pd.read_sql("SELECT * FROM [AnalysisReport]", conn)
                pred["type"] = "pred"

            if pred is None and obs is None:
                continue

            # ---------------------------------------------
            # ✅ CRITICAL FIX: enforce SimulationID alignment
            # ---------------------------------------------
            if pred is not None and obs is not None:

                pred_ids = set(pred["SimulationID"].dropna().unique())
                obs_ids = set(obs["SimulationID"].dropna().unique())

                # --- Diagnostics (optional but useful) ---
                missing_ids = pred_ids - obs_ids
                if missing_ids:
                    print(f"\n⚠️ Missing observed simulations in {sim.name}: {len(missing_ids)}")

                extra_obs_ids = obs_ids - pred_ids
                if extra_obs_ids:
                    print(f"\n⚠️ Dropping unmatched observed simulations in {sim.name}: {len(extra_obs_ids)}")

                # ✅ THE FIX — enforce alignment
                before = len(obs)
                obs = obs[obs["SimulationID"].isin(pred_ids)]
                after = len(obs)

                print(f"✅ Obs filtered by SimulationID: {after}/{before} rows kept")

                if after == 0:
                    obs = None  # clean downstream handling

            # ---------------------------------------------
            # ✅ Attach metadata (branch, file)
            # ---------------------------------------------
            if pred is not None:
                pred["branch"] = branch_name
                pred["file"] = sim.name

            if obs is not None:
                obs["branch"] = branch_name
                obs["file"] = sim.name

            # ---------------------------------------------
            # ✅ Combine
            # ---------------------------------------------
            if pred is not None and obs is not None:
                df = pd.concat([pred, obs], ignore_index=True)
            elif pred is not None:
                df = pred.copy()
            elif obs is not None:
                df = obs.copy()
            else:
                continue

            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data loaded for branch {branch_name}")

    return pd.concat(frames, ignore_index=True)
    
# def load_branch_data(branch_name, git_branch, config, apply_fn):
#     if should_run(branch_name, config):
#         reset_repo(config)
#         checkout_branch(git_branch, config)
#         build_apsim(config)

#     frames = []

#     for sim in config["sim_files"]:
#         db = branch_db_path(sim, branch_name)

#         # --- Run APSIM ---
#         if should_run(branch_name, config):
#             print("")
#             print(f"▶ Running APSIM [{branch_name}]: {sim.name}")
#             run_apsim(sim, config, apply_fn)

#             original_db = sim.with_suffix(".db")
#             branch_db = branch_db_path(sim, branch_name)

#             shutil.copyfile(original_db, branch_db)

#         else:
#             print(f"⏭ Skipping run [{branch_name}]: {sim.name}")

#         if not db.exists():
#             print(f"❌ DB not created: {db}")
#             continue

#         # ======================================================
#         # ✅ READ DATABASE
#         # ======================================================
#         with sqlite3.connect(db) as conn:

#             tables = pd.read_sql(
#                 "SELECT name FROM sqlite_master WHERE type='table';",
#                 conn
#             )["name"].tolist()

#             print(f"{sim.name} tables: {tables}")

#             # ---------------------------------------------
#             # ✅ Observed
#             # ---------------------------------------------
#             obs = None
#             if "Observed" in tables:
#                 obs = pd.read_sql("SELECT * FROM [Observed]", conn)
#                 obs["type"] = "obs"

#             # ---------------------------------------------
#             # ✅ Predictions
#             # ---------------------------------------------
#             pred = None
#             if "AnalysisReport" in tables:
#                 pred = pd.read_sql("SELECT * FROM [AnalysisReport]", conn)
#                 pred["type"] = "pred"

#             if pred is None and obs is None:
#                 continue
                
#             # ---------------------------------------------
#             # ✅ CHECK SimulationID alignment (per file)
#             # ---------------------------------------------
#             if pred is not None and obs is not None:

#                 pred_ids = set(pred["SimulationID"].drop_duplicates())
#                 obs_ids  = set(obs["SimulationID"].drop_duplicates())

#                 # simulations in pred but not in obs
#                 missing_ids = pred_ids - obs_ids

#                 if missing_ids:
#                     print(f"\n⚠️ Missing observed simulations in {sim.name}:")
#                     print(f"Count: {len(missing_ids)}")

#                     # attempt to print identifying info from pred
#                     cols_to_show = [
#                         c for c in [
#                             "SimulationID",
#                             "Simulation.Name",   # if present
#                             "Experiment",
#                         ] if c in pred.columns
#                     ]

#                     missing_rows = (
#                         pred[pred["SimulationID"].isin(missing_ids)]
#                         [cols_to_show]
#                         .drop_duplicates()
#                         .sort_values("SimulationID")
#                     )

#                     print(missing_rows.head(20)["Simulation.Name"].to_list())  # limit output     
                    
#             # ---------------------------------------------
#             # ✅ Attach metadata (branch, file)
#             # ---------------------------------------------
#             if pred is not None:
#                 pred["branch"] = branch_name
#                 pred["file"] = sim.name

#             if obs is not None:
#                 obs["branch"] = branch_name
#                 obs["file"] = sim.name

#             # ---------------------------------------------
#             # ✅ Combine
#             # ---------------------------------------------
#             df = pd.concat(
#                 [x for x in [pred, obs] if x is not None],
#                 ignore_index=True
#             )

#             frames.append(df)

#     if not frames:
#         raise RuntimeError(f"No data loaded for branch {branch_name}")

#     return pd.concat(frames, ignore_index=True)



#__________________________________________________________
# Enforce indicies on observed data
#----------------------------------------------------------

def enforce_indices_to_observed(df, indices_to_fill):

    keys = ["branch", "file", "SimulationID"]

    pred = df[df["type"] == "pred"].set_index(keys)
    obs  = df[df["type"] == "obs"].set_index(keys)

    meta = (
        pred
        .groupby(level=keys)[indices_to_fill]
        .first()
    )

    # ✅ vectorised alignment (clean + reliable)
    common_idx = obs.index.intersection(meta.index)

    obs.loc[common_idx, indices_to_fill] = meta.loc[common_idx, indices_to_fill]

    return pd.concat([pred.reset_index(), obs.reset_index()], ignore_index=True)


#__________________________________________________________
# Fuction which processes raw data into indexed format for graphing
#----------------------------------------------------------

def to_tidy(df, config, additional_index_maps=None):

    df = df.copy()

    # rename Simulation.Name
    if "Simulation.Name" in df.columns:
        df = df.rename(columns={"Simulation.Name": "SimulationName"})
        
    # ---------------------------------------------
    # ✅ Define columns that must NEVER be melted
    # ---------------------------------------------
    protected_cols = [
        "SimulationID",
        "CheckpointID",
        "Clock.Today",
        config["stage_name_var"],
        config["cultivar_col"],
        "SimulationName",
        "Experiment",
        "Zone",
        "file",
        "branch",
        "type"
    ]

    # include any that actually exist
    protected_cols = [c for c in protected_cols if c in df.columns]

    # ---------------------------------------------
    # ✅ Identify numeric columns
    # ---------------------------------------------
    
    possible_numeric = df.columns.difference(protected_cols)

    for col in possible_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---------------------------------------------
    # ✅ Value columns = numeric BUT NOT protected
    # ---------------------------------------------
    value_cols = [
        c for c in numeric_cols
        if c not in protected_cols
    ]

    # ---------------------------------------------
    # ✅ ID columns = everything else
    # ---------------------------------------------
    id_cols = [c for c in df.columns if c not in value_cols]
    
    # ---------------------------------------------
    # ✅ Enforce indices BEFORE melt
    # ---------------------------------------------
    indices_to_fill = ['Experiment','SimulationName',config["cultivar_col"]]
    df = enforce_indices_to_observed(df, indices_to_fill)
    
    col = config["cultivar_col"]

    df[col] = df.groupby(
        ["branch", "file", "SimulationID"]
    )[col].transform(lambda x: x.ffill().bfill())
                                     
    # ---------------------------------------------
    # ✅ Melt
    # ---------------------------------------------
    tidy = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="variable",
        value_name="value"
    )
    
    # ---------------------------------------------
    # ✅ Add development type index
    # ---------------------------------------------
    if additional_index_maps:
        for new_col, spec in additional_index_maps.items():
    
            source_col = spec.get("source")
            mapping = spec.get("map")
    
            if source_col in tidy.columns:
                tidy[new_col] = tidy[source_col].map(mapping)

    return tidy.dropna(subset=["value"])

# %% [markdown]
# # Data - functions to get data for graphing

# %%
# %%writefile C:\GitHubRepos\APSIMTools\GraphLib\apsim_tools\data.py


import pandas as pd
import numpy as np

#__________________________________________________________
# Get data from harvest events
#----------------------------------------------------------

def get_harvest_aligned(tidy, config, variable, filters=None):
    """
    Align data at harvest stage (HarvestRipe).

    Behaviour:
    - Applies optional filters
    - Filters to HarvestRipe stage (implicit)
    - If observed data exists:
        → returns aligned obs vs pred
    - If NOT:
        → returns predictions only (obs = NaN)

    Always returns one row per simulation.
    """

    # -------------------------------
    # APPLY FILTERS
    # -------------------------------
    df = apply_filters(tidy, filters)
    
    # -------------------------------
    # HARVEST FILTER (implicit)
    # -------------------------------
    if config["stage_name_var"] not in df.columns:
        raise KeyError(f"Missing {config['stage_name_var']} in tidy data")

    df = df[df[config["stage_name_var"]] == config["harvest_stage"]]

    # -------------------------------
    # Slice out variable to graph
    # -------------------------------
    df = df[df.variable == variable].copy()

    # -------------------------------
    # SPLIT OBS / PRED
    # -------------------------------
    obs = df[df["type"] == "obs"].copy()
    pred = df[df["type"] == "pred"].copy()

    # -------------------------------
    # COLLAPSE TO ONE ROW PER SIMULATION
    # -------------------------------
    keys = ["branch", "file", "SimulationID"]

    pred = pred.sort_values("Clock.Today").groupby(keys).tail(1)

    if obs.empty:
        # ✅ No observed data → prediction only mode
        print("\nNo observed data — returning predictions only")

        pred = pred.copy()
        pred["pred"] = pred["value"]
        pred["obs"] = np.nan

        return pred

    # collapse obs as well
    obs = obs.sort_values("Clock.Today").groupby(keys).tail(1)

    # -------------------------------
    # SAFE MERGE (1:1)
    # -------------------------------
    aligned = pred.merge(
        obs,
        on=keys,
        how="inner",
        suffixes=("_pred", "_obs")
    )

    # -------------------------------
    # BUILD OUTPUT
    # -------------------------------
    aligned["pred"] = aligned["value_pred"]
    aligned["obs"] = aligned["value_obs"]

    return aligned

#__________________________________________________________
# Get data from all daily measurements
#----------------------------------------------------------

def get_daily_obs_pred_exact(tidy, variable, filters=None):

    df = apply_filters(tidy, filters)

    keys = ["branch","file","SimulationID","Clock.Today"]

    obs = df[
        (df["type"]=="obs") &
        (df["variable"]==variable)
    ][keys + ["value"]].rename(columns={"value":"obs"})

    pred = df[
        (df["type"]=="pred") &
        (df["variable"]==variable)
    ][keys + ["value"]].rename(columns={"value":"pred"})

    # ✅ THIS is the key — inner join
    aligned = obs.merge(pred, on=keys, how="inner")

    return aligned

#__________________________________________________________
# Filter out unwanted data
#----------------------------------------------------------

def apply_filters(tidy, filters):

    if filters is None:
        return tidy.copy()

    df = tidy.copy()
    keys = ["branch", "file", "SimulationID"]

    for key, values in filters.items():

        # ✅ CASE 1: key is a COLUMN
        if key in df.columns:

            valid_sims = (
                df[df[key].isin(values)][keys]
                .drop_duplicates()
            )

        # ✅ CASE 2: key is a MELTED VARIABLE
        elif key in df["variable"].unique():

            pred_only = df[df["type"] == "pred"]

            mask = (
                (pred_only["variable"] == key)
                & (pred_only["value"].isin(values))
            )

            valid_sims = (
                pred_only.loc[mask, keys]
                .drop_duplicates()
            )

        # ❗ CASE 3: key not found → explicit failure (important!)
        else:
            raise KeyError(f"Filter key '{key}' not found in columns or variables")

        # ✅ Filter at simulation level (correct)
        df = df.merge(valid_sims, on=keys, how="inner")

        if df.empty:
            print(f"⚠️ Filter removed all data for {key}={values}")
            return df

    return df



def get_stage_timeseries(tidy, config, variable, filters=None):

    df = apply_filters(tidy, filters)

    keys = ["branch", "file", "SimulationID", "Clock.Today"]

    # -------------------------------
    # Extract predicted variable
    # -------------------------------
    pred = df[
        (df["type"] == "pred") &
        (df["variable"] == variable)
    ][keys + ["value"]].rename(columns={"value": "pred"})

    # -------------------------------
    # Extract stage (pred only)
    # -------------------------------
    stage = df[
        (df["type"] == "pred") &
        (df["variable"] == config["stage_var"])
    ][keys + ["value"]].rename(columns={"value": "stage"})

    # -------------------------------
    # Extract observations
    # -------------------------------
    obs = df[
        (df["type"] == "obs") &
        (df["variable"] == variable)
    ][keys + ["value"]].rename(columns={"value": "obs"})

    # -------------------------------
    # Merge pred + stage
    # -------------------------------
    dfm = pred.merge(stage, on=keys, how="inner")

    # -------------------------------
    # Attach obs (LEFT — sparse)
    # -------------------------------
    dfm = dfm.merge(obs, on=keys, how="left")

    # -------------------------------
    # ✅ FIX DUPLICATES (CRITICAL)
    # -------------------------------
    dfm = (
        dfm
        .groupby(keys, as_index=False)
        .agg({
            "pred": "max",
            "stage": "first",
            "obs": "first"
        })
    )

    # -------------------------------
    # sort cleanly
    # -------------------------------
    dfm = dfm.sort_values(["SimulationID", "stage"])

    return dfm


# %% [markdown]
# # Graphing

# %%
# %%writefile C:\GitHubRepos\APSIMTools\GraphLib\apsim_tools\graphing.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import math

from apsim_tools.data import (
    get_harvest_aligned,
    get_daily_obs_pred_exact,
    get_stage_timeseries
)

from apsim_tools.style import (
    Colors,
    Markers,
    Lines
)


#__________________________________________________________
#  Function plot observed vs predicted by branch
#----------------------------------------------------------

def plot_obs_pred_by_branch(
    tidy,
    config,
    variable,
    mode="harvest",
    filters=None,
    color_by=None,
    marker_by=None,
    size_by=None,
    show_ellipses=False,
    additional_index_maps=None
):
    
    # -------------------------------
    # ALIGN DATA
    # -------------------------------
    if mode == "harvest":
        pivot = get_harvest_aligned(tidy, config, variable, filters)

    elif mode == "daily":
        pivot = get_daily_obs_pred_exact(tidy, variable, filters)

    else:
        raise ValueError("mode must be 'harvest' or 'daily'")

    # -------------------------------
    # ATTACH METADATA
    # -------------------------------
    meta_cols = [
        "branch",
        "file",
        "SimulationID",
        "Experiment",
        config["cultivar_col"],
        ]


    # ✅ always include grouping dimensions automatically
    for col in [color_by, marker_by, size_by]:
        if col and col in tidy.columns:
            meta_cols.append(col)
    
    meta_cols = list(set(meta_cols))  # remove duplicates
    meta_cols = [c for c in meta_cols if c in tidy.columns]

    
    meta = tidy[meta_cols].drop_duplicates()
    
    pivot = pivot.merge(
        meta,
        on=["branch", "file", "SimulationID"],
        how="left"
    )

    # -------------------------------
    # BUILD GROUPINGS
    # -------------------------------
    branches = sorted(pivot["branch"].unique())

    # Color
    color_map = {}
    if color_by:
        color_vals = sorted(pivot[color_by].dropna().unique())
        colors = list(Colors.values())
        color_map = {v: colors[i % len(colors)] for i, v in enumerate(color_vals)}

    # Markers and fills
    if marker_by:
        marker_styles = list(Markers.values())
        n_markers = len(marker_styles)

        marker_vals = sorted(pivot[marker_by].dropna().unique())

        marker_map = {}
        fill_map = {}

        for i, v in enumerate(marker_vals):
            marker_map[v] = marker_styles[i % n_markers]

            # ✅ cycle fill AFTER cycling markers
            fill_map[v] = (i // n_markers) % 2 == 0


    # Size
    size_map = {}
    if size_by:
        size_vals = sorted(pivot[size_by].dropna().unique())
        sizes = np.linspace(30, 120, len(size_vals))
        size_map = {v: sizes[i] for i, v in enumerate(size_vals)}

    # -------------------------------
    # CREATE FIGURE (NO constrained_layout)
    # -------------------------------
    # max_panel_size = 25

    n = len(branches)
    
    fig_height = 10
    
    fig_width = fig_height / n

    fig, axes = plt.subplots(
        1, n,
        figsize=(fig_height, fig_width),
        sharex=True, sharey=True,
        constrained_layout=True
    )

    if len(branches) == 1:
        axes = [axes]

    # -------------------------------
    # SCATTER PLOT
    # -------------------------------
    max_val = np.nanmax([pivot["obs"], pivot["pred"]])
    lims = [0, max_val]

    group_cols = [c for c in [color_by, marker_by, size_by] if c]

    for ax, branch in zip(axes, branches):

        g = pivot[pivot["branch"] == branch]

        grouped = g.groupby(group_cols) if group_cols else [(None, g)]
        
        # -------------------------------
        # ADD Stats
        # -------------------------------
        
        stats = compute_stats(g)
                
        if stats:
            stats_text = (
                f"NSE = {stats['NSE']:.2f}\n"
                f"Bias = {stats['Bias']:.2f}"
            )
        
        ax.text(
            0.98, 0.02,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )
        
        # -------------------------------
        # ADD points
        # -------------------------------
        for key, sub in grouped:

            color_val = sub[color_by].iloc[0] if color_by else None
            marker_val = sub[marker_by].iloc[0] if marker_by else None
            size_val = sub[size_by].iloc[0] if size_by else None
            
            facecolor = color_map.get(color_val, "grey")
            edgecolor = color_map.get(color_val, "grey")

            if marker_by:
                filled = fill_map.get(marker_val, True)
                if not filled:
                    facecolor = "none"

            ax.scatter(
                sub["obs"],
                sub["pred"],
                facecolors=facecolor,
                edgecolors=edgecolor,
                marker=marker_map.get(marker_val, "o"),
                s=size_map.get(size_val, 50),
                alpha=0.7
            )

        # -------------------------------
        # ADD ELLIPSES (grouped by color_by)
        # -------------------------------
        if color_by:
            if show_ellipses:
                grouped_color = g.groupby(color_by)

                for val, sub in grouped_color:

                    x = sub["obs"]
                    y = sub["pred"]

                    color = color_map.get(val, "grey")

                    add_confidence_ellipse(ax, x, y, color)

        ax.text(
            0.02, 0.98, branch,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold"
        )
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel(f"Observed {variable}")
        ax.set_ylabel(f"Predicted {variable}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
        ax.margins(0.05)
    # -------------------------------
    # BUILD LEGENDS (NO POSITION YET)
    # -------------------------------
    legend_objects = []
    y_offset = 0
    # Color legend
    if color_by and color_map:
        handles = [
            plt.Line2D([0], [0],
                marker='o',
                color='w',
                markerfacecolor=color_map[val],
                label=str(val),
                markersize=8
            )
            for val in sorted(color_map)
        ]
        
        ncols = min(len(handles), 6)
        nrows = math.ceil(len(handles) / ncols)
        y_offset += nrows * 0.035 * n

        legend_objects.append(
            fig.legend(handles=handles, title=color_by, loc="upper center",
                       bbox_to_anchor=(0.5, 1.0 + y_offset),
                       ncol=ncols, frameon=False)
        )

    # Marker legend
    if marker_by and marker_map:
        handles = [
            plt.Line2D(
                [0], [0],
                marker=marker_map[val],
                linestyle='None',
                markerfacecolor=(
                    color_map.get(val, 'black') if fill_map[val] else 'none'
                ),
                markeredgecolor=color_map.get(val, 'black'),
                label=str(val),
                markersize=8
            )
            for val in sorted(marker_map)
        ]

        ncols = min(len(handles), 6)
        nrows = math.ceil(len(handles) / ncols)
        y_offset += nrows * 0.035 * n + .035

        legend_objects.append(
            fig.legend(handles=handles, title=marker_by, loc="upper center",
                       bbox_to_anchor=(0.5, 1 + y_offset),
                       ncol=ncols, frameon=False)
        )

    # -------------------------------
    # ✅ DYNAMIC LAYOUT (KEY PART)
    # -------------------------------
    # if legend_objects:

    #     # compute sizes
    #     fig.canvas.draw()
    #     renderer = fig.canvas.get_renderer()
    #     inv = fig.transFigure.inverted()

    #     heights = []
    #     for leg in legend_objects:
    #         bbox = leg.get_window_extent(renderer).transformed(inv)
    #         heights.append(bbox.height)

    #     pad = 0.01
    #     total_height = sum(heights) + pad * len(heights)

    #     # ✅ reserve space ABOVE panels
    #     top_margin = max(0.2, 1 - total_height)

    #     #plt.subplots_adjust(top=top_margin)

    #     # ✅ stack legends in reserved space
    #     y = 1.0
    #     for leg, h in zip(legend_objects, heights):
    #         leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
    #         y -= (h + pad)

    return fig

#__________________________________________________________
# Add variance ellipses
#----------------------------------------------------------

# The ellipse dimensions are based on:

# ✅ multivariate normal covariance structure
# ✅ scaled by standard deviations along principal axes

# ✅ n_std = 1

# ~68% of data (if Gaussian)

# ✅ n_std = 2

# ~95% of data
# 👉 your current default → typical “confidence ellipse”

# ✅ n_std = 3

# ~99.7% of data

def add_confidence_ellipse(ax, x, y, color, n_std=2.0, alpha=0.2):

    if len(x) < 2:
        return

    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # eigen decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # angle
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # full ellipse dimensions
    width  = 2 * n_std * np.sqrt(vals[0])
    height = 2 * n_std * np.sqrt(vals[1])

    ellipse = Ellipse(
        (mean_x, mean_y),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=2
    )
    ax.add_patch(ellipse)

    # -------------------------------
    # ✅ AXIS LINES (TRIMMED PROPERLY)
    # -------------------------------

    # semi-axis lengths (correct)
    major_len = n_std * np.sqrt(vals[0])
    minor_len = n_std * np.sqrt(vals[1])

    major_vec = vecs[:, 0]
    minor_vec = vecs[:, 1]

    # major axis (exactly to ellipse edge)
    ax.plot(
        [mean_x - major_vec[0] * major_len,
         mean_x + major_vec[0] * major_len],
        [mean_y - major_vec[1] * major_len,
         mean_y + major_vec[1] * major_len],
        linestyle="-",
        linewidth=2,
        color=color,
        alpha=alpha   # ✅ same as ellipse
    )

    # minor axis
    ax.plot(
        [mean_x - minor_vec[0] * minor_len,
         mean_x + minor_vec[0] * minor_len],
        [mean_y - minor_vec[1] * minor_len,
         mean_y + minor_vec[1] * minor_len],
        linestyle="-",
        linewidth=2,
        color=color,
        alpha=alpha   # ✅ same as ellipse
    )
    
#__________________________________________________________
# calculate stats
#----------------------------------------------------------

def compute_stats(df):

    obs = df["obs"].values
    pred = df["pred"].values

    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[mask]
    pred = pred[mask]

    if len(obs) < 2:
        return {}

    residuals = pred - obs

    # metrics
    bias = np.mean(residuals)
    rmse = np.sqrt(np.mean(residuals**2))

    # NSE
    denom = np.sum((obs - np.mean(obs))**2)
    nse = 1 - np.sum((obs - pred)**2) / denom if denom > 0 else np.nan

    # R²
    r = np.corrcoef(obs, pred)[0, 1]
    r2 = r**2

    return {
        "NSE": nse,
        "Bias": bias,
        "RMSE": rmse,
        "R2": r2
    }

#__________________________________________________________
# Time series graph Function
#----------------------------------------------------------
def plot_stage_timeseries(
    tidy,
    variable,
    config,
    filters=None,
    color_by=None,
    marker_by=None,
    linestyle_by=None,
    panels_by=None,
    max_cols=3,
    panel_scale=1.0   # ✅ NEW
):


    # -------------------------------
    # LINK linestyle to marker
    # -------------------------------
    if marker_by and linestyle_by is None:
        linestyle_by = marker_by

    df = get_stage_timeseries(tidy, config, variable, filters)

    # -------------------------------
    # ATTACH METADATA
    # -------------------------------
    meta_cols = ["branch", "file", "SimulationID"]
    for col in [color_by, marker_by, linestyle_by, panels_by]:
        if col and col not in meta_cols:
            meta_cols.append(col)

    meta = tidy[meta_cols].drop_duplicates()
    df = df.merge(meta, on=["branch", "file", "SimulationID"], how="left")

    # -------------------------------
    # RESIDUALS
    # -------------------------------
    df["residual"] = df["pred"] - df["obs"]
    
    # -------------------------------
    # GLOBAL Y LIMITS
    # -------------------------------

    # absolute (pred/obs)
    ymin_main = np.nanmin([df["pred"], df["obs"]])
    ymax_main = np.nanmax([df["pred"], df["obs"]])

    # residual
    ymin_res = np.nanmin(df["residual"])
    ymax_res = np.nanmax(df["residual"])
    
    pad_main = 0.05 * (ymax_main - ymin_main)
    pad_res  = 0.05 * (ymax_res  - ymin_res)

    ymin_main -= pad_main
    ymax_main += pad_main

    ymin_res -= pad_res
    ymax_res += pad_res
    
    max_abs = max(abs(ymin_res), abs(ymax_res))
    
    # -------------------------------
    # STYLE MAPS
    # -------------------------------
    def build_map(values, base_dict):
        values = sorted(values)
        base_vals = list(base_dict.values())
        return {v: base_vals[i % len(base_vals)] for i, v in enumerate(values)}

    color_map, marker_map, line_map = {}, {}, {}

    if color_by:
        color_map = build_map(df[color_by].dropna().unique(), Colors)

    if marker_by:
        marker_map = build_map(df[marker_by].dropna().unique(), Markers)

    if linestyle_by:
        line_map = build_map(df[linestyle_by].dropna().unique(), Lines)

    # -------------------------------
    # PANEL SETUP
    # -------------------------------
    panel_vals = sorted(df[panels_by].dropna().unique()) if panels_by else [None]

    n_panels = len(panel_vals)
    ncols = min(max_cols, n_panels)
    n_panel_rows = math.ceil(n_panels / max_cols)
    nrows = n_panel_rows * 2

    panel_size = 2.8 * panel_scale

    height_ratios = []
    for _ in range(n_panel_rows):
        height_ratios.extend([1.0, 0.6])   # ✅ main + residual

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        sharex=True,
        gridspec_kw={
            'height_ratios': height_ratios,
            'hspace': 0.10,   # ✅ GOOD balance
            'wspace': 0.15
        }
    )

    axes = axes.reshape(nrows, ncols)

    # -------------------------------
    # PLOTTING
    # -------------------------------
    for i, panel_val in enumerate(panel_vals):

        row = (i // ncols) * 2
        col = i % ncols

        ax_main = axes[row, col]
        ax_res  = axes[row + 1, col]

        gdf = df[df[panels_by] == panel_val] if panels_by else df

        for _, sub in gdf.groupby(["branch", "file", "SimulationID"]):

            sub = sub.sort_values("stage")

            color_val = sub[color_by].iloc[0] if color_by else None
            marker_val = sub[marker_by].iloc[0] if marker_by else None
            linestyle_val = sub[linestyle_by].iloc[0] if linestyle_by else None

            color = color_map.get(color_val, "grey")
            marker = marker_map.get(marker_val, "o")
            linestyle = line_map.get(linestyle_val, "-")

            mask = sub["obs"].notna()

            # --- main ---
            ax_main.plot(
                sub["stage"], sub["pred"],
                color=color, linestyle=linestyle,
                linewidth=1, alpha=0.35
            )

            ax_main.scatter(
                sub.loc[mask, "stage"],
                sub.loc[mask, "obs"],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=40
            )
            
            plt.xlim(0,11)

            # --- residual ---
            ax_res.scatter(
                sub.loc[mask, "stage"],
                sub.loc[mask, "residual"],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=40
            )
            
            plt.xlim(0,11)

        # -----------------------
        # TITLES
        # -----------------------
        if panel_val is not None:
            ax_main.text(
                0.02, 0.95, str(panel_val),
                transform=ax_main.transAxes,
                ha="left", va="top",
                fontsize=11, fontweight="bold"
            )

        # -----------------------
        # AXIS CLEANUP ✅
        # -----------------------
        if col == 0:
            ax_main.set_ylabel(variable)
            ax_res.set_ylabel("Residual")
        else:
            ax_main.set_ylabel("")
            ax_res.set_ylabel("")
            ax_main.set_yticklabels([])
            ax_res.set_yticklabels([])

        # only bottom row gets x labels
        if row >= nrows - 2:
            ax_res.set_xlabel("Stage")
        else:
            ax_res.set_xlabel("")

        ax_main.grid(alpha=0.3)

        ax_res.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_res.grid(alpha=0.3)
        
        ax_main.set_ylim(ymin_main, ymax_main)
        ax_res.set_ylim(-max_abs, max_abs)

    # -------------------------------
    # REMOVE EMPTY AXES
    # -------------------------------
    total_slots = ncols * n_panel_rows
    for j in range(n_panels, total_slots):
        r = (j // ncols) * 2
        c = j % ncols
        fig.delaxes(axes[r, c])
        fig.delaxes(axes[r + 1, c])

    # ============================================================
    # ✅ LEGEND (your working version)
    # ============================================================
    legend_objects = []

    if color_by and color_map:
        handles = [
            plt.Line2D([0], [0],
                marker='o',
                color='w',
                markerfacecolor=color_map[val],
                label=str(val),
                markersize=8
            )
            for val in sorted(color_map)
        ]
        legend_objects.append(
            fig.legend(handles=handles, title=color_by,
                       loc="upper center",
                       ncol=min(len(handles), 6), frameon=False)
        )

    if marker_by and marker_map:
        handles = [
            plt.Line2D(
                [0], [0],
                marker=marker_map[val],
                linestyle=line_map.get(val, "-"),
                markerfacecolor="none",
                markeredgecolor="black",
                color="black",
                label=str(val),
                markersize=8
            )
            for val in sorted(marker_map)
        ]
        legend_objects.append(
            fig.legend(handles=handles, title=marker_by,
                       loc="upper center",
                       ncol=min(len(handles), 6), frameon=False)
        )

    # ✅ dynamic layout
    if legend_objects:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()

        heights = [
            leg.get_window_extent(renderer).transformed(inv).height
            for leg in legend_objects
        ]

        pad = 0.01
        total_height = sum(heights) + pad * len(heights)

        top_margin = max(0.2, 1 - total_height)
        plt.subplots_adjust(top=top_margin)

        y = 1.0
        for leg, h in zip(legend_objects, heights):
            leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
            y -= (h + pad)

    return fig

# %% [markdown]
# # Test Example - run simulations and process data

# %% [markdown]
# ## read in libraries

# %%
import sys
sys.path.append(r"C:\GitHubRepos\APSIMTools\GraphLib")

# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# graph models
from apsim_tools.graphing import (
    plot_obs_pred_by_branch,
    plot_stage_timeseries
)

from apsim_tools.style import ( 
    Colors, 
    Markers,
    Lines
)

from apsim_tools.runner import (
    load_all,
    validate_run_branches,
    to_tidy
)

# %% [markdown]
# ## Branch and file Settings

# %%
# ======================
# CONFIG
# ======================
BRANCHES = {
    "master": "UoM_Wheat",
    "working": "WheatNeil",
    "working V2": "WheatHamish"
}

# ======================
# RUN CONTROL - Specify which branches to (re)run
# ======================

# Options:
RUN_BRANCHES = []                    # run nothing (use existing DBs)
#RUN_BRANCHES = list(BRANCHES.keys())   # run all branches
#RUN_BRANCHES = ["master"]
#RUN_BRANCHES = ["working"]
#RUN_BRANCHES = ["working V2"]

SIM_FILES = [
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\GxExM\GxExM.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Dookie2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Dookie2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\WaggaWagga2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\WaggaWagga2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Gnarwarre2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Gnarwarre2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\GrassPatch2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\GrassPatch2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Fords2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Turretfield2024.apsimx')
]

CONFIG = {
    "branches": BRANCHES,
    "run_branches": RUN_BRANCHES,
    "sim_files": SIM_FILES,
    "repo_path": Path(r"C:\GitHubRepos\ApsimX"),
    "apsim_exe": r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe",
    "apsim_solution": r"C:\GitHubRepos\ApsimX\ApsimX.sln",
    "stage_var": "Wheat.Phenology.Stage",
    "stage_name_var": "Wheat.Phenology.CurrentStageName",
    "harvest_stage": "HarvestRipe",
    "cultivar_col": "Wheat.SowingData.Cultivar"
}

validate_run_branches(config=CONFIG)

# %% [markdown]
# ## Additionl index mapping

# %%
# Specifiy map for cultivars to winter or spring type
Development_type={
'29B':'Spring',
'5A':'Spring',
'60A':'Spring',
'BigRed':'Winter',
'Corack':'Spring',
'Espada':'Spring',
'Gauntlet':'Spring',
'Gregory':'Spring',
'Hartog':'Spring',
'Illabo':'Winter',
'Janz':'Spring',
'Kittyhawk':'Winter',
'Mace':'Spring',
'Meering':'Spring',
'Mowhawk':'Winter',
'Osprey':'Winter',
'Rosella':'Winter',
'Scepter':'Spring',
'Scout':'Spring',
'Spitfire':'Spring',
'Stockade':'Spring',
'Sunbee':'Spring',
'Sunmaster':'Spring',
'Sunstate':'Spring',
'UOM001_3_47':'Winter',
'UOM001_9_1':'Winter',
'Wedgetail':'Winter',
'Whistler':'Winter',
'Wyalkatchem':'Spring',
'Wylah':'Winter',
'Yitpi':'Spring',
'Zanzibar':'Spring',
}

# Specify map for each experiment to project grouping
Project_group = {
    'Minnipa2014':'GxExM',
    'Minnipa2015':'GxExM',
    'Gatton2014Irrigated':'GxExM',
    'Gatton2014':'GxExM',
    'Gatton2015':'GxExM',
    'Junee2014':'GxExM',
    'Temora2015':'GxExM',
    'DookieWWHI2024':'WWHI',
    'DookieWWHI2025':'WWHI',
    'WaggaWagga2024':'WWHI',
    'WaggaWagga2025':'WWHI',
    'GrassPatch2024':'WWHI',
    'GrassPatch2025':'WWHI',
    'Turretfield2024':'WWHI',
    'Fords2025':'WWHI',
    'DookieEVA2024':'EVA',
    'DookieEVA2025':'EVA',
    'Gnarwarre2024':'EVA',
    'Gnarwarre2025':'EVA'
}

# Pack maps together ready to be inserted as indexes 
additional_index_maps = {
    "DevelopmentType": {
        "source": "Wheat.SowingData.Cultivar",
        "map": Development_type
    },
    "ProjectGroup": {
        "source": "Experiment",
        "map": Project_group
    }
}


# %% [markdown]
# ## Write file for command line tool to apply to each .apsimx file to be run.
# If you want to make standard modifications to all files run it can be done here

# %%
# this function writes and apply file that the CLI
def write_apply_file(sim_file):
    """
    Create an APSIM CLI apply file which:
    - removes specified reports
    - injects AnalysisReport
    - sets variables
    - saves and runs simulation
    """
    report_library = r"C:\GitHubRepos\APSIMTools\Report_lib.apsimx"
    apply_file = sim_file.with_name(f"_apply_{sim_file.stem}.txt")

    lines = []

    # ---------------------------------------------
    # Add AnalysisReport to all Simulation nodes
    # ---------------------------------------------
    lines.append(f"add [AnalysisReport] from {report_library} to all [Zone]")
    
    # ---------------------------------------------
    # Inject Spectral model into each simulation
    # ---------------------------------------------
    lines.append("add [Spectral] to all [Zone]")
    
    # ---------------------------------------------
    # Hartogify cultivars
    # ---------------------------------------------
    lines.append("add new SetModelParamsBySimulation to [Zone] name ConstantBaseCv")
    lines.append("[ConstantBaseCv].SetEventName = [Plant].PlantSowing")
    lines.append(f"[ConstantBaseCv].ParameterFile = Inputs/{sim_file.stem}_ConstantPhenology.csv")

    # ---------------------------------------------
    # Save + run
    # ---------------------------------------------
    lines.append(f"save {sim_file}")
    lines.append(f"run {sim_file}")

    # Write file
    apply_file.write_text("\n".join(lines))

    return apply_file


# %% [markdown]
# ## Run simulations and read in raw .db data and process into tidy format

# %%
# ======================
# EXECUTE PIPELINE
# ======================

# Read in raw data
raw = load_all(config=CONFIG, apply_fn=write_apply_file)

# ✅ Ensure SimulationName exists (from AnalysisReport)
if "Simulation.Name" in raw.columns:
    raw = raw.rename(columns={"Simulation.Name": "SimulationName"})

# ✅ Convert to tidy format
tidy = to_tidy(raw, config=CONFIG, additional_index_maps=additional_index_maps)

# %% [markdown]
# # Test Example - created graphs

# %% [markdown]
# ## Harvest graphs

# %% [markdown]
# ### Yield

# %%
graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Grain.Wt",
    mode='harvest',
    filters = {"branch":['master']},
    color_by = "Experiment",
    marker_by = "DevelopmentType",
    size_by=None
)
graph.savefig("Yield WWHI.jpg")

# %%
graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Grain.Wt",
    mode='harvest',
    filters = {"ProjectGroup":['WWHI']},
    color_by = "Experiment",
    marker_by = "DevelopmentType",
    size_by=None
)
graph.savefig("Yield GxExM.jpg")

# %%
plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Grain.Wt",
    mode='harvest',
    filters=None, #filters = {'Wheat.SowingData.Cultivar':['Meering','Mowhawk']}
    color_by = "DevelopmentType",
    marker_by = "Wheat.SowingData.Cultivar",
    size_by=None,
    show_ellipses=True    
)
plt.show()

# %% [markdown]
# ### Biomass

# %%
plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.AboveGround.Wt",
    mode='harvest',
    filters=None, #filters = {'Wheat.SowingData.Cultivar':['Meering','Mowhawk']}
    color_by = "Experiment",
    marker_by = "DevelopmentType",
    size_by=None
)
plt.show()

# %%
plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.AboveGround.Wt",
    mode='harvest',
    filters=None, #filters = {'Wheat.SowingData.Cultivar':['Meering','Mowhawk']}
    color_by = "DevelopmentType",
    marker_by = "Wheat.SowingData.Cultivar",
    size_by=None,
    show_ellipses=True
)
plt.show()

# %% [markdown]
# ## Growth cycle analysis

# %% [markdown]
# ### Haun Stage

# %% [markdown]
# #### obs vs pred daily

# %%
plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Phenology.HaunStage",
    mode="daily",
    filters=None, #filters={"Experiment": ["DookieWWHI2025"]},
    color_by = "Experiment",
    marker_by = "DevelopmentType",
    size_by = None
)
plt.show()

# %%
plot_obs_pred_by_branch(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Phenology.HaunStage",
    mode="daily",
    filters=None, #filters={"Experiment": ["DookieWWHI2025"]},
    color_by = "DevelopmentType",
    marker_by = "Wheat.SowingData.Cultivar",
    size_by = None,
    show_ellipses=True
)
plt.show()

# %% [markdown]
# #### All expts

# %%
plot_stage_timeseries(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Phenology.HaunStage",
    filters=None, #filters={"Experiment": ["DookieWWHI2025"]},
    color_by="Experiment",
    marker_by="DevelopmentType",
    panels_by="branch",
    max_cols=3,
    panel_scale=2,
)

plt.show()

# %% [markdown]
# #### By expt

# %%
plot_stage_timeseries(
    tidy = tidy,
    config = CONFIG,
    variable = "Wheat.Phenology.HaunStage",
    color_by="DevelopmentType",
    marker_by=None,
    panels_by="Experiment",
    filters={"branch": ["working"]},
    max_cols=4,
    panel_scale=0.8,
)
plt.show()
