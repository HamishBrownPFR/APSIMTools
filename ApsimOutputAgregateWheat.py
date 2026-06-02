# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Constants

# %%
from pathlib import Path
import pandas as pd
import sqlite3
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import shutil

import warnings

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

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

Lines = {1: '-',
 2: '--',
 3: '-,',
 4: ':',
 5: '-',
 6: '--',
 7: '-,',
 8: ':',
 9: '-',
 10: '--',
 11: '-,',
 12: ':',
 13: '-',
 14: '--',
 15: '-,',
 16: ':'}


# %% [markdown]
# # Settings

# %%
# ======================
# CONFIG
# ======================

CROP = 'Wheat'

BRANCHES = {
    "master": "DookieWheat2024",
    "working": "WheatMergeBranches"
}

REPO_PATH = Path(r"C:\GitHubRepos\ApsimX")

SIM_FILES = [
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\GxExM\GxExM.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Dookie2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Dookie2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\WaggaWagga2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\WaggaWagga2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Gnarwarre2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\Gnarwarre2025.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\GrassPatch2024.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\UoM_WinterVsSpring\GrassPatch2025.apsimx')
]

APSIM_EXE = r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe"

APSIM_SOLUTION = r"C:\GitHubRepos\ApsimX\ApsimX.sln"

REPORT_LIBRARY = r"C:\GitHubRepos\APSIMTools\Report_lib.apsimx"


# %% [markdown]
# # Helpers for setting up and running .apsim files

# %%
def validate_run_branches():
    invalid = [b for b in RUN_BRANCHES if b not in BRANCHES]
    if invalid:
        raise ValueError(f"Invalid RUN_BRANCHES: {invalid}")

# ======================
# APSIM / DB UTILITIES
# ======================

def checkout_branch(branch):
    subprocess.run(["git", "checkout", branch],
                   cwd=REPO_PATH,
                   check=True)

def build_apsim():
    subprocess.run(
        ["dotnet", "build", APSIM_SOLUTION, "-c", "Release"],
        cwd=REPO_PATH,
        check=True
    )
    return 
    
def run_apsim(sim_file):
    """
    High-level runner:
    - writes apply file
    - runs APSIM
    """

    apply_file = write_apply_file(sim_file)
    return run_apsim_with_apply(sim_file, apply_file)

def run_apsim_with_apply(sim_file, apply_file):
    """
    Executes APSIM using a pre-generated apply file.
    """

    print(f"▶ Running APSIM: {sim_file.name}")

    result = subprocess.run(
        [
            APSIM_EXE,
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

def reset_repo():
    print("🔄 Resetting tracked files only")

    subprocess.run(
        ["git", "reset", "--hard"],
        cwd=REPO_PATH,
        check=True
    )

def read_table(db_file, table):
    # if not db_file.exists():
    #     return None
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)

  
    df.columns = [
        c.strip().replace('"', '').replace("'", '')
        for c in df.columns
    ]

    return df

def should_run(branch_name):
    return branch_name in RUN_BRANCHES

def write_apply_file(sim_file):
    """
    Create an APSIM CLI apply file which:
    - removes specified reports
    - injects AnalysisReport
    - sets variables
    - saves and runs simulation
    """

    apply_file = sim_file.with_name(f"_apply_{sim_file.stem}.txt")

    lines = []

    # ---------------------------------------------
    # Add AnalysisReport to all Simulation nodes
    # ---------------------------------------------
    lines.append(f"add [AnalysisReport] from {REPORT_LIBRARY} to all [Zone]")
    
    # ---------------------------------------------
    # Inject Spectral model into each simulation
    # ---------------------------------------------
    lines.append("add [Spectral] to all [Zone]")

    # ---------------------------------------------
    # Save + run
    # ---------------------------------------------
    lines.append(f"save {sim_file}")
    lines.append(f"run {sim_file}")

    # Write file
    apply_file.write_text("\n".join(lines))

    return apply_file



# %% [markdown]
# # load_branch_data

# %%
def load_branch_data(branch_name, git_branch):
    if should_run(branch_name):
        reset_repo()
        checkout_branch(git_branch)
        build_apsim()

    frames = []

    for sim in SIM_FILES:
        db = branch_db_path(sim, branch_name)

        # --- Run APSIM ---
        if should_run(branch_name):
            print("")
            print(f"▶ Running APSIM [{branch_name}]: {sim.name}")
            run_apsim(sim)

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
            # ✅ CHECK SimulationID alignment (per file)
            # ---------------------------------------------
            if pred is not None and obs is not None:

                pred_ids = set(pred["SimulationID"].drop_duplicates())
                obs_ids  = set(obs["SimulationID"].drop_duplicates())

                # simulations in pred but not in obs
                missing_ids = pred_ids - obs_ids

                if missing_ids:
                    print(f"\n⚠️ Missing observed simulations in {sim.name}:")
                    print(f"Count: {len(missing_ids)}")

                    # attempt to print identifying info from pred
                    cols_to_show = [
                        c for c in [
                            "SimulationID",
                            "Simulation.Name",   # if present
                            "Experiment",
                            "TOS",
                            "Variety",
                            "WaterTrt"
                        ] if c in pred.columns
                    ]

                    missing_rows = (
                        pred[pred["SimulationID"].isin(missing_ids)]
                        [cols_to_show]
                        .drop_duplicates()
                        .sort_values("SimulationID")
                    )

                    print(missing_rows.head(20))  # limit output     
                    
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
            df = pd.concat(
                [x for x in [pred, obs] if x is not None],
                ignore_index=True
            )

            frames.append(df)

    if not frames:
        raise RuntimeError(f"No data loaded for branch {branch_name}")

    return pd.concat(frames, ignore_index=True)

def load_all():
    return pd.concat([
        load_branch_data(name, git_branch)
        for name, git_branch in BRANCHES.items()
    ], ignore_index=True)


# %% [markdown]
# # Run selected branches and simulations

# %%
# ======================
# RUN CONTROL
# ======================

# Options:
RUN_BRANCHES = []                    # run nothing (use existing DBs)
# RUN_BRANCHES = list(BRANCHES.keys())   # run all branches
# RUN_BRANCHES = ["master"]
# RUN_BRANCHES = ["working"]

validate_run_branches()


# ======================
# EXECUTE PIPELINE
# ======================

raw = load_all()


# %% [markdown]
# # Enforce indicies on observed data

# %%
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


# %% [markdown]
# # to_tidy

# %%
def to_tidy(df):

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
        f"{CROP}.Phenology.CurrentStageName",
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
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

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
    indices_to_fill = ['Experiment','SimulationName',f'{CROP}.SowingData.Cultivar']
    df = enforce_indices_to_observed(df, indices_to_fill)
    
    # ---------------------------------------------
    # ✅ Melt
    # ---------------------------------------------
    tidy = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="variable",
        value_name="value"
    )

    return tidy.dropna(subset=["value"])


# %%
# Tidy raw data

# %%
# ✅ Ensure SimulationName exists (from AnalysisReport)
if "Simulation.Name" in raw.columns:
    raw = raw.rename(columns={"Simulation.Name": "SimulationName"})

# ✅ Convert to tidy format
tidy = to_tidy(raw)


# %% [markdown]
# # Functions to get data for graphing

# %%
def get_harvest_aligned(tidy, variable, filters=None):
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
    if f"{CROP}.Phenology.CurrentStageName" not in df.columns:
        raise KeyError(f"Missing {CROP}.Phenology.CurrentStageName in tidy data")

    df = df[df[f"{CROP}.Phenology.CurrentStageName"] == "HarvestRipe"]

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


# %%
def get_daily_aligned(tidy, variable, filters=None):
    """
    Align data at daily resolution.

    - applies filters
    - aligns obs/pred on SimulationID + date
    """

    df = apply_filters(tidy, filters)

    df = df[df["variable"] == variable].copy()

    obs = df[df["type"] == "obs"]
    pred = df[df["type"] == "pred"]

    aligned = pred.merge(
        obs,
        on=["file", "SimulationID", "Clock.Today"],
        how="inner",
        suffixes=("_pred", "_obs")
    )

    aligned["branch"] = aligned["branch_pred"]
    aligned["pred"] = aligned["value_pred"]
    aligned["obs"] = aligned["value_obs"]

    return aligned


# %%
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


# %% [markdown]
# # Function plot observed vs predicted by branch

# %%
def plot_obs_pred_by_branch(
    tidy,
    variable,
    mode="harvest",
    filters=None,
    color_by=None,
    marker_by=None,
    size_by=None,
):
    
    # -------------------------------
    # ALIGN DATA
    # -------------------------------
    if mode == "harvest":
        pivot = get_harvest_aligned(tidy, variable, filters)

    elif mode == "daily":
        pivot = get_daily_aligned(tidy, variable, filters)

    else:
        raise ValueError("mode must be 'harvest' or 'daily'")

    # -------------------------------
    # ATTACH METADATA
    # -------------------------------
    meta = tidy[[
        "branch", "file", "SimulationID",
        "Experiment", f"{CROP}.SowingData.Cultivar"
    ]].drop_duplicates()

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
    panel_size = 5
    n = len(branches)

    fig_width = panel_size * n
    fig_height = panel_size * 1.2

    fig, axes = plt.subplots(
        1, n,
        figsize=(fig_width, fig_height),
        sharex=True, sharey=True
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

        legend_objects.append(
            fig.legend(handles=handles, title=color_by, loc="upper center",
                       ncol=min(len(handles), 6), frameon=False)
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

        legend_objects.append(
            fig.legend(handles=handles, title=marker_by, loc="upper center",
                       ncol=min(len(handles), 6), frameon=False)
        )

    # -------------------------------
    # ✅ DYNAMIC LAYOUT (KEY PART)
    # -------------------------------
    if legend_objects:

        # compute sizes
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()

        heights = []
        for leg in legend_objects:
            bbox = leg.get_window_extent(renderer).transformed(inv)
            heights.append(bbox.height)

        pad = 0.01
        total_height = sum(heights) + pad * len(heights)

        # ✅ reserve space ABOVE panels
        top_margin = max(0.2, 1 - total_height)

        plt.subplots_adjust(top=top_margin)

        # ✅ stack legends in reserved space
        y = 1.0
        for leg, h in zip(legend_objects, heights):
            leg.set_bbox_to_anchor((0.5, y), transform=fig.transFigure)
            y -= (h + pad)

    return fig

# %% [markdown]
# # Yield Graph

# %%
plot_obs_pred_by_branch(
    tidy,
    "Wheat.Grain.Wt",
    color_by = "Experiment",
    marker_by = "Wheat.SowingData.Cultivar"
)
plt.show()

# %%
plot_obs_pred_by_branch(
    tidy,
    "Wheat.Grain.Wt",
    color_by = "Wheat.SowingData.Cultivar",
    marker_by = "Experiment",
    filters = {'Wheat.SowingData.Cultivar':['Meering','Mowhawk']}
)
plt.show()

# %%
plot_obs_pred_by_branch(
    tidy,
    "Wheat.AboveGround.Wt",
    color_by = "Experiment",
    marker_by = "Wheat.SowingData.Cultivar",
    mode="daily"
)
plt.show()
