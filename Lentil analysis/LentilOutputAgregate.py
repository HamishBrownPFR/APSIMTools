# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Set up simulations and data

# ## read in libraries

# +
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
# -

# ## Branch and file Settings

# +
# ======================
# CONFIG
# ======================
BRANCHES = {
    "master": "Lentil",
}

# ======================
# RUN CONTROL - Specify which branches to (re)run
# ======================

# Options:
#RUN_BRANCHES = []                    # run nothing (use existing DBs)
RUN_BRANCHES = list(BRANCHES.keys())   # run all branches
#RUN_BRANCHES = ["master"]
#RUN_BRANCHES = ["working"]
#RUN_BRANCHES = ["working V2"]

REPO_PATH = Path(r"C:\GitHubRepos\ApsimX")

SIM_FILES = [
    Path(REPO_PATH) / 'Prototypes/Lentil/Lentil.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/FAHMA/FAHMA_Lentil.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2019_NSW_Greenethorpe_Mixed_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_Vic_Kalkee_Lentil_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_SA_Riverton_Lentil_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_NSW_WaggaWagga_Lentil_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_NSW_Methul_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_Vic_Ouyen_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_NSW_RankinsSprings_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2022_SA_Warnertown_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2023_SA_Pinery_Lentil_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2023_Vic_Dooen_Lentil_Detailed.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2023_SA_Warnertown_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2023_Vic_Ouyen_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2023_Qld_Gatton_Mixed_Light.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2024_NSW_Greenethorpe_Mixed_NFix.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2024_SA_Warnertown_Lentil_Satellite.apsimx',
    Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2024_Vic_Walpeup_Lentil_Satellite.apsimx'
]


CONFIG = {
    "branches": BRANCHES,
    "run_branches": RUN_BRANCHES,
    "sim_files": SIM_FILES,
    "repo_path": Path(r"C:\GitHubRepos\ApsimX"),
    "apsim_exe": r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe",
    "apsim_solution": r"C:\GitHubRepos\ApsimX\ApsimX.sln",
    "stage_var": "Lentil.Phenology.Stage",
    "stage_name_var": "Lentil.Phenology.CurrentStageName",
    "harvest_stage": "HarvestRipe",
    "cultivar_col": "Lentil.SowingData.Cultivar"
}

validate_run_branches(config=CONFIG)
# -

# ## additional index mapping

# +
# Specifiy map for cultivars to winter or spring type
project_group={
 'Caragabal19':'Historic',
 'Emerald19':'Historic',
 'Gatton19':'Historic',
 'Greenethorpe19':'Historic',
 'GreenethorpePhen19':'Historic',
 'Mildura19':'Historic',
 'Millmerran19':'Historic',
 'Walgett19':'Historic',
 'Wellcamp25':'FAHMA',
 'Roberts1988':'ContEnv',
 'Roberts1986':'ContEnv',
 'Summerfield1985':'ContEnv',
 'Rajandran2022':'ContEnv',
 'Dooen2001':'Historic',
 'Roseworthy':'Historic',
 'Gatton24':'FAHMA',
 'Gatton23':'FAHMA',
 'ForestHill25':'FAHMA',
 '2019_NSW_Greenethorpe_Mixed_Detailed':'NaPA',
 '2022_Vic_Kalkee_Lentil_Detailed':'NaPA',
 '2022_SA_Riverton_Lentil_Detailed':'NaPA',
 '2022_NSW_WaggaWagga_Lentil_Detailed':'NaPA',
 '2022_NSW_Methul_Lentil_Satellite':'NaPA',
 '2022_Vic_Ouyen_Lentil_Satellite':'NaPA',
 '2022_NSW_RankinsSprings_Lentil_Satellite':'NaPA',
 '2022_SA_Warnertown_Lentil_Satellite':'NaPA',
 '2023_SA_Pinery_Lentil_Detailed':'NaPA',
 '2023_Vic_Dooen_Lentil_Detailed':'NaPA',
 '2023_SA_Warnertown_Lentil_Satellite':'NaPA',
 '2023_Vic_Ouyen_Lentil_Satellite':'NaPA',
 '2023_Qld_Gatton_Mixed_Light':'NaPA',
 '2024_NSW_Greenethorpe_Mixed_NFix':'NaPA',
 '2024_SA_Warnertown_Lentil_Satellite':'NaPA',
 '2024_Vic_Walpeup_Lentil_Satellite':'NaPA',
}

# Pack maps together ready to be inserted as indexes 
additional_index_maps = {
    "ProjectGroup": {
        "source": "Experiment",
        "map": project_group
    }
}


# -

# ## Write file for command line tool to apply to each .apsimx file to be run.
# If you want to make standard modifications to all files run it can be done here

def write_apply_file(sim_file):
    """
    Create an APSIM CLI apply file which:
    - removes specified reports
    - injects AnalysisReport
    - sets variables
    - saves and runs simulation
    """
    report_library = r"C:/GitHubRepos/ApsimX/Prototypes/Lentil/Report_lib.apsimx"
    apply_file = sim_file.with_name(f"_apply_{sim_file.stem}.txt")

    lines = []

    # ---------------------------------------------
    # Add AnalysisReport to all Simulation nodes
    # ---------------------------------------------
    lines.append(f"add [AnalysisReport] from {report_library} to all [Zone]")
    
    # ---------------------------------------------
    # Inject Spectral model into each simulation
    # ---------------------------------------------
    # lines.append("add [Spectral] to all [Zone]")

    # ---------------------------------------------
    # Save + run
    # ---------------------------------------------
    lines.append(f"save {sim_file}")
    lines.append(f"run {sim_file}")

    # Write file
    apply_file.write_text("\n".join(lines))

    return apply_file


# ## Run simulations and read in raw .db data and process into tidy format

# +
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
# -

# # Harvest predictions

# ## Phenology

# ### Budding DAS

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Phenology.StartBuddingDAS",
    mode='harvest',
    filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# ### Flowering DAS

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Phenology.StartFloweringDAS",
    mode='harvest',
    filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# ### Podding DAS

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Phenology.StartPoddingDAS",
    mode='harvest',
    filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Phenology.StartPoddingDAS",
    mode='harvest',
    filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# ## Grain

# ### Yield (g/m2)

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Grain.Wt",
    mode='harvest',
    #filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# ### Grain number

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.Grain.Number",
    mode='harvest',
    #filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# ## Biomas

# ### Above ground Wt (g/m2)

graph = plot_obs_pred_by_branch(
    tidy = tidy,
    config=CONFIG,
    variable = "Lentil.AboveGround.Wt",
    mode='harvest',
    #filters = {"Lentil.SowingData.Cultivar": ["HallmarkXT"]},
    color_by = "Experiment",
    marker_by = "Lentil.SowingData.Cultivar",
    size_by=None,
)

# # Work out list of files to run for each cultivar for phenology fitting

obs = tidy[tidy.type=='obs']
phenVars = ['Lentil.Phenology.StartBuddingDAS',
             'Lentil.Phenology.StartFloweringDAS',
             'Lentil.Phenology.StartPoddingDAS']
phenObs = obs[obs.variable.isin(phenVars)]

phenObs#[phenObs.file=="2024_Vic_Walpeup_Lentil_Satellite.apsimx"]

phenFrame = phenObs.groupby(['Lentil.SowingData.Cultivar','file']).count().index.to_frame()

CvList = phenFrame.index.get_level_values(0).drop_duplicates().values

# +
CvFileDict = {}

for Cv in CvList:
    CvFileDict[Cv] = phenFrame.loc[Cv,'file'].drop_duplicates().to_list()
# -

CvFileDict

# +

obs_raw = raw[raw["type"] == "obs"]

cols = obs_raw.columns
print(cols)

obs_raw.head(20)


# +

obs_raw[
    obs_raw["file"] == "2024_Vic_Walpeup_Lentil_Satellite.apsimx"
].head(20)


# +

raw_obs = raw[raw["type"] == "obs"]

sat_obs = raw_obs[
    raw_obs["file"] == "2024_Vic_Walpeup_Lentil_Satellite.apsimx"
].copy()


# +

# columns that could contain real observations

# ONLY variables you actually care about
obs_vars = [
    'Lentil.Phenology.StartBuddingDAS',
    'Lentil.Phenology.StartFloweringDAS',
    'Lentil.Phenology.StartPoddingDAS'
]

value_cols = [c for c in obs_vars if c in sat_obs.columns]


# detect rows with any non-NaN values
has_real_data = sat_obs[value_cols].notna().any(axis=1)

print("Rows with real data:", has_real_data.sum())
print("Total rows:", len(sat_obs))

# -

sat_obs[value_cols]

import sqlite3
db_file =  Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2024_Vic_Walpeup_Lentil_Satellite.db'
with sqlite3.connect(db_file) as conn:
    df = pd.read_sql(f"SELECT * FROM Observed", conn)

df.columns

df.SimulationID.drop_duplicates()

import sqlite3
db_file =  Path(REPO_PATH) / 'Prototypes/Lentil/NaPA/2024_Vic_Walpeup_Lentil_Satellite.db'
with sqlite3.connect(db_file) as conn:
    df = pd.read_sql(f"SELECT * FROM AnalysisReport", conn)

df.SimulationID.drop_duplicates()

# ### 
