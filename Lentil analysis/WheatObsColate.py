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
# # Set Up

# %% [markdown]
# ## Libraries

# %%
import sys
sys.path.append(r"C:\GitHubRepos\APSIMTools\GraphLib")

# %load_ext autoreload
# %autoreload 2

from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
import subprocess
import shutil
import warnings
from matplotlib.lines import Line2D


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
reset_repo,
checkout_branch,
build_apsim,
run_apsim
)



# %% [markdown]
# ## Grouping functions

# %%
def value_at_stage(target, tol_low, tol_high):

    def fn(df, variable):

        window = df.loc[
            (df["Wheat.Phenology.Stage"] > tol_low) &
            (df["Wheat.Phenology.Stage"] < tol_high)
        ]

        window = window[window[variable].notna()]

        if window.empty:
            return np.nan

        idx = (
            window["Wheat.Phenology.Stage"]
            .sub(target)
            .abs()
            .idxmin()
        )

        return window.loc[idx, variable]

    return fn

def MeanValue(df, variable):
    return df[variable].mean()

def SumValues(df, variable):
    return df[variable].sum()


# %% [markdown]
# ## data class

# %%
@dataclass
class AnalysisData:

    pred: pd.DataFrame
    obs: pd.DataFrame

    def derive(self, name, fn):

        derived = fn(self.obs)

        if name in self.obs.columns:

            self.obs[name] = (
                self.obs[name]
                .combine_first(derived)
            )

        else:

            self.obs[name] = derived

    def filter(self, fn):

        filterMask = fn(self.obs)

        return filterMask

    def fill_triangular_set(self, top, left, right):

        cols = [top, left, right]
    
        missing = [c for c in cols if c not in self.obs.columns]
    
        if missing:
            print(f"Missing columns: {missing}")
            return self
    
        a = self.obs[top]
        b = self.obs[left]
        c = self.obs[right]
    
        # fill top
        self.obs[top] = a.combine_first(b * c)
    
        # fill left
        self.obs[left] = b.combine_first(
            a.div(c.replace(0, np.nan))
        )
    
        # fill right
        self.obs[right] = c.combine_first(
            a.div(b.replace(0, np.nan))
        )
        
    def aggregate_sim_values(self, var, outName, source="obs", fn = MeanValue, filter_fn = None):
        df = getattr(self, source)
        mask = pd.Series(True, index=df.index)
        if filter_fn:
            mask = filter_fn(df)
    
        grouped = (df.loc[mask].groupby("Simulation.Name"))
    
        values = grouped.apply(lambda g: fn(g, var))
    
        self.obs[outName] = (self.obs["Simulation.Name"].map(values))

    def summary(self):
        print(f"Pred rows : {len(self.pred):,}")
        print(f"Obs rows  : {len(self.obs):,}")

        print(f"Pred sims : {self.pred['SimulationID'].nunique():,}")
        print(f"Obs sims  : {self.obs['SimulationID'].nunique():,}")


ANALYSIS_KEYS = [
    "file",
    "SimulationID",
    "Clock.Today"
]


# %% [markdown]
# ## Load branch data function

# %%
def load_branch_data(config, apply_fn):
    git_branch = config['git_branch']

    if config['run_sims'] == True:
        reset_repo(config)
        checkout_branch(git_branch, config)
        build_apsim(config)

    pred_frames = []
    obs_frames = []

    for sim in config["sim_files"]:

        db = sim.parent / f"{sim.stem}.db"

        # --- Run APSIM ---
        if config['run_sims'] == True:
            print("")
            print(f"▶ Running APSIM [{git_branch}]: {sim.name}")
            run_apsim(sim, config, apply_fn)

            original_db = sim.with_suffix(".db")
            hold_db = sim.parent / f"{sim.stem}_{git_branch}_hold.db"

            shutil.copyfile(original_db, hold_db)

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
            if config["obs_table_name"] in tables:
                obs = pd.read_sql(f"SELECT * FROM [{config['obs_table_name']}]", conn)
                obs = obs.drop(columns=["SimulationName"], errors="ignore")
                
            # ---------------------------------------------
            # ✅ Predictions
            # ---------------------------------------------
            pred = None
            if config["pred_table_name"] in tables:
                pred = pd.read_sql(f"SELECT * FROM [{config['pred_table_name']}]", conn)
                
            if pred is None and obs is None:
                continue
            
            # ---------------------------------------------
            # ✅ Diagnose SimulationID alignment
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

            # ---------------------------------------------
            # ✅ Attach metadata (branch, file)
            # ---------------------------------------------
            if pred is not None:
                pred["file"] = sim.name
                pred["branch"] = git_branch
                pred_frames.append(pred)
                if "Clock.Today" in pred.columns:
                    pred["Clock.Today"] = pd.to_datetime(pred["Clock.Today"])

            if obs is not None:
                obs["file"] = sim.name
                obs_frames.append(obs)
                obs["branch"] = git_branch
                if "Clock.Today" in obs.columns:
                    obs["Clock.Today"] = pd.to_datetime(obs["Clock.Today"])

    if not obs_frames:
        raise RuntimeError(f"No obs data loaded")
    if not pred_frames:
        raise RuntimeError(f"No pred data loaded")
    
    return AnalysisData(
                        pred=pd.concat(pred_frames, ignore_index=True),
                        obs=pd.concat(obs_frames, ignore_index=True)
                        )


# %% [markdown]
# # Read in data

# %% [markdown]
# ## Set branch and file data

# %%
SIM_FILES = [
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Wheat.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\FAR\FAR.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\GxExM\GxExM.apsimx'),
    Path(r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Pask\PaskExperiments.apsimx'),
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
    "git_branch":  "UoM_Wheat",
    "run_sims": False,
    "sim_files": SIM_FILES,
    "repo_path": Path(r"C:\GitHubRepos\ApsimX"),
    "apsim_exe": r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe",
    "apsim_solution": r"C:\GitHubRepos\ApsimX\ApsimX.sln",
    "obs_table_name": "Observed",
    "pred_table_name": "ObsAnalysisReport"
}


# %% [markdown]
# ## Write apply file

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
    lines.append("delete all [Report]")
    lines.append(f"add [ObsAnalysisReport] from {report_library} to all [Zone]")

    # ---------------------------------------------
    # remove predicted Observed tables
    # ---------------------------------------------
    lines.append("delete all [PredictedObserved]")
    
    # ---------------------------------------------
    # Save + run
    # ---------------------------------------------
    lines.append(f"save {sim_file}")
    lines.append(f"run")

    # Write file
    apply_file.write_text("\n".join(lines))

    return apply_file


# %% [markdown]
# ## Read in data

# %%
data = load_branch_data(CONFIG, write_apply_file)


# %% [markdown]
# # Tidy up observed data frame

# %% [markdown]
# ## deal with unuseful and duplicate columns

# %%
def drop_unused_cols(data):
    Keep_columns = [
    'branch',
    'file',
    'SimulationID',
    'Clock.Today',
    'Spectral.NDVI',
    'sum(ObservedLayers.SWmm)',
    'sum(Soil.Water.MM)',
    'SW90cm',
    'TotalSW90cm',
    'Wheat.AboveGround.Live.NConc',
    'Wheat.AboveGround.Live.WSCc',
    'Wheat.AboveGround.N',
    'Wheat.AboveGround.NConc',
    'Wheat.AboveGround.WSC',
    'Wheat.AboveGround.Wt',
    'Wheat.AboveGround.Wt_1',
    'Wheat.Ear.N',
    'Wheat.Ear.NConc',
    'Wheat.Ear.Nconc',
    'Wheat.Ear.WSC',
    'Wheat.Ear.WSCc',
    'Wheat.Ear.Wt',
    'Wheat.Ear.WtProportion',
    'Wheat.Grain.FWt',
    'Wheat.Grain.FWt15',
    'Wheat.Grain.LiveFWt',
    'Wheat.Grain.Moisture',
    'Wheat.Grain.N',
    'Wheat.Grain.NConc',
    'Wheat.Grain.Nconc',
    'Wheat.Grain.Number',
    'Wheat.Grain.Protein',
    'Wheat.Grain.Screenings',
    'Wheat.Grain.Size',
    'Wheat.Grain.WSC',
    'Wheat.Grain.WSCc',
    'Wheat.Grain.Wt',
    'Wheat.Grain.Yield',
    'Wheat.Leaf.CoverGreen',
    'Wheat.Leaf.CoverTotal',
    'Wheat.Leaf.Dead.CID',
    'Wheat.Leaf.Dead.N',
    'Wheat.Leaf.Dead.NConc',
    'Wheat.Leaf.Dead.Nconc',
    'Wheat.Leaf.Dead.WSC',
    'Wheat.Leaf.Dead.WSCc',
    'Wheat.Leaf.Dead.Wt',
    'Wheat.Leaf.DeadCohortNo',
    'Wheat.Leaf.DeadWtProportion',
    'Wheat.Leaf.ExtinctionCoefficient',
    'Wheat.Leaf.Height',
    'Wheat.Leaf.Height_1',
    'Wheat.Leaf.LAI',
    'Wheat.Leaf.Ligules',
    'Wheat.Leaf.Live.CID',
    'Wheat.Leaf.Live.N',
    'Wheat.Leaf.Live.NConc',
    'Wheat.Leaf.Live.StorageWt',
    'Wheat.Leaf.Live.WSC',
    'Wheat.Leaf.Live.WSCc',
    'Wheat.Leaf.Live.Wt',
    'Wheat.Leaf.LiveWtProportion',
    'Wheat.Leaf.N',
    'Wheat.Leaf.NConc',
    'Wheat.Leaf.SpecificAreaCanopy',
    'Wheat.Leaf.Stem.Number',
    'Wheat.Leaf.StemNumberPerPlant',
    'Wheat.Leaf.StemNumberPerPlant.Total.Tillers',
    'Wheat.Leaf.StemPopulation',
    'Wheat.Leaf.Tips',
    'Wheat.Leaf.TrueStemPopulation',
    'Wheat.Leaf.Wt',
    'Wheat.Phenology.CAMP.TSHS',
    'Wheat.Phenology.CurrentStageName',
    'Wheat.Phenology.FinalLeafNumber',
    'Wheat.Phenology.FlagLeafDAS',
    'Wheat.Phenology.FloweringDAS',
    'Wheat.Phenology.HaunStage',
    'Wheat.Phenology.HeadingDAS',
    'Wheat.Phenology.MaturityDAS',
    'Wheat.Phenology.TerminalSpikeletDAS',
    'Wheat.Phenology.Zadok.Stage',
    'Wheat.Phenology.Zadok.Stage-NotModeled',
    'Wheat.Population',
    'Wheat.SowingData.Cultivar',
    'Wheat.SowingData.Population',
    'Wheat.SpecificAreaExpanding',
    'Wheat.Spike.CID',
    'Wheat.Spike.HeadNumber',
    'Wheat.Spike.Live.StorageWt',
    'Wheat.Spike.N',
    'Wheat.Spike.NConc',
    'Wheat.Spike.Nconc',
    'Wheat.Spike.WSC',
    'Wheat.Spike.WSCc',
    'Wheat.Spike.Wt',
    'Wheat.Stem.AreaIndex',
    'Wheat.Stem.CID',
    'Wheat.Stem.Live.StorageWt',
    'Wheat.Stem.N',
    'Wheat.Stem.NConc',
    'Wheat.Stem.Nconc',
    'Wheat.Stem.WSC',
    'Wheat.Stem.WSCc',
    'Wheat.Stem.WSCConc',
    'Wheat.Stem.Wt',
    'Wheat.Stem.WtProportion',
    'Wheat.StemPlusChaff',
    'Wheat.StemPlusSpike',
    'Wheat.Structure.TotalStemPopn',
    '[Wheat].Leaf.StemPopulation',
    'Wleat.Leaf.Wt']
    
    data.obs = data.obs.loc[:,Keep_columns].copy()

    remove_cols = [
    'branch',
    'file',
    'SimulationID',
    'Clock.Today'
    ]
    
    numeric_cols = [
        c for c in Keep_columns
        if c not in remove_cols
    ]

    for c in numeric_cols:
        data.obs[c] = pd.to_numeric(data.obs[c], errors="coerce")
    return data

data = drop_unused_cols(data)


# %%
def standardise_cols(data):
    COLUMN_MAP = {'[Wheat].Leaf.StemPopulation': 'Wheat.Leaf.StemPopulation',
    'Wheat.AboveGround.Wt_1': 'Wheat.AboveGround.Wt',
    'Wheat.Ear.Nconc': 'Wheat.Ear.NConc',
    'Wheat.Grain.FWt': 'Wheat.Grain.Yield',
    'Wheat.Grain.FWt15': 'Wheat.Grain.Yield',
    'Wheat.Grain.LiveFWt': 'Wheat.Grain.Yield',
    'Wheat.Grain.Moisture': 'Wheat.Grain.WaterContent',
    'Wheat.Grain.Nconc': 'Wheat.Grain.NConc',
    'Wheat.Leaf.Dead.Nconc': 'Wheat.Leaf.Dead.NConc',
    'Wheat.Leaf.Height_1': 'Wheat.Leaf.Height',
    'Wheat.Leaf.Stem.Number': 'Wheat.Leaf.StemPopulation',
    'Wheat.Leaf.StemNumberPerPlant.Total.Tillers': 'Wheat.Leaf.StemNumberPerPlant.StemNumberPerPlant.Total.Tillers',
    'Wheat.Leaf.TrueStemPopulation': 'Wheat.Leaf.StemPopulation',
    'Wheat.SowingData.Population': 'Wheat.Population',
    'Wheat.Spike.Nconc': 'Wheat.Spike.NConc',
    'Wheat.Stem.Nconc': 'Wheat.Stem.NConc',
    'Wheat.Stem.WSCConc': 'Wheat.Stem.WSCc',
    'Wheat.StemPlusChaff': 'Wheat.StemPlusSpike',
    'Wheat.Structure.TotalStemPopn': 'Wheat.Leaf.StemPopulation',
    'Wleat.Leaf.Wt': 'Wheat.Leaf.Wt'}
    
    data.obs = data.obs.rename(columns=COLUMN_MAP)
    return data
data = standardise_cols(data)


# %%
def merge_duplicate_cols(data):

    result = pd.DataFrame(index=data.obs.index)

    for col in pd.unique(data.obs.columns):

        matches = data.obs.loc[:, data.obs.columns == col]

        if matches.shape[1] == 1:
            result[col] = matches.iloc[:, 0]

        else:
            conflicts = (matches.notna().sum(axis=1) > 1)

            if conflicts.any():
                print(f"Warning: {col} has {conflicts.sum()} rows with multiple values")
            
            result[col] = matches.bfill(axis=1).iloc[:, 0]
            
    data.obs = result
    return data

data = merge_duplicate_cols(data)


# %% [markdown]
# ## remove rows from predicted where wheat is not sown

# %%
def remove_fallow_rows(data):

    data.pred = data.pred.loc[
        (data.pred["Wheat.DaysAfterSowing"] > 0)
        |
        (data.pred["Wheat.Phenology.CurrentStageName"] == "HarvestRipe")
    ]

    return data

data = remove_fallow_rows(data)    


# %% [markdown]
# ## Tidy up indexing in observed file

# %%
def fill_obs_metadata(data):

    cols = [
        "Experiment",
        "Simulation.Name",
        "Wheat.SowingData.Cultivar",
        "Wheat.Population",
        "IWeather.Latitude",
        "IWeather.Longitude"
    ]

    keys = ["file", "SimulationID"]

    meta = (
        data.pred
        [keys + cols]
        .drop_duplicates()
    )

    data.obs = (
        data.obs
        .drop(columns=cols, errors="ignore")
        .merge(meta, on=keys, how="left")
    )

    return data

data = fill_obs_metadata(data)

# %% [markdown]
# ## add in custom indexing

# %%
DevMap = {
"Accroc":"Winter",
"Adv08_0008":"Winter",
"Anapurna":"Winter",
"Ararat":"Spring",
"Atlanta":"Spring",
"Axe":"Spring",
"Batavia":"Spring",
"Battenspring":"Spring",
"Batten":"Winter",
"Beaufort":"Spring",
"Bennett":"Winter",
"BigRed":"Winter",
"Bolac":"Spring",
"Braewood":"Spring",
"Calabro":"Winter",
"Calingiri":"Spring",
"Catalina":"Spring",
"Catapult":"Spring",
"Cesario":"Winter",
"Claire":"Winter",
"Conquest":"Spring",
"Corack":"Spring",
"Crusader":"Spring",
"Crw247":"Spring",
"Cutlass":"Spring",
"Dekan":"Spring",
"Derrimut":"Spring",
"Discovery":"Spring",
"Drysdale":"Spring",
"Eaglehawk":"Spring",
"Einstein":"Winter",
"Ellison":"Spring",
"Forrest":"Spring",
"Gamenya":"Spring",
"Gauntlet":"Spring",
"Gladius":"Spring",
"Gorgan":"Spring",
"Graham":"Winter",
"Gregory":"Spring",
"Gutha":"Spring",
"H45":"Spring",
"H46":"Spring",
"Har1685":"Spring",
"Hartog":"Spring",
"Hume":"Spring",
"Illabo":"Winter",
"Istabraq":"Spring",
"Janz":"Spring",
"Kellalac":"Spring",
"Kennedy":"Spring",
"Kerrin":"Winter",
"Keyu13":"Spring",
"Kinsei":"Spring",
"Kittyhawk":"Winter",
"Konya":"Spring",
"Lancer":"Spring",
"Lincoln":"Spring",
"Livingston":"Spring",
"Mace":"Spring",
"Magenta":"Spring",
"Manning":"Winter",
"Matong":"Spring",
"Meering":"Spring",
"Mercury":"Spring",
"Merinda":"Spring",
"Mowhawk":"Winter",
"Nighthawk":"Spring",
"Osprey":"Winter",
"Otane":"Spring",
"Ouyen":"Spring",
"Pascal":"Spring",
"Peake":"Spring",
"Relay":"Winter",
"Revenue":"Winter",
"Rockstar":"Spring",
"Rongotea":"Spring",
"Rosario":"Spring",
"Rosella":"Winter",
"Ruby":"Spring",
"Savannah":"Winter",
"Scepter":"Spring",
"Scout":"Spring",
"Scythe":"Spring",
"Sorrial":"Winter",
"Spear":"Spring",
"Spitfire":"Spring",
"Stockade":"Spring",
"Strzelecki":"Spring",
"Sunbri":"Spring",
"Sunmaster":"Spring",
"Sunstate":"Spring",
"Suntop":"Spring",
"Trojan":"Spring",
"Uom001_3_47":"Winter",
"Uom001_9_1":"Winter",
"Ventura":"Spring",
"Voltron":"Winter",
"Wakanui":"Winter",
"Waugh":"Winter",
"Wedgetail":"Winter",
"Whistler":"Winter",
"Wilgoyne":"Spring",
"Wills":"Spring",
"Wyalkatchem":"Spring",
"Wylah":"Winter",
"Yecora":"Spring",
"Yitpi":"Spring",
"Young":"Spring",
"Zanzibar":"Spring",
"Zyatt":"Winter",
}

TestSetMap = {
'APS14':'TestSet',
'APS26':'TestSet',
'APS2':'TestSet',
'APS6':'TestSet',
'Cunderdin97':'TestSet',
'Gatton2009':'TestSet',
'Gatton2009TOS1CvKelallac':'TestSet',
'Gatton2009TOS1CvMckellar':'TestSet',
'Gatton2009TOS2CvKelallac':'TestSet',
'Gatton2009TOS2CvMckellar':'TestSet',
'Gatton2009TOS3CvKelallac':'TestSet',
'Gatton2009TOS3CvMckellar':'TestSet',
'Gatton2011':'TestSet',
'Gatton2011TOS2CvEaglehawk':'TestSet',
'Gatton2014AE':'TestSet',
'Gatton2014AEV1P1CvBaxter':'TestSet',
'Gatton2014AEV1P1CvDrysdale':'TestSet',
'Gatton2014AEV1P1CvHartog':'TestSet',
'Gatton2014AEV1P1CvWestonia':'TestSet',
'Gatton2014AEV1P2CvBaxter':'TestSet',
'Gatton2014AEV1P2CvDrysdale':'TestSet',
'Gatton2014AEV1P2CvHartog':'TestSet',
'Gatton2014AEV1P2CvWestonia':'TestSet',
'Gatton2014AEV2P1CvBaxter':'TestSet',
'Gatton2014AEV2P1CvDrysdale':'TestSet',
'Gatton2014AEV2P1CvHartog':'TestSet',
'Gatton2014AEV2P1CvWestonia':'TestSet',
'Gatton2014AEV2P2CvBaxter':'TestSet',
'Gatton2014AEV2P2CvDrysdale':'TestSet',
'Gatton2014AEV2P2CvHartog':'TestSet',
'Gatton2014AEV2P2CvWestonia':'TestSet',
'Gatton2014':'TestSet',
'Gatton2014TOS11-AprCvBaxter':'TestSet',
'Gatton2014TOS11-AprCvDrysdale':'TestSet',
'Gatton2014TOS11-AprCvHartog':'TestSet',
'Gatton2014TOS11-AprCvWestonia':'TestSet',
'Gatton2014TOS12-AugCvBaxter':'TestSet',
'Gatton2014TOS12-AugCvDrysdale':'TestSet',
'Gatton2014TOS12-AugCvHartog':'TestSet',
'Gatton2014TOS12-AugCvWestonia':'TestSet',
'Gatton2014TOS13-MayCvBaxter':'TestSet',
'Gatton2014TOS13-MayCvDrysdale':'TestSet',
'Gatton2014TOS13-MayCvHartog':'TestSet',
'Gatton2014TOS13-MayCvWestonia':'TestSet',
'Gatton2014TOS16-JulCvBaxter':'TestSet',
'Gatton2014TOS16-JulCvDrysdale':'TestSet',
'Gatton2014TOS16-JulCvHartog':'TestSet',
'Gatton2014TOS16-JulCvWestonia':'TestSet',
'Gatton94':'TestSet',
'Gatton94CvBataviaTOS4_Jul':'TestSet',
'Gatton94CvHartogTOS4_Jul':'TestSet',
'GattonRowSpacing':'TestSet',
'Ginninderra1991':'TestSet',
'Gorgan05':'TestSet',
'Griffith1983CVYecoraTOS15-Apr':'TestSet',
'Jamma':'TestSet',
'Konya09':'TestSet',
'Konya11':'TestSet',
'Leeston2013':'TestSet',
'Leeston2014':'TestSet',
'Lincoln1991':'TestSet',
'Lincoln1991Irrig02':'TestSet',
'Lincoln1991Irrig04':'TestSet',
'Lincoln1991Irrig09':'TestSet',
'Lincoln1991Irrig10':'TestSet',
'Lincoln1991Irrig12':'TestSet',
'Lincoln1991Irrig13':'TestSet',
'Lincoln1991Irrig14':'TestSet',
'Lincoln1992':'TestSet',
'Lincoln1994':'TestSet',
'Lincoln2010':'TestSet',
'Lincoln2014':'TestSet',
'Lincoln2015':'TestSet',
'Lincoln2021':'TestSet',
'Lincoln2023':'TestSet',
'Lincoln2024':'TestSet',
'Linconln2015Nit0IrrFull':'TestSet',
'Linconln2015Nit0IrrNil':'TestSet',
'Linconln2015Nit250IrrFull':'TestSet',
'Linconln2015Nit250IrrNil':'TestSet',
'Linconln2015Nit50IrrFull':'TestSet',
'Linconln2015Nit50IrrNil':'TestSet',
'Lonzee04':'TestSet',
'Lonzee06':'TestSet',
'Lonzee08':'TestSet',
'MaricopaFACE92_93':'TestSet',
'MaricopaFACE93_94':'TestSet',
'MaricopaFACE95_96':'TestSet',
'MaricopaFACE96_97':'TestSet',
'Mer73':'TestSet',
'Mer86':'TestSet',
'Mouse':'TestSet',
'PalmerstonNorth1989':'TestSet',
'TraitMod2015':'TestSet',
'TraitMod2016':'TestSet',
'Wagga1991':'TestSet',
'Wagga2013':'TestSet',
'Wagga2014':'TestSet',
'Wakanui2015':'TestSet',
'Wakanui2016':'TestSet',
'Wakanui2017':'TestSet',
'Wheat_Beverley90_Early':'TestSet',
'Wheat_Beverley90_Late':'TestSet',
'Wheat_Beverley90_n15':'TestSet',
'Wheat_Beverley90_n30':'TestSet',
'Wheat_Beverley90_n60':'TestSet',
'Wheat_Corrigin_10mmBasal':'TestSet',
'Wheat_Corrigin_10mmBasalTopDress':'TestSet',
'Wheat_Corrigin_40mmBasal':'TestSet',
'Wheat_Corrigin_40mmBasalTopDress':'TestSet',
'Wheat_Corrigin_DryBasal':'TestSet',
'Wheat_Corrigin_DryBasalTopDress':'TestSet',
'Wheat_Moora94_N0':'TestSet',
'Wheat_Moora94_N50':'TestSet',
'Wheat_Moora95_N0':'TestSet',
'Wheat_Moora95_N80':'TestSet',
'Wheat_Wongan83_Single':'TestSet',
'Wheat_Wongan84_N000':'TestSet',
'Wheat_Wongan84_N050':'TestSet',
'Wheat_Wongan84_N300':'TestSet',
'Wheat_Wongan84_N325':'TestSet',
'Wongan83':'TestSet',
'YarrabahCreek':'TestSet',
'Yucheng02':'TestSet',
'Yucheng03':'TestSet',
'Yucheng04':'TestSet',
'AGR ESW W23-01':'FAR',
'AGR ESW W23-01TOS1CvLancer':'FAR',
'AGR ESW W23-01TOS1CvLongsword':'FAR',
'AGR ESW W23-01TOS1CvRaider':'FAR',
'AGR ESW W23-01TOS1CvSunmaster':'FAR',
'AGR ESW W23-01TOS1CvVixen':'FAR',
'AGR ESW W23-01TOS2CvLancer':'FAR',
'AGR ESW W23-01TOS2CvLongsword':'FAR',
'AGR ESW W23-01TOS2CvRaider':'FAR',
'AGR ESW W23-01TOS2CvSunmaster':'FAR',
'AGR ESW W23-01TOS2CvVixen':'FAR',
'FAR DMC W20-03':'FAR',
'FAR DMC W20-03MgmtGrazedCvTabasco':'FAR',
'FAR DMC W20-03MgmtHigh InputCvTabasco':'FAR',
'FAR DMC W20-03MgmtStandardCvTabasco':'FAR',
'FAR DMC W20-05':'FAR',
'FAR DMC W20-06':'FAR',
'FAR ESW W23-02':'FAR',
'FAR ESW W23-02GrazeGrazeSeed180CvLongsword':'FAR',
'FAR ESW W23-02GrazeGrazeSeed180CvRaider':'FAR',
'FAR ESW W23-02GrazeGrazeSeed30CvLongsword':'FAR',
'FAR ESW W23-02GrazeGrazeSeed30CvRaider':'FAR',
'FAR ESW W23-02GrazeGrazeSeed90CvLongsword':'FAR',
'FAR ESW W23-02GrazeGrazeSeed90CvRaider':'FAR',
'FAR ESW W23-02GrazeNoneSeed180CvLongsword':'FAR',
'FAR ESW W23-02GrazeNoneSeed180CvRaider':'FAR',
'FAR ESW W23-02GrazeNoneSeed30CvLongsword':'FAR',
'FAR ESW W23-02GrazeNoneSeed30CvRaider':'FAR',
'FAR ESW W23-02GrazeNoneSeed90CvLongsword':'FAR',
'FAR ESW W23-02GrazeNoneSeed90CvRaider':'FAR',
'FAR HYC W17-01-1':'FAR',
'FAR HYC W17-01-1MgmtGrazedCvConqueror':'FAR',
'FAR HYC W17-01-1MgmtGrazedCvGenius':'FAR',
'FAR HYC W17-01-1MgmtHigh InputCvConqueror':'FAR',
'FAR HYC W17-01-1MgmtHigh InputCvGenius':'FAR',
'FAR HYC W17-01-1MgmtStandardCvConqueror':'FAR',
'FAR HYC W17-01-1MgmtStandardCvGenius':'FAR',
'FAR HYC W17-01-2MgmtHigh InputCvADV11.9419':'FAR',
'FAR HYC W17-01-2MgmtHigh InputCvAGTW0001':'FAR',
'FAR HYC W17-01-2MgmtHigh InputCvAGTW0002':'FAR',
'FAR HYC W17-01-2':'FAR',
'FAR HYC W17-01-2MgmtHigh InputCvConqueror':'FAR',
'FAR HYC W17-01-2MgmtHigh InputCvGenius':'FAR',
'FAR HYC W17-01-2MgmtStandardCvADV11.9419':'FAR',
'FAR HYC W17-01-2MgmtStandardCvAGTW0001':'FAR',
'FAR HYC W17-01-2MgmtStandardCvAGTW0002':'FAR',
'FAR HYC W17-01-2MgmtStandardCvConqueror':'FAR',
'FAR HYC W17-01-2MgmtStandardCvGenius':'FAR',
'FAR HYC W17-02-1CvAGTW0001':'FAR',
'FAR HYC W17-02-1':'FAR',
'FAR HYC W17-02-1CvAsano':'FAR',
'FAR HYC W17-02-1CvBA 26.35':'FAR',
'FAR HYC W17-02-1CvConqueror':'FAR',
'FAR HYC W17-02-1CvCordiale':'FAR',
'FAR HYC W17-02-1CvGenius':'FAR',
'FAR HYC W17-02-1CvHereford':'FAR',
'FAR HYC W17-02-1CvMercedes':'FAR',
'FAR HYC W17-02-1CvOakley':'FAR',
'FAR HYC W17-02-1CvViscount':'FAR',
'FAR HYC W17-02-1CvXi19':'FAR',
'FAR HYC W17-02-2CvADV14.1292':'FAR',
'FAR HYC W17-02-2CvADV14.1335':'FAR',
'FAR HYC W17-02-2':'FAR',
'FAR HYC W17-02-2CvApache':'FAR',
'FAR HYC W17-02-2CvAsano':'FAR',
'FAR HYC W17-02-2CvBA 26.35':'FAR',
'FAR HYC W17-02-2CvCS170':'FAR',
'FAR HYC W17-02-2CvCS3250.30':'FAR',
'FAR HYC W17-02-2CvCS611':'FAR',
'FAR HYC W17-02-2CvCS98152.79':'FAR',
'FAR HYC W17-02-2CvCSQ496.88':'FAR',
'FAR HYC W17-02-2CvCSR65':'FAR',
'FAR HYC W17-02-2CvCordiale':'FAR',
'FAR HYC W17-02-2CvEDGE W12-090-04':'FAR',
'FAR HYC W17-02-2CvEDGE06-018b-10':'FAR',
'FAR HYC W17-02-2CvHereford':'FAR',
'FAR HYC W17-02-2CvKowari (Trit)':'FAR',
'FAR HYC W17-02-2CvOakley':'FAR',
'FAR HYC W17-02-2CvSolist':'FAR',
'FAR HYC W17-02-2CvTabasco':'FAR',
'FAR HYC W17-02-2CvTuareg':'FAR',
'FAR HYC W17-02-2CvViscount':'FAR',
'FAR HYC W17-02-2CvXi19':'FAR',
'FAR HYC W17-08':'FAR',
'FAR HYC W18-02-1':'FAR',
'FAR HYC W18-02-2':'FAR',
'FAR HYC W18-02-2Seeds100CvBennett':'FAR',
'FAR HYC W18-02-2Seeds175CvBennett':'FAR',
'FAR HYC W18-02-2Seeds250CvBennett':'FAR',
'FAR HYC W18-02-2a':'FAR',
'FAR HYC W18-08-1':'FAR',
'FAR HYC W19-01-1':'FAR',
'FAR HYC W19-01-1FungicideFullCvConqueror':'FAR',
'FAR HYC W19-01-1FungicideFullCvGenius':'FAR',
'FAR HYC W19-01-1FungicideFullCvTabasco':'FAR',
'FAR HYC W19-01-1FungicideNoneCvConqueror':'FAR',
'FAR HYC W19-01-1FungicideNoneCvGenius':'FAR',
'FAR HYC W19-01-1FungicideNoneCvTabasco':'FAR',
'FAR HYC W19-02-1':'FAR',
'FAR HYC W19-03-1GrazedGS16 30 40NSeeds100CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16 30 40NSeeds175CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16 30 40NSeeds250CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16 30Seeds100CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16 30Seeds175CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16 30Seeds250CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16Seeds100CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16Seeds175CvBennett':'FAR',
'FAR HYC W19-03-1GrazedGS16Seeds250CvBennett':'FAR',
'FAR HYC W19-06-1':'FAR',
'FAR HYC W19-08-1':'FAR',
'FAR NEV FRO WB23-01':'FAR',
'FAR NEV FRO WB23-01Def1TOS1CvVixen':'FAR',
'FAR NEV FRO WB23-01Def1TOS2CvVixen':'FAR',
'FAR NEV FRO WB23-01Def2TOS1CvVixen':'FAR',
'FAR NEV FRO WB23-01Def2TOS2CvVixen':'FAR',
'FAR NEV FRO WB23-01DefControlTOS1CvDenison':'FAR',
'FAR NEV FRO WB23-01DefControlTOS1CvVixen':'FAR',
'FAR NEV FRO WB23-01DefControlTOS2CvDenison':'FAR',
'FAR NEV FRO WB23-01DefControlTOS2CvVixen':'FAR',
'FAR NEV FRO WB23-01DefControlTOS3CvDenison':'FAR',
'FAR NEV FRO WB23-01DefControlTOS3CvVixen':'FAR',
'FAR NSW W23-03MgmtHigh InputCvAGTW0005':'FAR',
'FAR NSW W23-03':'FAR',
'FAR NSW W23-03MgmtHigh InputCvLongford':'FAR',
'FAR NSW W23-03MgmtLow InputCvAGTW0005':'FAR',
'FAR NSW W23-03MgmtLow InputCvLongford':'FAR',
'FAR NSW W23-03MgmtStrategicCvAGTW0005':'FAR',
'FAR NSW W23-03MgmtStrategicCvLongford':'FAR',
'FAR NSW W23-03MgmtTacticalCvAGTW0005':'FAR',
'FAR NSW W23-03MgmtTacticalCvLongford':'FAR',
'FAR NSW W23-05':'FAR',
'FAR RRC W20-03':'FAR',
'FAR RRC W20-03MgmtGrazedCvBeckom':'FAR',
'FAR RRC W20-03MgmtGrazedCvGregory':'FAR',
'FAR RRC W20-03MgmtHigh InputCvBeckom':'FAR',
'FAR RRC W20-03MgmtHigh InputCvGregory':'FAR',
'FAR RRC W20-03MgmtStandardCvBeckom':'FAR',
'FAR RRC W20-03MgmtStandardCvGregory':'FAR',
'FAR RRC W20-05':'FAR',
'FAR RRC W20-06-1':'FAR',
'FAR RRC W21-01CvAGFWH004418':'FAR',
'FAR RRC W21-01CvAGFWH004618':'FAR',
'FAR RRC W21-01':'FAR',
'FAR RRC W21-01CvAurora':'FAR',
'FAR RRC W21-01CvBeckom':'FAR',
'FAR RRC W21-01CvBitalli':'FAR',
'FAR RRC W21-01CvCoota':'FAR',
'FAR RRC W21-01CvGraham':'FAR',
'FAR RRC W21-01CvL13070-027':'FAR',
'FAR RRC W21-01CvLPB16-0582':'FAR',
'FAR RRC W21-01CvLPB17-5691':'FAR',
'FAR RRC W21-01CvLongford':'FAR',
'FAR RRC W21-01CvReflection':'FAR',
'FAR RRC W21-01CvSUN1087I':'FAR',
'FAR RRC W21-01CvSavello':'FAR',
'FAR RRC W21-01CvShabras':'FAR',
'FAR RRC W21-01CvTabasco':'FAR',
'FAR RRC W21-01CvV11068-085-047':'FAR',
'FAR RRC W21-01CvV12167-048':'FAR',
'FAR RRC W21-01CvWestcourt':'FAR',
'FAR RRC W21-03':'FAR',
'FAR RRC W21-06':'FAR',
'FAR RRC W21-07':'FAR',
'FAR RRC W22-02FungicideFullCvAGTW0005':'FAR',
'FAR RRC W22-02':'FAR',
'FAR RRC W22-02FungicideFullCvLongford':'FAR',
'FAR RRC W22-02FungicideFullCvReflection':'FAR',
'FAR RRC W22-02FungicideFullCvTabasco':'FAR',
'FAR RRC W22-02FungicideNoneCvAGTW0005':'FAR',
'FAR RRC W22-02FungicideNoneCvLongford':'FAR',
'FAR RRC W22-02FungicideNoneCvReflection':'FAR',
'FAR RRC W22-03':'FAR',
'FAR RRC W22-05-1':'FAR',
'FAR SAC W18-01MgmtGrazedCvAGTW0002':'FAR',
'FAR SAC W18-01':'FAR',
'FAR SAC W18-01MgmtGrazedCvConqueror':'FAR',
'FAR SAC W18-01MgmtHigh InputCvAGTW0002':'FAR',
'FAR SAC W18-01MgmtHigh InputCvConqueror':'FAR',
'FAR SAC W18-01MgmtStandardCvAGTW0002':'FAR',
'FAR SAC W18-01MgmtStandardCvConqueror':'FAR',
'FAR SAC W18-02':'FAR',
'FAR SAC W18-02FungicideNoneCvAdagio':'FAR',
'FAR SAC W18-02FungicideNoneCvAsano':'FAR',
'FAR SAC W18-02FungicideNoneCvCoolah':'FAR',
'FAR SAC W18-02FungicideNoneCvGenius':'FAR',
'FAR SAC W18-02FungicideNoneCvHereford':'FAR',
'FAR SAC W18-02FungicidePlusCvAdagio':'FAR',
'FAR SAC W18-02FungicidePlusCvAsano':'FAR',
'FAR SAC W18-02FungicidePlusCvCoolah':'FAR',
'FAR SAC W18-02FungicidePlusCvGenius':'FAR',
'FAR SAC W18-02FungicidePlusCvHereford':'FAR',
'FAR SAC W18-06':'FAR',
'FAR SAC W19-01':'FAR',
'FAR SAC W19-01MgmtGrazedCvConqueror':'FAR',
'FAR SAC W19-01MgmtGrazedCvTabasco':'FAR',
'FAR SAC W19-01MgmtHigh InputCvConqueror':'FAR',
'FAR SAC W19-01MgmtHigh InputCvTabasco':'FAR',
'FAR SAC W19-01MgmtStandardCvConqueror':'FAR',
'FAR SAC W19-01MgmtStandardCvTabasco':'FAR',
'FAR SAC W19-02':'FAR',
'FAR SAC W19-02FungicideFullCvAdagio':'FAR',
'FAR SAC W19-02FungicideFullCvAsano':'FAR',
'FAR SAC W19-02FungicideFullCvHereford':'FAR',
'FAR SAC W19-02FungicideFullCvTabasco':'FAR',
'FAR SAC W19-02FungicideNoneCvAdagio':'FAR',
'FAR SAC W19-02FungicideNoneCvAsano':'FAR',
'FAR SAC W19-02FungicideNoneCvHereford':'FAR',
'FAR SAC W19-02FungicideNoneCvTabasco':'FAR',
'FAR SAC W19-06':'FAR',
'FAR SAC W20-03-1':'FAR',
'FAR SAC W20-03-1MgmtGrazedCvAdagio':'FAR',
'FAR SAC W20-03-1MgmtGrazedCvTabasco':'FAR',
'FAR SAC W20-03-1MgmtHigh InputCvAdagio':'FAR',
'FAR SAC W20-03-1MgmtHigh InputCvTabasco':'FAR',
'FAR SAC W20-03-1MgmtStandardCvAdagio':'FAR',
'FAR SAC W20-03-1MgmtStandardCvTabasco':'FAR',
'FAR SAC W20-03-2':'FAR',
'FAR SAC W20-03-2MgmtHigh InputCvCobra':'FAR',
'FAR SAC W20-03-2MgmtLow InputCvCobra':'FAR',
'FAR SAC W20-03-2MgmtStandardCvCobra':'FAR',
'FAR SAC W20-05-1':'FAR',
'FAR SAC W20-06-1':'FAR',
'FAR SAC W21-05-1':'FAR',
'FAR SAC W22-03-1':'FAR',
'FAR SAC W22-03-2':'FAR',
'FAR SAC W22-05-1':'FAR',
'FAR SAC W23-05':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvADV11.9419':'FAR',
'FAR TAS W16-01-1':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvCS3250.30':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvCS98152.79':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvCSQ496.88':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvConqueror':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvEDGE06-018b-10':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvEDGE06-025-03':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvEDGE06-039-13':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvGenius':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvLPB11-0140':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvV08126-64':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvV10006-026':'FAR',
'FAR TAS W16-01-1MgmtGrazedCvV10083-050':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvADV11.9419':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvCS3250.30':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvCS98152.79':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvCSQ496.88':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvConqueror':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvEDGE06-018b-10':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvEDGE06-025-03':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvEDGE06-039-13':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvGenius':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvLPB11-0140':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvV08126-64':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvV10006-026':'FAR',
'FAR TAS W16-01-1MgmtHigh InputCvV10083-050':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvADV11.9419':'FAR',
'FAR TAS W16-01-2':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvCS3250.30':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvCS98152.79':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvCSQ496.88':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvConqueror':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvEDGE06-018b-10':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvEDGE06-025-03':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvEDGE06-039-13':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvGenius':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvLPB11-0140':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvV08126-64':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvV10006-026':'FAR',
'FAR TAS W16-01-2MgmtHigh InputCvV10083-050':'FAR',
'FAR TAS W16-01-2MgmtStandardCvADV11.9419':'FAR',
'FAR TAS W16-01-2MgmtStandardCvCS3250.30':'FAR',
'FAR TAS W16-01-2MgmtStandardCvCS98152.79':'FAR',
'FAR TAS W16-01-2MgmtStandardCvCSQ496.88':'FAR',
'FAR TAS W16-01-2MgmtStandardCvConqueror':'FAR',
'FAR TAS W16-01-2MgmtStandardCvEDGE06-018b-10':'FAR',
'FAR TAS W16-01-2MgmtStandardCvEDGE06-025-03':'FAR',
'FAR TAS W16-01-2MgmtStandardCvEDGE06-039-13':'FAR',
'FAR TAS W16-01-2MgmtStandardCvGenius':'FAR',
'FAR TAS W16-01-2MgmtStandardCvLPB11-0140':'FAR',
'FAR TAS W16-01-2MgmtStandardCvV08126-64':'FAR',
'FAR TAS W16-01-2MgmtStandardCvV10006-026':'FAR',
'FAR TAS W16-01-2MgmtStandardCvV10083-050':'FAR',
'FAR TAS W16-06':'FAR',
'FAR TAS W16-06PGRDoubleLateSeeds200CvManning':'FAR',
'FAR TAS W16-06PGRDoubleMidSeeds200CvManning':'FAR',
'FAR TAS W16-06PGREarly1Seeds200CvManning':'FAR',
'FAR TAS W16-06PGREarly2Seeds200CvManning':'FAR',
'FAR TAS W16-06PGRNoneSeeds100CvManning':'FAR',
'FAR TAS W16-06PGRNoneSeeds150CvManning':'FAR',
'FAR TAS W16-06PGRNoneSeeds200CvManning':'FAR',
'FAR TAS W16-06PGRNoneSeeds50CvManning':'FAR',
'FAR TAS W16-06PGRSingleLateSeeds200CvManning':'FAR',
'FAR TAS W16-06PGRSingleMidSeeds200CvManning':'FAR',
'FAR TAS W16-06PGRTripleEarlySeeds200CvManning':'FAR',
'FAR TAS W16-06PGRTripleLateSeeds200CvManning':'FAR',
'FAR TAS W16-08':'FAR',
'FAR TAS W21-06-1':'FAR',
'FAR TAS W23-03':'FAR',
'FAR TAS W23-03MgmtHigh InputCvLongford':'FAR',
'FAR TAS W23-03MgmtLow InputCvLongford':'FAR',
'FAR TAS W23-03MgmtStrategicCvLongford':'FAR',
'FAR TAS W23-03MgmtTacticalCvLongford':'FAR',
'FAR VIC W22-03-1':'FAR',
'FAR VIC W22-03-2':'FAR',
'FAR VIC W22-05-1':'FAR',
'FAR VIC W23-03a':'FAR',
'FAR WAA W20-01b':'FAR',
'FAR WAA W20-01bMgmtGrazedCvLRPB TBC':'FAR',
'FAR WAA W20-01bMgmtHigh InputCvLRPB TBC':'FAR',
'FAR WAA W20-01bMgmtStandardCvLRPB TBC':'FAR',
'FAR WAA W22-01':'FAR',
'FAR WAA W22-01MgmtGrazedCvDenison':'FAR',
'FAR WAA W22-01MgmtHigh InputCvDenison':'FAR',
'FAR WAA W22-01MgmtStandardCvDenison':'FAR',
'FAR WAA W23-03':'FAR',
'FAR WAA W23-03MgmtHigh InputCvDenison':'FAR',
'FAR WAA W23-03MgmtLow InputCvDenison':'FAR',
'FAR WAA W23-03MgmtStrategicCvDenison':'FAR',
'FAR WAA W23-03MgmtTacticalCvDenison':'FAR',
'FAR WAA W23-05':'FAR',
'FAR WAE W21-05':'FAR',
'FAR WAE W21-05CvDenison':'FAR',
'FAR WAE W21-05CvMagenta':'FAR',
'FAR WAE W21-05CvValiant CL':'FAR',
'FAR WAE W22-01':'FAR',
'FAR WAE W22-02':'FAR',
'FAR WAE W22-02MgmtGrazedCvDenison':'FAR',
'FAR WAE W22-02MgmtHigh InputCvDenison':'FAR',
'FAR WAE W22-02MgmtStandardCvDenison':'FAR',
'FAR WAE W22-03':'FAR',
'FAR WAE W22-03CvLTU001-038':'FAR',
'FAR WAE W22-03CvLTU001-039':'FAR',
'FAR WAE W22-03CvLTU001-066':'FAR',
'FAR WAE W22-03CvLTU001-092':'FAR',
'FAR WAE W22-03CvLTU002-18-01':'FAR',
'FAR WAE W22-04':'FAR',
'FAR WAE W22-04CvDenison':'FAR',
'FAR WAE W22-04CvDevil':'FAR',
'FAR WAE W22-04CvSting':'FAR',
'FAR WAE W22-04CvVixen':'FAR',
'FAR WAG W22-01':'FAR',
'FAR WAG W22-01CvCoota':'FAR',
'FAR WAG W22-01CvLongsword':'FAR',
'FAR WAG W22-01CvSunflex':'FAR',
'FAR WAG W22-01CvV12167-048':'FAR',
'FAR WAG W22-01CvValiant':'FAR',
'FAR WAG W22-03CvBoree':'FAR',
'FAR WAG W22-03CvCoota':'FAR',
'FAR WAG W22-03CvDS Bennett':'FAR',
'FAR WAG W22-03':'FAR',
'FAR WAG W22-03CvLongsword':'FAR',
'FAR WAG W22-03CvRGT Accroc':'FAR',
'FAR WAG W22-03CvRocksta':'FAR',
'FAR WAG W22-03CvSunflex':'FAR',
'FAR WAG W22-03CvV12167-048':'FAR',
'FAR WAG W22-03CvValiant':'FAR',
'FAR WAG W22-03CvVixen':'FAR',
'Pask LC07':'TestSet',
'Pask TT06':'TestSet',
'Pask TT07':'TestSet',
'Gatton2014CV29B':'GxExM',
'Gatton2014CV5A':'GxExM',
'Gatton2014CV60A':'GxExM',
'Gatton2014CVEspada':'GxExM',
'Gatton2014CVGauntlet':'GxExM',
'Gatton2014CVScout':'GxExM',
'Gatton2014CVSpitfire':'GxExM',
'Gatton2014CVSunbee':'GxExM',
'Gatton2014CVSunstate':'GxExM',
'Gatton2014IrrigatedCV29B':'GxExM',
'Gatton2014IrrigatedCV5A':'GxExM',
'Gatton2014IrrigatedCV60A':'GxExM',
'Gatton2014IrrigatedCVEspada':'GxExM',
'Gatton2014IrrigatedCVGauntlet':'GxExM',
'Gatton2014IrrigatedCVScout':'GxExM',
'Gatton2014IrrigatedCVSpitfire':'GxExM',
'Gatton2014IrrigatedCVSunbee':'GxExM',
'Gatton2014IrrigatedCVSunstate':'GxExM',
'Gatton2014Irrigated':'GxExM',
'Gatton2015CV29B':'GxExM',
'Gatton2015CV5A':'GxExM',
'Gatton2015CV60A':'GxExM',
'Gatton2015CVEspada':'GxExM',
'Gatton2015CVGaunlet':'GxExM',
'Gatton2015CVGauntlet':'GxExM',
'Gatton2015CVScout':'GxExM',
'Gatton2015CVSpitfire':'GxExM',
'Gatton2015CVSunbee':'GxExM',
'Gatton2015CVSunstate':'GxExM',
'Gatton2015':'GxExM',
'Junee2014CV29B':'GxExM',
'Junee2014CV5A':'GxExM',
'Junee2014CV60A':'GxExM',
'Junee2014CVEspada':'GxExM',
'Junee2014CVGauntlet':'GxExM',
'Junee2014CVScout':'GxExM',
'Junee2014CVSpitfire':'GxExM',
'Junee2014CVSunbee':'GxExM',
'Junee2014CVSunstate':'GxExM',
'Junee2014':'GxExM',
'Minnipa2014CV29B':'GxExM',
'Minnipa2014CV5A':'GxExM',
'Minnipa2014CV60A':'GxExM',
'Minnipa2014CVEspada':'GxExM',
'Minnipa2014CVGauntlet':'GxExM',
'Minnipa2014CVScout':'GxExM',
'Minnipa2014CVSpitfire':'GxExM',
'Minnipa2014CVSunbee':'GxExM',
'Minnipa2014CVSunstate':'GxExM',
'Minnipa2014':'GxExM',
'Minnipa2015CV29B':'GxExM',
'Minnipa2015CV5A':'GxExM',
'Minnipa2015CV60A':'GxExM',
'Minnipa2015CVEspada':'GxExM',
'Minnipa2015CVGauntlet':'GxExM',
'Minnipa2015CVScout':'GxExM',
'Minnipa2015CVSpitfire':'GxExM',
'Minnipa2015CVSunbee':'GxExM',
'Minnipa2015CVSunstate':'GxExM',
'Minnipa2015':'GxExM',
'Temora2015CV29B':'GxExM',
'Temora2015CV5A':'GxExM',
'Temora2015CV60A':'GxExM',
'Temora2015CVEspada':'GxExM',
'Temora2015CVGauntlet':'GxExM',
'Temora2015CVScout':'GxExM',
'Temora2015CVSpitfire':'GxExM',
'Temora2015CVSunbee':'GxExM',
'Temora2015CVSunstate':'GxExM',
'Temora2015':'GxExM',
'DookieEVA2024':'WWHI',
'DookieWWHI2024':'WWHI',
'DookieEVA2025':'WWHI',
'DookieWWHI2025':'WWHI',
'WaggaWagga2024':'WWHI',
'WaggaWagga2025':'WWHI',
'Gnarwarre2024':'WWHI',
'Gnarwarre2025':'WWHI',
'GrassPatch2024':'WWHI',
'GrassPatch2025':'WWHI',
'Fords2025':'WWHI',
'Turretfield2024':'WWHI',
'Turretfield2024CvKittyhawkSow15-May':'WWHI',
'Turretfield2024CvScepterSow16-Apr':'WWHI',
}


# Pack maps together ready to be inserted as indexes 
additional_index_maps = {
    "DevelopmentType": {
        "source": "Wheat.SowingData.Cultivar",
        "map": DevMap
    },
    "ProjectGroup": {
        "source": "Experiment",
        "map": TestSetMap
    }
}


# %%
def add_index_maps(data, additional_index_maps):

    for new_col, spec in additional_index_maps.items():

        source_col = spec["source"]
        mapping = spec["map"]

        if source_col not in data.obs.columns:
            print(
                f"Warning: '{source_col}' not found "
                f"for index '{new_col}'"
            )
            continue

        data.obs[new_col] = (
            data.obs[source_col]
            .map(mapping)
        )

    return data

data = add_index_maps(data, additional_index_maps)


# %% [markdown]
# # Merge predicted values in with observations 

# %% [markdown]
# ## Merge simple variables

# %%
def attach_pred_vars(data, variables):

    keys = [
        "file",
        "SimulationID",
        "Clock.Today"
    ]

    pred_subset = data.pred[
        keys + variables
    ]

    data.obs = data.obs.merge(
                pred_subset,
                how="left",
                on=keys
    )
    return data


# %%
phenology_preds = ["Wheat.DaysAfterSowing",
"Wheat.Phenology.AccumulatedTT",
"Wheat.Phenology.CurrentPhase.Name",
"Wheat.Phenology.CurrentStageName",
"Wheat.Phenology.Photoperiod",
"Wheat.Phenology.PTQ",
"Wheat.Phenology.Stage",
"Wheat.Phenology.ThermalTime",
"Wheat.Phenology.Zadok.Stage"]

data = attach_pred_vars(data, phenology_preds)


# %% [markdown]
# ## Derive summary met variables

# %%
def add_weather_predictors(
    data,
    mean_windows=(7, 30)
):

    sim_keys = ["file", "SimulationID"]

    derived_vars = []

    mean_vars = [
        "Wheat.Phenology.PTQ",
        "Wheat.Phenology.ThermalTime",
        "IWeather.MinT",
        "IWeather.MaxT",
        "IWeather.MeanT",
        "IWeather.Radn",
        "IWeather.VPD",
        "IWeather.Wind",
        "IWeather.CO2"
    ]

    # -------------------------------------
    # Rolling means
    # -------------------------------------
    for var in mean_vars:

        if var not in data.pred.columns:
            continue

        for window in mean_windows:

            new_name = f"{var}.Mean{window}"

            data.pred[new_name] = (
                data.pred
                .groupby(sim_keys)[var]
                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=1
                    ).mean()
                )
            )

            derived_vars.append(new_name)

    # -------------------------------------
    # Rolling rainfall sums
    # -------------------------------------
    if "IWeather.Rain" in data.pred.columns:

        for window in mean_windows:

            new_name = f"IWeather.Rain.Sum{window}"

            data.pred[new_name] = (
                data.pred
                .groupby(sim_keys)["IWeather.Rain"]
                .transform(
                    lambda x:
                    x.rolling(
                        window,
                        min_periods=1
                    ).sum()
                )
            )

            derived_vars.append(new_name)

    # -------------------------------------
    # Accumulations since sowing
    # -------------------------------------
    for var in ["IWeather.Radn", "IWeather.Rain"]:

        if var not in data.pred.columns:
            continue

        acc_name = f"{var}.Accum"

        data.pred[acc_name] = (
            data.pred
            .groupby(sim_keys)[var]
            .cumsum()
        )

        derived_vars.append(acc_name)

    return data, derived_vars

data, met_vars = add_weather_predictors(data, mean_windows=(7, 30))

data = attach_pred_vars(data, met_vars)

# %% [markdown]
# # Derive additional data

# %% [markdown]
# ## Agregate variables

# %%
data.derive('Wheat.Leaf.Wt',
                lambda df:
            df["Wheat.Leaf.Live.Wt"] +
            df["Wheat.Leaf.Dead.Wt"])

# %%
data.derive('Wheat.AboveGround.Wt',
                lambda df:
            df["Wheat.Leaf.Wt"] +
            df["Wheat.Stem.Wt"] +
            df["Wheat.Spike.Wt"] +
            df["Wheat.Grain.Wt"])

# %%
data.derive('Wheat.Leaf.StemNumberPerPlant',
                lambda df:
        df['Wheat.Leaf.StemNumberPerPlant.StemNumberPerPlant.Total.Tillers'] + 1)

# %%
data.derive('Wheat.StemPlusSpikeWt',
                lambda df:
            df["Wheat.Stem.Wt"] +
            df["Wheat.Spike.Wt"])

# %% [markdown]
# ## Complete traingular sets

# %% [markdown]
# The variable at the top of the triangle is derived as to product of the two variables at the bottom.
# The variables at the bottom are derived as the top variable divided the the other bottom variable 
#         A
#        / \
#       /   \
#      B  x  C
#
# A = B * C
#
# B = A / C
#
# C = A / B

# %%
data.fill_triangular_set('Wheat.Leaf.StemPopulation',
                            'Wheat.Leaf.StemNumberPerPlant',
                            'Wheat.Population')

# %%
data.fill_triangular_set('Wheat.Leaf.LAI',
                            'Wheat.Leaf.Live.Wt',
                            'Wheat.Leaf.SpecificAreaCanopy')

# %%
data.fill_triangular_set('Wheat.AboveGround.N',
                            'Wheat.AboveGround.Wt',
                            'Wheat.AboveGround.NConc')

# %%
data.fill_triangular_set('Wheat.Leaf.Live.N',
                            'Wheat.Leaf.Live.Wt',
                            'Wheat.Leaf.Live.NConc')

# %%
data.fill_triangular_set('Wheat.Leaf.Dead.N',
                            'Wheat.Leaf.Dead.Wt',
                            'Wheat.Leaf.Dead.NConc')

# %%
data.fill_triangular_set('Wheat.Stem.N',
                            'Wheat.Stem.Wt',
                            'Wheat.Stem.NConc')

# %%
data.fill_triangular_set('Wheat.Ear.N',
                            'Wheat.Ear.Wt',
                            'Wheat.Ear.NConc')

# %%
data.fill_triangular_set('Wheat.Spike.N',
                            'Wheat.Spike.Wt',
                            'Wheat.Spike.NConc')

# %% [markdown]
# ## Calculate Ratios

# %%
data.derive(
    "Spike/Stem",
    lambda df:
        df["Wheat.Spike.Wt"] /
        df["Wheat.Stem.Wt"]
)

# %%
data.derive('Wheat.Spike.WtProportion',
                lambda df:
        df["Wheat.Spike.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %%
data.derive('Wheat.Stem.WtProportion',
                lambda df:
        df["Wheat.Stem.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %%
data.derive('Wheat.Leaf.WtProportion',
                lambda df:
        df["Wheat.Leaf.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %%
data.derive('Wheat.Leaf.LiveWtProportion',
                lambda df:
        df["Wheat.Leaf.Live.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %%
data.derive('Wheat.Leaf.DeadWtProportion',
                lambda df:
        df["Wheat.Leaf.Dead.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %%
data.derive('Wheat.Ear.WtProportion',
                lambda df:
        df["Wheat.Ear.Wt"] /
        df["Wheat.AboveGround.Wt"])

# %% [markdown]
# ## Met normalisations

# %%
data.derive('SLA * Radn',
                lambda df:
        df["Wheat.Leaf.SpecificAreaCanopy"] * 
        df['IWeather.Radn.Mean30'])

# %%
data.derive('SLA * MinT',
                lambda df:
        df["Wheat.Leaf.SpecificAreaCanopy"] * 
        df['IWeather.MinT.Mean7'])

# %%
data.derive('SLA * MaxT',
                lambda df:
        df["Wheat.Leaf.SpecificAreaCanopy"] * 
        df['IWeather.MaxT.Mean7'])

# %%
data.derive('SLA * PTQ',
                lambda df:
        df["Wheat.Leaf.SpecificAreaCanopy"] * 
        df['Wheat.Phenology.PTQ.Mean7'])

# %% [markdown]
# ## Agregate sim values

# %%
data.aggregate_sim_values('Wheat.Leaf.StemNumberPerPlant','Wheat.Leaf.StemNumberPerPlant.Final',
                          filter_fn = lambda df: df["Wheat.Phenology.Stage"] > 7.5)

# %%
data.aggregate_sim_values('Wheat.Leaf.StemPopulation','Wheat.Leaf.StemPopulation.Final',
                   filter_fn = lambda df: df["Wheat.Phenology.Stage"] > 7.5)

# %%
data.aggregate_sim_values('Wheat.Stem.Wt','Wheat.Stem.Wt.Anthesis',
                   fn = value_at_stage(8,6.5,8.5))

# %%
data.aggregate_sim_values('Wheat.Spike.Wt','Wheat.Spike.Wt.Anthesis',
                   fn = value_at_stage(8,6.5,8.5))

# %%
data.derive('Wheat.StemPlusSpikeWt.Anthesis',
                lambda df:
            df["Wheat.Stem.Wt.Anthesis"] +
            df["Wheat.Spike.Wt.Anthesis"])

# %%
data.derive('Wheat.StemPlusSpikeWt.Anthesis',
            lambda df: df['Wheat.Stem.Wt.Anthesis'] * 1.4) 
            #  Based on analysis below, spike wt = 0.4 * stem wt at anthesis

# %%
data.derive('Wheat.GrainNoPerGofStem',
            lambda df: df['Wheat.Grain.Number']/df['Wheat.StemPlusSpikeWt.Anthesis'] ) 
            #  Based on analysis below, spike wt = 0.4 * stem wt at anthesis

# %%
data.aggregate_sim_values('Wheat.Phenology.PTQ','Wheat.Phenology.PTQ.Critical',
                   filter_fn = lambda df: (df["Wheat.Phenology.Stage"] > 5.9) & 
                         (df["Wheat.Phenology.Stage"] < 8.1))

# %%
data.aggregate_sim_values('IWeather.Radn','IWeather.Radn.Critical',
                          source = 'pred',
                          fn = SumValues,
                           filter_fn = lambda df: (df["Wheat.Phenology.Stage"] > 5.9) & 
                         (df["Wheat.Phenology.Stage"] < 8.1))

# %%
data.aggregate_sim_values('IWeather.MinT','IWeather.MinT.Critical',
                          source = 'pred',
                          fn = MeanValue,
                           filter_fn = lambda df: (df["Wheat.Phenology.Stage"] > 5.9) & 
                         (df["Wheat.Phenology.Stage"] < 8.1))

# %%
data.aggregate_sim_values('IWeather.MaxT','IWeather.MaxT.Critical',
                          source = 'pred',
                          fn = MeanValue,
                           filter_fn = lambda df: (df["Wheat.Phenology.Stage"] > 5.9) & 
                         (df["Wheat.Phenology.Stage"] < 8.1))

# %%
data.aggregate_sim_values('IWeather.MeanT','IWeather.MeanT.Critical',
                          source = 'pred',
                          fn = MeanValue,
                           filter_fn = lambda df: (df["Wheat.Phenology.Stage"] > 5.9) & 
                         (df["Wheat.Phenology.Stage"] < 8.1))

# %% [markdown]
# # Graphing

# %% [markdown]
# ## functions and settings

# %%
DevCols = {"Spring":"Orange",
           "Winter":"Blue"}

TestSetAlphas = {"WWHI":1.0,
                 "GxExM":0.4,
                 "TestSet":0.1,
                 "FAR":0.1}

TestSetMarkers = {"WWHI":'s',
                 "GxExM":'o',
                 "TestSet":'^',
                 "FAR":'v'}

TestSetSizes = {"WWHI":10,
                 "GxExM":50,
                 "TestSet":100,
                 "FAR":200}

plot_order = {
    'FAR': 0,
    'TestSet': 1,
    'GxExM': 2,
    'WWHI': 3
}


# %% [markdown]
# ## Marker plot xy function

# %%
def add_plot_legend(ax=None):

    if ax is None:
        ax = plt.gca()

    dev_handles = [
        Line2D([0], [0],
               marker='o',
               linestyle='None',
               color=colour,
               markersize=8,
               label=dev)
        for dev, colour in DevCols.items()
    ]

    test_handles = [
        Line2D([0], [0],
               marker=TestSetMarkers[test],
               linestyle='None',
               color='black',
               alpha=TestSetAlphas[test],
               markersize=8,
               label=test)
        for test in TestSetMarkers
    ]

    leg1 = ax.legend(
        handles=dev_handles,
        title="Development",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00)
    )

    ax.add_artist(leg1)

    ax.legend(
        handles=test_handles,
        title="Test Set",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.65)
    )

    plt.subplots_adjust(right=0.8)


# %%
def plotxy_markers(yvar, 
           xvar = 'Wheat.Phenology.Stage',
           filter_fn = None,
           color_by = "DevelopmentType",
           marker_by = 'ProjectGroup',
           color_map = DevCols,
           marker_map = TestSetMarkers,
           alpha_map = TestSetAlphas,
           size_map = TestSetSizes, 
           addLeg = True,
           xmin = 2, xmax = 12,
           ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    filtered_data = data.obs
    if filter_fn:
        Mask = data.filter(filter_fn)
        filtered_data = data.obs.loc[Mask,:]
    
    for m in marker_map.keys():
        plotData = filtered_data.loc[data.obs.loc[:,marker_by] == m,[color_by, xvar,yvar]].dropna()
        color_ix =  plotData.loc[:,color_by]
        colors = color_ix.map(lambda c: color_map[c])
        marker = marker_map[m]
        alpha = alpha_map[m]
        size = size_map[m]
        ax.scatter(plotData[xvar],plotData[yvar],c=colors, marker=marker, alpha = alpha, s = size)
    if addLeg:
        add_plot_legend(ax)
    if xmax and xmin:
        ax.set_xlim(xmin,xmax)
    ax.set_ylabel(yvar)
    ax.set_xlabel(xvar)
    return ax


# %% [markdown]
# ## Experiment plot function

# %%
def plotxy(yvar,
            xvar = "Wheat.Phenology.Stage",
            filter_fn = None,
            color_by = 'Experiment', 
            addLeg = True,
            ncols=np.nan,
            xmin = 2, xmax = 12,
           ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    cpos=1
    mpos=1
    
    Mask = pd.Series(True, index=data.obs.index)
    if filter_fn:
        Mask = data.filter(filter_fn)
    
    plotData = data.obs.loc[Mask,[color_by,xvar,yvar]].dropna()
        
    groups = plotData.loc[:,color_by].drop_duplicates()
    for g in groups:
        subPlotData = plotData.loc[plotData.Experiment==g,:]
        xdata = pd.to_numeric(subPlotData.loc[:,xvar])
        ydata = pd.to_numeric(subPlotData.loc[:,yvar])
        ax.plot(xdata,ydata,Markers[mpos],color=Colors[cpos],label=g)
        cpos+=1
        mpos+=1
        if mpos>13:
            mpos=1
        if cpos>28:
            cpos=1
    if addLeg == True:
        if np.isnan(ncols):
             ncols = np.ceil(groups.size/17)
        ax.legend(bbox_to_anchor=(1.15, 1),numpoints=1,ncols=ncols)
    if xmax and xmin:
        ax.set_xlim(xmin,xmax)
    ax.set_ylabel(yvar)
    ax.set_xlabel(xvar)
    return ax


# %% [markdown]
# ## Pannel for variable

# %%
def plot_panel(
    yvar,
    xvars,
    ncols=3,
    markers=False,
    **kwargs
):

    nplots = len(xvars)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5*ncols, 4*nrows),
        constrained_layout=True
    )

    axes = np.array(axes).flatten()

    for ax, xvar in zip(axes, xvars):
        if markers == False:
            plotxy(yvar=yvar,xvar=xvar,ax=ax,addLeg=False,xmax=None,**kwargs)
        else:
            plotxy_markers(yvar=yvar,xvar=xvar,ax=ax,xmax=None,addLeg=False,**kwargs)

        ax.text(0.05,0.95,xvar,transform=ax.transAxes)

    # Hide unused panels
    for ax in axes[nplots:]:
        ax.set_visible(False)

    return fig


# %% [markdown]
# ## Pannel for experiment

# %%
def plotxy_experments(yvar,
                      xvar = 'Wheat.Phenology.Stage',
                      ncols = 4):
    plotData = data.obs.loc[:,['Experiment','Simulation.Name',xvar,yvar]].dropna(how='any')
    experiments = plotData.Experiment.drop_duplicates()
    nplots = len(experiments)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3*ncols, 3*nrows),
        constrained_layout=True
    )
    ymax = plotData.loc[:,yvar].max()*1.1
    ymin = plotData.loc[:,yvar].min()*0.9
    xmax = plotData.loc[:,xvar].max()*1.1
    xmin = plotData.loc[:,xvar].min()*0.9
    axes = np.array(axes).flatten()
    for ax, e in zip(axes, experiments):
        setdata = plotData.loc[plotData.Experiment==e,:]
        sims = setdata.loc[:,'Simulation.Name'].drop_duplicates()
        colors = [Colors[x] for x in range(1,len(sims)+1)]
        cmap = dict(zip(sims,colors))
        colser = [cmap[x] for x in setdata.loc[:,'Simulation.Name']]
        ax.scatter(setdata[xvar],setdata[yvar],c=colser,s=100) 
        ax.text(0.05,0.95,e,transform=ax.transAxes)
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
    
    # Hide unused panels
    for ax in axes[nplots:]:
        ax.set_visible(False)


# %% [markdown]
# ## calculated means then graph xy

# %%
def first_nonblank(s):
    s = s.replace("", np.nan).dropna()
    return s.iloc[0] if len(s) else np.nan

def xyMeans(xvar,yvar,    
    ncols=2,
    addLeg = True,
    cult_cols=True):

    vars = [xvar,yvar,'Simulation.Name','Experiment','Wheat.SowingData.Cultivar']
    
    All = data.obs.loc[:,vars]
    means = All.groupby('Simulation.Name',as_index=False).agg({
        xvar: "mean",
        yvar: "mean",
        "Experiment": first_nonblank,
        "Wheat.SowingData.Cultivar": first_nonblank}).dropna()
    
    fig, ax = plt.subplots()
    cpos=1
    mpos=1
    experiments = means.Experiment.drop_duplicates()
    for e in experiments:
        plotData = means.loc[means.Experiment==e,:]
        xdata = pd.to_numeric(plotData.loc[:,xvar])
        ydata = pd.to_numeric(plotData.loc[:,yvar])
        if cult_cols==False:
            plt.plot(xdata,ydata,Markers[mpos],color=Colors[cpos],label=e)
            cpos+=1
            mpos+=1
            if mpos>13:
                mpos=1
            if cpos>28:
                cpos=1
        else:
            cultivars =  plotData.loc[:,"Wheat.SowingData.Cultivar"]
            default_color = "lightgrey"
            colors = cultivars.map(
                lambda c: DevCols[DevMap[c]]
                if c in DevMap
                else default_color)
            e_marker = TestSetMarkers[TestSetMap[e]]
            e_alpha = TestSetAlphas[TestSetMap[e]]
            e_size = TestSetSizes[TestSetMap[e]]
            plt.scatter(xdata,ydata,c=colors, marker=e_marker,alpha = e_alpha,s = e_size)
            add_plot_legend(ax)
            addLeg = False
    if addLeg == True:
        if np.isnan(ncols):
             ncols = np.ceil(experiments.size/17)
        plt.legend(bbox_to_anchor=(1.15, 1),numpoints=1,ncols=ncols)
    plt.ylabel(yvar)
    plt.xlabel(xvar)


# %% [markdown]
# # Spike Wt

# %% [markdown]
# ## Spike/Stem

# %%
plotxy_markers("Spike/Stem")
plt.plot([5.8,6.0,7.0,8.0,11.0],
         [0.0,.02,.35,.45,.45],'-')

# %%
plotxy("Spike/Stem")
plt.plot([5.8,6.0,7.0,8.0,11.0],
         [0.0,.02,.35,.4,.4],'-')

# %% [markdown]
# ## Spike/totalDM

# %%
plotxy_markers('Wheat.Spike.WtProportion')
plt.plot([3.0,5.5, 6,7.0,8.0],
         [0,0,0.04,.22,.22],'-')
plt.plot([3.0,5.5, 6,7.0,8.0],
         np.multiply([0,0,0.04,.22,.22],0.75),'-')

# %%
plotxy('Wheat.Spike.WtProportion')
plt.plot([3.0,5.5, 6,7.0,8.0],
         [0,0,0.04,.22,.22],'-')
plt.plot([3.0,5.5, 6,7.0,8.0],
         np.multiply([0,0,0.04,.22,.22],0.75),'-')

# %% [markdown]
# # Stem Wt

# %% [markdown]
# ## proportion

# %%
plotxy_markers('Wheat.Stem.WtProportion')
plt.plot([3.0,5.0, 6.0,8.0],
         [0.0,0.36,.65,.65],'-')
plt.plot([3.0,5.0, 6.0,8.0],
         np.multiply([0.0,0.36,.65,.65],0.7),'-')

# %%
plotxy('Wheat.Stem.WtProportion')
plt.plot([3.0,5.0, 6.0,8.0],
         [0.0,0.36,.65,.65],'-')
plt.plot([3.0,5.0, 6.0,8.0],
         np.multiply([0.0,0.36,.65,.65],0.7),'-')
plt.ylim(0,.9)

# %% [markdown]
# ## Allometric

# %%
plotxy_markers('Wheat.Stem.Wt',
               xvar='Wheat.AboveGround.Wt',
               filter_fn=lambda df: df["Wheat.Phenology.Stage"] < 8.5,
              xmax = 3000)
xs = range(0,2800,10)
const = .135
power = 1.2
ys = [const * np.power(x,power) for x in xs]
plt.plot(xs,ys,'-')

# %% [markdown]
# # Leaf Wt

# %% [markdown]
# ## Total

# %%
plotxy_markers('Wheat.Leaf.WtProportion')
plt.plot([3.0,4.0,5.0,6.0,8.0],
         [1,.9,.7,.3,.15],'-')

# %%
plotxy_markers('Wheat.Leaf.Wt',
              xvar = 'Wheat.Phenology.AccumulatedTT',
              xmax = 3100)

# %%
plotxy_markers('Wheat.Leaf.Wt')

# %% [markdown]
# ## Live

# %%
plotxy_markers('Wheat.Leaf.LiveWtProportion')
plt.plot([3.0,4.0,5.0,6.0,8.0],
         [1,.9,.7,.3,.15],'-')

# %%
plotxy_markers('Wheat.Leaf.Live.Wt',
              xvar = 'Wheat.Phenology.AccumulatedTT',
              xmax = 3100)

# %%
plotxy_markers('Wheat.Leaf.Live.Wt')

# %% [markdown]
# ## Dead Leaf

# %%
plotxy_markers('Wheat.Leaf.DeadWtProportion')

# %%
plotxy_markers('Wheat.Leaf.Dead.Wt',
              xvar = 'Wheat.Phenology.AccumulatedTT',
              xmax = 3100)

# %%
plotxy_markers('Wheat.Leaf.Dead.Wt')

# %% [markdown]
# # Ear Wt

# %%
plotxy_markers('Wheat.Ear.WtProportion')
plt.plot([3.0,5.8,7.0,8.0,9.0,10,11],
         [0,0,.15,.2,.25,0.6,0.6],'-')

# %% [markdown]
# # Leaf Area Index

# %%
plotxy_markers('Wheat.Leaf.LAI',
              xvar = 'Wheat.Phenology.AccumulatedTT',
              xmax = 3100)

# %%
plotxy_markers('Wheat.Leaf.LAI')

# %% [markdown]
# # Specific Leaf Area

# %% [markdown]
# ## Raw

# %%
plotxy_markers('Wheat.Leaf.SpecificAreaCanopy')
plt.ylim(0,0.04)

# %%
plotxy('Wheat.Leaf.SpecificAreaCanopy')
plt.ylim(0,0.04)

# %%
plotxy_markers('Wheat.Leaf.SpecificAreaCanopy',
              xvar= 'Wheat.Phenology.AccumulatedTT', 
              xmax = 3100)
plt.ylim(0,0.04)

# %%
RMeanVars = [
'Wheat.Phenology.PTQ.Mean30',
'Wheat.Phenology.ThermalTime.Mean30',
'IWeather.MinT.Mean30',
'IWeather.MaxT.Mean30',
'IWeather.MeanT.Mean30',
'IWeather.Radn.Mean30'
]

fig = plot_panel(
    'Wheat.Leaf.SpecificAreaCanopy',
    RMeanVars,
    ncols=2,
    markers=True)


# %% [markdown]
# ## Radn normed 

# %%
plotxy_markers('SLA * Radn')
plt.ylim(0,0.8)

# %%
plotxy_markers('SLA * Radn',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.8)

# %%
plotxy('SLA * Radn',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.8)

# %% [markdown]
# ## MinT normed 

# %%
plotxy_markers('SLA * MinT')
plt.ylim(0,0.6)

# %%
plotxy_markers('SLA * MinT',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.6)

# %%
plotxy('SLA * MinT',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.6)

# %% [markdown]
# ## MaxT normed 

# %%
plotxy_markers('SLA * Radn')
plt.ylim(0,0.8)

# %%
plotxy_markers('SLA * MaxT',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.8)

# %%
plotxy('SLA * MaxT',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.8)

# %% [markdown]
# ## PTQ normed 

# %%
plotxy_markers('SLA * PTQ')
plt.ylim(0,0.1)

# %%
plotxy_markers('SLA * PTQ',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.1)

# %%
plotxy('SLA * PTQ',
              xvar='Wheat.Phenology.AccumulatedTT',
              xmax=3100)
plt.ylim(0,0.1)

# %% [markdown]
# ## All experiments

# %%
plotxy_experments('Wheat.Leaf.SpecificAreaCanopy')

# %% [markdown]
# # Stem Number

# %% [markdown]
# ## Stem Number per plant

# %%
plotxy_markers('Wheat.Leaf.StemNumberPerPlant')

# %%
plotxy('Wheat.Leaf.StemNumberPerPlant')

# %%
plotxy_experments('Wheat.Leaf.StemNumberPerPlant')

# %% [markdown]
# ## Stem Population

# %%
plotxy_markers('Wheat.Leaf.StemPopulation')

# %%
plotxy('Wheat.Leaf.StemPopulation')

# %%
plotxy_experments('Wheat.Leaf.StemPopulation')

# %% [markdown]
# ## relationships

# %%
xyMeans(
xvar = 'Wheat.Population',
yvar = 'Wheat.Leaf.StemNumberPerPlant.Final',)
xs = range(40,300)
ys = [1000/(x+0) for x in xs]
plt.plot(xs,ys,'-',color='k',label='y=600/x')

# %%
xyMeans(
xvar = 'Wheat.Population',
yvar = 'Wheat.Leaf.StemNumberPerPlant.Final',
cult_cols=False)
xs = range(40,300)
ys = [1000/(x+0) for x in xs]
plt.plot(xs,ys,'-',color='k',label='y=600/x')

# %%
xyMeans(
xvar = 'Wheat.Population',
yvar = 'Wheat.Leaf.StemPopulation',
cult_cols=False)
xs = [0,300]
ys = [600,600]
plt.plot(xs,ys,'-',color='k',label='y=x')

# %%
xyMeans(
xvar = 'Wheat.Stem.Wt.Anthesis',
yvar = 'Wheat.Leaf.StemPopulation',
cult_cols=False)

# %%
xyMeans(
xvar = 'Wheat.Stem.Wt.Anthesis',
yvar = 'Wheat.Leaf.StemNumberPerPlant.Final',
cult_cols=True)

# %% [markdown]
# # Organ N content

# %% [markdown]
# ## Leaf

# %%
plotxy_markers('Wheat.Leaf.Live.NConc')
plt.plot([3.0,5.0,5.5,9.5,11.0],
         [0.055,.055,.045,.035,.005],'-',color='k')

# %%
plotxy_markers('Wheat.Leaf.Dead.NConc')
plt.plot([3.0,5.0,5.5,9.5,11.0],
         [0.055,.055,.045,.035,.005],'-',color='k')

# %% [markdown]
# ## Stem

# %%
plotxy_markers('Wheat.Stem.NConc')
plt.plot([3.0,4.5,6.0,9.5,11.0],
         [0.055,.055,.02,.012,.005],'-',color='k')

# %%
plotxy('Wheat.Stem.NConc')
plt.plot([3.0,4.5,6.0,9.5,11.0],
         [0.055,.055,.02,.012,.005],'-',color='k')

# %% [markdown]
# ## Ear

# %%
plotxy('Wheat.Ear.NConc')
plt.plot([6.0,11.0],
         [0.02,.02],'-',color='k')
plt.ylim(0,.08)

# %%
plotxy('Wheat.Spike.NConc')
plt.plot([6.0,8.5,10.2],
         [0.024,.024,.005],'-',color='k')
plt.ylim(0,.03)

# %% [markdown]
# ## AboveGround

# %%
plotxy('Wheat.AboveGround.NConc',xvar='Wheat.AboveGround.Wt',xmin=0,xmax=3100,ncols=2)
def funct(b1,b2,b3,x):
    return b1 + np.exp(b2+(b3*x))
xs = range(0,3000,10)
ys = [funct(0.81,1.68,-0.00152,x)/100 for x in xs]                       
plt.plot(xs,ys,'-',color='k')
ys = [funct(0.35,1.55,-0.00738,x)/100 for x in xs]                       
plt.plot(xs,ys,'--',color='k')

# %%
plotxy_markers('Wheat.AboveGround.NConc',xvar='Wheat.AboveGround.Wt',xmin=0,xmax=3100)
def funct(b1,b2,b3,x):
    return b1 + np.exp(b2+(b3*x))
xs = range(0,3000,10)
ys = [funct(0.81,1.68,-0.00152,x)/100 for x in xs]                       
plt.plot(xs,ys,'-',color='k')
ys = [funct(0.35,1.55,-0.00738,x)/100 for x in xs]                       
plt.plot(xs,ys,'--',color='k')

# %% [markdown]
# ## Grain

# %%
plotxy_markers('Wheat.Grain.NConc',xvar='Wheat.Grain.Wt',xmin=0,xmax=3100)

# %%
plotxy_markers('Wheat.Grain.NConc',xvar='Wheat.Grain.Size',xmin=0,xmax=3100)

# %% [markdown]
# # Grain Number

# %%
plotxy_markers('Wheat.AboveGround.NConc',xvar='Wheat.AboveGround.Wt',xmin=0,xmax=3100)

# %%
xyMeans(
xvar = 'Wheat.StemPlusSpikeWt.Anthesis',
yvar = 'Wheat.Grain.Number',
cult_cols=True)
xs=[0,1600]
plt.plot(xs,np.multiply(xs,26),'-')
plt.plot(xs,np.multiply(xs,26*1.4),'--')
plt.plot(xs,np.multiply(xs,26*.6),'--')
plt.ylim(0,40000)

# %%
xyMeans(
xvar = 'Wheat.StemPlusSpikeWt.Anthesis',
yvar = 'Wheat.Grain.Number',
cult_cols=False)
xs=[0,1600]
plt.plot(xs,np.multiply(xs,20),'-')
plt.plot(xs,np.multiply(xs,26),'--')
plt.plot(xs,np.multiply(xs,12),'--')
plt.ylim(0,40000)

# %%
CMeanVars = [
'Wheat.Phenology.PTQ.Critical',
'IWeather.MinT.Critical',
'IWeather.MaxT.Critical',
'IWeather.MeanT.Critical',
'IWeather.Radn.Critical'
]

fig = plot_panel(
    'Wheat.Grain.Number',
    CMeanVars,
    ncols=2,
    markers=True)

# %%
CMeanVars = [
'Wheat.Phenology.PTQ.Critical',
'IWeather.MinT.Critical',
'IWeather.MaxT.Critical',
'IWeather.MeanT.Critical',
'IWeather.Radn.Critical'
]

fig = plot_panel(
    'Wheat.GrainNoPerGofStem',
    CMeanVars,
    ncols=2,
    markers=True)
