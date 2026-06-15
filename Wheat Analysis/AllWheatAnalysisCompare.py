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

# %%
import subprocess
import os
import shutil
import sqlite3
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MathsUtilities as MUte
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import mplcursors
# #%matplotlib widget
# %matplotlib inline

# %% [markdown]
# # Constants

# %%
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

Markers = {1: 'o',
 2: '^',
 3: 's',
 4: '*',
 5: '>',
 6: 'v',
 7: '+',
 8: 'X',
 9: '<',
 10: 'p',
 11: '8',
 12: 'd',
 13:'P',
 14:'D',
 15:'o',
 16:'^'}

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

SensibilityFolders = ['CO2AndTranspirationEfficiency',
'CO2AndTemperatureInteractions',
'ProteinAccumulation',
'LeafAppearance',
'TerminalWaterStress',
'DetailedDynamics']

branches = {"master":"master",#"master", 
            "dev":"WinterCerealWheatRelease", 
            "working":"FittingWheat"}   # branches to test
            
simulation_files = [
    r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\GxExM\GxExM.apsimx',
    r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Wheat.apsimx',
    #r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\FAR\FAR.apsimx',
    r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Pask\PaskExperiments.apsimx',
    #r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Dookie2024\Dookie2024.apsimx',
    #r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Dookie2024\WaggaWagga2024.apsimx',
     #r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Dookie2024\GrassPatch2024.apsimx',
     #r'C:\GitHubRepos\ApsimX\Tests\Validation\Wheat\Dookie2024\GrassPatch2025.apsimx'
]

apsim_exe = r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe"  # adjust path to APSIM executable
msbuild_path = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
apsimSolution_path = r"C:\GitHubRepos\ApsimX\ApsimX.sln"
repo_path = r"C:\GitHubRepos\ApsimX"

# %% [markdown]
# # Set Branch and Run Simulation sets - Functions declaired

# %%
apsim_exe = r"C:\GitHubRepos\ApsimX\bin\Release\net8.0\Models.exe"  # adjust path to APSIM executable
msbuild_path = r"C:\Program Files\Microsoft Visual Studio\2022\Professional\MSBuild\Current\Bin\MSBuild.exe"
apsimSolution_path = r"C:\GitHubRepos\ApsimX\ApsimX.sln"
repo_path = r"C:\GitHubRepos\ApsimX"

def set_branch_and_build(branch):
    try:
        # Checkout branch
        subprocess.run(
            ["git", "checkout", branches[branch]],
            cwd=repo_path,   # ensure you’re in the repo
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git checkout failed for branch {branch}: {e}")

    try:
        # Force rebuild to avoid cached binaries
        subprocess.run(
            ["dotnet", "build", apsimSolution_path, "-c", "Release"],
            cwd=repo_path,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Build failed for branch {branch}: {e}")
    
def run_branch(branch, sim_file):
    print(f"\n=== Running {sim_file} on branch: {branch} ===")

    db_file = os.path.splitext(sim_file)[0] + ".db"
    branch_db_file = os.path.splitext(sim_file)[0] + f"_{branch}.db"

    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Deleted old {db_file}")

    # Pass relative path to APSIM
    rel_sim_file = os.path.relpath(sim_file, repo_path)

    result = subprocess.run(
        [apsim_exe, rel_sim_file],
        cwd=repo_path,
        env=os.environ,
        capture_output=True,
        text=True
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Simulation failed for {branch}: {result.stderr}")

    shutil.copyfile(db_file, branch_db_file)
    print(f"Saved results to {branch_db_file}")


# %% [markdown]
# # Run all branches

# %%
# for branch in branches.keys():
#     set_branch_and_build(branch)
#     for sim_file in simulation_files:
#         run_branch(branch, sim_file)

# %% [markdown]
# # Run Working branch

# %%
branch = "working"
set_branch_and_build(branch)
for sim_file in simulation_files:
    run_branch(branch, sim_file)


# %% [markdown]
# # Get Harvest predictions and observations - Functions

# %%
#Set up simulations index table
def GetSimulationData(sim_files, branch):
    allSimulations = {}
    for sim_file in sim_files:
        branch_db_file = os.path.splitext(sim_file)[0] + f"_{branch}.db"
        if not os.path.exists(branch_db_file):
            print(f"Missing DB for {sim_file} ({branch})")
            continue

        try:
            with sqlite3.connect(branch_db_file) as con:
                Simulations = pd.read_sql("SELECT * FROM _Simulations", con)
        except Exception as e:
            print(f"Error reading {branch_db_file}: {e}")
            continue

        Simulations.set_index('ID', inplace=True)
        Simulations.sort_index(axis=0, inplace=True)
        allSimulations[os.path.basename(sim_file)] = Simulations

    if allSimulations:
        return pd.concat(allSimulations.values(),
                         keys=allSimulations.keys(),
                         names=['SimulationFile', 'SimulationID'])
    else:
        return pd.DataFrame()

SimulationData = {}
for branch in branches.keys():
    SimulationData[branch] = GetSimulationData(simulation_files,branch)


# %%
def GetHarvestPred(sim_files, branch, SimulationData):
    allHarvestPred = {}
    for sim_file in sim_files:
        branch_db_file = os.path.splitext(sim_file)[0] + f"_{branch}.db"
        if not os.path.exists(branch_db_file):
            print(f"Missing DB for {sim_file} ({branch})")
            continue

        try:
            with sqlite3.connect(branch_db_file) as con:
                HarvestPred = pd.read_sql("SELECT * FROM HarvestReport", con)
        except Exception as e:
            print(f"Error reading {branch_db_file}: {e}")
            continue

        fileName = os.path.basename(sim_file)

        # Attach simulation names from metadata
        HarvestPred['SimulationName'] = [
            SimulationData.loc[(fileName, x), 'Name']
            for x in HarvestPred['SimulationID']
        ]

        # Ensure FolderName exists
        if 'FolderName' not in HarvestPred.columns:
            HarvestPred['FolderName'] = ''

        # Filter out sensibility tests
        HarvestPred = HarvestPred.loc[
            ~HarvestPred['FolderName'].isin(SensibilityFolders)
        ].copy()

        # Clean up cultivar names
        if 'Wheat.SowingData.Cultivar' in HarvestPred.columns:
            HarvestPred['Wheat.SowingData.Cultivar'] = HarvestPred['Wheat.SowingData.Cultivar'].str.title()

        # Replace missing Experiment values
        mask = HarvestPred['Experiment'].isna()
        HarvestPred.loc[mask, 'Experiment'] = HarvestPred.loc[mask, 'FolderName']

        # Drop all‑NaN columns
        HarvestPred.dropna(how='all', axis=1, inplace=True)
        HarvestPred.set_index('SimulationName',inplace=True)

        allHarvestPred[fileName] = HarvestPred

    if allHarvestPred:
        AllHarvestPredData = pd.concat(
            allHarvestPred.values(),
            keys=allHarvestPred.keys(),
            names=['FileName']
        )
        #AllHarvestPredData['FileName'] = AllHarvestPredData.index.get_level_values(0)
        AllHarvestPredData.rename(columns={'Simulation.Name': 'SimulationName'})
        return AllHarvestPredData
    else:
        return pd.DataFrame()
    
HarvestPred = {}
for branch in branches.keys():
    HarvestPred[branch] = GetHarvestPred(simulation_files, branch, SimulationData[branch])


# %%
#Set up additional index table
def unique_sorted(series, placeholder=None):
    vals = series.drop_duplicates().values
    if placeholder is None:
        return sorted([x for x in vals if x is not None])
    else:
        return sorted([str(x) if x is not None else placeholder for x in vals])

def makeLists(HarvestPred, branches):
    Latitudes, Longitudes, Countries, Regions, States, Cultivars, Experiments, Folders, Files = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for branch in branches.keys():
        Latitudes[branch]   = unique_sorted(HarvestPred[branch]['IWeather.Latitude'])
        Longitudes[branch]  = unique_sorted(HarvestPred[branch]['IWeather.Longitude'])
        Countries[branch]   = unique_sorted(HarvestPred[branch]['LocationInfo.Script.Country'], placeholder="Unknown")
        Regions[branch]     = unique_sorted(HarvestPred[branch]['LocationInfo.Script.Region'], placeholder="Unknown")
        States[branch]      = unique_sorted(HarvestPred[branch]['LocationInfo.Script.State'], placeholder="Unknown")
        Cultivars[branch]   = unique_sorted(HarvestPred[branch]['Wheat.SowingData.Cultivar'])
        Experiments[branch] = unique_sorted(HarvestPred[branch]['Experiment'], placeholder="Unknown")
        Folders[branch]     = unique_sorted(HarvestPred[branch]['FolderName'])
        Files[branch]       = unique_sorted(HarvestPred[branch].index.get_level_values(0)) 
    return Latitudes, Longitudes, Countries, Regions, States, Cultivars, Experiments, Folders, Files

Latitudes, Longitudes, Countries, Regions, States, Cultivars, Experiments, Folders, Files = makeLists(HarvestPred, branches)


# %%
def get_APSIMdb(branch, sim_file, table):
    branch_db_file = os.path.splitext(sim_file)[0] + f"_{branch}.db"
    if not os.path.exists(branch_db_file):
        print(f"Missing DB for {sim_file} ({branch})")

    fileName = os.path.basename(sim_file)

    try:
        with sqlite3.connect(branch_db_file) as con:
            APSIMdataTable = pd.read_sql("SELECT * FROM " + table, con)
    except Exception as e:
        print(f"Error reading {branch_db_file}: {e}")
    return APSIMdataTable


# %%
SowIndices = ['IWeather.Latitude',
    'IWeather.Longitude',
    'LocationInfo.Script.Country',
    'LocationInfo.Script.Region',
    'LocationInfo.Script.State',
    'Wheat.SowingData.Cultivar',
    'Experiment',
    'FolderName']


def GetObserved(sim_files, branch, simulationData, harvestPred):
    # Build lookup once
    sim_lookup = (
        simulationData.reset_index()[['SimulationFile', 'SimulationID', 'Name']]
        .rename(columns={'SimulationFile': 'File', 'Name': 'SimulationName'})
    )
    sim_lookup['SimulationID'] = pd.to_numeric(sim_lookup['SimulationID'], errors='coerce').astype('Int64')

    allObserved = {}
    for sim_file in sim_files:
        fileName = os.path.basename(sim_file)
        observed = get_APSIMdb(branch, sim_file, "Observed").copy()

        observed.drop('SimulationName',axis=1,errors='ignore',inplace=True)
        observed.drop('SimulationName',axis=1,errors='ignore')
        observed['File'] = fileName
        observed['SimulationID'] = pd.to_numeric(observed['SimulationID'], errors='coerce').astype('Int64')

        observed = observed.merge(sim_lookup, on=['File', 'SimulationID'], how='left')

        sowing_info = harvestPred.reset_index()[['SimulationName'] + SowIndices].drop_duplicates()
        observed = observed.merge(sowing_info, on='SimulationName', how='left')
        
        observed['Clock.Today'] = pd.to_datetime(observed['Clock.Today'], errors='coerce').dt.normalize()
        observed['SimulationName'] = observed['SimulationName'].astype('string')

        observed.set_index(['SimulationName', 'Clock.Today'], drop=False, inplace=True)
        observed.sort_index(axis=0, inplace=True)
        observed.sort_index(axis=1, inplace=True)

        allObserved[fileName] = observed

    if allObserved:
        return pd.concat(allObserved.values(), keys=allObserved.keys(), names=['File', 'SimulationName', 'Clock.Today'])
    else:
        return pd.DataFrame()

Observed = {}
for branch in branches.keys():
    Observed[branch] = GetObserved(simulation_files, branch, SimulationData[branch], HarvestPred[branch])

# %%
#Make table of Observations tagged with a 'Wheat.Phenology.CurrentStageName' value equal to 'HarvestRipe' which indicates it is the final harvest value
HarvestObs = {}
for branch in branches.keys():
    HarvestObs[branch] = Observed[branch].loc[Observed[branch].loc[:,'Wheat.Phenology.CurrentStageName']=='HarvestRipe',:].dropna(axis=1,how='all').dropna(axis=0,how='all')

HarvVars = ['Wheat.AboveGround.N',
'Wheat.AboveGround.NConc',
'Wheat.AboveGround.Wt',
'Wheat.Ear.N',
'Wheat.Ear.NConc',
'Wheat.Ear.Wt',
'Wheat.Grain.N',
'Wheat.Grain.NConc',
'Wheat.Grain.Number',
'Wheat.Grain.Protein',
'Wheat.Grain.Size',
'Wheat.Grain.Wt',
'Wheat.Leaf.N',
'Wheat.Leaf.StemPopulation',
'Wheat.Leaf.Wt',
'Wheat.Phenology.CAMP.TSHS',
'Wheat.Phenology.EmergenceDAS',
'Wheat.Phenology.FinalLeafNumber',
'Wheat.Phenology.FlagLeafDAS',
'Wheat.Phenology.FloweringDAS',
'Wheat.Phenology.HeadingDAS',
'Wheat.Phenology.MaturityDAS',
'Wheat.Phenology.TerminalSpikeletDAS',
'Wheat.Spike.HeadNumber',
'Wheat.Spike.N',
'Wheat.Spike.Wt',
'Wheat.Stem.N',
'Wheat.Stem.NConc',
'Wheat.Stem.Wt']


# %% [markdown]
# # Make Harvest Obs Pred graphs for all variables - Function defined

# %%
def harvestObsPredGraph():
    HarvMar = {1:'o',2:'x'}
    checkObsPred = pd.DataFrame()
    legpos = {"master":[0.01,0.89],"dev":[0.6,0.2],"working":[.6,0.4]}
    HarvGraphs =  plt.figure(figsize=(20,20))
    pos=1
    obsPred = pd.DataFrame()
    for var in HarvVars:
        ax = HarvGraphs.add_subplot(6,6,pos)
        axmax = 0.01
        cpos = 1
        msv = 3
        alpv = 1

        for branch in branches:
            for file in Files[branch]:
                try:
                    obsPred = pd.DataFrame(HarvestObs[branch].loc[:,var].dropna().copy())
                    obsPred.columns = ['obs']
                    obsPred.obs = pd.to_numeric(obsPred.obs)
                    for s in obsPred.index:
                        try:
                            obsPred.loc[s,'pred'] = pd.to_numeric(HarvestPred[branch].loc[s[:2],var])
                        except:
                            do = 'nothing'
                except:
                    do = 'nothing'

                obsPred.dropna(inplace=True)

                if not obsPred.empty:
                    ax.plot(obsPred.obs, obsPred.pred,
                            HarvMar.get(cpos,'o'),
                            ms=msv, color=Colors[cpos],
                            alpha=alpv, markeredgewidth=1)

            # Regression stats
            if not obsPred.empty and obsPred.pred.dropna().size > 2:
                n = len(obsPred.obs)
                RegStats = MUte.MathUtilities.CalcRegressionStats('',obsPred.pred,obsPred.obs)
                fitR2 = f"{branch}\n$NSE$ = {RegStats.NSE:.3f}\nn = {n}"
                ax.text(legpos[branch][0], legpos[branch][1], fitR2,
                        transform=ax.transAxes, ha='left', va='top', color=Colors[cpos])

            if not obsPred.empty:
                axmax = max(obsPred.obs.max(), obsPred.pred.max(), axmax)

            cpos += 1
            alpv *= 0.9

        ax.text(0.01,0.99,var,transform=ax.transAxes,ha='left',va='top')
        ax.set_ylim(0, axmax*1.05)
        ax.set_xlim(0, axmax*1.05)
        pos += 1


# %% [markdown]
# # Harvest Obs Pred

# %%
harvestObsPredGraph()


# %% [markdown]
# # Get Daily predictions and observations - Functions

# %%
def getDailyPred(sim_files, branch, SimulationData):

    # Build a lookup from SimulationData: (File, SimulationID) -> SimulationName
    sim_lookup = (
        SimulationData.reset_index()[['SimulationFile', 'SimulationID', 'Name']]
        .rename(columns={'SimulationFile': 'File', 'Name': 'SimulationName'})
    )
    # Coerce SimulationID to integer-like dtype to match tables
    sim_lookup['SimulationID'] = pd.to_numeric(sim_lookup['SimulationID'], errors='coerce').astype('Int64')

    frames = []

    for sim_file in sim_files:
        fileName = os.path.basename(sim_file)

        # --- Load DailyReport ---
        dailyPred = get_APSIMdb(branch, sim_file, "DailyReport").copy()

        # Stamp file and coerce types before merging
        dailyPred['File'] = fileName
        dailyPred['SimulationID'] = pd.to_numeric(dailyPred['SimulationID'], errors='coerce').astype('Int64')

        # Merge SimulationName using (File, SimulationID)
        dailyPred = dailyPred.merge(sim_lookup, on=['File', 'SimulationID'], how='left')

        # Ensure FolderName exists
        if 'FolderName' not in dailyPred.columns:
            dailyPred['FolderName'] = ''

        # Filter out sensibility folders
        dailyPred = dailyPred[~dailyPred['FolderName'].isin(SensibilityFolders)]

        # Patch missing Experiment values
        missing_exp = dailyPred['Experiment'].isna()
        dailyPred.loc[missing_exp, 'Experiment'] = dailyPred.loc[missing_exp, 'FolderName']

        # --- Load NDVI info and merge by File/SimulationID/Clock.Today, keeping 'Spectral.NDVI' ---
        try:
            # Normalize keys in dailyPred *before* merging
            dailyPred['Clock.Today'] = pd.to_datetime(dailyPred['Clock.Today'], errors='coerce').dt.normalize()
            dailyPred['SimulationID'] = pd.to_numeric(dailyPred['SimulationID'], errors='coerce').astype('Int64')

            NDVIPred = None
            ndvi_source = None

            # 1) Preferred source: NDVIDailyReport (files 2–5)
            try:
                tmp = get_APSIMdb(branch, sim_file, "NDVIDailyReport").copy()
                NDVIPred = tmp
                ndvi_source = "NDVIDailyReport"
            except Exception as e_ndvi:
                # 2) Fallback: DailyReport (file 1 stores Spectral.NDVI here)
                try:
                    tmp = get_APSIMdb(branch, sim_file, "DailyReport").copy()
                    NDVIPred = tmp
                    ndvi_source = "DailyReport"
                    print(f"[INFO] Using NDVI from DailyReport for {fileName} (no NDVIDailyReport).")
                except Exception as e_daily:
                    print(f"[INFO] Skipping NDVI for {fileName}: NDVIDailyReport err='{e_ndvi}', DailyReport err='{e_daily}'")

            if NDVIPred is not None:
                # Stamp keys & normalize types
                NDVIPred['File'] = fileName
                NDVIPred['SimulationID'] = pd.to_numeric(NDVIPred['SimulationID'], errors='coerce').astype('Int64')
                NDVIPred['Clock.Today'] = pd.to_datetime(NDVIPred['Clock.Today'], errors='coerce').dt.normalize()

                # Align sensibility filter if FolderName exists
                if 'FolderName' in NDVIPred.columns:
                    NDVIPred = NDVIPred[~NDVIPred['FolderName'].isin(SensibilityFolders)]

                # Detect any NDVI-like column and standardize to 'Spectral.NDVI'
                possible_ndvi_cols = ['Spectral.NDVI', 'NDVIModel.Script.NDVI', 'NDVI']
                ndvi_col = next((c for c in possible_ndvi_cols if c in NDVIPred.columns), None)

                if ndvi_col is None:
                    print(f"[WARN] No NDVI column found in {ndvi_source} for {fileName}. Columns: {list(NDVIPred.columns)}")
                else:
                    # Select and rename to Spectral.NDVI for downstream compatibility
                    NDVIPred = NDVIPred[['File', 'SimulationID', 'Clock.Today', ndvi_col]].rename(
                        columns={ndvi_col: 'Spectral.NDVI'}
                    )

                    # Merge with suffixes to avoid accidental overwrites
                    dailyPred = dailyPred.merge(
                        NDVIPred,
                        on=['File', 'SimulationID', 'Clock.Today'],
                        how='left',
                        suffixes=('', '_ndvi')  # if dailyPred already had Spectral.NDVI
                    )

                    # Coalesce: prefer the NDVI values from the merge into a single 'Spectral.NDVI'
                    if 'Spectral.NDVI_ndvi' in dailyPred.columns:
                        if 'Spectral.NDVI' not in dailyPred.columns:
                            dailyPred['Spectral.NDVI'] = np.nan
                        dailyPred['Spectral.NDVI'] = dailyPred['Spectral.NDVI'].fillna(dailyPred['Spectral.NDVI_ndvi'])
                        dailyPred.drop(columns=['Spectral.NDVI_ndvi'], inplace=True)

                    # Diagnostics for this file
                    nn = int(dailyPred['Spectral.NDVI'].notna().sum())
                    tot = int(len(dailyPred))
                    print(f"[Diag] {fileName} ({ndvi_source}): Spectral.NDVI non-null after merge = {nn}/{tot}")

                    # Key match diagnostics (helps catch residual mismatches)
                    try:
                        keys_daily = dailyPred[['File','SimulationID','Clock.Today']].drop_duplicates()
                        keys_ndvi  = NDVIPred[['File','SimulationID','Clock.Today']].drop_duplicates()
                        diag = keys_daily.merge(keys_ndvi, on=['File','SimulationID','Clock.Today'], how='left', indicator=True)
                        print(f"[Diag] {fileName} key match:", diag['_merge'].value_counts().to_dict())
                        nonmatching = diag[diag['_merge'] == 'left_only'].head(5)
                        if not nonmatching.empty:
                            print(f"[Diag] {fileName} examples with no NDVI match:")
                            print(nonmatching.to_string(index=False))
                    except Exception as e_diag:
                        print(f"[Diag] Key diagnostics skipped for {fileName}: {e_diag}")
        except Exception as e:
            print(f"[INFO] NDVI handling block failed for {fileName}: {e}")


        frames.append(dailyPred)

    # --- Combine all simulations and set MultiIndex with SimulationName ---
    AllDailyPredData = (
        pd.concat(frames, ignore_index=True)
          .set_index(['File', 'SimulationName', 'Clock.Today'])
          .sort_index()
    )

    # Diagnostics: how many rows still lack SimulationName?
    missing_names = AllDailyPredData.index.get_level_values('SimulationName').isna().sum()
    if missing_names > 0:
        # Print a few examples to help track any remaining mismatch
        sample = (
            pd.concat(frames, ignore_index=True)
              .loc[lambda d: d['SimulationName'].isna(), ['File', 'SimulationID']]
              .drop_duplicates()
              .head(10)
        )
        print(f"[WARN] {missing_names} rows missing SimulationName after merge. Examples:")
        print(sample.to_string(index=False))

    return AllDailyPredData

# %% [markdown]
# # Read daily predicted data

# %%
# Rebuild DailyPred using the patched function
DailyPred = {}
for branch in branches.keys():
    DailyPred[branch] = getDailyPred(simulation_files, branch, SimulationData[branch])

# %%
DailyObsIndices = [#'Clock.Today',
                'Wheat.Phenology.CurrentStageName',
                'IWeather.Latitude',
                'IWeather.Longitude',
                'LocationInfo.Script.Country',
                'LocationInfo.Script.Region',
                'LocationInfo.Script.State',
                'Wheat.SowingData.Cultivar',
                'Experiment',
                'FolderName']

DailyObsVars = ['Spectral.NDVI',
                'Wheat.AboveGround.N',
                'Wheat.AboveGround.NConc',
                'Wheat.AboveGround.Wt',
                'Wheat.Ear.N',
                'Wheat.Ear.NConc',
                'Wheat.Ear.Wt',
                'Wheat.Grain.N',
                'Wheat.Grain.NConc',
                'Wheat.Grain.Wt',
                'Wheat.Leaf.CoverGreen',
                'Wheat.Leaf.CoverTotal',
                'Wheat.Leaf.Dead.N',
                'Wheat.Leaf.Dead.NConc',
                'Wheat.Leaf.Dead.Wt',
                'Wheat.Leaf.Height',
                'Wheat.Leaf.LAI',
                'Wheat.Leaf.Live.N',
                'Wheat.Leaf.Live.NConc',
                'Wheat.Leaf.Live.StorageWt',
                'Wheat.Leaf.Live.Wt',
                'Wheat.Leaf.N',
                'Wheat.Leaf.SpecificAreaCanopy',
                'Wheat.Leaf.StemNumberPerPlant',
                'Wheat.Leaf.StemPopulation',
                'Wheat.Leaf.Wt',
                'Wheat.Phenology.HaunStage',
                'Wheat.Phenology.Zadok.Stage',
                'Wheat.Spike.N',
                'Wheat.Spike.NConc',
                'Wheat.Spike.Live.StorageWt',
                'Wheat.Spike.Wt',
                'Wheat.Stem.N',
                'Wheat.Stem.NConc',
                'Wheat.Stem.Live.StorageWt',
                'Wheat.Stem.Wt',
                '(Wheat.Leaf.Transpiration + ISoilWater.Es + MicroClimate.PrecipitationInterception)',
                'sum(Soil.Water.MM)',
                'Soil.Water.Volumetric(1)']

DailyObsIndicesAndVars = DailyObsIndices+DailyObsVars

DailyObs = {}
for branch in branches.keys():
    DailyObs[branch] = Observed[branch].loc[Observed[branch].loc[:, 'Clock.Today'].notnull(), DailyObsIndicesAndVars]
    DailyObs[branch].dropna(axis=1,how="all",inplace=True)
    DailyObs[branch].dropna(axis=0,how="all",inplace=True)

# %%
# IndexVars = ['IWeather.MaxT',
# 'IWeather.MinT',
# 'IWeather.Radn',
# 'SimulationID',
# 'Wheat.DaysAfterSowing',
# 'Wheat.Phenology.AccumulatedTT',
# 'Wheat.Phenology.PTQ',
# 'Wheat.Phenology.CurrentPhaseName',
# 'Wheat.Phenology.CurrentStageName',
# 'Wheat.Phenology.Stage']

# # 1. Insert IndexVars into DailyObs
# for branch in branches.keys():

#     pred = DailyPred[branch].reset_index()   # File, SimulationName, Clock.Today
#     obs  = DailyObs[branch].reset_index()    # SimulationName, Clock.Today

#     merged = obs.merge(
#         pred[IndexVars + ['SimulationName', 'Clock.Today']],
#         on=['SimulationName', 'Clock.Today'],
#         how='left'
#     )

#     # Put merged values back into DailyObs
#     available = [c for c in IndexVars if c in merged.columns]
#     DailyObs[branch][available] = merged[available].values



IndexVars = [
    'IWeather.MaxT', 'IWeather.MinT', 'IWeather.Radn', 'SimulationID',
    'Wheat.DaysAfterSowing', 'Wheat.Phenology.AccumulatedTT', 'Wheat.Phenology.PTQ',
    'Wheat.Phenology.CurrentPhaseName', 'Wheat.Phenology.CurrentStageName', 'Wheat.Phenology.Stage'
]

# 1. Insert IndexVars into DailyObs
for branch in branches.keys():
    pred = DailyPred[branch].reset_index()   # File, SimulationName, Clock.Today
    obs  = DailyObs[branch].reset_index()    # SimulationName, Clock.Today

    # --- Normalize merge key dtypes on both sides ---
    pred['Clock.Today'] = pd.to_datetime(pred['Clock.Today'], errors='coerce').dt.normalize()
    obs['Clock.Today']  = pd.to_datetime(obs['Clock.Today'],  errors='coerce').dt.normalize()

    pred['SimulationName'] = pred['SimulationName'].astype('string')
    obs['SimulationName']  = obs['SimulationName'].astype('string')

    # --- Build safe subset of pred for the merge ---
    vars_available_in_pred = [c for c in IndexVars if c in pred.columns]
    pred_subset = pred[vars_available_in_pred + ['SimulationName', 'Clock.Today']]

    # Optional info to see what's missing in pred
    missing_in_pred = [c for c in IndexVars if c not in pred.columns]
    if missing_in_pred:
        print(f"[Info] {branch}: IndexVars missing in pred: {missing_in_pred}")

    # --- Merge and write back ---
    merged = obs.merge(
        pred_subset,
        on=['SimulationName', 'Clock.Today'],
        how='left'
    )

    # Put merged values back into DailyObs (only those present)
    available = [c for c in IndexVars if c in merged.columns]
    DailyObs[branch][available] = merged[available].values

    # --- Diagnostics ---
    cols_in_merged = [c for c in vars_available_in_pred if c in merged.columns]
    matched = (merged[cols_in_merged].notna().any(axis=1).sum()) if cols_in_merged else 0
    print(f"[Diag] {branch}: merged {matched}/{len(merged)} obs rows with at least one IndexVar.")

    diag = obs.merge(
        pred[['SimulationName','Clock.Today']].drop_duplicates(),
        on=['SimulationName','Clock.Today'],
        how='left',
        indicator=True
    )
    print("[Diag] key coverage:", diag['_merge'].value_counts().to_dict())
    no_key_match = diag[diag['_merge'] == 'left_only'].head(5)
    if not no_key_match.empty:
        print("[Diag] examples with no key match:")
        print(no_key_match[['SimulationName','Clock.Today']].to_string(index=False))


# %%
def mode_non_blank(x: pd.Series):
    """Return the most frequent non-blank/non-NaN value in x; np.nan if none."""
    s = x.dropna()
    if s.empty:
        return np.nan
    # Remove empty/whitespace strings
    if s.dtype == object:
        s = s[s.astype(str).str.strip() != ""]
        if s.empty:
            return np.nan
    counts = s.value_counts()
    return counts.index[0] if len(counts) else np.nan

def ensure_unique_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a possibly non-unique MultiIndex (File, SimulationName, Clock.Today)
    to unique rows by aggregating:
      - numeric cols => mean
      - non-numeric cols => first non-blank / non-NaN
    Returns a DataFrame indexed by (File, SimulationName, Clock.Today).
    """

    # Make sure index has the right levels; if not, try to set from columns
    needed = ['File', 'SimulationName', 'Clock.Today']
    if (df.index.nlevels < 3) or (df.index.names[:3] != needed):
        idx_cols = [c for c in needed if c in df.columns]
        if len(idx_cols) == 3:
            df = df.set_index(idx_cols)

    # Already unique? Bail early
    if df.index.is_unique:
        return df

    # Build aggregation dict for numeric vs non-numeric
    num_cols = df.select_dtypes(include='number').columns.tolist()
    # Treat 'object' and 'category' as non-numeric text/categorical
    non_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    agg_dict = {c: 'mean' for c in num_cols}
    for c in non_cols:
        agg_dict[c] = mode_non_blank

    # For any other dtypes (e.g. datetime), take first non-NaN
    other_cols = [c for c in df.columns if c not in num_cols + non_cols]
    for c in other_cols:
        agg_dict[c] = mode_non_blank

    # Group by full index and aggregate
    df_unique = (
        df.groupby(level=['File', 'SimulationName', 'Clock.Today'])
          .agg(agg_dict)
    )

    return df_unique


def _ensure_unique_index(df, keep_cols):
    """
    Ensure df has a unique MultiIndex by grouping on all index levels and taking the first row.
    keep_cols: columns to keep in the grouped result
    """
    if not df.index.is_unique:
        return (
            df[keep_cols]
            .groupby(level=list(range(df.index.nlevels)))
            .first()
        )
    return df[keep_cols]


# %%
# Make DailyObs unique
for branch in branches.keys():
    DailyObs[branch] = ensure_unique_daily(DailyObs[branch])
    DailyPred[branch] = ensure_unique_daily(DailyPred[branch])
    

    for df_name in ("DailyObs", "DailyPred"):
            df = locals()[df_name][branch]
            if df.index.nlevels >= 3 and list(df.index.names[:3]) == ['File','SimulationName','Clock.Today']:
                idx = df.index
                new_idx = pd.MultiIndex.from_arrays(
                    [idx.get_level_values(0),
                     idx.get_level_values(1),
                     pd.to_datetime(idx.get_level_values(2), errors='coerce').normalize()],
                    names=idx.names
                )
                locals()[df_name][branch] = df.copy()
                locals()[df_name][branch].index = new_idx



# %%
# 2. Compute residuals by aligning on a UNIQUE MultiIndex and avoid reindex issues
DailyResidules = {}

for branch in branches.keys():
    obs  = DailyObs[branch]
    pred = DailyPred[branch]

    ind_col = 'Wheat.Phenology.Stage'
    # set intersection for column selection
    keep_obs_cols  = list(set(DailyObsVars + ['Experiment', ind_col]) & set(obs.columns))
    keep_pred_cols = list(set(DailyObsVars + ['Experiment', ind_col]) & set(pred.columns))

    # 1) Enforce unique index on both sides
    obs_u  = _ensure_unique_index(obs,  keep_obs_cols)
    pred_u = _ensure_unique_index(pred, keep_pred_cols)

    # 2) Align on common unique index
    common_idx = obs_u.index.intersection(pred_u.index)
    if len(common_idx) == 0:
        DailyResidules[branch] = pd.DataFrame(
            columns=DailyObsVars + ['Experiment', ind_col]
        ).set_index(obs_u.index[:0])
        continue

    # 3) Compute residuals (Obs - Pred)
    obs_vars  = obs_u.loc[common_idx, DailyObsVars].apply(pd.to_numeric, errors='coerce')
    pred_vars = pred_u.loc[common_idx, DailyObsVars].apply(pd.to_numeric, errors='coerce')
    res = obs_vars - pred_vars  # columns retain original names

    # 4) Carry group and index columns by aligned arrays to avoid reindex
    if 'Experiment' in obs_u.columns:
        res['Experiment'] = obs_u.loc[common_idx, 'Experiment'].to_numpy()
    elif 'Experiment' in pred_u.columns:
        res['Experiment'] = pred_u.loc[common_idx, 'Experiment'].to_numpy()
    else:
        res['Experiment'] = np.nan

    if ind_col in obs_u.columns:
        res[ind_col] = pd.to_numeric(obs_u.loc[common_idx, ind_col], errors='coerce').to_numpy()
    elif ind_col in pred_u.columns:
        res[ind_col] = pd.to_numeric(pred_u.loc[common_idx, ind_col], errors='coerce').to_numpy()
    else:
        res[ind_col] = np.nan

    # 5) Residuals retain the unique MultiIndex
    DailyResidules[branch] = res

# %%
#Make data frame with factor information for each simulation
FactorList = [ 'Experiment',
 'Cm',
 'Date',
 'Irr',
 'Irrig',
 'N',
 'NRate',
 'Nit',
 'P',
 'Popn',
 'Removal',
 'RowSpace',
 'SD',
 'Soil',
 'Sow',
 'SowN',
 'Stubble',
 'TOS',
 'TopN',
 'Treatment',
 'V',
 'Water']

Factors = HarvestPred['master'].loc[:,FactorList].copy()
Factors.set_index('Experiment',append=True,inplace=True)
Factors=Factors.reorder_levels(['FileName','Experiment','SimulationName'])
Factors.sort_index(inplace=True)
CondensedFactors = pd.DataFrame(index = Factors.index,columns = ['fName1','fValue1'])
for s in Factors.index:
    fs = Factors.loc[s,:].dropna().to_dict()
    fCount = 1
    for key, value in fs.items():
        CondensedFactors.loc[s,'fName'+str(fCount)] = key
        CondensedFactors.loc[s,'fValue'+str(fCount)] = value
        fCount +=1
valueLabs = ['fValue1','fValue2','fValue3','fValue4']
indexLabs = ['fIndex1','fIndex2','fIndex3','fIndex4']
CondensedFactors.loc[:,indexLabs]=1

Experiments = list(HarvestPred['master'].loc[:,'Experiment'].drop_duplicates().values)

# # put simulation names in as first factor level for sims that are not in a experiment
for e in Experiments:
    expMask = CondensedFactors.index.get_level_values(1)==e
    fValues1 = CondensedFactors.loc[expMask,'fValue1']
    if True in pd.isna(fValues1.values):
        CondensedFactors.loc[expMask,'fValue1'] = CondensedFactors.loc[expMask,:].index.get_level_values(2)
        
# #assign numeric index to each factor level
def dicMaker(fLevels):
    thisDic = {}
    num = 1
    for f in fLevels:
        thisDic.update({f:num})
        if num <28:
            num+=1
        else:
            num = 1
    return thisDic

for e in Experiments:
    expMask = CondensedFactors.index.get_level_values(1)==e
    for v in valueLabs:
        fLevels = CondensedFactors.loc[expMask,v].drop_duplicates().values
        if False in pd.isna(fLevels):
            levelDic = dicMaker(fLevels)
            CondensedFactors.loc[expMask,v.replace('Value','Index')] = [levelDic[x] for x in CondensedFactors.loc[expMask,v]]

# %% [markdown]
# # Make Daily Data Structures

# %%
MasterDaily = {"obs":DailyObs['master'],"pred":DailyPred['master'],"res":DailyResidules['master']}
DevDaily = {"obs":DailyObs['dev'],"pred":DailyPred['dev'],"res":DailyResidules['dev']}
WorkingDaily = {"obs":DailyObs['working'],"pred":DailyPred['working'],"res":DailyResidules['working']}
DailyPair = {"master":MasterDaily,"dev":DevDaily,"working":WorkingDaily}


# %% [markdown]
# # Make time course graphs - functions defined

# %%
def timeSeriesByTreatmentCompact(
    ind, var, gro,
    legends=True,        # figure-level legend mapping datasets to line styles
    cols_per_row=12,     # up to 12 panels per row
    cell_w=2.2,          # per-panel width (inches)
    cell_h=1.8,          # per-panel height (inches)
    top_margin=0.08,     # figure top margin for spacing
    right_margin=0.86,   # room for right-side legend (0–1 in figure coords)
    yfix=True,           # fix y-axis per experiment block
    xfix=True            # fix x-axis to [1, 113] with 1.0 ticks
):
    """
    Compact figure of time series panels:
    - Panels are per (experiment, treatment) with master/dev/working overlays.
    - Up to `cols_per_row` panels per row; a new block starts for each experiment.
    - If yfix=True: per-experiment fixed y-limits, applied to all panels in that experiment.
    - If xfix=True: x-range fixed to [1, 113] with tick step 1.0; no x-axis labels; x tick labels only on leftmost panels.
    """

    # --- Helpers -------------------------------------------------------------
    def series_min_max(series_like):
        """
        Return scalar (min, max) after coercing to numeric.
        - Treat zeros as missing (to match your plotting replacement of 0 -> NaN in preds).
        - Ignore NaN; return None, None if no valid data.
        """
        try:
            s = pd.to_numeric(series_like, errors='coerce')
            s = s.replace(0, np.nan)
            s = s.dropna()
            if s.empty:
                return None, None
            mn, mx = float(s.min()), float(s.max())
            if not np.isfinite(mn) or not np.isfinite(mx):
                return None, None
            return mn, mx
        except Exception:
            return None, None

    # --- Branch detection ----------------------------------------------------
    candidates = ("master", "dev", "working")
    branches = [dp for dp in candidates if dp in DailyPair]
    if "master" not in branches:
        raise KeyError("Expected 'master' dataset in DailyPair.")

    # --- Base plotting set from master (prefer obs; fallback to res) ---------
    res_master = DailyPair["master"].get("res", pd.DataFrame())
    base_df = DailyPair["master"]["obs"]
    if not ({ind, var, gro} <= set(base_df.columns)):
        base_df = res_master

    plotSet = base_df.dropna(subset=[var, ind])[[ind, var, gro]]

    # --- Experiments with >1 point ------------------------------------------
    groups = [
        g for g in plotSet[gro].dropna().unique()
        if plotSet[plotSet[gro] == g].shape[0] > 1
    ]
    if not groups:
        raise ValueError("No experiments found with >1 point.")

    m_obs  = DailyPair["master"]["obs"]
    m_pred = DailyPair["master"]["pred"]

    # --- Build panels per experiment: mapping g -> list of treatments s ------
    group_to_sims = {}
    total_panels = 0
    for g in groups:
        obs_group = m_obs[m_obs[gro] == g] if gro in m_obs.columns else m_obs
        sims = []
        if hasattr(obs_group.index, "nlevels") and obs_group.index.nlevels > 1:
            sims = list(obs_group.index.get_level_values(1).unique())

        # Keep sims that actually have obs (ind+var) data
        valid_sims = []
        for s in sims:
            try:
                df = m_obs.xs(s, level=1)[[ind, var]].dropna()
                if not df.empty:
                    valid_sims.append(s)
            except Exception:
                pass
        group_to_sims[g] = valid_sims
        total_panels += len(valid_sims)

    if total_panels == 0:
        raise ValueError("No valid (experiment, treatment) panels found.")

    # --- Figure size & GridSpec layout --------------------------------------
    rows_per_group = {
        g: int(np.ceil(len(sims) / cols_per_row)) if sims else 1
        for g, sims in group_to_sims.items()
    }
    total_rows = sum(rows_per_group.values())

    fig_w = max(8.0, cols_per_row * cell_w)   # width in inches
    fig_h = max(6.0, total_rows * cell_h)     # height in inches

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows=total_rows, ncols=cols_per_row, figure=fig)

    # --- Dataset styles ------------------------------------------------------
    styles = {
        "master":   dict(ls="-", lw=1.3,   alpha=0.6),
        "dev":      dict(ls="--",  lw=1.3, alpha=0.8),
        "working":  dict(ls=":",  lw=1.3,   alpha=1.0),
    }

    # --- Plot panels per experiment -----------------------------------------
    start_row = 0
    for g in groups:
        sims = group_to_sims[g]
        nrows_block = rows_per_group[g]
        if not sims:
            start_row += nrows_block
            continue

        # Compute per-experiment Y limits ONCE
        if yfix:
            ymins_all, ymaxs_all = [], []
            for s in sims:
                # obs (master)
                try:
                    mn, mx = series_min_max(m_obs.xs(s, level=1)[var])
                    if mn is not None: ymins_all.append(mn)
                    if mx is not None: ymaxs_all.append(mx)
                except Exception:
                    pass
                # preds (each branch)
                for dp in branches:
                    try:
                        mn, mx = series_min_max(DailyPair[dp]["pred"].xs(s, level=1)[var])
                        if mn is not None: ymins_all.append(mn)
                        if mx is not None: ymaxs_all.append(mx)
                    except Exception:
                        pass

            # If nothing valid, fall back to [0, 1]
            if not ymins_all or not ymaxs_all:
                ymin_g, ymax_g = 0.0, 1.0
            else:
                ymin_g = min(ymins_all)
                ymax_g = max(ymaxs_all)
                # If degenerate or inverted, fix
                if not np.isfinite(ymin_g) or not np.isfinite(ymax_g) or ymax_g <= ymin_g:
                    ymin_g, ymax_g = 0.0, max(1.0, float(ymax_g))
                else:
                    # small padding
                    pad = 0.01 * (ymax_g - ymin_g)
                    ymin_g -= pad
                    ymax_g += pad
        else:
            ymin_g = ymax_g = None  # per-panel autoscale

        # Place each sim panel within this group's block
        for i, s in enumerate(sims):
            r = start_row + (i // cols_per_row)
            c = i % cols_per_row
            ax = fig.add_subplot(gs[r, c])

            # Factor styling (color/marker) per simulation
            scol, smar = '#000000', 'o'
            try:
                nlvls = getattr(CondensedFactors.index, 'nlevels', 1)
                if nlvls > 2:
                    cf = CondensedFactors.xs(s, level=2)
                else:
                    cf = CondensedFactors.xs(s)
                scol = Colors.get(cf.get('fIndex1', 1).values[0], '#000000')
                smar = Markers.get(cf.get('fIndex2', 1).values[0], 'o')
            except Exception:
                pass

            # Observations (master)
            try:
                obs = m_obs.xs(s, level=1)[[ind, var]].dropna()
                ax.plot(obs[ind], obs[var], smar, ms=3.2, color=scol,
                        label=str(s) if isinstance(s, str) else str(s))
            except Exception:
                pass

            # Predictions in each branch
            for dp in branches:
                try:
                    pred = DailyPair[dp]["pred"].xs(s, level=1)[[ind, var]].dropna()
                    pred[var] = pred[var].replace(0, np.nan)
                    st = styles.get(dp, dict(ls="-", lw=1))
                    ax.plot(pred[ind], pred[var], st["ls"], lw=st["lw"],
                            color=scol, alpha=st.get("alpha", 1.0))
                except Exception:
                    pass

            # --- Axis limits and ticks ---
            # Y-limits: APPLY SAME LIMITS FOR ALL PANELS IN THIS EXPERIMENT
            if yfix and (ymin_g is not None and ymax_g is not None):
                ax.set_ylim(ymin_g, ymax_g)
            else:
                # per-panel autoscale, but make sure not to vary via padding
                pass

            # X-limits and ticks
            if xfix:
                ax.set_xlim(1, 11.5)
                ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.set_xlabel("")  # no x-axis label
            ax.tick_params(axis='x', labelbottom=(c == 0), labelsize=7)

            # Cosmetics for small panels
            ax.set_title(str(s), fontsize=8, pad=2)
            ax.tick_params(axis='y', labelsize=7)
            # Reduce y-axis label clutter: show only on first column
            if c == 0:
                ax.set_ylabel(var, fontsize=8)
                ax.tick_params(axis='y', labelleft=True, labelsize=7)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', labelleft=False, labelsize=7)

        # advance row offset for next experiment block
        start_row += nrows_block

    # --- Figure-level legend on the right -----------------------------------
    if legends:
        fig.subplots_adjust(top=1 - top_margin, right=right_margin)
        legend_elems = []
        if "master" in branches:
            legend_elems.append(Line2D([0], [0], color='k', lw=2, ls='-', alpha=0.6, label='Master'))
        if "dev" in branches:
            legend_elems.append(Line2D([0], [0], color='k', lw=1.2, ls='--',  alpha=0.8, label='Dev'))
        if "working" in branches:
            legend_elems.append(Line2D([0], [0], color='k', lw=2, ls=':',   alpha=1.0, label='Working'))
        if legend_elems:
            fig.legend(legend_elems, [le.get_label() for le in legend_elems],
                       loc='center left', bbox_to_anchor=(right_margin + 0.02, 0.5),
                       fontsize=9, title='Dataset', frameon=True)

    plt.tight_layout(rect=[0, 0, right_margin, 1 - top_margin])
    plt.show()
    plt.close(fig)
    return fig


# %%
def residualGraph(ind, var, gro, figsize=(15, 10), legends=True):
    fig = plt.figure(figsize=figsize)

    # Which datasets do we have?
    candidates = ("master", "dev", "working")
    branches = [dp for dp in candidates if dp in DailyPair]  # defensive
    if not branches:
        raise KeyError("No datasets found in DailyPair among: master, dev, working.")

    # --- Helpers -------------------------------------------------------------
    def get_res_series(dp, vname):
        """Return residual series for var from res if present,
        else compute obs - pred via inner-join on the shared keys."""
        res_df = DailyPair[dp].get("res", pd.DataFrame())
        if vname in res_df.columns:
            return res_df[vname].dropna()

        # fallback: use index intersection if both frames carry the same MultiIndex
        obs = DailyPair[dp]["obs"]; pred = DailyPair[dp]["pred"]
        if vname not in obs.columns or vname not in pred.columns:
            return pd.Series(dtype=float)

        # Case 1: both use (File, SimulationName, Clock.Today) index
        if (hasattr(obs.index, "names") and hasattr(pred.index, "names") and
            list(obs.index.names[:3]) == ['File','SimulationName','Clock.Today'] and
            list(pred.index.names[:3]) == ['File','SimulationName','Clock.Today']):
            common_idx = obs.index.intersection(pred.index)
            if len(common_idx) == 0:
                return pd.Series(dtype=float)
            o = pd.to_numeric(obs.loc[common_idx, vname], errors='coerce')
            p = pd.to_numeric(pred.loc[common_idx, vname], errors='coerce')
            return (o - p).dropna()

        # Case 2: fall back to merging on shared columns
        ok_cols = [c for c in ['File','SimulationName','Clock.Today'] if c in obs.columns and c in pred.columns]
        if ok_cols:
            o = obs[ok_cols + [vname]].dropna()
            p = pred[ok_cols + [vname]].dropna()
            merged = o.merge(p, on=ok_cols, how='inner', suffixes=('_obs','_pred'))
            if merged.empty:
                return pd.Series(dtype=float)
            s = pd.to_numeric(merged[f'{vname}_obs'], errors='coerce') - pd.to_numeric(merged[f'{vname}_pred'], errors='coerce')
            return s.dropna()

        return pd.Series(dtype=float)


    def get_xmax_from_branch(dp):
        """Try to derive max x for the index variable. Return None if not available."""
        pred = DailyPair[dp]["pred"]
        obs  = DailyPair[dp]["obs"]

        series = None
        if ind in pred.columns:
            series = pred[ind].dropna()
        elif ind in obs.columns:
            series = obs[ind].dropna()

        if series is None or series.empty:
            return None
        try:
            return series.max()
        except Exception:
            return None

    def is_numeric_scalar(x):
        return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)

    # --- Compute global y-limits from residuals ------------------------------
    res_list = [get_res_series(dp, var) for dp in branches]
    if all(s.empty for s in res_list):
        raise KeyError(f"Residuals for '{var}' are empty or missing in all datasets: {branches}")
    
    branches = [dp for dp, s in zip(branches, res_list) if not s.empty]
    res_list = [s for s in res_list if not s.empty]

    # Safe max/min across possibly empty series
    ymax_candidates = [s.max() for s in res_list if not s.empty]
    ymin_candidates = [s.min() for s in res_list if not s.empty]

    ymax = (max(ymax_candidates) if ymax_candidates else 0) * 1.01
    ymin = (min(ymin_candidates) if ymin_candidates else 0) * 1.01

    absMax = max(ymax, -ymin)
    ymin   = -absMax  # symmetric limits

    # --- Layout --------------------------------------------------------------
    n = len(branches)
    pos = 1

    for dp in branches:
        ax = fig.add_subplot(1, n, pos)

        # Build plotting frame
        res_df  = DailyPair[dp].get("res", pd.DataFrame())
        base_df = res_df

        # We need ind, var, gro in the plotting set; if missing, rebuild from obs/pred
        cols_needed = [ind, var, gro]
        if any(c not in base_df.columns for c in cols_needed):
            obs  = DailyPair[dp]["obs"]
            pred = DailyPair[dp]["pred"]

            missing_in_obs  = [c for c in [ind, var, gro] if c not in obs.columns]
            missing_in_pred = [c for c in [ind, var]      if c not in pred.columns]

            if missing_in_obs:
                raise KeyError(f"{dp}: missing columns in obs: {missing_in_obs}")
            if missing_in_pred:
                # we can still plot obs residuals as obs - pred[var] only if var present;
                # otherwise, we cannot compute residuals
                raise KeyError(f"{dp}: missing columns in pred: {missing_in_pred}")

            align  = obs[[ind, var, gro]].dropna().copy()
            pred_v = pred.reindex(align.index)[[ind, var]]

            # Compute residuals and keep required columns
            align['__res__'] = align[var] - pred_v[var]
            base_df = align.rename(columns={'__res__': var})

        # Filter to rows that actually have ind + var
        plotSet = base_df.dropna(subset=[ind, var])[[ind, var, gro]]
        groups  = plotSet[gro].dropna().unique()

        colPos    = 1
        markerPos = 1

        if dp == "master":
            ax.set_ylabel(f"{var} (Obs - Pred)")
            ax.set_title(var, fontsize=30)

        # plot per group (and optionally per sim if MultiIndex is present)
        for g in groups:
            groupSet = plotSet[plotSet[gro] == g]
            has_multi = hasattr(groupSet.index, 'nlevels') and groupSet.index.nlevels > 1

            sims = groupSet.index.get_level_values(1).unique() if has_multi else [None]

            for s in sims:
                if s is not None and hasattr(res_df, 'xs') and res_df.index.nlevels > 1:
                    try:
                        data = res_df.xs(s, level=1)[[ind, var]].dropna()
                    except Exception:
                        data = groupSet[[ind, var]].dropna()
                else:
                    data = groupSet[[ind, var]].dropna()

                x = data[ind]
                y = data[var]

                ax.plot(
                    x, y,
                    Markers.get(markerPos, 'o') + '-',
                    lw=1, ms=7,
                    color=Colors.get(colPos, '#000000'),
                    label=str(g)
                )

            colPos += 1
            if colPos > 8:
                colPos = 1
                markerPos += 1

        # Regression annotation (based on obs/pred)
        allobs  = DailyPair[dp]["obs"][var].dropna()
        allpred = DailyPair[dp]["pred"].reindex(allobs.index)[var]
        if len(allobs) > 2:
            RegStats = MUte.MathUtilities.CalcRegressionStats('', allpred, allobs)
            txt = f"{dp}\n$NSE$ = {RegStats.NSE:.3f}\nn = {len(allobs)}"
            ax.text(0.05, 0.95, txt, transform=ax.transAxes, ha='left', va='top')

        # y-limits: symmetric
        ax.set_ylim(ymin, absMax)

        # x-limit: compute per axis from data type
        xmax_dp = get_xmax_from_branch(dp)
        if is_numeric_scalar(xmax_dp):
            ax.set_xlim(0, float(xmax_dp) * 1.01)
        else:
            # For datetimes or unknown types, let Matplotlib autoscale
            ax.set_xlim(auto=True)

        ax.set_xlabel(ind)
        pos += 1


    if legends:
        # Use the final axis (last subplot) only
        last_ax = fig.axes[-1]
        handles, labels = last_ax.get_legend_handles_labels()

        # Deduplicate labels while preserving first occurrence
        by_label = {}
        for h, lab in zip(handles, labels):
            if lab not in by_label:
                by_label[lab] = h

        # Make room on the right for legend
        fig.subplots_adjust(right=0.82)

        last_ax.legend(
            list(by_label.values()), list(by_label.keys()),
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # 1.02 pushes just outside the right edge of the axis
            fontsize=12, title='Experiment', frameon=True,
            borderaxespad=0.0
        )


    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


# %%
def makeGraphs(var):
    try:
        fig1 = residualGraph(
            ind='Wheat.Phenology.Stage',
            var=var,
            gro='Experiment',
            figsize=(20, 10))
        
        fig2 = timeSeriesByTreatmentCompact(
            ind='Wheat.Phenology.Stage',
            var=var,
            gro='Experiment')
        
        return None  # success

    except Exception as e:
        return e


# %% [markdown]
# # Make data frame of harvest ObsPre pairs so we can graph with experiment color mark up

# %%
def buildObsPredHarvest(HarvestObs, HarvestPred, HarvVars, branches):
    """
    Build observed–predicted harvest dataframe for each branch.

    Returns
    -------
    ObsPredHarvest : dict
        keyed by branch, values are DataFrames indexed by
        (File, Experiment, SimulationName)
        with columns: ['var', 'obs', 'pred']
    """
    ObsPredHarvest = {}

    for branch in branches.keys():
        records = []

        obs_df_branch = HarvestObs[branch]
        pred_df_branch = HarvestPred[branch]

        for var in HarvVars:
            if var not in obs_df_branch.columns:
                continue
            if var not in pred_df_branch.columns:
                continue

            # Observed harvest values
            obs_var = (
                obs_df_branch[var]
                .dropna()
                .to_frame(name='obs')
            )
            obs_var['obs'] = pd.to_numeric(obs_var['obs'], errors='coerce')

            # Work safely with index levels
            obs_idx = obs_var.index.to_frame(index=False)

            for i, idx_row in obs_idx.iterrows():
                file = idx_row['File']
                sim  = idx_row['SimulationName']

                try:
                    pred_row = pred_df_branch.loc[(file, sim)]

                    pred_val = pd.to_numeric(pred_row[var], errors='coerce')
                    experiment = pred_row['Experiment']

                    records.append({
                        'File': file,
                        'Experiment': experiment,
                        'SimulationName': sim,
                        'var': var,
                        'obs': obs_var.iloc[i]['obs'],
                        'pred': pred_val
                    })

                except KeyError:
                    # No prediction for this (File, SimulationName)
                    continue

        if records:
            df = pd.DataFrame(records)
            df.set_index(['File', 'Experiment', 'SimulationName'], inplace=True)
            ObsPredHarvest[branch] = df
        else:
            ObsPredHarvest[branch] = pd.DataFrame(
                columns=['var', 'obs', 'pred']
            )

    return ObsPredHarvest


# %%
ObsPredHarvest = buildObsPredHarvest(
    HarvestObs=HarvestObs,
    HarvestPred=HarvestPred,
    HarvVars=HarvVars,
    branches=branches
)


# %% [markdown]
# ## Make Harvest ObsPre graph for each branch with experiment color mark up

# %%
def harvestObsPredBySimulationGraph(
    var,
    experiments,
    figsize=(18, 6),
    legends=True
):
    """
    Observed vs Predicted harvest plot (residualGraph-style).

    Panels:
        One per branch (master, dev, working)

    Points:
        One per simulation, coloured/marked by Experiment.

    Interactivity:
        Optional hover tooltips showing SimulationName, Experiment, obs, pred
    """

    # --- Pre-pass: determine global axis limits across all branches ---
    all_vals = []

    for branch in branches.keys():
        if branch not in ObsPredHarvest:
            continue
        df = ObsPredHarvest[branch]
        df = df[df['var'] == var]
        if not df.empty:
            all_vals.append(df['obs'])
            all_vals.append(df['pred'])

    if not all_vals:
        print(f"No data available for variable: {var}")
        return

    all_vals = pd.concat(all_vals).dropna()
    vmin, vmax = all_vals.min(), all_vals.max()

    # --- Figure setup ---
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, len(branches), figure=fig, wspace=0.10)

    ColorList = list(Colors.values())
    MarkerList = [m for m in Markers.values() if m != '']

    n_colors = len(ColorList)
    n_markers = len(MarkerList)

    legend_handles = {}
    scatter_meta = []   # <--- holds metadata for tooltips
    panel = 0

    for branch in branches.keys():
        ax = fig.add_subplot(gs[0, panel])
        panel += 1

        df = ObsPredHarvest.get(branch, pd.DataFrame())
        df = df[df['var'] == var]

        if df.empty:
            ax.set_title(f"{branch} (no data)")
            continue

        #experiments = df.index.get_level_values('Experiment').unique()

        # Build colour/marker cycling
        exp_style = {}
        for i, exp in enumerate(experiments):
            color = ColorList[i % n_colors]
            marker = MarkerList[(i // n_colors) % n_markers]
            exp_style[exp] = (color, marker)

        for exp in experiments:
            sub = df.loc[df.index.get_level_values('Experiment') == exp]

            color, marker = exp_style[exp]

            filled_markers = [
                'o', 's', '^', 'v', '<', '>', 'd', 'D', 'p', 'P', 'h', 'H', '8'
            ]
            edgecolor = 'none' if marker in filled_markers else None

            sc = ax.scatter(
                sub['obs'],
                sub['pred'],
                c=color,
                marker=marker,
                s=25,
                alpha=0.85,
                edgecolors=edgecolor
            )

            # Legend handle
            if legends and exp not in legend_handles:
                legend_handles[exp] = Line2D(
                    [0], [0],
                    marker=marker,
                    linestyle='None',
                    markerfacecolor=color,
                    markeredgecolor='none',
                    markeredgewidth=0,
                    markersize=6,
                    label=exp
                )

        # 1:1 line and shared limits
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal', adjustable='box')

        ax.set_title(branch)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")

    # --- Legend across bottom ---
    if legends and legend_handles:
        n_items = len(legend_handles)
        ncol = min(8, n_items)

        fig.legend(
            handles=list(legend_handles.values()),
            loc='lower center',
            ncol=ncol,
            title='Experiment',
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            handletextpad=0.8,
            columnspacing=1.5,
            labelspacing=0.8,
            borderaxespad=-0.6
        )

        plt.subplots_adjust(bottom=0.32)


# %%
ExptsWithData = ObsPredHarvest['master'].loc[ObsPredHarvest['master'].loc[:,'var'] == 'Wheat.Grain.NConc',:].index.get_level_values(1).drop_duplicates()

# %%
harvestObsPredBySimulationGraph('Wheat.Grain.NConc',ExptsWithData)

# %%
harvestObsPredBySimulationGraph('Wheat.Grain.N',ExptsWithData)

# %%
harvestObsPredBySimulationGraph('Wheat.Grain.Wt',ExptsWithData)

# %%
harvestObsPredBySimulationGraph('Wheat.Grain.Number',ExptsWithData)

# %% [markdown]
# # Leaf.LAI

# %%
var = 'Wheat.Leaf.LAI'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Spectral.NDVI

# %%
var = 'Spectral.NDVI'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.CoverGreen

# %%
var = 'Wheat.Leaf.CoverGreen'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.CoverTotal

# %%
var = 'Wheat.Leaf.CoverTotal'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Live.Wt

# %%
var = 'Wheat.Leaf.Live.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %%
# Leaf.Wt

# %%
var = 'Wheat.Leaf.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.SpecificAreaCanopy

# %%
var = 'Wheat.Leaf.SpecificAreaCanopy'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.StemNumberPerPlant

# %%
var = 'Wheat.Leaf.StemNumberPerPlant'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.StemPopulation

# %%
var = 'Wheat.Leaf.StemPopulation'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # AboveGround.Wt

# %%
var = 'Wheat.AboveGround.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Ear.Wt

# %%
var = 'Wheat.Ear.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Grain.Wt

# %%
var = 'Wheat.Grain.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Dead.Wt

# %%
var = 'Wheat.Leaf.Dead.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Live.StorageWt

# %%
var = 'Wheat.Leaf.Live.StorageWt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Live.Wt

# %%
var = 'Wheat.Leaf.Live.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Wt

# %%
var = 'Wheat.Leaf.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Spike.Live.StorageWt

# %%
var = 'Wheat.Spike.Live.StorageWt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Spike.Wt

# %%
var = 'Wheat.Spike.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Stem.Live.StorageWt

# %%
var = 'Wheat.Stem.Live.StorageWt'
err = makeGraphs(var)
err = makeGraphs('Wheat.Stem.Live.StorageWt')
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Stem.Wt

# %%
var = 'Wheat.Stem.Wt'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # AboveGround.N

# %%
var = 'Wheat.AboveGround.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # AboveGround.NConc

# %%
var = 'Wheat.AboveGround.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Ear.N

# %%
var = 'Wheat.Ear.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Ear.NConc

# %%
var = 'Wheat.Ear.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Grain.N

# %%
var = 'Wheat.Grain.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Grain.NConc

# %%
var = 'Wheat.Grain.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Dead.N

# %%
var = 'Wheat.Leaf.Dead.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Dead.NConc

# %%
var = 'Wheat.Leaf.Dead.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Live.N

# %%
var = 'Wheat.Leaf.Live.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Live.NConc

# %%
var = 'Wheat.Leaf.Live.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.N

# %%
var = 'Wheat.Leaf.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Spike.N

# %%
var = 'Wheat.Spike.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # spike.NConc

# %%
var = 'Wheat.Spike.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Stem.N

# %%
var = 'Wheat.Stem.N'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Stem.NConc

# %%
var = 'Wheat.Stem.NConc'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Phenology.HaunStage

# %%
var = 'Wheat.Phenology.HaunStage'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Leaf.Height

# %%
var = 'Wheat.Leaf.Height'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")

# %% [markdown]
# # Phenology.Zadok.Stage

# %%
var = 'Wheat.Phenology.Zadok.Stage'
err = makeGraphs(var)
if err is not None:
    print(f"{var} failed with error: {err}")
