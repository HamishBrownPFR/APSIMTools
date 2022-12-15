# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import xmltodict, json
import ast
import numbers
# %matplotlib inline

# +
def AddOrModifyCultivars(NewCultivarParams):
    ## Go through and overwrite the parameter values for existing cultivars
    ## or if cultivar is currently listed as an alias remove that. 
    ModifiedCultivarList = []
    CultivarsToModifyOrAdd = NewCultivarParams.index
    ParametersToModifyOrAdd = NewCultivarParams.columns
    Pos = 0
    for model in Wheat['Children'][0]['Children']:
        #find cultivar folder
        if model['Name']  == 'Cultivars':
            #Go into each regonds folder
            CultFolder=model
            CultivarFolderPos = Pos
            Pos2 =0
            for region in CultFolder['Children']:
                #Go into each cultivar
                RegionFolder = region['Children']
                RegionPos = Pos2
                Pos3 = 0
                #First find the Alias list for each cultivar and remove any cultivars that are to be modified
                for cult in RegionFolder:
                    try:
                        Aliai = cult['Children']
                        RetainedAliai = []
                        for a in Aliai:
                            if (a['Name'] not in CultivarsToModifyOrAdd) | (a["$type"] != "Models.Core.Alias, Models"):
                                RetainedAliai.append(a)
                    except:
                        Aliai = 'None'
                    Wheat['Children'][0]['Children'][CultivarFolderPos]['Children'][RegionPos]['Children'][Pos3]['Children'] = RetainedAliai
                    cultName = cult['Name']
                    #Then go through each cultivar and Add or Modity parameter commands
                    ModifiedParameterList = []
                    if cultName in CultivarsToModifyOrAdd:
                        cultCommands = cult['Command']
                        CultivarPos = Pos3
                        #Make a new param list to replace existing
                        NewCommands = []
                        #Overwrite commands we want to modify
                        for com in cultCommands:
                            paramName = com.split('=')[0].replace(' ','')
                            if paramName in ParametersToModifyOrAdd:
                                newCommand = paramName + ' = ' + str(NewCultivarParams.loc[cultName,paramName])
                                NewCommands.append(newCommand)    
                                ModifiedParameterList.append(paramName)
                            #Add params that are not in our modify table back into cultivar params as they were
                            if paramName not in ParametersToModifyOrAdd: 
                                NewCommands.append(com)
                        #Add commands that are not currently in cultivar
                        for paramName in ParametersToModifyOrAdd:
                            if paramName not in ModifiedParameterList:
                                newCommand = paramName + ' = ' + str(NewCultivarParams.loc[cultName,paramName])
                                NewCommands.append(newCommand)    

                        # Replace existing cultivar description with modified one
                        Wheat['Children'][0]['Children'][CultivarFolderPos]\
                        ['Children'][RegionPos]['Children'][CultivarPos]\
                        ['Command'] = NewCommands
                        ModifiedCultivarList.append(cultName)
                    Pos3 +=1
                Pos2+=1
        Pos+=1

    # Create empty folder to put new cultivars into
    CultivarsToAdd = {"$type": "Models.PMF.CultivarFolder, Models",
                      "Name": "Just Added",
                      "Children": []}
    AddedCultivarList = []
    ## Go through and add in cultivars that do not currently exist
    for cultName in CultivarsToModifyOrAdd:
        if cultName not in ModifiedCultivarList:
            comToAdd = {"$type": "Models.PMF.Cultivar, Models",
                      "Command": [],
                      "Name": cultName,
                      }
            for paramName in ParametersToModifyOrAdd:
                pCommand = paramName + ' = ' + str(NewCultivarParams.loc[cultName,paramName])
                comToAdd['Command'].append(pCommand)
            CultivarsToAdd['Children'].append(comToAdd)
            AddedCultivarList.append(cultName)


    ## Add empty folder to put new cultivars into
    Wheat['Children'][0]['Children']\
    [CultivarFolderPos]['Children'].append(CultivarsToAdd)

def AddMaxLAI(CultivarMaxLAR):
    ## Go through and overwrite the parameter values for existing cultivars
    ## or if cultivar is currently listed as an alias remove that. 
    ModifiedCultivarList = []
    Pos = 0
    for model in Wheat['Children'][0]['Children']:
        #find cultivar folder
        if model['Name']  == 'Cultivars':
            #Go into each regonds folder
            CultFolder=model
            CultivarFolderPos = Pos
            Pos2 =0
            for region in CultFolder['Children']:
                #Go into each cultivar
                RegionFolder = region['Children']
                RegionPos = Pos2
                Pos3 = 0
                #First find the Alias list for each cultivar and remove any cultivars that are to be modified
                for cult in RegionFolder:
                    cultCommands = cult['Command']
                    CultivarPos = Pos3
                    pointer = '[Structure].MaxLAR.FixedValue = '
                    try:
                        MaxLARcommand = pointer + str(CultivarMaxLAR.loc[cult['Name'],'MaxLAR'])
                    except:
                        MaxLARcommand = pointer + str(0.021)
                    Wheat['Children'][0]['Children'][CultivarFolderPos]\
                    ['Children'][RegionPos]['Children'][CultivarPos]\
                    ['Command'].append(MaxLARcommand)
                    Pos4 = 0
                    for com in cultCommands:
                        CommandPos = Pos4
                        if '[Structure].Phyllochron.BasePhyllochron.FixedValue' in com:
                            del Wheat['Children'][0]['Children'][CultivarFolderPos]\
                            ['Children'][RegionPos]['Children'][CultivarPos]\
                            ['Command'][CommandPos]
                        Pos4 +=1
                    Pos3 +=1
                Pos2+=1
        Pos+=1

def findNextChild(Parent,ChildName):
    if len(Parent['Children']) >0:
        for child in range(len(Parent['Children'])):
            if Parent['Children'][child]['Name'] == ChildName:
                return Parent['Children'][child]
    else:
        return Parent[ChildName]

def findModel(Parent,PathElements):
    for pe in PathElements:
        Parent = findNextChild(Parent,pe)
    return Parent

def replaceModel(Parent,modelPath,New):
    PathElements = modelPath.split('.')
    try:
        test = findModel(Parent,PathElements[:-1])[PathElements[-1]]
    except:
        print('Could not find parent node of model to over write for ' + modelPath)
        raise
    findModel(Parent,PathElements[:-1])[PathElements[-1]] = New
    
def removeModel(Parent,modelPath):
    PathElements = modelPath.split('.')
    Parent = findModel(Parent,PathElements[:-1])
    pos = 0
    counter = 0
    for c in Parent['Children']:
        if c['Name'] == PathElements[-1]:
            pos = counter
        counter +=1
    del Parent['Children'][pos]
            
def addModel(Parent,modelPath,New):
    PathElements = modelPath.split('.')
    Parent = findModel(Parent,PathElements)
    NewDict = ast.literal_eval(New.strip().replace('\n','').replace(' ',''))
    Parent['Children'].append(NewDict)



# +
## Read in the base Wheat.json file
with open('C:\Users\cflhxb\Dropbox\APSIMPotato\PotatoSimpleLeaf.apsimx','r') as PotatoAPSIMX:
    Potato = json.load(PotatoAPSIMX)
    PotatoAPSIMX.close()

## First set MaxLAR at the rate estimated from the existing phyllochron and PTQ values in simulations.  
## This assumes that some of the cultivar variation in phyllochorn will be due to differences in PTQ at specific locations
CultivarMaxLAR = pd.read_excel('C:/GitHubRepos/npi/Simulation/ModelFitting/Phyllochron/CultivarMaxLAR.xlsx',index_col=0)

AddMaxLAI(CultivarMaxLAR)
    
## Then over write with the parameters measured in controled environmet for genotypes where that data exists    
CultivarParamFiles = ['C:/GitHubRepos/npi/Analysis/Controlled Environment/CEParamFitsAus.xlsx',
                      'C:/GitHubRepos/npi/Analysis/Controlled Environment/CEParamFitsNZL.xlsx']
    
for f in CultivarParamFiles: 
    NewCultivarParams = pd.read_excel(f, sheet_name='FittedParams')
    NewCultivarParams.set_index('Cultivar',inplace=True)
    AddOrModifyCultivars(NewCultivarParams)

## Finally set any crop level parameter changes.
NewParams = pd.read_excel('C:/GitHubRepos/npi/Simulation/ModelFitting/ParameterOverwrites.xlsx',
                          sheet_name='NewParams')

for n in NewParams.index:
    if NewParams.loc[n,'Action'] == 'Remove':
        removeModel(Wheat,NewParams.loc[n,'Path'])
    elif NewParams.loc[n,'Action'] == 'Overwrite':
        if isinstance(NewParams.loc[n,'NewValue'],numbers.Number):
            newValue = NewParams.loc[n,'NewValue']
        else:
            newValue = ast.literal_eval(NewParams.loc[n,'NewValue'])
        replaceModel(Wheat,
                     NewParams.loc[n,'Path'],
                     newValue)
    elif NewParams.loc[n,'Action'] == 'Add':
        addModel(Wheat,NewParams.loc[n,'Path'], NewParams.loc[n,'NewValue'])
    
    else:
        print('Model changes failed')

## Save the altered wheat.json file
with open('C:\GitHubRepos\ApsimX\Models\Resources\Wheat.json','w') as WheatJSON:
    json.dump(Wheat,WheatJSON,indent=2)
