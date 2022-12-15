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
## Read in the base Wheat.json file
with open('C:/GitHubRepos/ApsimX/Tests/Validation/Potato/PotatoExpandedValidation.apsimx','r') as PotatoAPSIMX:
    Potato = json.load(PotatoAPSIMX)
    PotatoAPSIMX.close()
count = 0
def FixModel(model):
    if (model['Name'] == 'PlantandHarvest') & (model['$type'] == "Models.Manager, Models"):
        model['Code'] = "\r\nusing System.Xml.Serialization;\r\nusing APSIM.Shared.Utilities;\r\nusing Models.PMF;\r\nusing Models.Core;\r\nusing System;\r\n        \r\nnamespace Models\r\n{\r\n    [Serializable] \r\n    [System.Xml.Serialization.XmlInclude(typeof(Model))]\r\n    public class Script : Model\r\n    {\r\n        [Link] Plant Potato;\r\n        [Link] Clock Clock;\r\n        [Link] Zone zone;\r\n        [Link] Summary Summary;\r\n        \r\n        [Description(\"Planting Date\")]\r\n        public DateTime PlantingDate { get; set; }\r\n        [Description(\"Cultivar\")]\r\n        [Display(Type=DisplayType.CultivarName)]\r\n        public string CultivarName { get; set; }\r\n        [Description(\"Spacing (mm) between plants within rows\")]\r\n        public double InterRowPlantSpace { get; set; }\r\n        [Description(\"Spacing (mm) between rows\")]\r\n        public double RowWidth { get; set; }\r\n        [Description(\"Depth (mm) tubers are planted\")]\r\n        public double PlantingDepth { get; set; }\r\n        [Description(\"Number of stems per seed tuber\")]\r\n        public double StemNumberPerSeedTuber { get; set; }\r\n        [Description(\"Final leaf number.  Leave at 0 if not known\")]\r\n        public double SetFinalLeafNumber { get; set; }\r\n        [Description(\"HarvestDate Date\")]\r\n        public DateTime HarvestDate { get; set; }\r\n\r\n        public string FinalTag {get; set;}\r\n        public string PlantTag {get; set;}\r\n\r\n        [EventSubscribe(\"DoManagement\")]\r\n        private void OnDoManagement(object sender, EventArgs e)\r\n        {\r\n            double PlantPopulation = 1 / ((RowWidth / 1000) * (InterRowPlantSpace / 1000));\r\n\r\n            PlantTag = \"\";\r\n            if (Clock.Today.Date == PlantingDate)\r\n              {\r\n                PlantTag = \"planting\";\r\n                Potato.Sow(population: PlantPopulation, cultivar: CultivarName, depth: PlantingDepth, rowSpacing: RowWidth, budNumber: StemNumberPerSeedTuber);\r\n              }\r\n            FinalTag = \"\";   \r\n            if (Clock.Today == HarvestDate) \r\n            {\r\n                FinalTag = \"yield\";\r\n                Potato.RemoveBiomass(\"Harvest\");\r\n                Potato.EndCrop();\r\n                Clock.EndDate = Clock.Today.AddDays(1);\r\n                Summary.WriteMessage(this, \"Manager has sent a harvest event\");\r\n            }\r\n            if (SetFinalLeafNumber > 0)\r\n                zone.Set(\"Potato.Structure.FinalLeafNumber.FixedValue\", SetFinalLeafNumber);\r\n        }\r\n    }\r\n}\r\n"   
        print(count)
        count +=1

def CheckAllNodesAndFix(Parent):
    count = 0
    for child in Parent['Children']:
        FixModel(child)
        try:
            if len(child['Children']) > 0:
                CheckAllNodesAndFix(child)
        except:
            do = 'Nothing'
            #print(child['Name'])

CheckAllNodesAndFix(Potato)

with open('C:/GitHubRepos/ApsimX/Tests/Validation/Potato/PotatoExpandedValidation.apsimx','w') as PotatoAPSIMX:
    json.dump(Potato,PotatoAPSIMX,indent=2)
# -

Potato
