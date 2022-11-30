import ee, os
import subprocess
import datetime


ee.Initialize()

def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True,text=True)
    return p.stdout.read()

batcmd="earthengine ls projects/cipalawan/assets/palawanAlerts/"

result = system_call(batcmd).split("\n")

ft = ee.FeatureCollection(result[0])

for i in range(1,len(result),1):
    if len(result[i]) > 0:
        fc = ee.FeatureCollection(result[i])
        ft = ft.merge(fc)


# get the date for today
date = datetime.datetime.now()
year = date.year
month = date.month
day = date.day
name = str(year)+str(month)+str(day)

# create the export task
exportTask = ee.batch.Export.table.toAsset(collection= ee.FeatureCollection(ft),description= name,assetId="projects/cipalawan/assets/palawanAlerts/alerts_"+name)
# run the export task
exportTask.start()
