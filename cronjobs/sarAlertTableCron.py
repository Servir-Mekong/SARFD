import ee
import sys
from datetime import date
import math, time
import ee.mapclient
import subprocess


ee.Initialize()
PRJ = "EPSG:32650"
# Apply a ///MMU to a binary image
def apply_mmu (binary, band_name, mmu_value):

  img = binary

  binary = binary.updateMask(binary)

  # Get the connected pixel count
  mmu = binary.connectedPixelCount(mmu_value)

  # Get the connected pixel count
  mmu = binary.connectedPixelCount(mmu_value, True).reproject(PRJ, None, 10).gte(mmu_value).toInt8();

  # Create the cleaned layer
  mmu_applied = img.multiply(mmu).unmask(0);

  return mmu_applied.rename([band_name]).toInt8();

cdate = date.today()
endDate = ee.Date(cdate.strftime("%Y-%m-%d"))
collection = ee.ImageCollection("projects/cipalawan/assets/alertModelPalawanv1").filterDate("2023-01-01",endDate).aggregate_histogram("system:index").getInfo()

proc = subprocess.Popen(["earthengine ls projects/cipalawan/assets/palawanAlerts"], stdout=subprocess.PIPE, shell=True,universal_newlines=True)
(out, err) = proc.communicate()


shpfiles = []
out = out.split("\n")
for items in out:
  try:
    fname = items.split("/")
    shpfiles.append(fname[4])
  except:
    pass

rasterfiles = list(collection.keys())
collectionList = rasterfiles

for x in shpfiles:
  for z in collection:
    if x == z:
      collectionList.remove(x)


island = ee.FeatureCollection("projects/cipalawan/assets/palawanIsland")
islandBox = ee.FeatureCollection("projects/cipalawan/assets/palawanIslandBox")
riceMap = ee.Image("projects/cipalawan/assets/staticMaps/riceMapv1").lt(40)


counter = 0
for item in collectionList:
  print("processing",item)

  name = item.split("_")
  ymd = name[4]
  year = ymd[0:4]
  month = ymd[4:6]
  day = ymd[6:8]

  image = ee.Image("projects/cipalawan/assets/alertModelPalawanv1/" + item)
  prop = image.toDictionary()
  image = image.reduce(ee.Reducer.mean())
  index = image.get("system:index")
  t = image.get("system:time_start")

  alertImg = image.gt(60).selfMask()
  alertImg = alertImg.updateMask(riceMap)
  value = alertImg.reduceRegion(geometry=islandBox.geometry(),reducer=ee.Reducer.sum(),scale=10,maxPixels=1e13).getInfo()

  if int(value.get("mean")) > 0:

    mmu =  apply_mmu (alertImg, "mmu", 25)
    alertImg = alertImg.updateMask(mmu)


    alertSHP = alertImg.reduceToVectors(geometry=image.geometry(),scale=10, maxPixels=1e13,tileScale=16)
    alertSHP = alertSHP.filterBounds(island).filterBounds(islandBox)

    alertSHP = image.reduceRegions(collection=alertSHP, reducer=ee.Reducer.mean(), scale=10, crs=PRJ, tileScale=16)
    alertSHP = alertSHP.set("system:time_start",t).set("index",index).set(prop)

    def setTime(feat):
      return feat.set("system:time_start",t)\
                 .set("year",year)\
                 .set("month",month)\
                 .set("day",day)\
                 .set("type","sar")

    alertSHP = alertSHP.map(setTime)

    exportTask= ee.batch.Export.table.toAsset(collection= ee.FeatureCollection(alertSHP),description="table"+item,assetId="projects/cipalawan/assets/palawanAlerts/"+item)
    exportTask.start()
    counter+=1
