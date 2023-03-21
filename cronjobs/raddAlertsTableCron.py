
import ee
import sys
import datetime
from datetime import date, timedelta
from dateutil.relativedelta import *
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


island = ee.FeatureCollection("projects/cipalawan/assets/palawanIsland")
islandBox = ee.FeatureCollection("projects/cipalawan/assets/palawanIslandBox")

radd = ee.ImageCollection('projects/radar-wur/raddalert/v1')
geography = 'asia' # 'sa' (south america), 'africa' (africa), 'asia' (asia & pacific)


radd_alert =  ee.ImageCollection(radd.filterMetadata('layer','contains','alert')\
                           .filterMetadata('geography','contains',geography)\
                           .sort('system:time_end', False))

radd_alert= radd_alert.select("Alert")
collection = radd_alert.aggregate_histogram("system:index").getInfo()

proc = subprocess.Popen(["earthengine ls projects/cipalawan/assets/raddPalawan"], stdout=subprocess.PIPE, shell=True,universal_newlines=True)
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
#print(len(collectionList))
for x in shpfiles:
  for z in collection:
    if x == z:
      collectionList.remove(x)

island = ee.FeatureCollection("projects/cipalawan/assets/palawanIsland")
islandBox = ee.FeatureCollection("projects/cipalawan/assets/palawanIslandBox")
riceMap = ee.Image("projects/cipalawan/assets/Temp/riceMapv1").lt(40)


counter = 0
for item in collectionList:

  year = item[5:9]
  month = item[9:11]
  day = item[11:13]
  t = ee.Date.fromYMD(int(year),int(month),int(day))
  if int(year) == 2023:
    print("processing",item)
    startDate = datetime.datetime(int(year), int(month), int(day))+ relativedelta(days=-7)
    endDate = datetime.datetime(int(year), int(month), int(day) )

    startDOY = startDate.timetuple().tm_yday
    endDOY = endDate.timetuple().tm_yday

    image = ee.Image("projects/radar-wur/raddalert/v1/" + item).select("Date")

    beforeDates = str(int(year)-2000) + str(startDOY).zfill(3)
    afterDates = str(int(year)-2000) + str(endDOY).zfill(3)

    alertBefore = image.gt(ee.Number.parse(beforeDates))
    alertAfter = image.lt(ee.Number.parse(afterDates))
    alert = alertBefore.add(alertAfter).eq(2)
    alert = alert.selfMask()

    value = alert.reduceRegion(geometry=islandBox.geometry(),reducer=ee.Reducer.sum(),scale=10,maxPixels=1e13).getInfo()

    if int(value.get("Date")) > 0:

      def setTime(feat):
        return feat.set("year",year)\
                   .set("month",month)\
                   .set("day",day)\
                   .set("type","RADD")

      mmu =  apply_mmu (alert, "mmu", 15)
      alertImg = alert.updateMask(mmu)

      alertSHP = alertImg.reduceToVectors(geometry=islandBox,scale=10, maxPixels=1e13,tileScale=16)
      alertSHP = alertSHP.filterBounds(island).filterBounds(islandBox)
      alertSHP = alertSHP.map(setTime)

      if int(alertSHP.size().getInfo()) > 0:
        exportTask= ee.batch.Export.table.toAsset(collection= ee.FeatureCollection(alertSHP),description="table"+item,assetId="projects/cipalawan/assets/raddPalawan/"+item)
        exportTask.start()
