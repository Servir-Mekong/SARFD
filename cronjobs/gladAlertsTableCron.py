
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

gladAlert = ee.Image(ee.ImageCollection('projects/glad/alert/UpdResult').select('alertDate23').max()).clip(island)
#gladAlert = ee.Image(ee.ImageCollection('projects/glad/alert/2021final').select('alertDate21').max()).clip(island)

cdate = date.today()
month = cdate.month -1
#endDate = ee.Date(cdate.strftime("%Y-%m-%d"))

year = 2023
startDate = datetime.datetime(year, month, 1)
endDate = datetime.datetime(year, month,1 )  + relativedelta(months=+1)

startDOY = startDate.timetuple().tm_yday
endDOY = endDate.timetuple().tm_yday

alertBefore = gladAlert.gt(startDOY)
alertAfter = gladAlert.lt(endDOY)
alert = alertBefore.add(alertAfter).eq(2)
alert = alert.selfMask()

value = alert.reduceRegion(geometry=islandBox.geometry(),reducer=ee.Reducer.sum(),scale=10,maxPixels=1e13).getInfo()
print( value.get("alertDate23"))
if int(value.get("alertDate23")) > 0:
  mmu =  apply_mmu (alert, "mmu", 25)
  alert = alert.updateMask(mmu)
  alertSHP = alert.reduceToVectors(geometry=island,scale=30, maxPixels=1e13,tileScale=16)
  alertSHP = alertSHP.filterBounds(island).filterBounds(islandBox)
  alertSHP = gladAlert.reduceRegions(collection=alertSHP, reducer=ee.Reducer.mode(), scale=10, crs=PRJ, tileScale=16)


  def setTime(feat):
    doy = feat.get("mode")
    return feat.set("year",year)\
             .set("month",month)\
             .set("doy",doy)\
             .set("type","glad")

  alertSHP = alertSHP.map(setTime)

  outputName = "gladAlerts" + str(year) + str(month).zfill(2)

  exportTask= ee.batch.Export.table.toAsset(collection= ee.FeatureCollection(alertSHP),description=outputName,assetId="projects/cipalawan/assets/gladPalawan/"+outputName)
  exportTask.start()
