import ee, subprocess

ee.Initialize()


# set projection
PRJ = "EPSG:32650"

# function to calculate minimum mapping unit using connected pixel count.
def apply_mmu (binary, band_name, mmu_value):

  img = binary;
  binary = binary.updateMask(binary);

  #// Get the connected pixel count
  mmu = binary.connectedPixelCount(mmu_value);

  # Get the connected pixel count
  mmu = binary.connectedPixelCount(mmu_value, True).reproject(PRJ, None, 10).gte(mmu_value).toInt8();

  # Create the cleaned layer
  mmu_applied = img.multiply(mmu).unmask(0);

  return mmu_applied.rename([band_name]).toInt8();

# system call to get the list of feature collections
def system_call(command):
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True,text=True)
    return p.stdout.read()

# import project shapefiles
projectArea = ee.FeatureCollection("projects/cipalawan/assets/Project_Area_Footprint");
forest =  ee.FeatureCollection("projects/cipalawan/assets/LAWIN/FC_PALAWAN2015").filterBounds(projectArea);
ancestraldomains = ee.FeatureCollection("projects/cipalawan/assets/Priority_ancestraldomains");

# import the alerts and filter for aoi
gplCollection = ee.ImageCollection("projects/cipalawan/assets/alertsv4").filterBounds(projectArea).filterBounds(forest).aggregate_histogram("system:index").getInfo()

# command to get list of alerts
batcmd="earthengine ls projects/cipalawan/assets/palawanAlerts/"

# get the current file list
result = system_call(batcmd).split("\n")

# add all filenames to a list
ftFiles = []
for item in result:
    try:
        ftFiles.append(item.split("/")[4])
    except:
        pass

# add the processed imagery to a list
gplList1 = []
gplList2 = []
for item in gplCollection:
    gplList1.append(item)
    gplList2.append(item)

# remove images that have already been processed from the list
for x in ftFiles:
    for z in gplList2:
        if x == z:
            gplList1.remove(x)


# process imagery that was not processed yet
for item in gplList1:
    # get the image
    img = ee.Image("projects/cipalawan/assets/alertsv4/"+item)
    # get the name
    name = img.get("system:index").getInfo()

    # get date information
    date = ee.Date(img.get("system:time_start"))
    year = date.get("year")
    month = date.get("month")
    day= date.get("day")
    doy = date.format("D")

    # apply a theshold and mmu
    img = img.gt(50).selfMask();
    mmu =  apply_mmu (img, "mmu", 50);
    img = img.updateMask(mmu).clip(forest).clip(projectArea);

    # calculate area for each alert
    def calcArea(feat):
        area = feat.area(1)
        return feat.set("size",area).set("year",year).set("month",month).set("day",day).set("doy",doy)

    # vectorize the data
    alertshp = img.reduceToVectors(geometry=projectArea,scale=10,maxPixels=1e13,tileScale=16);
    # calculate the area of each polygon
    alertshp = alertshp.map(calcArea)

    # create a list to set unique ID to each feature
    feature_list = alertshp.toList(alertshp.size());
    indexes = ee.List.sequence(1, alertshp.size());

    def setId(t):
        t = ee.List(t);
        f = ee.Feature(t.get(0));
        idx = ee.Number(t.get(1));
        return f.set("ID", idx);

    with_ids = feature_list.zip(indexes).map(setId)
    alertshp = ee.FeatureCollection(with_ids);

    # create export task
    exportTask = ee.batch.Export.table.toAsset(collection= ee.FeatureCollection(alertshp),description= name,assetId="projects/cipalawan/assets/palawanAlerts/"+name)
    # start the task
    exportTask.start()
