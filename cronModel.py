import ee
import sys
from datetime import date
import math, time

# initialize EE
ee.Initialize()

def submitJob(item,mpl):

    # set some ee.Model parameters
    bands = ['VH_after0','VH_after1','VH_before0', 'VH_before1','VH_before2','VV_after0','VV_after1','VV_before0', 'VV_before1', 'VV_before2']
    PROJECT = 'cipalawan';
    MODEL_NAME = 'alerts';
    VERSION_NAME = 'sarModelv18'

    # Load the trained model and use it for prediction.
    model = ee.Model.fromAiPlatformPredictor(
    projectName= PROJECT,
    modelName= MODEL_NAME,
    version= VERSION_NAME,
    inputTileSize= [128,128],
    inputOverlapSize= [16,16],
    proj= ee.Projection('EPSG:4326').atScale(10),
    fixInputProj= True,
    outputBands= {'landclass': {'type': ee.PixelType.float(),'dimensions': 1 }})
    MODE = 'DESCENDING'

    # Get the projection that is needed for the study area
    projection = ee.Projection('EPSG:32650')


    # Import Sentinel-1 Collection
    s1 =  ee.ImageCollection('COPERNICUS/S1_GRD')\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
			.filter(ee.Filter.eq('orbitProperties_pass', MODE))\
			.filter(ee.Filter.eq('instrumentMode', 'IW'))\
			.map(erodeGeometry)\
			.map(terrainCorrection)\
			.map(applySpeckleFilter)\
			.map(addRatio)

    # get the image that hasnt been processed
    img = ee.Image(s1.filter(ee.Filter.eq("system:index",item)).first())
    # get the geometry
    geom = img.geometry()
    # get the properties
    prop = img.toDictionary()
    # get the name of the image
    name = img.get("system:index").getInfo()
    # get the timestamp
    timeStamp = img.get("system:time_start")
    date = ee.Date(timeStamp)

    # filter for area of the image
    s1Item = s1.filterBounds(geom)
    # get the before images
    beforeSeries = createSeriesBefore(s1Item,date.advance(-1,"days"))
    # get the after image
    afterSeries = createSeriesAfter(s1Item,date.advance(1,"days")).select(["VV_after1","VH_after1"])
    # this is the current image
    after = img.select(["VV","VH"],["VV_after0","VH_after0"])

    # combine images into a single image
    image = beforeSeries.addBands(after).addBands(afterSeries).unmask(0,False)
    # convert to float for include_preprocessing
    image = image.select(bands).toFloat()

    # run the cnn
    prediction = ee.Image(model.predictImage(image.toArray()).arrayFlatten([["landclass","other"]]).toFloat())
    prediction = prediction.select("landclass").multiply(100).toInt().clip(mpl)

    # set the export location
    outputName = "projects/cipalawan/assets/alertsv4/" + name

    geom = geom.transform("EPSG:4326",0.01)
    output = ee.Image(prediction).clip(geom)
    output = clipEdge(output)
    output = output.set(prop).set("system:time_start",timeStamp)

    # create the export task
    export_task = ee.batch.Export.image.toAsset(image=output, description="alerts "+ name, assetId=outputName,region=geom.getInfo()['coordinates'], maxPixels=1e13,scale=10 )
    # run the export task
    export_task.start()



def createSeriesBefore(collection,date,iters=3,nday =12):

    iterations = list(range(1,iters*nday,nday))
    names = list(["_before{:01d}".format(x) for x in range(0,iters,1)])
    #print(iterations)
    #print(names)

    def returnCollection(day,name):
	    #print(day,name)
	    start = ee.Date(date).advance(-day,"days").advance(-nday,"days")
	    end = ee.Date(date).advance(-day,"days")
	    bandNames = ["VV"+name,"VH"+name]
	    #print(bandNames)
	    return ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH"],bandNames)\
				  .set("system:time_start",start)

    return toBands(ee.ImageCollection.fromImages(list(map(returnCollection,iterations,names))))

def createSeriesAfter(collection,date,iters=2,nday =12):


    iterations = list(range(1,iters*nday,nday))
    names = list(["_after{:01d}".format(x) for x in range(0,iters,1)])

    def returnCollection(day,name):
	    start = ee.Date(date).advance(day,"days")
	    end = ee.Date(date).advance(day,"days").advance(nday,"days")
	    bandNames = ["VV"+name,"VH"+name]

	    return ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH"],bandNames)\
				  .set("system:time_start",start)

    return toBands(ee.ImageCollection.fromImages(list(map(returnCollection,iterations,names))))
#
# * Clips 5km edges

def erodeGeometry(image):
    return image.clip(image.geometry().buffer(-1000))

def clipEdge(image):
    return image.clip(image.geometry().buffer(-500))


def applySpeckleFilter(img):

    vv = img.select('VV')
    vh = img.select('VH')
    vv = speckleFilter(vv).rename('VV');
    vh = speckleFilter(vh).rename('VH');
    return ee.Image(ee.Image.cat(vv,vh).copyProperties(img,['system:time_start'])).clip(img.geometry()).copyProperties(img);


def speckleFilter(image):
    """ apply the speckle filter """
    ksize = 3
    enl = 7;

    geom = image.geometry()

    # Convert image from dB to natural values
    nat_img = toNatural(image);

    # Square kernel, ksize should be odd (typically 3, 5 or 7)
    weights = ee.List.repeat(ee.List.repeat(1,ksize),ksize);

    # ~~(ksize/2) does integer division in JavaScript
    kernel = ee.Kernel.fixed(ksize,ksize, weights, (ksize//2), (ksize//2), False);

    # Get mean and variance
    mean = nat_img.reduceNeighborhood(ee.Reducer.mean(), kernel);
    variance = nat_img.reduceNeighborhood(ee.Reducer.variance(), kernel);

    # "Pure speckle" threshold
    ci = variance.sqrt().divide(mean);# square root of inverse of enl

    # If ci <= cu, the kernel lies in a "pure speckle" area -> return simple mean
    cu = 1.0/math.sqrt(enl);

    # If cu < ci < cmax the kernel lies in the low textured speckle area
    # -> return the filtered value
    cmax = math.sqrt(2.0) * cu;

    alpha = ee.Image(1.0 + cu*cu).divide(ci.multiply(ci).subtract(cu*cu));
    b = alpha.subtract(enl + 1.0);
    d = mean.multiply(mean).multiply(b).multiply(b).add(alpha.multiply(mean).multiply(nat_img).multiply(4.0*enl));
    f = b.multiply(mean).add(d.sqrt()).divide(alpha.multiply(2.0));

    # If ci > cmax do not filter at all (i.e. we don't do anything, other then masking)
    # Compose a 3 band image with the mean filtered "pure speckle",
    # the "low textured" filtered and the unfiltered portions
    out = ee.Image.cat(toDB(mean.updateMask(ci.lte(cu))),toDB(f.updateMask(ci.gt(cu)).updateMask(ci.lt(cmax))),image.updateMask(ci.gte(cmax)));

    return out.reduce(ee.Reducer.sum()).clip(geom);


def toNatural(img):
    """Function to convert from dB to natural"""
    return ee.Image(10.0).pow(img.select(0).divide(10.0));

def toDB(img):
    """ Function to convert from natural to dB """
    return ee.Image(img).log10().multiply(10.0);


def toBands(collection):

    def createStack(img,prev):
	    return ee.Image(prev).addBands(img)

    stack = ee.Image(collection.iterate(createStack,ee.Image(1)))
    stack = stack.select(ee.List.sequence(1, stack.bandNames().size().subtract(1)));
    return stack;

# Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
def terrainCorrection(image):
    date = ee.Date(image.get('system:time_start'))
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom)  # 30m srtm
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

    #Article ( numbers relate to chapters)
    #2.1.1 Radar geometry
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000).get('aspect')

    #2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    # 2.1.3 Model geometry
    # reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    #convert all to radians
    phi_rRad = phi_r.multiply(math.pi / 180)
    alpha_sRad = alpha_s.multiply(math.pi / 180)
    theta_iRad = theta_i.multiply(math.pi / 180)
    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

    # slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

    # slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

    # local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
    theta_liaDeg = theta_lia.multiply(180 / math.pi)

    # 2.2
    # Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos())
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

    # Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
    denominator = (ninetyRad.subtract(theta_iRad)).tan()
    volModel = (nominator.divide(denominator)).abs()

    # apply model
    gamma0_Volume = gamma0.divide(volModel)
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

    # we add a layover/shadow maskto the original implmentation
    # layover, where slope > radar viewing angle
    alpha_rDeg = alpha_r.multiply(180 / math.pi)
    layover = alpha_rDeg.lt(theta_i);

    # shadow where LIA > 90
    shadow = theta_liaDeg.lt(85)

    # calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad)\
			    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

    return output.select(['VV', 'VH'], ['VV', 'VH']).set("system:time_start",date).clip(imgGeom ).copyProperties(image)


# add the ratio band to the image
def addRatio(img):
    geom = img.geometry()
    vv = toNatural(img.select(['VV'])).rename(['VV']);
    vh = toNatural(img.select(['VH'])).rename(['VH']);
    ratio = vh.divide(vv).rename(['ratio']);
    return ee.Image(ee.Image.cat(vv,vh,ratio).copyProperties(img,['system:time_start'])).clip(geom).copyProperties(img);



# import the administrative boundaries
WDPAareas = ee.FeatureCollection("WCMC/WDPA/current/polygons");
mpl = WDPAareas.filter(ee.Filter.eq("ORIG_NAME","Mt. Mantalingahan Protected Landscape")).geometry().buffer(1000);
mpl= ee.FeatureCollection("projects/cipalawan/assets/palawanIsland").geometry().buffer(1000);

# we only have descening imagery
MODE = 'DESCENDING'

region = mpl

# Declare start and end period
cdate = date.today()
start = "2022-10-01"
end = "2022-11-15"

s1Collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(start,end)\
						      .filterBounds(region)\
						      .sort("system:time_start")\
						      .filter(ee.Filter.eq('orbitProperties_pass', MODE))\
						      .sort("system:time_start",True)\
						      .aggregate_histogram("system:index").getInfo()


gplCollection = ee.ImageCollection("projects/cipalawan/assets/alertsv4").aggregate_histogram("system:index").getInfo()


# add s1 filenames in colelction to two different lists
s1List1 = []
s1List2 = []
for item in s1Collection:
                s1List1.append(item)
                s1List2.append(item)

# add the processed imagery to a list
gplList = []
for item in gplCollection:
    gplList.append(item)

# remove images that have already been processed from the list
for x in s1List1:
    for z in gplList:
        if x == z:
            s1List2.remove(x)


# process imagery that has not been processed
for item in s1List2:
    print(counter,item)
    submitJob(item,mpl)
