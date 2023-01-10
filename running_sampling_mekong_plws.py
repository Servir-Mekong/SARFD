import ee
import math
import numpy as np
import random
from numpy.random import seed
from numpy.random import rand

ee.Initialize()

seed(10)
values = rand(50000)



def main ():

    outputBucket = "bucketsvmk"
    folder = "khnpl_alerts/alertsPLWSSamples"
    #cam = ee.FeatureCollection("projects/servir-mekong/admin/KHM_adm0");
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017");
    #cambodia = countries.filter(ee.Filter.inList("country_na",["Cambodia","Thailand","Laos","Vietnam","Burma"]));
    #cam = countries.filter(ee.Filter.inList("country_na",["Cambodia"]));

    featureNames = ['VH_after0','VH_after1',
                    'VH_before0', 'VH_before1','VH_before2',
                    'VV_after0','VV_after1',
                    'VV_before0', 'VV_before1', 'VV_before2',
                    'alert','other']

    # Define kernel size
    kernel_size = 128
    image_kernel = get_kernel(kernel_size)

    # Get the projection that is needed for the study area
    projection = ee.Projection('EPSG:32648')


    MODE = 'DESCENDING'
    year = 2020

    # stratified samples were created in different files
    ft = ee.FeatureCollection("projects/cemis-camp/assets/digitized/plws_all").filter(ee.Filter.eq("year",year))

    sample_locations = ft.randomColumn().sort("random") #sample_locations.sort("random")


    beforeDate = ee.Date.fromYMD(year,1,1)
    afterDate = ee.Date.fromYMD(year,1,1)

    # Import Sentinel-1 Collection
    s1 =  ee.ImageCollection('COPERNICUS/S1_GRD')\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
			.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
			.filter(ee.Filter.eq('orbitProperties_pass', MODE))\
			.filter(ee.Filter.eq('instrumentMode', 'IW'))\
			.map(terrainCorrection)\
			.map(applySpeckleFilter)\
			.map(addRatio)
            #.filterBounds(cam)\


    sample_locations = sample_locations.toList(10000)
    end_list = sample_locations.size().getInfo()

    s = 1
    step = 1
    start = s*step
    for i in range(s,2926,1):

        print(year, i)

        feature = ee.Feature(sample_locations.get(i))

        doy = int(feature.get("doy").getInfo())

        beforeDate = ee.Date.fromYMD(year,1,1).advance(doy,"day").advance(-1,"day")
        afterDate = ee.Date.fromYMD(year,1,1).advance(doy,"day").advance(1,"day")


        label = ee.Image("projects/cemis-camp/assets/digitized/raster" + str(year)).eq(doy).rename("alert") #
        other = label.remap([0,1],[1,0]).rename(["other"])

        def addRandomPoints(feature):
            bounds = feature.geometry().buffer(30).bounds()
            points = ee.FeatureCollection.randomPoints(region=bounds, points=50, seed=11232*i)
            return points



        points = addRandomPoints(feature)


        before = createSeriesBefore(s1.filterBounds(points.geometry()),beforeDate,i)
        after = createSeriesAfter(s1.filterBounds(points.geometry()),afterDate,i)


        image = before.addBands(after).addBands(label).addBands(other).unmask(0,False)

        neighborhood = image.neighborhoodToArray(image_kernel)
        trainingData = neighborhood.sample(region = points,scale= 10,tileScale= 16, geometries= False)

        sample = ee.Feature(trainingData.first())

        if values[i]<=0.1:
            trainFilePrefix = "/validation/validPLWS_"+str(year) + "_" + str(i).zfill(4)
        elif values[i]>0.1 and values[i] <= 0.3:
            trainFilePrefix = "/testing/testPLWS_"+str(year) + "_"+ str(i).zfill(4)
        else:
            trainFilePrefix = "/training/trainPLWS_"+str(year) + "_"  + str(i).zfill(4)


        trainingTask = ee.batch.Export.table.toCloudStorage(collection= ee.FeatureCollection(trainingData),
								description= "trainpatch"+str(i),
								fileNamePrefix= folder+ trainFilePrefix,
								bucket= outputBucket,
								fileFormat= 'TFRecord',
								selectors= featureNames)

        trainingTask.start()




def createSeriesBefore(collection,date,val,iters=3,nday =24):

    iterations = []

    # Set a length of the list to 10
    for i in range(0, iters):
	    # any random numbers from 0 to 1000
	    iterations.append(random.randint(0, 100))


    names = ["_before{:01d}".format(x) for x in range(0,iters,1)]
    #print(iterations)


    imgList = []

    for n in range(0,3):
        day = iterations[n]
        name = names[n]
        start = ee.Date(date).advance(-day,"days")
        end = ee.Date(date).advance(-day+nday,"days")
        bandNames = ["VV"+name,"VH"+name,"ratio"+name]
        img = ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH","ratio"],bandNames)\
				  .set("system:time_start",start)
        imgList.append(img)


    return toBands(ee.ImageCollection.fromImages(imgList))


def createSeriesAfter(collection,date,val,iters=2,nday =24):

    random.seed(val)
    iterations = []
    # Set a length of the list to 10
    for i in range(0, iters):
	    # any random numbers from 0 to 1000
	    iterations.append(random.randint(0, 10))

    names = ["_after{:01d}".format(x) for x in range(0,iters,1)]

    imgList = []
    for n in range(0,2):
        day = iterations[n]
        name = names[n]
        start = ee.Date(date).advance(day,"days")
        end = ee.Date(date).advance(day+nday,"days")
        bandNames = ["VV"+name,"VH"+name,"ratio"+name]
        img =  ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH","ratio"],bandNames)\
				  .set("system:time_start",start)
        imgList.append(img)

    return toBands(ee.ImageCollection.fromImages(imgList))

"""
def createSeriesBefore(collection,date,val,iters=3,nday =14):

    iterations = [0,14,28]
    names = ["_before{:01d}".format(x) for x in range(0,iters,1)]


    imgList = []

    for n in range(0,iters):
        day = iterations[n]
        name = names[n]
        start = ee.Date(date).advance(-day-nday,"days")
        end = ee.Date(date).advance(-day,"days")
        bandNames = ["VV"+name,"VH"+name,"ratio"+name]
        img = ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH","ratio"],bandNames)\
				  .set("system:time_start",start)
        imgList.append(img)


    return toBands(ee.ImageCollection.fromImages(imgList))


def createSeriesAfter(collection,date,val,iters=2,nday =14):

    iterations = [0,14]
    names = ["_after{:01d}".format(x) for x in range(0,iters,1)]

    imgList = []
    for n in range(0,iters):
        day = iterations[n]
        name = names[n]
        start = ee.Date(date).advance(day,"days")
        end = ee.Date(date).advance(day+nday,"days")
        bandNames = ["VV"+name,"VH"+name,"ratio"+name]
        img =  ee.Image(collection.filterDate(start,end).mean())\
				  .select(["VV","VH","ratio"],bandNames)\
				  .set("system:time_start",start)
        imgList.append(img)

    return toBands(ee.ImageCollection.fromImages(imgList))
"""

# Produces a kernel of a given sized fro sampling in GEE
def get_kernel (kernel_size):
    eelist = ee.List.repeat(1, kernel_size)
    lists = ee.List.repeat(eelist, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)
    return kernel

# Scale the integer values to a range between 1 and 0
def scale_sentinel_values (image):
    return image.unmask(-50).clamp(-50, 1).unitScale(-50, 1).set('system:time_start', image.date())




# Implementation by Andreas Vollrath (ESA), inspired by Johannes Reiche (Wageningen)
def terrainCorrection(image):
    date = ee.Date(image.get('system:time_start'))
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom)  # 30m srtm
    #srtm = ee.Image('projects/cipalawan/assets/output_COP30').clip(imgGeom)
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

    output = output.where(output.gt(10),0)
    output = output.where(output.lt(-25),0)

    return output.select(['VV', 'VH'], ['VV', 'VH']).set("system:time_start",date)


def applySpeckleFilter(img):
    vv = img.select('VV')
    vh = img.select('VH')
    vv = speckleFilter(vv).rename('VV');
    vh = speckleFilter(vh).rename('VH');
    return ee.Image.cat(vv,vh).copyProperties(img,['system:time_start']);


def speckleFilter(image):
    """ apply the speckle filter """
    ksize = 3
    enl = 7;

    # Convert image from dB to natural values
    nat_img = toNatural(image);

    # Square kernel, ksize should be odd (typically 3, 5 or 7)
    weights = ee.List.repeat(ee.List.repeat(1,ksize),ksize);

    # ~~(ksize/2) does integer division in JavaScript
    kernel = ee.Kernel.fixed(ksize,ksize, weights, ~~(ksize//2), ~~(ksize//2), False);

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

    return out.reduce(ee.Reducer.sum());

def addRatio(img):
    vv = toNatural(img.select(['VV'])).rename(['VV']);
    vh = toNatural(img.select(['VH'])).rename(['VH']);
    ratio = vh.divide(vv).rename(['ratio']);
    return ee.Image.cat(vv,vh,ratio).copyProperties(img,['system:time_start']);


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




if __name__ == "__main__":
    print('Program started..')
    main()
    print('\nProgram completed.')
