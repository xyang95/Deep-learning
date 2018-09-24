
from netCDF4 import Dataset, date2index, num2date, date2num
import numpy as np
import matplotlib.pyplot as plt
from datetime  import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import interpolation
import numpy.ma as ma
import pickle
from sys import stdout
import argparse
import xarray as xr

# http://tds0.ifremer.fr/thredds/CORIOLIS-GLOBAL-CORA-OBS/CORIOLIS-GLOBAL-CORA04.2-OBS/CORIOLIS-GLOBAL-CORA04.2-OBS_FULL_TIME_SERIE.html?dataset=CORIOLIS-GLOBAL-CORA04.2-OBS_FULL_TIME_SERIE
sss_ds = Dataset('raw-data/coriolis/CORIOLIS-GLOBAL-CORA04.2-OBS_FULL_TIME_SERIE_1489953709377.nc')
sss_ds_interp_file = 'raw-processed/coriolis/coriolis-global-cora04.2-interpolate-land.pickle'

# http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_PHYS_001_020
# ftp://ftp.mfcglo-obs.cls.fr/Core/GLOBAL_ANALYSIS_PHYS_001_020/dataset-armor-3d-v4-cmems-v2/
sss_ds2 = Dataset('raw-data/sss/dataset-armor-3d-v4-cmems-v2_1491082996207.nc')

# http://marine.copernicus.eu/services-portfolio/access-to-products/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHYS_001_015
# ftp://data.ncof.co.uk/Core/GLOBAL_ANALYSIS_FORECAST_PHYS_001_015/MetO-GLO-PHYS-daily/


# https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.html
sst_ds = Dataset('raw-data/sst/sst.wkmean.1990-present.nc')

# https://www.esrl.noaa.gov/psd/repository/entry/show?entryid=b5492d1c-7d9c-47f7-b058-e84030622bbd
landmask_ds = Dataset('raw-data/lsmask.nc')


def ma2d_interp(array):
    valid = np.where(array.mask == False)
    w = array.shape[0]
    h = array.shape[1]
    grd = np.array(np.mgrid[0:w, 0:h])
    grd2 = np.vstack((grd[0][valid], grd[1][valid]))
    pts = grd2.T.flatten().reshape(int(grd2.size/2), 2)
    return griddata(pts, array[valid].T.flatten(), (grd[0], grd[1]), method='linear')

def ma_interp(array):
    if len(array.shape) == 2:
        return ma2d_interp(array)
    if len(array.shape) == 3:
        # assume first dimention shouldn't be interpolated
        output = np.empty_like(array, subok=False)
        for i in range(array.shape[0]):
            output[i] = ma2d_interp(array[i])
    return output

def array2d_reduce(array, zoom):
    output = ma.masked_all([int(array.shape[0] / zoom), int(array.shape[1] / zoom)])
    for i in range(int(array.shape[0] / zoom)):
        for j in range(int(array.shape[1] / zoom)):
            x = i * zoom
            y = j * zoom
            output[i, j] = array[x:x+zoom,y:y+zoom].mean()
    return output

def array_reduce(array, zoom):
    if len(array.shape) == 2:
        return array2d_reduce(array, zoom)
    if len(array.shape) == 3:
        # assume first dimention shouldn't be zoomed
        output = ma.masked_all([array.shape[0], int(array.shape[1] / zoom), int(array.shape[2] / zoom)])
        for i in range(array.shape[0]):
            output[i] = array2d_reduce(array[i], zoom)
        return output

def get_date_range(timevar):
    start = num2date(timevar[0], timevar.units)
    end = num2date(timevar[-1], timevar.units)
    return (start, end)

def year_fraction(date):
    year = date.year
    this_year_start = datetime(year=year, month=1, day=1)
    next_year_start = datetime(year=year+1, month=1, day=1)
    days_elapsed = date.timetuple().tm_yday - 0.5
    days_total = (next_year_start - this_year_start).days
    return days_elapsed/days_total



################################################################################
# Date range                                                                   #
################################################################################

sst_start, sst_end = get_date_range(sst_ds.variables['time'])
sss_start, sss_end = get_date_range(sss_ds.variables['time'])
sss2_start, sss2_end = get_date_range(sss_ds2.variables['time'])

# sst (noaa) date is at beginning of range, sss (copernicus) date is middle of
# range. Weekly data is centered on Wednesday, use center as date value.
sst_start += timedelta(days=3)
sst_end += timedelta(days=3)

# align dates (weekly)
ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
sst_offset = 0
start_day = sst_start
while start_day < sss_start:
    sst_offset += 1
    start_day += ONE_WEEK

end_day = min(sst_end, sss2_end)


################################################################################
# Precip                                                                       #
################################################################################

# get prediction locations
point = xr.open_dataset('raw-data/precip/precip.V1.0.2013.nc')
a = point.sel(time=slice('2013-01-01'))
mask_lat = a.variables['lat'][:]
mask_lon = a.variables['lon'][:]
points_idx = np.where(a.variables['precip'][:] >=0)
b = np.array((mask_lat[points_idx[1]], mask_lon[points_idx[2]])).T
b = np.round(np.float64(b))
#c = np.round(np.float64(b), decimals=1)
points = np.unique(b, axis=0)


DAYS_IN_YEAR = 365
def day2index(date):
    frac = year_fraction(date)
    index = round(frac * DAYS_IN_YEAR - 0.5)
    return int(index)


nweeks = int((end_day - start_day).days / 7) + 1
A_precip_mean = np.zeros((points.shape[0], nweeks))

P_mean = np.zeros((points.shape[0], DAYS_IN_YEAR))

for i in range(points.shape[0]):
    LAT = points[i][0]
    LON = points[i][1]
    print(LAT, ',', LON)
    date = datetime(1981, 1, 1)
    P = [[] for _ in range(DAYS_IN_YEAR)]
    year = 0
    while date < datetime(2011, 1, 1):
        if date.year != year:
            year = date.year
            precip_ds = Dataset('raw-data/precip/precip.V1.0.' + str(year) + '.nc')
            precip = precip_ds.variables['precip'][:]
            time = precip_ds.variables['time'][:]
            lat = precip_ds.variables['lat'][:]
            lon = precip_ds.variables['lon'][:]
            #
        t = date2index(date, precip_ds.variables['time'])
        lats = np.where((lat >= LAT-0.5) & (lat < LAT+0.5))[0]
        lons = np.where((lon >= LON-0.5) & (lon < LON+0.5))[0]
        one_degree_square = precip[t][lats][:,lons]
        if one_degree_square.mask.all():
            print('Missing data for', date.date().isoformat())
        else:
            mean_precip = one_degree_square.mean()
            P[day2index(date)].append(mean_precip)
            #
        date += ONE_DAY
        #
    for j in range(DAYS_IN_YEAR):
        P_mean[i, j] = sum(P[j])/len(P[j])

        # Calculate the total mean precip for the given time period
    date = start_day - ONE_DAY
    weekday = 0
    wkprecip = 0
    P2 = np.zeros(nweeks)
    j = 0
    while j < P2.size:
        wkprecip += P_mean[i, day2index(date)]
        date += ONE_DAY
        weekday += 1
        if weekday == 7:
            P2[j] = wkprecip
            weekday = 0
            wkprecip = 0
            j += 1
        #
    A_precip_mean[i] = P2

with open('train-data/usapcp-daily-mean.pickle', 'wb') as fd:
    pickle.dump(P_mean, fd)

with open('train-data/usprecip-mean.pickle', 'wb') as fd:
    pickle.dump(A_precip_mean, fd)

# precip data
# 1948 - 2006:
#   https://www.esrl.noaa.gov/psd/data/gridded/data.unified.daily.conus.html
# 2007 - present:
#   https://www.esrl.noaa.gov/psd/thredds/catalog/Datasets/cpc_us_precip/RT/catalog.html


nweeks = int((end_day - start_day).days / 7) + 1
A_precip = np.zeros((points.shape[0], nweeks))

for i in range(points.shape[0]):
    LAT = points[i][0]
    LON = points[i][1]
    print(LAT, ',', LON)
    date = start_day - ONE_DAY
    P = np.zeros(nweeks)
    j = 0
    weekday = 0
    wkprecip = 0
    year = 0
    while j < P.size:
        if date.year != year:
            year = date.year
            precip_ds = Dataset('raw-data/precip/precip.V1.0.' + str(year) + '.nc')
            precip = precip_ds.variables['precip'][:]
            time = precip_ds.variables['time'][:]
            lat = precip_ds.variables['lat'][:]
            lon = precip_ds.variables['lon'][:]
        #
        hours = date2num(date, precip_ds.variables['time'].units)
        t = np.where(time == hours)[0][0]
        lats = np.where((lat >= LAT-0.5) & (lat < LAT+0.5))[0]
        lons = np.where((lon >= LON-0.5) & (lon < LON+0.5))[0]
        one_degree_square = precip[t][lats][:,lons]
        if one_degree_square.mask.all():
            print('Missing data for', date.date().isoformat())
        else:
            mean_precip = one_degree_square.mean()
            wkprecip += mean_precip
        #
        date += ONE_DAY
        weekday += 1
        if weekday == 7:
            P[j] = wkprecip
            weekday = 0
            wkprecip = 0
            j += 1
        #
    A_precip[i] = P

with open('train-data/new-usprecip.pickle', 'wb') as fd:
        pickle.dump(A_precip, fd)



################################################################################
# Temperature                                                                  #
################################################################################
# ftp://ftp.cpc.ncep.noaa.gov/precip/wd52ws/global_temp/
# process-cpc-global-temp.sh

