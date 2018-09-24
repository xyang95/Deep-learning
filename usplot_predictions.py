from __future__ import print_function
from common import *
from netCDF4 import Dataset, date2index, num2date, date2num
from datetime import datetime, timedelta
import argparse
import matplotlib as m
import matplotlib.pyplot as plt
import pickle
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from os import mkdir
from os.path import isdir
import pickle
from common import *
import glob
import re
import xarray as xr
import datetime as date
import keras

def combine_weeks(myarray,a_type):
    # format precip data to be sum of i'th week + (i+1)'th week
    if a_type == 'precip':
        newarray = np.zeros((myarray.shape[0],myarray.shape[1]-1),dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:,i] = myarray[:,i] + myarray[:,i+1]

    if a_type == 'temp':
        newarray = np.zeros((myarray.shape[0],myarray.shape[1]-1),dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:,i] = (myarray[:,i] + myarray[:,i+1])/2

    # merge sst or sss data into 2 week periods from 1 week periods (obsolete)
    # if len(myarray) % 2 == 1:
    #   myarray = myarray[:-1]
    # if a_type == 'ss':
    #   newarray = np.zeros((int(myarray.shape[0]/2),myarray.shape[1]),dtype=np.float32)
    #   for i in range(len(newarray)):
    #       newarray[i,:] = np.divide(np.add(myarray[i*2,:],myarray[i*2+1,:]),2.)

    return newarray

def generate_windows(x):
    result = []
    for i in range(x.shape[0]+1-window_size):
        result.append(x[i:i+window_size, ...])
    return np.stack(result)

#get date
target_date = None
# target_date = date.date(2018, 4, 17)
if not target_date:
    # default to next Tuesday (weekday #1)
    today = datetime.now().date()
    days_ahead = (1 - today.weekday() + 7) % 7
    target_date = today + timedelta(days=days_ahead)
datestr = target_date.strftime('%Y%m%d')

def compile_input(datestr, combine=False, i_sss=False, i_precip=False, i_time=False):


    ## Load training data
    # load time data
    training_time_data_file = 'train-data/time.pickle' # 1414 x 1
    training_time_vectors = pickle.load(open(training_time_data_file, 'rb'))
    training_time_vectors = np.reshape(training_time_vectors,(training_time_vectors.shape[0],1))

    # load sst data
    training_sst_data_file = 'train-data/sst.pickle' # 1414 x 8099
    training_sst_vectors = pickle.load(open(training_sst_data_file, 'rb'))

    # load sss data
    training_sss_data_file = 'train-data/sss.pickle' # 1414 x 8099
    training_sss_vectors = pickle.load(open(training_sss_data_file, 'rb'))

    # load precipitation data
    training_location_precip_file = 'train-data/usprecip.pickle' # 514 x 1299
    training_precip_data = pickle.load(open(training_location_precip_file, 'rb'))
    if combine:
        training_precip_data = combine_weeks(training_precip_data,'precip')
    training_precip_data = training_precip_data.T

    # load temperature data

    # make precip data only as long as temp data
    training_precip_data = training_precip_data[:training_precip_data.shape[0],:]

    # ensure same length vectors and normalize
    training_time_vectors = training_time_vectors[:training_precip_data.shape[0],:]
    training_sst_vectors = training_sst_vectors[:training_precip_data.shape[0],:]
    training_sss_vectors = training_sss_vectors[:training_precip_data.shape[0],:]
    training_precip_input = training_precip_data[:training_precip_data.shape[0],:]

    ## Load input data
    # load sst data
    sst_data_file = 'predict-data/'+datestr+'/sst.pickle' # 10 x 8099
    sst_vectors = pickle.load(open(sst_data_file, 'rb'))
    sst_vectors = (sst_vectors - np.amin(training_sst_vectors)) * 1./(np.amax(training_sst_vectors) - np.amin(training_sst_vectors))

    # compile input data
    input_data = sst_vectors
    if i_sss:
        sss_data_file = 'predict-data/'+datestr+'/sss.pickle' # 10 x 8099
        sss_vectors = pickle.load(open(sss_data_file, 'rb'))
        sss_vectors = (sss_vectors - np.amin(training_sss_vectors)) * 1./(np.amax(training_sss_vectors) - np.amin(training_sss_vectors))
        input_data = np.concatenate((input_data, sss_vectors), axis=1)
    if i_precip:
        location_precip_file = 'predict-data/'+datestr+'/precip.pickle' # 10 x 514
        precip_data = pickle.load(open(location_precip_file, 'rb'))
        precip_data = (precip_data - np.amin(training_precip_input)) * 1./(np.amax(training_precip_input) - np.amin(training_precip_input))
        input_data = np.concatenate((input_data, precip_data), axis=1)
    if i_time:
        time_data_file = 'predict-data/'+datestr+'/time.pickle' # 10 x 1
        time_vectors = pickle.load(open(time_data_file, 'rb'))
        time_vectors = np.reshape(time_vectors,(time_vectors.shape[0],1))
        input_data = np.concatenate((input_data, time_vectors), axis=1)

    return input_data

for weeks in ('35', '71'):
    if weeks == '35':
        window_size = 10
        model_names = glob.glob('train-results/k_model_*.h5')
        N_models = len(model_names)
        precip_wk34_predictions = np.zeros((N_models, 935))
        precip_wk56_predictions = np.zeros((N_models, 935))
        for i in range(N_models):
            model_name = model_names[i]
            model = keras.models.load_model(model_name)

            result = re.search('time=(.*).h5', model_name)
            input_set = int(result.group(1))

            input_data = compile_input(datestr, combine=False, i_sss=False, i_precip=False, i_time=bool(input_set))
            input_data = generate_windows(input_data)

            # Run predictions
            prediction_i = model.predict(input_data)
            prediction_i = np.reshape(prediction_i, (4, 935))
            precip_wk34_predictions[i, :] = np.sum(prediction_i[0:2, :], axis=0)
            precip_wk56_predictions[i, :] = np.sum(prediction_i[2:4, :], axis=0)

        precip_wk34_prediction = np.mean(precip_wk34_predictions, axis=0)
        precip_wk34_prediction = precip_wk34_prediction.clip(0)
        precip_wk56_prediction = np.mean(precip_wk56_predictions, axis=0)
        precip_wk56_prediction = precip_wk56_prediction.clip(0)
    elif weeks == '71':
        window_size = 10
        model_names = glob.glob('train91nc-results/k_model_*.h5')
        N_models = len(model_names)
        precip_wk78_predictions = np.zeros((N_models, 935))
        precip_wk91_predictions = np.zeros((N_models, 935))
        for i in range(N_models):
            model_name = model_names[i]
            model = keras.models.load_model(model_name)

            result = re.search('time=(.*).h5', model_name)
            input_set = int(result.group(1))

            input_data = compile_input(datestr, combine=False, i_sss=False, i_precip=False, i_time=bool(input_set))
            input_data = generate_windows(input_data)

            # Run predictions
            prediction_i = model.predict(input_data)
            prediction_i = np.reshape(prediction_i, (4, 935))
            precip_wk78_predictions[i, :] = np.sum(prediction_i[0:2, :], axis=0)
            precip_wk91_predictions[i, :] = np.sum(prediction_i[2:4, :], axis=0)

        precip_wk78_prediction = np.mean(precip_wk78_predictions, axis=0)
        precip_wk78_prediction = precip_wk78_prediction.clip(0)
        precip_wk91_prediction = np.mean(precip_wk91_predictions, axis=0)
        precip_wk91_prediction = precip_wk91_prediction.clip(0)

point = xr.open_dataset('raw-data/precip/precip.V1.0.2013.nc')
a = point.sel(time=slice('2013-01-01'))
mask_lat = a.variables['lat'][:]
mask_lon = a.variables['lon'][:]
points_idx = np.where(a.variables['precip'][:] >= 0)
b = np.array((mask_lat[points_idx[1]], mask_lon[points_idx[2]])).T
#points = np.array((points_idx[1], points_idx[2])).T
c = np.around(b)
points1 = np.unique(c, axis=0)
points = np.zeros((935,2), dtype=np.int)
for i in range(len(points)):
    points[i] = [int(x) for x in points1[i]]
points[:,0] = 56 - points[:,0]
points[:,1] = points[:,1] - 234
xmax = points[:,0].max() + 1
ymax = points[:,1].max() + 1

for weeks in ('34', '56', '78', '910'):
    a = np.ma.masked_all((xmax + 1, ymax + 1))
    if weeks == '34':
        prediction = precip_wk34_prediction
    elif weeks == '56':
        prediction = precip_wk56_prediction
    elif weeks == '78':
        prediction = precip_wk78_prediction
    elif weeks == '910':
        prediction = precip_wk91_prediction
    for loc in range(len(points)):
        index = points[loc]
        a[tuple(index)] = prediction[loc]

    plt.figure()
    plt.title(str(target_date) + ' ' + 'apcp' + ' weeks ' + weeks)
    plt.imshow(a, cmap=plt.get_cmap('BrBG'), vmin=0, vmax=60)
    plt.colorbar()
    plt.pause(0.01)
    plt.savefig('uspredictions/' + datestr + '/' + 'apcp' + '-wks' + weeks + '-' + datestr + '.png', bbox_inches='tight')

ONE_WEEK = timedelta(days=7)
ONE_DAY = timedelta(days=1)
DAYS_IN_YEAR = 365

def year_fraction(date):
    year = date.year
    this_year_start = datetime(year=year, month=1, day=1)
    next_year_start = datetime(year=year+1, month=1, day=1)
    days_elapsed = date.timetuple().tm_yday - 0.5
    days_total = (next_year_start - this_year_start).days
    return days_elapsed/days_total

def day2index(date):
    frac = year_fraction(date)
    index = round(frac * DAYS_IN_YEAR - 0.5)
    return int(index)

for weeks in ('34', '56', '78', '910'):
    with open('train-data/' + 'usapcp' + '-daily-mean.pickle', 'rb') as fd:
        A_mean = pickle.load(fd)
    a = np.ma.masked_all((xmax + 1, ymax + 1))
    for loc in range(len(points)):
        index = points[loc]
        if weeks == '34':
            date = target_date + 2 * ONE_WEEK
            prediction = precip_wk34_prediction
        elif weeks == '56':
            date = target_date + 4 * ONE_WEEK
            prediction = precip_wk56_prediction
        elif weeks == '78':
            date = target_date + 6 * ONE_WEEK
            prediction = precip_wk78_prediction
        elif weeks == '910':
            date = target_date + 8 * ONE_WEEK
            prediction = precip_wk91_prediction
        total = 0
        for j in range(14):
            total += A_mean[loc, day2index(date)]
            date += ONE_DAY
        a[tuple(index)] = prediction[loc] - total

    plt.figure()
    plt.title(str(target_date) + ' ' + 'apcp' + ' weeks ' + weeks + ' anom')
    plt.imshow(a, cmap=plt.get_cmap('BrBG'), vmin=-40, vmax=40)
    plt.colorbar()
    plt.pause(0.001)
    plt.savefig('uspredictions/' + datestr + '/' + 'apcp' + '-wks' + weeks + '-' + datestr + '-anom.png', bbox_inches='tight')

