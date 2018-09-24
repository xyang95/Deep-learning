import pickle
import keras
import numpy as np
import matplotlib.pyplot as plt
import click
import pprint
import random
import itertools
import os
import re
import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import load_model
from keras import metrics
from shutil import copyfile

batch_size = 128
train_ratio = .89
# tsteps = 1
window_size = 10
valid_last = False
number_points = 3538


def combine_weeks(myarray, a_type):
    # format precip data to be of i'th week + (i+1)'th week
    if a_type == 'precip':
        newarray = np.zeros((myarray.shape[0], myarray.shape[1] - 1), dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:, i] = myarray[:, i] + myarray[:, i + 1]

    if a_type == 'temp':
        newarray = np.zeros((myarray.shape[0], myarray.shape[1] - 1), dtype=np.float32)
        for i in range(newarray.shape[1]):
            newarray[:, i] = (myarray[:, i] + myarray[:, i + 1]) / 2

    # merge sst or sss data into 2 week periods from 1 week periods (obsolete)
    # if len(myarray) % 2 == 1:
    #   myarray = myarray[:-1]
    # if a_type == 'ss':
    #   newarray = np.zeros((int(myarray.shape[0]/2),myarray.shape[1]),dtype=np.float32)
    #   for i in range(len(newarray)):
    #       newarray[i,:] = np.divide(np.add(myarray[i*2,:],myarray[i*2+1,:]),2.)

    return newarray


def generate_windows(x):
    '''Sliding window over the data of length window_size'''
    result = []
    for i in range(x.shape[0] - window_size):
        result.append(x[i:i + window_size, ...])
    return np.stack(result)


def pearson_loss(y_true, y_pred):
    # TODO: fix this. replace tf. with K.
    x = y_pred
    y = y_true
    cov = K.mean(K.multiply(x - K.tile(K.mean(x, 1, keepdims=True), [1, num_labels]),
                            y - K.tile(K.mean(y, 1, keepdims=True), [1, num_labels])), 1)
    varx = K.mean(K.square(x - K.tile(K.mean(x, 1, keepdims=True), [1, num_labels])), 1)
    vary = K.mean(K.square(y - K.tile(K.mean(y, 1, keepdims=True), [1, num_labels])), 1)
    pearson = cov / K.sqrt(K.multiply(varx, vary))
    ploss = K.mean(pearson * -1) + 1
    return ploss


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def mean_loss_i(i):
    def i_error(y_true, y_pred):
        start = i * number_points
        end = (i + 1) * number_points
        y_true = y_true[..., start:end]
        y_pred = y_pred[..., start:end]
        err = K.abs(y_true - y_pred)
        return K.mean(err)

    return i_error


def val_i_error(y_true, y_pred, i):
    start = i * number_points
    end = (i + 1) * number_points
    y_true = y_true[..., start:end]
    y_pred = y_pred[..., start:end]
    err = np.abs(y_true - y_pred)
    mean_err = np.mean(err)
    return mean_err

def points_i_error(y_true, y_pred, i):
    one = i
    two = i + number_points
    three = i + 2 * number_points
    four =  i + 3 * number_points
    y_true = y_true[..., [one, two, three, four]]
    y_pred = y_pred[..., [one, two, three, four]]
    err = np.abs(y_true - y_pred)
    mean_err = np.mean(err)
    return mean_err

def points_i_error_nomean(y_true, y_pred, i):
    one = i
    two = i + number_points
    three = i + 2 * number_points
    four =  i + 3 * number_points
    # y_tru = y_true[..., [one, two, three, four]]
    # y_pre = y_pred[..., [one, two, three, four]]
    y_tru = y_true[..., two]
    y_pre = y_pred[..., two]
    err = np.abs(y_tru - y_pre)
    return err



def train(x, y, activation='lrelu', epochs=200, units=300, depth=3):
    # Set up number of metrics to be used
    N_metrics = int(y.shape[1] / number_points)
    metrics_vector = '['
    for i in range(N_metrics):
        metrics_vector += 'mean_loss_i(' + str(i) + ')'
        if i < N_metrics - 1:
            metrics_vector += ', '
    metrics_vector += ']'

    # Set up lrelu usage
    lrelu = False
    if activation == 'lrelu':
        lrelu = True
        activation = 'linear'

    num_weeks = x.shape[0]

    x_train = x[:int(train_ratio * num_weeks), ...]
    x_test = x[int(train_ratio * num_weeks):, ...]
    y_train = y[:int(train_ratio * num_weeks), ...]
    y_test = y[int(train_ratio * num_weeks):, ...]
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # learning_rate = 0.4
    # lr_end_fraction = 0.01
    # decay_rate = (1-lr_end_fraction)/lr_end_fraction/epochs

    model = Sequential()
    model.add(Dense(units, input_shape=x_train.shape[1:], activation=activation))
    if lrelu:
        model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dropout(0.25))

    for i in range(depth - 2):
        model.add(Dense(units, activation=activation))
        if lrelu:
            model.add(LeakyReLU(alpha=.2))
        model.add(Dropout(0.25))

    # model.add(LSTM(50,
    #                input_shape=(tsteps, x.shape[1]),
    #                batch_size=batch_size,
    #                return_sequences=True,
    #                stateful=True))
    # model.add(LSTM(50,
    #                return_sequences=False,
    #                stateful=True))
    model.add(Dense(y_train.shape[1], activation='linear'))
    model.add(LeakyReLU(alpha=.05))
    # sgd = keras.optimizers.SGD(lr=learning_rate, decay=decay_rate, nesterov=False)
    # model.compile(loss=keras.losses.mean_absolute_error, optimizer=sgd)
    adam_opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    # model.compile(loss=keras.losses.mean_absolute_error, optimizer=adam_opt, metrics=eval(metrics_vector))
    model.compile(loss=keras.losses.mean_squared_error, optimizer=adam_opt)
    # model.summary()

    # tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        # callbacks=[tbCallBack],
                        validation_data=(x_test, y_test),
                        shuffle=False)

    val_prediction = model.predict(x_test)
    val_metrics = np.zeros(N_metrics)
    val_points_error_nomean = np.zeros((y_test.shape[0], number_points))
    for i in range(N_metrics):
        val_metrics[...,i] = val_i_error(y_test, val_prediction, i)
    # for i in range(number_points):
    #     val_points_error[i] = points_i_error(y_test, val_prediction, i)
    for i in range(number_points):
        val_points_error_nomean[..., i] = points_i_error_nomean(y_test, val_prediction, i)
    return model, history, val_metrics, val_points_error_nomean


def reshape_data(x, y):
    x = generate_windows(x)
    # Truncate first few y values since there aren't enough preceding weeks to predict on.
    y = y[window_size:, ...]
    # Shuffle data or not
    if valid_last:
        indices = np.arange(x.shape[0])
    else:
        # np.random.seed(seed=1337)
        indices = np.random.permutation(x.shape[0])
        # np.random.seed()
    x = x[indices, ...]
    y = y[indices, ...]
    return x, y


def accuracy(predictions, labels):
    return (np.mean(np.abs(predictions - labels)))


def compile_input(n_weeks_predict, combine=False, i_sss=False, i_precip=False, i_time=False):
    # Load data
    # load time data
    time_data_file = 'train-data/time.pickle'  # 1414 x 1
    time_vectors = pickle.load(open(time_data_file, 'rb'))
    time_vectors = np.reshape(time_vectors, (time_vectors.shape[0], 1))

    # load sst data
    sst_data_file = 'train-data/sst.pickle'  # 1414 x 8099
    sst_vectors = pickle.load(open(sst_data_file, 'rb'))

    # load sss data
    sss_data_file = 'train-data/sss.pickle'  # 1414 x 8099
    sss_vectors = pickle.load(open(sss_data_file, 'rb'))

    # load precipitation data
    location_precip_file = 'train-data/afprecip.pickle'  # 514 x 1299
    precip_data = pickle.load(open(location_precip_file, 'rb'))
    if combine:
        precip_data = combine_weeks(precip_data, 'precip')
    precip_data = precip_data.T

    # load temperature data

    # make precip data only as long as temp data
    precip_data = precip_data[:precip_data.shape[0], :]

    # ensure same length vectors
    time_vectors = time_vectors[:precip_data.shape[0], :]
    sst_vectors = sst_vectors[:precip_data.shape[0], :]
    sst_vectors = (sst_vectors - np.amin(sst_vectors)) * 1. / (np.amax(sst_vectors) - np.amin(sst_vectors))
    sss_vectors = sss_vectors[:precip_data.shape[0], :]
    sss_vectors = (sss_vectors - np.amin(sss_vectors)) * 1. / (np.amax(sss_vectors) - np.amin(sss_vectors))
    precip_input = precip_data[:precip_data.shape[0], :]
    precip_input = (precip_input - np.amin(precip_input)) * 1. / (np.amax(precip_input) - np.amin(precip_input))

    max_weeks_predict = np.amax(n_weeks_predict)
    # Can't use the last weeks because there wouldn't be enough data to predict.
    sst_vectors = sst_vectors[:-(1 + max_weeks_predict), ...] #1404x2040
    sss_vectors = sss_vectors[:-(1 + max_weeks_predict), ...] #1404x2040
    precip_input = precip_input[:-(1 + max_weeks_predict), ...] #1404x935
    time_vectors = time_vectors[:-(1 + max_weeks_predict), ...] #1404x1

    # compile input data
    input_data = sss_vectors
    if i_sss:
        input_data = np.concatenate((input_data, sss_vectors), axis=1)
    if i_precip:
        input_data = np.concatenate((input_data, precip_input), axis=1)
    if i_time:
        input_data = np.concatenate((input_data, time_vectors), axis=1) #1404x5016(2040 2040 935 1)

    # compile output data
    precip_data_all = np.zeros((sst_vectors.shape[0], precip_data.shape[1], len(n_weeks_predict)))  # (t,loc,wk) 1404 935 4
    for i in range(len(n_weeks_predict)):
        week = n_weeks_predict[i]
        # offset precip data
        precip_data_all[:, :, i] = precip_data[week:-(1 + max_weeks_predict - week), :]  # (t,loc,wk)
    precip_data_all = np.rollaxis(precip_data_all, 2, 1)  # (t,wk,loc) 1404 4 935
    precip_data_all = precip_data_all.reshape(precip_data_all.shape[0], -1)  # (t,wkloc) 1404 3740

    output_data_all = precip_data_all

    return input_data, output_data_all


def main():
    # Clean out old temp model files

    # Define weeks to predict


    n_weeks_predict = [9, 10, 11, 12]

    input_sets = list(itertools.product([0, 1], repeat=3))
    activations = ['elu', 'lrelu']  # ['relu', 'linear', 'tanh', 'sigmoid', 'elu', 'lrelu']

    n_random_tries = 30
    histories = []
    models = []
    val_points_error_total = np.zeros((50, 153, number_points))
    for i in range(n_random_tries):
        print('Training model ' + str(i))

        # input_set = input_sets[random.randint(0,7)]
        # input_data, output_data = compile_input(n_weeks_predict, i_sss=bool(input_set[0]), i_precip=bool(input_set[1]), i_time=bool(input_set[2]))
        input_set = random.randint(0, 1)
        input_data, output_data = compile_input(n_weeks_predict, combine=False, i_sss=False, i_precip=False,
                                                i_time=bool(input_set))

        activation = activations[random.randint(0, len(activations) - 1)]
        units = random.randint(300, 800)
        layers = random.randint(3, 7)
        epochs = random.randint(600, 800)
        # epochs = random.randint(1,1)

        model, history, val_metrics, val_points_error_nomean = train(*reshape_data(input_data, output_data),
                                            activation=activation,
                                            units=units,
                                            depth=layers,
                                            epochs=epochs
                                            )

        model_name = 'train-results/af_tmp_k_model_' + str(i) + '_time=' + str(input_set) + '.h5'
        model.save(model_name)
        models.append(model_name)

        history_i = [
            input_set,
            activation,
            units,
            layers,
            epochs,
            history.history['loss'][-1],
            history.history['val_loss'][-1]
        ]
        history_i.extend(list(val_metrics))
        # history_i.extend(list(val_points_error))
        # history_i.extend(list(val_points_error_nomean))
        val_points_error_total[i] = val_points_error_nomean

        histories.append(history_i)
        print(history_i)

    # print(history.history)
    print('')
    print('All Results:')
    # pprint.pprint(histories)
    [print(item) for item in histories]
    val_losses = [item[6] for item in histories]
    val_losses = np.asarray(val_losses)
    best_inds = val_losses.argsort()[:10]

    # Remove old models

    # List and copy best models
    print('')
    print('Best Results:')
    for i in range(len(best_inds)):
        print(histories[best_inds[i]])
        result = re.search('time=(.*).h5', models[best_inds[i]])
        input_set = int(result.group(1))
        model_name = 'train-results/af91_k_model_' + str(i) + '_time=' + str(input_set) + '.h5'
        copyfile(models[best_inds[i]], model_name)
    with open('error/best_inds1.pickle', 'wb') as fd:
        pickle.dump(best_inds, fd)
    with open('error/afoints_error_totals.pickle', 'wb') as fd:
        pickle.dump(val_points_error_total, fd)
    val_points_error = val_points_error_total[[best_inds]]
    with open('error/afpoints_errors.pickle', 'wb') as fd:
        pickle.dump(val_points_error, fd)
    # Clean out temp files
    tmp_models = glob.glob('train-results/af_tmp_k_model_*.h5')
    for item in tmp_models:
        os.remove(item)

    # num_weeks = input_data.shape[0]
    # x = generate_windows(input_data[int(train_ratio*num_weeks):-13,...])
    # y = precip_data[int(train_ratio*num_weeks)+window_size:-13,...]
    # val_predict = model.predict(x)
    # # loss = model.evaluate(x,y)

    # valid_labels = y

    # plt.figure(1, figsize=(16, 8))
    # plot_title = 'yo dawg'

    # plot_locations = [17,118,137,158,293,305,309,432,445,507]
    # plot_index = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13]
    # for i in range(len(plot_locations)):
    #     ax = plt.subplot(2,7,plot_index[i])
    #     ax.cla()
    #     ax.set_title('Location %d: %.2f' % (plot_locations[i]
    #         ,accuracy(val_predict[:,plot_locations[i]],valid_labels[:,plot_locations[i]]
    #     )))
    #     ax.plot(valid_labels[:,plot_locations[i]])
    #     ax.plot(val_predict[:,plot_locations[i]],linewidth=2.0)

    # ax = plt.subplot(2,7,7)
    # plt.show()


if __name__ == '__main__':
    main()