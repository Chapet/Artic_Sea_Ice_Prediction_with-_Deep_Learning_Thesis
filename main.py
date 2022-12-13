import os
import sys
import time
from netCDF4 import Dataset
import numpy as np
import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import UpSampling2D, concatenate, MaxPooling2D, Input, Conv2D
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.metrics import BinaryAccuracy
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

#### GPU SETUP ####
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
###################

#### CONSTANTS ####
SIC_LOCATION_NC = "/media/loulaurent/HDD/SIC/v2p0/"
SIC_LOCATION_NUMPY = "/media/loulaurent/HDD/SIC/npy/"
SIC_LOCATION_MODELS = "/media/loulaurent/HDD/SIC/models/"
SIC_LOCATION_RESULTS = "/media/loulaurent/HDD/SIC/results/"
SIC_LOCATION_PLOTS = "/media/loulaurent/HDD/SIC/plots/"
SIC_LOCATION_NCDF = "/media/loulaurent/HDD/SIC/ncdf/"

###################
# reminder: my_func // x, my_variable // Model, MyClass // MY_CTT // module.py, my_mod.py // package, mypackage
# nvtop to monitor gpu usage
###################

################## ON HOLD #################

def diff(li1, li2):  # Python code to get difference of two lists not using set()
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


def forecast_accuracy(forecast, actual):
    np.seterr(divide='ignore', invalid='ignore')
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE cannot be used because there are zeros
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE cannot be used because there are zeros
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    mse = np.mean((forecast - actual) ** 2)  # MSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    return {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'mse': mse, 'rmse': rmse, 'corr': corr}


def load_dfs(start=1976, end=2021):
    dfs = {}
    for year in range(start, end + 1):
        df = pd.read_pickle("./res/pickle/df_" + str(year) + ".pkl")
        dfs[year] = df

    return dfs


def get_day_information(file):
    data = Dataset(file, mode='r')
    time_ = data.variables['time'][:]
    date1 = datetime.date.fromtimestamp(time_[0]) + datetime.timedelta(
        days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)

    conc_array = data.variables['ice_conc'][:][0].ravel()
    conc_array = conc_array.filled(0)  # replace missing values with zeros
    conc_array = np.divide(conc_array, 100)  # normalize concentration between 0 and 1
    conc_array = np.array(conc_array)
    conc_array = conc_array.reshape((432, 432))

    day_of_year_str = date1.strftime("%j")  # e.g. '065'
    day_of_year = int(day_of_year_str)  # e.g. 65

    day_array1 = np.empty((432, 432))
    day_array1.fill(day_of_year)

    no_ice_array = np.zeros((432, 432))
    ice_array = np.zeros((432, 432))
    for row in range(432):
        for col in range(432):
            e1 = conc_array[row][col]
            if e1 >= 0.15:
                ice_array[row][col] = 1
            else:
                no_ice_array[row][col] = 1

    final_array = np.stack([conc_array, day_array1, no_ice_array, ice_array], axis=2)  # shape (432,432,4)
    return final_array


def svm_1y():
    from sklearn.svm import SVR
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    dfs = load_dfs(start=1979, end=2021)
    for year, df in dfs.items():
        list_6m = []
        list_september = []
        for key, value in sorted(df[0].items(), key=lambda x: x[0]):  # key = date_str, value = SIE
            sorted_month = int(key[5:7])
            if 1 < sorted_month < 8:
                list_6m.append(value * 625)
            elif sorted_month == 9:
                list_september.append(value * 625)

        if year < 2016:
            X_train.append(list_6m)
            y_train.append(np.mean(list_september))
        else:
            X_test.append(list_6m)
            y_test.append(np.mean(list_september))

    print("X_train:")
    print(X_train)
    print("y_train:")
    print(y_train)
    print("X_test:")
    print(X_test)
    print("y_test:")
    print(y_test)
    # print([x * 625 for x in y_test])

    svr = SVR(kernel='linear')  # see appendix C of guide to SVM -> visibly every case is different
    svr.fit(X_train, y_train)

    forecast = svr.predict(X_test)
    print("forecast:")
    print(forecast)
    actual = y_test
    # print(svr.score(X_test, y_test))  # coefficient of determination of the prediction

    acc = forecast_accuracy(forecast, actual)
    for e in acc.items():
        print(e)


################### KEEP ###################

# takes the epoch time and returns the year, month and day as integers
def epoch_to_datetime(epoch_time):
    timestamp = datetime.datetime.fromtimestamp(epoch_time)
    year = int(timestamp.strftime("%Y"))
    month = int(timestamp.strftime("%m"))
    day = int(timestamp.strftime("%d"))
    return year, month, day

# takes a filename from the npy folder and returns the corresponding epoch
def file_to_epoch(filename):
    # filename format is ice_class_nh_YYYYMMDD.npy OR ice_conc_nh_YYYYMMDD.npy
    y = int(filename[-12:-8])
    m = int(filename[-8:-6])
    d = int(filename[-6:-4])
    return datetime.datetime(y, m, d).timestamp()


def load_day(epoch_time):
    y, m, d = epoch_to_datetime(epoch_time)  # year, month, day of the target day
    day_path = SIC_LOCATION_NUMPY + "{}/{:02d}/".format(y, m) + "ice_class_nh_{}{:02d}{:02d}.npy".format(y, m, d)
    if os.path.exists(day_path):
        day = np.load(day_path)
        return True, day
    return False, 0


def train(num_epoch, model, loss_fn, acc_metric, optimizer, lead_time):
    print("Starting training loop")

    epoch_start = 284040000  # start at epoch of 1/1/1979 then increment day by day
    epoch_end = 1483185600  # end at epoch of 31/12/2016
    epoch_end_full = 1640952000  # end at epoch of 31/12/2021
    epoch_day = 86400  # one day is 86400 sec
    epoch_month = 30 * epoch_day  # prediction time of one month

    for epoch_it in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch_it + 1}")

        # instead of going over the years, give the start years and the duration, then use the epoch
        # to transform into date

        for epoch_time in range(epoch_start + (lead_time + 1) * epoch_month, epoch_end + epoch_day, epoch_day):
            # load day to predict
            exists, target_day = load_day(epoch_time)
            if not exists:
                continue  # if the day is not available, it can't be used for training

            # try to load day to predict from (1 month before), if not +-1, if still not, skip this day
            exists, day_1m = load_day(epoch_time - lead_time * epoch_month)
            if not exists:
                exists, day_1m = load_day(epoch_time - lead_time * epoch_month + epoch_day)
                if not exists:
                    exists, day_1m = load_day(epoch_time - lead_time * epoch_month - epoch_day)
                    if not exists:
                        continue  # if the day 30, 31 or 29 day before is not available, skip this day for training

            # try to load day to predict from (2 month before), if not +-1, if still not, skip the day for training
            exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month)
            if not exists:
                exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month + epoch_day)
                if not exists:
                    exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month - epoch_day)
                    if not exists:
                        continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

            # if we arrive here, we have target_day and day_1m which have the format (432, 432)

            # actual training
            X_train = np.reshape(np.stack((day_2m, day_1m), axis=2), (1, 432, 432, 2))
            y_train = np.reshape(target_day, (1, 432, 432, 1))
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_ds = train_ds.batch(1)  # .shuffle(buffer_size=1024) entre train_ds et batch
            loss_saved = 0

            for batch_idx, (x_batch, y_batch) in enumerate(train_ds):  # training over 1 batch
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True)
                    loss = loss_fn(y_batch, y_pred)

                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))
                acc_metric.update_state(y_batch, y_pred)
                loss_saved = loss

            # print(f"metric: {acc_metric.result()} and loss: {loss_saved}")

    model.save(SIC_LOCATION_MODELS + "50ep_3dates_1mSpacing_" + str(lead_time) + "m")
    print("model saved")

    # # evaluates the model on the test set
    # model = load_model(SIC_LOCATION_MODELS + "50ep_3dates_1mSpacing")
    # print("model loaded")
    #
    # # test loop
    # for epoch_time in range(epoch_end + epoch_day, epoch_end_full + epoch_day, epoch_day):
    #     # load day to predict
    #     exists, target_day = load_day(epoch_time)
    #     if not exists:
    #         continue  # if the day is not available, it can't be used for training
    #
    #     # try to load day to predict from (1 month before), if not +-1, if still not, skip this day
    #     exists, day_1m = load_day(epoch_time - epoch_month)
    #     if not exists:
    #         exists, day_1m = load_day(epoch_time - epoch_month + epoch_day)
    #         if not exists:
    #             exists, day_1m = load_day(epoch_time - epoch_month - epoch_day)
    #             if not exists:
    #                 continue  # if the day 30,31 or 29 day before is not available, skip this day for training
    #
    #     # try to load day to predict from (2 month before), if not +-1, if still not, skip this day
    #     exists, day_2m = load_day(epoch_time - 2 * epoch_month)
    #     if not exists:
    #         exists, day_2m = load_day(epoch_time - 2 * epoch_month + epoch_day)
    #         if not exists:
    #             exists, day_2m = load_day(epoch_time - 2 * epoch_month - epoch_day)
    #             if not exists:
    #                 continue  # if the day 30,31 or 29 day before is not available, skip this day for training
    #
    #     # if we arrive here, we have target_day and day_1m which have the format (432, 432)
    #
    #     # actual testing
    #     X_test = np.reshape(np.stack((day_2m, day_1m), axis=2), (1, 432, 432, 2))
    #     y_test = np.reshape(target_day, (1, 432, 432, 1))
    #     test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    #     test_ds = test_ds.batch(1)  # .shuffle(buffer_size=1024) entre train_ds et batch
    #
    #     for batch_idx, (x_batch, y_batch) in enumerate(test_ds):  # training over 1 batch
    #         y_pred = model(x_batch)
    #         acc_metric.update_state(y_batch, y_pred)
    #
    # test_acc = acc_metric.result()
    # print(f"Metric over test set: {test_acc}")
    # acc_metric.reset_states()


def go_through_files():
    def map_fun(elem):
        if elem < 0.5:
            return 0
        return 1

    apply_map = np.vectorize(map_fun)

    # script going through the SIC files and doing something (delete sh, convert to .np)
    for y in range(1979, 2022):
        for m in range(1, 13):
            for d in range(1, 32):
                var = "cdr"
                if y > 2015:
                    var = "icdr"

                filename = "ice_conc_nh_ease2-250_{}-v2p0_{}{:02d}{:02d}1200.nc".format(var, y, m, d)
                path_to_file = SIC_LOCATION_NC + "{}/{:02d}/".format(y, m) + filename

                if os.path.exists(path_to_file):
                    data = Dataset(path_to_file, mode='r')
                    conc_array = data.variables['ice_conc'][:][0].ravel()
                    conc_array = conc_array.filled(0)  # replace missing values with zeros (mainly water areas)
                    conc_array = np.divide(conc_array, 100)  # normalize concentration between 0 and 1

                    conc_array = np.array(conc_array)
                    conc_array = apply_map(conc_array)  # applying the function map_fun to all elements of the array
                    conc_array = conc_array.reshape((432, 432))

                    npy_path = SIC_LOCATION_NUMPY + "{}/{:02d}/".format(y, m)
                    if not os.path.exists(npy_path):
                        os.makedirs(npy_path)
                    np.save(npy_path + "ice_class_nh_{}{:02d}{:02d}.npy".format(y, m, d), conc_array)


def interpolate_gaps():  # there shouldn't be any interpolation yet
    # when there is a gap, register a start_gap event
    # when the gap ends, register an end_gap event & interpolate missing values

    # special cases = days missing at the start or at the end -> just add them at the beginning
    pass

    # start at epoch of 1/1/1979 then increment day by day
    epoch_cur = 284040000
    gap_size = 0  # used to count & check if we are currently in a gap
    # from 1/1/1979 to 31/12/2021 => 15705 days (1 day = 86400 seconds)
    for i in range(15705):
        # get year, month & day from the epoch time
        pass

        # check if the file corresponding to that date exists
        # yes -> if gap_size == 0 => do nothing (so this will not appear)
        #     -> if gap_size > 0 => end gap & compute
        # no -> increment gap_size

        epoch_cur += 86400  # increment day


def get_model(model_type, filter_size=3, n_filters_factor=1, n_output_classes=1):
    inputs = Input(shape=(432, 432, 2))

    if model_type == 1:
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)

        up8 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv5))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(int(64 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)

        final_layer_logits = Conv2D(n_output_classes, 1, activation='linear')(conv9)
        final_layer = tf.nn.sigmoid(final_layer_logits)  # to switch conc/class (relu/sigmoid)

        model = Model(inputs, final_layer)
        return model

    if model_type == 2:
        conv1 = Conv2D(32, kernel_size=1, strides=1, activation='relu', padding='same')(inputs)
        final_layer_logits = Conv2D(n_output_classes, kernel_size=1, activation='linear')(conv1)
        final_layer = tf.nn.sigmoid(final_layer_logits)  # to switch conc/class (relu/sigmoid)

        model = Model(inputs, final_layer)
        return model

    if model_type == 3:
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)

        up8 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv5))
        merge8 = concatenate([conv1, up8], axis=3)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        final_layer_logits = Conv2D(n_output_classes, 1, activation='linear')(conv8)
        final_layer = tf.nn.sigmoid(final_layer_logits)  # to switch conc/class (relu/sigmoid)

        model = Model(inputs, final_layer)
        return model

    # model is too big for available hardware
    if model_type == 4:  # model with more depth but not runnable due to hardware limitations
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv5 = Conv2D(int(256 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)

        up7 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv5))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(int(64 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2), interpolation='nearest')(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(int(64 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)

        final_layer_logits = Conv2D(n_output_classes, 1, activation='linear')(conv9)
        final_layer = tf.nn.sigmoid(final_layer_logits)  # to switch conc/class (relu/sigmoid)

        model = Model(inputs, final_layer)
        return model

    else:
        print("no model corresponding to that number")
        return 0


def create_heatpmap_df(metric):
    max_extent_months = np.load(SIC_LOCATION_RESULTS + "max_extent_months.npy")  # shape (12, 432, 432)
    result_df = np.zeros((12, 6))  # 12 months, 6 lead times
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for lead_time in range(1, 7):
        model = load_model(SIC_LOCATION_MODELS + "50ep_3dates_1mSpacing_" + str(lead_time) + "m")

        for month in range(0, 12):
            # parcourir les années et à chaque fois faire le calcul sur les jours dans les folders des mois

            # pour ce faire: on récup chaque jour via le file system. Avec la date on récupère l'epoch, puis on opère
            # de la même manière que dans la fonction train

            for year in os.listdir(SIC_LOCATION_NUMPY):
                if int(year) < 2016:  # only test years (2016-2021)
                    continue
                files = os.listdir(SIC_LOCATION_NUMPY + year + "/{:02d}/".format(month+1))
                filtered_files = filter(lambda file: file[4:9] == "class", files)  # get only the class training files
                for file in filtered_files:

                    epoch_time = file_to_epoch(file)
                    epoch_day = 86400  # one day is 86400 sec
                    epoch_month = 30 * epoch_day  # prediction time of one month

                    # load day to predict
                    exists, target_day = load_day(epoch_time)
                    if not exists:
                        continue  # if the day is not available, it can't be used for training

                    # try to load day to predict from (lead_times month(s) before), if not +-1, if still not, skip this day
                    exists, day_1m = load_day(epoch_time - lead_time * epoch_month)
                    if not exists:
                        exists, day_1m = load_day(epoch_time - lead_time * epoch_month + epoch_day)
                        if not exists:
                            exists, day_1m = load_day(epoch_time - lead_time * epoch_month - epoch_day)
                            if not exists:
                                continue  # if the day 30, 31 or 29 day before is not available, skip this day for training

                    # try to load day to predict from (lead_time+1 months before), if not +-1, if still not, skip the day for training
                    exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month)
                    if not exists:
                        exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month + epoch_day)
                        if not exists:
                            exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month - epoch_day)
                            if not exists:
                                continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

                    # actual testing
                    X_test = np.reshape(np.stack((day_2m, day_1m), axis=2), (1, 432, 432, 2))
                    y_test = np.reshape(target_day, (1, 432, 432, 1))
                    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                    test_ds = test_ds.batch(1)  # .shuffle(buffer_size=1024) entre train_ds et batch

                    for batch_idx, (x_batch, y_batch) in enumerate(test_ds):  # training over 1 batch
                        y_pred = model(x_batch)

                        # we update the metric but only take into account the cells in the observed max extent
                        metric.update_state(np.reshape(y_batch, (186624, 1)), np.reshape(y_pred, (186624, 1)),
                                            sample_weight=np.reshape(max_extent_months[month], 186624))


            result_df[(month) % 12][lead_time - 1] = metric.result()
            print(f"metric for the month \"{months[month]}\" with a lead time of {lead_time}: {metric.result()}")
            metric.reset_state()

    print(result_df)
    np.save(SIC_LOCATION_RESULTS + "full_U-Net_plotDF", result_df)


def plot_heatmap(model1, model2):  # model1 is the base model, model2 is the model to compare model1 with if model2 = 0, we simply display the result of model1
    sns.set_theme()

    save_fn = str(model1) + ".png"
    compMode = False
    result_icenet = [[96.9, 96.4, 96.4, 96.4, 96.3, 96.3],
                     [96.9, 96, 95.8, 95.8, 95.7, 95.7],
                     [96.9, 95.7, 95.3, 95, 95.1, 95.1],
                     [97.1, 95.9, 95.4, 95.3, 95.1, 95.1],
                     [97.5, 96.6, 96.3, 96, 95.7, 95.6],
                     [96, 94.5, 94.1, 94.1, 93.9, 93.7],
                     [94.2, 91.7, 90.7, 90.6, 90.9, 90.5],
                     [94, 92.1, 91, 90.5, 90.5, 90.2],
                     [94.3, 92.9, 92.2, 91.1, 90.4, 90.4],
                     [93, 92.4, 92, 91.8, 90.7, 89.8],
                     [95.4, 95.3, 94.6, 94.6, 94.7, 94.8],
                     [96.9, 96.4, 96.3, 96.2, 96.3, 96.3]]

    if model1 == 1 or model1 == 2 or model1 == 3 or model1 == 4:
        model1_df = np.load(SIC_LOCATION_RESULTS + "50ep_3dates_1mSpacing_plotDF.npy")  # to change

        if model1 == 1:
            t = "U-Net 1 model"
        elif model1 == 2:
            t = "U-Net 2 model"
        elif model1 == 3:
            t = "Simple Convolution model"
        elif model1 == 4:
            t = "Trivial model"

    elif model1 == 5:
        t = "Icenet"
        model1_df = np.empty((12, 6))
        for i in range(12):
            for j in range(6):
                model1_df[i, j] = result_icenet[i][j] / 100

    if model2 != 0:
        compMode = True
        save_fn = str(model1) + "vs" + str(model2) + ".png"
        t += " comparison with the "
        if model2 == 1 or model2 == 2 or model2 == 3 or model2 == 4:
            model2_df = np.load('./res/results/model_' + str(model2) + '_1_maxextent_correct.npy')

            if model2 == 1:
                t += "U-Net 1 model"
            elif model2 == 2:
                t += "U-Net 2 model"
            elif model2 == 3:
                t += "Simple Convolution model"
            elif model2 == 4:
                t += "Trivial model"

        elif model2 == 5:
            t += "Icenet"
            model2_df = np.empty((12, 6))
            for i in range(12):
                for j in range(6):
                    model2_df[i, j] = result_icenet[i][j] / 100

    x_labels = [1, 2, 3, 4, 5, 6]
    y_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if compMode:
        ax = sns.heatmap(model1_df[:, :] * 100 - model2_df[:, :] * 100, annot=True, vmin=-30, vmax=30,
                         xticklabels=x_labels, yticklabels=y_labels, fmt='.3g', cmap='RdYlGn')
    else:  # simply model1 values
        ax = sns.heatmap(model1_df[:, :] * 100, annot=True, vmin=50, vmax=100, xticklabels=x_labels,
                         yticklabels=y_labels, fmt='.3g', cmap='RdYlGn')

    ax.set_title(t, fontsize=14)

    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Lead Time (months)', fontweight='bold')
    ax.set_ylabel('Calendar month', fontweight='bold')
    cbar = ax.collections[0].colorbar
    cbar.set_label("Binary Accuracy (%)", fontsize=15)

    plt.savefig(SIC_LOCATION_PLOTS + save_fn, format='png')
    plt.show()


def get_max_extent(year_range):
    max_extent = np.zeros((432, 432))
    max_extent_months = np.zeros((12, 432, 432))
    for y in year_range:
        for m in ["%.2d" % i for i in range(1, 13)]:
            if not os.path.exists(SIC_LOCATION_NC + str(y) + '/' + m):
                continue
            days_files = os.listdir(SIC_LOCATION_NC + str(y) + '/' + m)

            for day in days_files:
                file = SIC_LOCATION_NC + str(y) + '/' + m + '/' + day
                day_data = Dataset(file, mode='r')
                day_conc = day_data.variables['ice_conc'][0]
                day_conc = np.array(day_conc)

                for i in range(432):
                    for j in range(432):
                        if day_conc[i][j] > 0.15:
                            max_extent[i][j] = 1
                            max_extent_months[int(m) - 1][i][j] = 1
                day_data.close()
        print(f"year {y} done")
    np.save(SIC_LOCATION_RESULTS + "max_extent_months.npy", max_extent_months)
    np.save(SIC_LOCATION_RESULTS + "max_extent.npy", max_extent)
    return max_extent


def create_ncdf_file(data_array, filename):  # data_array: a 432x432 npy array containing sea ice values
    ref = Dataset(SIC_LOCATION_NCDF + "ice_conc_nh_ease2-250_icdr-v2p0_201809061200.nc", mode='r')
    fn = SIC_LOCATION_NCDF + filename
    ds = Dataset(fn, mode='w', format='NETCDF4')

    time = ds.createDimension('time', None)
    xc = ds.createDimension('xc', 432)
    yc = ds.createDimension('yc', 432)

    xcs = ds.createVariable('xc', 'f4', ('xc',))
    xcs.units = 'km'
    xcs.axis = 'X'
    ycs = ds.createVariable('yc', 'f4', ('yc',))
    ycs.units = 'km'
    ycs.axis = 'Y'

    lats = ds.createVariable('lat', 'f4', ('yc', 'xc',))
    lats.unit = 'degrees_north'
    lons = ds.createVariable('lon', 'f4', ('yc', 'xc',))
    lons.units = 'degrees_east'

    times = ds.createVariable('time', 'f4', ('time',))
    times.axis = 'T'

    ice_conc = ds.createVariable('ice_conc', 'f4', ('time', 'yc', 'xc',))
    ice_conc.units = '%'
    ice_conc.grid_mapping = 'Lambert_Azimuthal_Grid'

    xcs[:] = ref.variables['xc'][:]
    ycs[:] = ref.variables['yc'][:]
    lats[:] = ref.variables['lat'][:][:]
    lons[:] = ref.variables['lon'][:][:]

    shape = data_array.shape
    ice_conc[0:shape[0], :, :] = data_array

    ds.close()
    ref.close()


def CNN_model(loss=BinaryCrossentropy(), metric=BinaryAccuracy(), learning_rate=1e-4):  # to switch conc/class mse+mse&acc/bin+bin

    model = get_model(1)  # select the model to use
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    model.summary(line_length=120)
    train(50, model, loss, metric, optimizer, 5)  # last var is lead_time

    return model


################### RUN ####################
start_time = time.time()
print(f"Run start: {datetime.datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")

np.set_printoptions(threshold=sys.maxsize)


CNN_model()
# get_max_extent(range(1979, 2022))
# create_heatpmap_df(BinaryAccuracy())
# plot_heatmap(2, 0)
# a = np.ones((432, 432))
# create_ncdf_file(a, "test.nc")

# go_through_files()


print(f"Run end: {datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}, "
      f"total elapsed time: {datetime.datetime.fromtimestamp(time.time() - start_time).strftime('%H:%M:%S')}")
