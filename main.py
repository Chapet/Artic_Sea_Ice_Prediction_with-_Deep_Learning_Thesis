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

    start_epoch_range = 0

    for epoch_it in range(start_epoch_range, num_epoch):
        print(f"\nStart of Training Epoch {epoch_it + 1}")

        # instead of going over the years, give the start years and the duration, then use the epoch
        # to transform into date

        # or epoch_time in range(epoch_start + (lead_time + 1) * epoch_month, epoch_end + epoch_day, epoch_day):
        for epoch_time in range(epoch_start + (11+lead_time) * epoch_month, epoch_end + epoch_day, epoch_day):
            # load day to predict
            exists, target_day = load_day(epoch_time)
            if not exists:
                continue  # if the day is not available, it can't be used for training

            day_12m = np.zeros((12, 432, 432))

            for i in range(12):
                # try to load day to predict from (2 month before), if not +-1, if still not, skip the day for training
                exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month)
                if not exists:
                    exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month + epoch_day)
                    if not exists:
                        exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month - epoch_day)
                        if not exists:
                            continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

            # # try to load day to predict from (1 month before), if not +-1, if still not, skip this day
            # exists, day_1m = load_day(epoch_time - lead_time * epoch_month)
            # if not exists:
            #     exists, day_1m = load_day(epoch_time - lead_time * epoch_month + epoch_day)
            #     if not exists:
            #         exists, day_1m = load_day(epoch_time - lead_time * epoch_month - epoch_day)
            #         if not exists:
            #             continue  # if the day 30, 31 or 29 day before is not available, skip this day for training
            #
            # # try to load day to predict from (2 month before), if not +-1, if still not, skip the day for training
            # exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month)
            # if not exists:
            #     exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month + epoch_day)
            #     if not exists:
            #         exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month - epoch_day)
            #         if not exists:
            #             continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

            # if we arrive here, we have target_day and day_1m which have the format (432, 432)

            # actual training
            X_train = np.reshape(np.stack(day_12m, axis=2), (1, 432, 432, 12))  # reverse day_12m has an impact ?
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

        model.save(SIC_LOCATION_MODELS + "50ep_12dates_1mSpacing_" + str(lead_time) + "m")

            # print(f"metric: {acc_metric.result()} and loss: {loss_saved}")

    model.save(SIC_LOCATION_MODELS + "50ep_12dates_1mSpacing_" + str(lead_time) + "m")
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


def get_model(model_type, filter_size=3, n_filters_factor=1, n_output_classes=1):
    inputs = Input(shape=(432, 432, 12))

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

        # model = load_model(SIC_LOCATION_MODELS + "50ep_12dates_1mSpacing_" + str(lead_time) + "m")

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

                    day_12m = np.zeros((12, 432, 432))

                    for i in range(12):
                        # try to load day to predict from (2 month before), if not +-1, if still not, skip the day for training
                        exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month)
                        if not exists:
                            exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month + epoch_day)
                            if not exists:
                                exists, day_12m[i] = load_day(epoch_time - (lead_time + i) * epoch_month - epoch_day)
                                if not exists:
                                    continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

                    # # try to load day to predict from (lead_times month(s) before), if not +-1, if still not, skip this day
                    # exists, day_1m = load_day(epoch_time - lead_time * epoch_month)
                    # if not exists:
                    #     exists, day_1m = load_day(epoch_time - lead_time * epoch_month + epoch_day)
                    #     if not exists:
                    #         exists, day_1m = load_day(epoch_time - lead_time * epoch_month - epoch_day)
                    #         if not exists:
                    #             continue  # if the day 30, 31 or 29 day before is not available, skip this day for training
                    #
                    # # try to load day to predict from (lead_time+1 months before), if not +-1, if still not, skip the day for training
                    # exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month)
                    # if not exists:
                    #     exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month + epoch_day)
                    #     if not exists:
                    #         exists, day_2m = load_day(epoch_time - (lead_time + 1) * epoch_month - epoch_day)
                    #         if not exists:
                    #             continue  # if the day 60, 61 or 59 day before is not available, skip the day for training

                    # actual testing
                    X_test = np.reshape(np.stack(day_12m, axis=2), (1, 432, 432, 12))
                    y_test = np.reshape(target_day, (1, 432, 432, 1))
                    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                    test_ds = test_ds.batch(1)  # .shuffle(buffer_size=1024) entre train_ds et batch

                    for batch_idx, (x_batch, y_batch) in enumerate(test_ds):  # training over 1 batch
                        # y_pred = model(x_batch)
                        y_pred = np.zeros((432, 432))

                        # we update the metric but only take into account the cells in the observed max extent
                        metric.update_state(np.reshape(y_batch, (186624, 1)), np.reshape(y_pred, (186624, 1)),
                                            sample_weight=np.reshape(max_extent_months[month], 186624))


            result_df[(month) % 12][lead_time - 1] = metric.result()
            print(f"metric for the month \"{months[month]}\" with a lead time of {lead_time}: {metric.result()}")
            metric.reset_state()

    print(result_df)
    np.save(SIC_LOCATION_RESULTS + "NoIce", result_df)


def plot_heatmap(compMode):  # model1 is the base model, model2 is the model to compare model1 with if model2 = 0, we simply display the result of model1
    sns.set_theme()

    result_icenet = [[0.969, 0.964, 0.964, 0.964, 0.963, 0.963],
                     [0.969, 0.960, 0.958, 0.958, 0.957, 0.957],
                     [0.969, 0.957, 0.953, 0.950, 0.951, 0.951],
                     [0.971, 0.959, 0.954, 0.953, 0.951, 0.951],
                     [0.975, 0.966, 0.963, 0.960, 0.957, 0.956],
                     [0.960, 0.945, 0.941, 0.941, 0.939, 0.937],
                     [0.942, 0.917, 0.907, 0.906, 0.909, 0.905],
                     [0.940, 0.921, 0.910, 0.905, 0.905, 0.902],
                     [0.943, 0.929, 0.922, 0.911, 0.904, 0.904],
                     [0.930, 0.924, 0.920, 0.918, 0.907, 0.898],
                     [0.954, 0.953, 0.946, 0.946, 0.947, 0.948],
                     [0.969, 0.964, 0.963, 0.962, 0.963, 0.963]]
    result_trivial = [[0.945, 0.945, 0.945, 0.945, 0.945, 0.945],
                     [0.933, 0.933, 0.933, 0.933, 0.933, 0.933],
                     [0.929, 0.929, 0.929, 0.929, 0.929, 0.929],
                     [0.941, 0.941, 0.941, 0.941, 0.941, 0.941],
                     [0.932, 0.932, 0.932, 0.932, 0.932, 0.932],
                     [0.882, 0.882, 0.882, 0.882, 0.882, 0.882],
                     [0.876, 0.876, 0.876, 0.876, 0.876, 0.876],
                     [0.885, 0.885, 0.885, 0.885, 0.885, 0.885],
                     [0.896, 0.896, 0.896, 0.896, 0.896, 0.896],
                     [0.867, 0.867, 0.867, 0.867, 0.867, 0.867],
                     [0.902, 0.902, 0.902,0.902 , 0.902, 0.902],
                     [0.926, 0.926, 0.926, 0.926, 0.926, 0.926]]
    result_UN1_1m = [[0.941, 0.920, 0.875, 0.846, 0.799, 0.852],
                     [0.924, 0.926, 0.919, 0.849, 0.701, 0.738],
                     [0.913, 0.921, 0.928, 0.913, 0.746, 0.671],
                     [0.933, 0.911, 0.925, 0.930, 0.870, 0.759],
                     [0.909, 0.879, 0.908, 0.926, 0.918, 0.867],
                     [0.835, 0.814, 0.835, 0.874, 0.904, 0.880],
                     [0.714, 0.708, 0.711, 0.721, 0.772, 0.841],
                     [0.536, 0.530, 0.523, 0.532, 0.579, 0.686],
                     [0.558, 0.502, 0.495, 0.498, 0.544, 0.637],
                     [0.678, 0.630, 0.603, 0.601, 0.655, 0.748],
                     [0.829, 0.806, 0.776, 0.777, 0.823, 0.846],
                     [0.902, 0.873, 0.862, 0.875, 0.895, 0.878]]
    result_UN2_1m = [[0.908, 0.785, 0.629, 0.558, 0.600, 0.729],
                     [0.933, 0.864, 0.746, 0.594, 0.522, 0.565],
                     [0.948, 0.919, 0.859, 0.748, 0.587, 0.516],
                     [0.931, 0.923, 0.926, 0.893, 0.774, 0.619],
                     [0.918, 0.872, 0.877, 0.901, 0.905, 0.817],
                     [0.894, 0.833, 0.811, 0.820, 0.837, 0.869],
                     [0.816, 0.730, 0.699, 0.692, 0.708, 0.711],
                     [0.794, 0.615, 0.538, 0.516, 0.512, 0.526],
                     [0.888, 0.723, 0.572, 0.505, 0.485, 0.489],
                     [0.859, 0.847, 0.787, 0.684, 0.622, 0.590],
                     [0.785, 0.690, 0.735, 0.816, 0.832, 0.806],
                     [0.851, 0.680, 0.601, 0.644, 0.770, 0.872]]
    result_simple_conv = [[0.941, 0.864, 0.728, 0.608, 0.596, 0.676],
                     [0.946, 0.920, 0.824, 0.640, 0.515, 0.528],
                     [0.938, 0.930, 0.909, 0.776, 0.579, 0.484],
                     [0.901, 0.892, 0.916, 0.904, 0.763, 0.580],
                     [0.873, 0.831, 0.843, 0.920, 0.896, 0.762],
                     [0.841, 0.798, 0.788, 0.852, 0.893, 0.862],
                     [0.740, 0.702, 0.683, 0.697, 0.753, 0.837],
                     [0.652, 0.549, 0.516, 0.518, 0.552, 0.676],
                     [0.735, 0.594, 0.511, 0.494, 0.516, 0.608],
                     [0.819, 0.774, 0.698, 0.654, 0.649, 0.723],
                     [0.836, 0.788, 0.805, 0.845, 0.846, 0.857],
                     [0.889, 0.776, 0.716, 0.700, 0.766, 0.815]]

    # model1_df = np.load(SIC_LOCATION_RESULTS + "full_U-Net_plotDF.npy")  # to change
    model1_df = np.asarray(result_UN1_1m)

    if compMode:
        # model2_df = np.load(SIC_LOCATION_RESULTS + "full_U-Net_plotDF.npy")
        model2_df = np.asarray(result_UN1_1m)

    x_labels = [1, 2, 3, 4, 5, 6]
    y_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if compMode:
        ax = sns.heatmap(model1_df[:, :] * 100 - model2_df[:, :] * 100, annot=True, vmin=-30, vmax=30,
                         xticklabels=x_labels, yticklabels=y_labels, fmt='.1f', cmap='RdYlGn')
    else:  # simply model1 values
        ax = sns.heatmap(model1_df[:, :] * 100, annot=True, vmin=50, vmax=100, xticklabels=x_labels,
                         yticklabels=y_labels, fmt='.3g', cmap='RdYlGn')

    # title override:
    save_fn = "U-Net1_1m_10epochs.png"
    t = "U-Net 1 - 1 day Training - 10 epochs"
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


# npy array to be used by the create_ncdf_file function for panoply visualization
def save_data_array(model_name, month):
    # possible models: "UN1-2", "UN1-12", "Actual", "Trivial"
    # possible month: "march", "june", "september"

    epoch_day = 86400  # one day is 86400 sec
    epoch_time_15march = 1615809600
    epoch_time_15june = 1623758400
    epoch_time_15sept = 1631707200

    if month == "march":
        chosen_epoch = epoch_time_15march
    elif month == "june":
        chosen_epoch = epoch_time_15june
    elif month == "september":
        chosen_epoch = epoch_time_15sept

    if model_name == "Actual":
        empty_bool, day = load_day(chosen_epoch)
        return day
    elif model_name == "Trivial":
        chosen_epoch -= 365 * epoch_day
        empty_bool, day = load_day(chosen_epoch)
        return day
    elif model_name == "UN1-2":
        model = load_model(SIC_LOCATION_MODELS + "50ep_3dates_1mSpacing_6m")
        it = 2
        day_list = np.zeros((2, 432, 432))
    elif model_name == "UN1-12":
        model = load_model(SIC_LOCATION_MODELS + "50ep_12dates_1mSpacing_6m")
        it = 12
        day_list = np.zeros((12, 432, 432))
    else:
        exit("The {model_name} function argument is wrong, should be \"UN1-2\" or \"UN1-12\"")

    for i in range(it):
        empty_bool, day_list[i] = load_day(chosen_epoch - 6 * 30 * epoch_day - i * 30 * epoch_day)

    days = np.reshape(np.stack(day_list, axis=2), (1, 432, 432, it))
    y_pred = model(days)
    return y_pred


def CNN_model(loss=BinaryCrossentropy(), metric=BinaryAccuracy(), learning_rate=1e-4):  # to switch conc/class mse+mse&acc/bin+bin

    model = get_model(1)  # select the model to use
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    model.summary(line_length=120)
    train(50, model, loss, metric, optimizer, 4)  # last var is lead_time
    # was at "Start of Epoch 15"

    return model


################### RUN ####################
start_time = time.time()
print(f"Run start: {datetime.datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}")

np.set_printoptions(threshold=sys.maxsize)


# CNN_model()
# get_max_extent(range(1979, 2022))
# create_heatpmap_df(BinaryAccuracy())
# plot_heatmap(False)
create_ncdf_file(save_data_array("Trivial", "march"), "march_Trivial.nc")  # <-- ici
# save_data_array("test")
# go_through_files()


print(f"Run end: {datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')}, "
      f"total elapsed time: {datetime.datetime.fromtimestamp(time.time() - start_time).strftime('%H:%M:%S')}")
