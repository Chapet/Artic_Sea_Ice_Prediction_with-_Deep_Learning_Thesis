import os
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


####################################################################################################
# You probably have to change the imports to fit your configuration, mines are a little messed up  #
####################################################################################################

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# Python code to get difference of two lists not using set()
def Diff(li1, li2):
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


def increment_date(d, m, days):  # increment the date, works with days & months but not years
    if d == days[m - 1]:
        m += 1
        d = 1
    else:
        d += 1

    return d, m


def decrement_date(d, m, days):  # increment the date, works with days & months but not years
    if d == 1:
        m -= 1
        d = days[m - 1]
    else:
        d -= 1

    return d, m


def date_to_string(d, m, y):
    if d < 10:
        d_str = "0" + str(d)
    else:
        d_str = str(d)
    if m < 10:
        m_str = "0" + str(m)
    else:
        m_str = str(m)

    return str(y) + "-" + m_str + "-" + d_str


def get_day_information(file):
    data = Dataset(file, mode='r')
    time_ = data.variables['time'][:]
    date1 = datetime.date.fromtimestamp(time_[0]) + datetime.timedelta(days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)

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


def get_autoreg(X_train, model):
    shape = X_train.shape  # shape = (#days, 432, 432)
    X_result = np.empty((shape[0], 432, 432))
    for i in range(1, shape[0]):
        x_next = model(np.reshape(X_train[i-1], (1, 432, 432)))
        X_result[i] = x_next[0, :, :, 0]
    return X_result


def get_model(model_type,  filter_size=3, n_filters_factor=1, n_output_classes=1):
    inputs = Input(shape=(432, 432, 1))

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

        up8 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='nearest')(conv5))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(int(128 * n_filters_factor), filter_size, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(int(64 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='nearest')(conv8))
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

        up8 = Conv2D(int(128 * n_filters_factor), 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2), interpolation='nearest')(conv5))
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


def training_loop(num_epoch, model, loss_fn, acc_metric, optimizer, x_class, y_class, lead_time):  # TODO attention epoch stil there
    print("Starting training loop")
    # training loop
    for epoch in range(num_epoch):
        print(f"\nStart of Training Epoch {epoch + 1}")

        # small = 2002-2015 & 2017-2018
        for y in range(1979, 2016):
            for num_batch in range(4):
                batch_data = np.load('./res/np/' + str(lead_time) + 'm_2class/' + str(y) + '_' + str(num_batch+1) + '.npy')  # shape (2, #days, 432, 432, 2)
                X_train = batch_data[0, :, :, :, x_class]
                y_train = batch_data[1, :, :, :, y_class]
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

                print(f"year {y}, batch {num_batch+1} with metric: {acc_metric.result()} and loss: {loss_saved}")

        train_acc = acc_metric.result()
        print(f"Metric over epoch {epoch+1}: {train_acc}")
        acc_metric.reset_states()

    model.save('./res/models/1_' + str(lead_time) + 'm_train' + str(x_class) + '_class' + str(y_class) + '_epoch5')  # TODO
    print("model saved")

    # evaluates the model on the test set
    # model = load_model('./res/models/2_1m_train0_class0_big')
    # print("model loaded")

    # test loop
    # for y in range(2016, 2022):
    #     for num_batch in range(4):
    #         batch_data = np.load('./res/np/' + str(lead_time) + 'm_2class/' + str(y) + '_' + str(num_batch + 1) + '.npy')  # shape (2, #days, 432, 432, 4)
    #         X_test = batch_data[0, :, :, :, x_class]
    #         y_test = batch_data[1, :, :, :, y_class]
    #
    #         test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    #         test_ds = test_ds.batch(1)  # .shuffle(buffer_size=1024) entre train_ds et batch
    #
    #         for batch_idx, (x_batch, y_batch) in enumerate(test_ds):
    #             y_pred = model(x_batch)
    #             acc_metric.update_state(y_batch, y_pred)
    #
    # test_acc = acc_metric.result()
    # print(f"Metric over test set: {test_acc}")
    # acc_metric.reset_states()


def SVM_1y():
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
                list_6m.append(value*625)
            elif sorted_month == 9:
                list_september.append(value*625)

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


def create_SIE_df(years):
    for y in years:
        base_path = "L:\\ice_conc_data\\" + str(y)
        df_dict = {}
        to_append_dict = {}

        for m in os.listdir(base_path):
            filler = "\\"
            path = base_path + filler + m
            files = os.listdir(path)  # get all filenames/dir_names in dir, in a list
            for f in files:
                data = Dataset(path + "\\" + f, mode='r')
                time = data.variables['time'][:]
                date = datetime.date.fromtimestamp(time[0]) + datetime.timedelta(
                    days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)
                conc_array = data.variables['ice_conc'][:][0].ravel()
                conc_array = conc_array.filled(0)  # replace missing values with zeros

                # grid_SIE_i = [0] * 100
                area = 1
                SIE_i = 0
                for i in range(len(conc_array)):
                    e = conc_array[i]
                    # row = i // 849
                    # col = i % 849
                    if e > 15:
                        SIE_i += area
                        # grid_SIE_i[row//85*10+col//85] += area

                date_str = date.strftime("%Y-%m-%d")
                df_dict[date_str + "_SIE"] = SIE_i

        # here check that there is the right amount of days for each month in df_dict
        # if not interpolate linearly to get all the value
        # so 1000 x 2000 -> x=1500, 1000 x1 x2 x3 2000 -> x1=1250 x2=1500 x3=1750, etc
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # we don't take 29/2 into account
        if str(y) + "-02-29_SIE" in df_dict:
            del df_dict[str(y) + "-02-29_SIE"]

        first_available_day_string = str(
            y) + "-01-13_SIE"  # there is only for which the first day available is outside the range [1,10]
        day_path = base_path + "\\01"
        for i in range(1, 10):
            if "ice_conc_nh_ease2-250_cdr-v2p0_" + str(y) + "010" + str(i) + "1200.nc" in os.listdir(day_path):
                first_available_day_string = str(y) + "-01-0" + str(i) + "_SIE"
                break
        print(first_available_day_string)
        marking = [-1, -1, 0,
                   first_available_day_string]  # interval of missing data [start, end, nb_missing_days, date_before_start]
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            for d in range(1, days[m - 1] + 1):  # for each day in the month
                cur_date_str = date_to_string(d, m, y) + "_SIE"
                if cur_date_str not in df_dict:  # if key does not exist (date is missing from data)
                    if marking[2] == 0:  # first day of data missing, mark start & end
                        marking[0] = cur_date_str
                        marking[1] = cur_date_str
                        marking[2] += 1  # increment nb_missing_days
                    else:  # not the first day, juste change end
                        marking[1] = cur_date_str
                        marking[2] += 1

                    if cur_date_str == str(y) + "-12-31_SIE":  # if the year is ending on a missing value
                        nb_missing_days = marking[2]
                        start_SIE = df_dict[marking[3]]
                        cur_d = int(marking[0][8:10])
                        cur_m = int(marking[0][5:7])
                        for i in range(1, nb_missing_days + 1):
                            df_dict[date_to_string(cur_d, cur_m, y) + "_SIE"] = start_SIE
                            cur_d, cur_m = increment_date(cur_d, cur_m, days)

                elif cur_date_str in df_dict:
                    if marking[2] != 0:  # we just got out of a missing date conflict
                        missing_d_start = int(marking[0][8:10])
                        missing_m_start = int(marking[0][5:7])

                        nb_missing_days = marking[2]

                        start_SIE = df_dict[marking[3]]
                        end_SIE = df_dict[cur_date_str]
                        interpolation_step = (end_SIE - start_SIE) / (nb_missing_days + 1)

                        cur_d = missing_d_start
                        cur_m = missing_m_start
                        for i in range(1, nb_missing_days + 1):
                            df_dict[date_to_string(cur_d, cur_m,
                                                   y) + "_SIE"] = start_SIE + interpolation_step * i  # input missing value value
                            cur_d, cur_m = increment_date(cur_d, cur_m, days)

                        marking[0] = -1
                        marking[1] = -1
                        marking[2] = 0
                    marking[3] = cur_date_str

        df = pd.DataFrame.from_dict(df_dict, orient='index')
        df.to_pickle("./res/pickle/df_" + str(y) + ".pkl")
        print(str(y) + " done")


def create_CNN_df(years):  # TODO: remove ?
    for y in years:  # for all the years
        base_path = "L:\\ice_conc_data\\" + str(y)
        npyear = np.zeros((365, 432, 432, 4))  # ice_conc, date, 01 ice >= 15%, 01 ice < 15%

        for m in os.listdir(base_path):  # for all the months
            path = base_path + "\\" + m
            files = os.listdir(path)  # get all filenames/dir_names in dir, in a list

            for f in files:  # for all the days
                data = Dataset(path + "\\" + f, mode='r')
                time = data.variables['time'][:]
                date = datetime.date.fromtimestamp(time[0]) + datetime.timedelta(days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)
                conc_array = data.variables['ice_conc'][:][0].ravel()
                conc_array = conc_array.filled(0)  # replace missing values with zeros
                conc_array = np.divide(conc_array, 100)  # normalize concentration between 0 and 1

                conc_array = np.array(conc_array)
                conc_array = conc_array.reshape((432, 432))

                day_of_year_str = date.strftime("%j")  # e.g. '065'
                day_of_year = int(day_of_year_str)  # e.g. 65
                if day_of_year == 60:  # skipping 29th of february
                    continue

                day_array = np.empty((432, 432))
                day_array.fill(day_of_year)

                no_ice_array = np.zeros((432, 432))
                ice_array = np.zeros((432, 432))
                for row in range(432):
                    for col in range(432):
                        e = conc_array[row][col]
                        if e >= 0.15:
                            ice_array[row][col] = 1
                        else:
                            no_ice_array[row][col] = 1

                final_array = np.stack([conc_array, day_array, no_ice_array, ice_array], axis=2)  # shape (432,432,4)

                if day_of_year > 60:
                    npyear[day_of_year - 2] = final_array
                else:
                    npyear[day_of_year - 1] = final_array

        np.save('.\\res\\np\\1m_2class\\' + str(y) + '.npy', npyear)
        print(str(y) + " done")


def batched_forecast(nb_lead_months, nb_lead_days):
    years = np.arange(1979, 2022)
    for y in years:  # for all the years
        nptrimestre = np.zeros((2, 92, 432, 432, 2))  # (x/y), days, gridxgrid, (ice_conc, date, 01 ice >= 15%)
        base_path = "L:\\ice_conc_data\\"
        check_save = [False, False, False]
        day_cnt = 0

        for m in ["%.2d" % i for i in range(1, 13)]:
            days_files = os.listdir(base_path + str(y) + '\\' + m)

            for day in days_files:
                y2_int = y+1 if int(m) > 12-nb_lead_months else y
                y2 = str(y2_int)
                m2_int = (int(m) + nb_lead_months) % 12 if y2_int != y else int(m) + nb_lead_months
                m2 = "%.2d" % m2_int
                d2 = str(int(day[-9:-7]) + nb_lead_days)
                d3 = "%.2d" % (int(d2) + 1)
                d4 = "%.2d" % (int(d2) - 1)
                cdr = "icdr" if y2_int >= 2016 else "cdr"
                file1 = base_path + str(y) + '\\' + m + '\\' + day
                file2 = base_path + str(y2) + '\\' + m2 + "\\ice_conc_nh_ease2-250_" + cdr + "-v2p0_" + y2 + m2 + d2 + "1200.nc"
                file3 = base_path + str(y2) + '\\' + m2 + "\\ice_conc_nh_ease2-250_" + cdr + "-v2p0_" + y2 + m2 + d3 + "1200.nc"
                file4 = base_path + str(y2) + '\\' + m2 + "\\ice_conc_nh_ease2-250_" + cdr + "-v2p0_" + y2 + m2 + d4 + "1200.nc"
                valid = file2

                check_day = False
                if os.path.exists(file2):
                    check_day = True
                elif os.path.exists(file3):
                    check_day = True
                    valid = file3
                elif os.path.exists(file4):
                    check_day = True
                    valid = file4

                if check_day:
                    data1 = Dataset(file1, mode='r')
                    data2 = Dataset(valid, mode='r')
                    time1 = data1.variables['time'][:]
                    time2 = data2.variables['time'][:]
                    date1 = datetime.date.fromtimestamp(time1[0]) + datetime.timedelta(days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)
                    date2 = datetime.date.fromtimestamp(time2[0]) + datetime.timedelta(days=365 * 8 + 2)  # missed 8 years to be up to EPOCH (with bisextile +2)

                    conc_array1 = data1.variables['ice_conc'][:][0].ravel()
                    conc_array1 = conc_array1.filled(0)  # replace missing values with zeros
                    conc_array1 = np.divide(conc_array1, 100)  # normalize concentration between 0 and 1
                    conc_array1 = np.array(conc_array1)
                    conc_array1 = conc_array1.reshape((432, 432))

                    conc_array2 = data2.variables['ice_conc'][:][0].ravel()
                    conc_array2 = conc_array2.filled(0)  # replace missing values with zeros
                    conc_array2 = np.divide(conc_array2, 100)  # normalize concentration between 0 and 1
                    conc_array2 = np.array(conc_array2)
                    conc_array2 = conc_array2.reshape((432, 432))

                    day_of_year_str1 = date1.strftime("%j")  # e.g. '065'
                    day_of_year1 = int(day_of_year_str1)  # e.g. 65
                    day_of_year_str2 = date2.strftime("%j")  # e.g. '065'
                    day_of_year2 = int(day_of_year_str2)  # e.g. 65

                    day_array1 = np.empty((432, 432))
                    day_array1.fill(day_of_year1)
                    day_array2 = np.empty((432, 432))
                    day_array2.fill(day_of_year2)

                    ice_array1 = np.zeros((432, 432))
                    ice_array2 = np.zeros((432, 432))
                    for row in range(432):
                        for col in range(432):
                            e1 = conc_array1[row][col]
                            e2 = conc_array2[row][col]
                            if e1 >= 0.15:
                                ice_array1[row][col] = 1
                            if e2 >= 0.15:
                                ice_array2[row][col] = 1

                    final_array1 = np.stack([conc_array1, ice_array1], axis=2)  # shape (432,432,2)
                    final_array2 = np.stack([conc_array2, ice_array2], axis=2)  # shape (432,432,2)

                    nptrimestre[0][day_cnt] = final_array1
                    nptrimestre[1][day_cnt] = final_array2
                    day_cnt += 1
            # nptrimestre = np.zeros((2, 92, 432, 432, 4))  # (x/y), days, gridxgrid, (ice_conc, date, 01 ice >= 15%)

            if m == "03" or m == "04" and not check_save[0]:
                check_save[0] = True
                ntrim = np.empty((2, day_cnt, 432, 432, 2))
                ntrim[0] = nptrimestre[0][:day_cnt][:][:][:]
                ntrim[1] = nptrimestre[1][:day_cnt][:][:][:]
                np.save('.\\res\\np\\' + str(nb_lead_months) + 'm_2class\\' + str(y) + "_1" + '.npy', ntrim)
                nptrimestre = np.zeros((2, 92, 432, 432, 2))
                day_cnt = 0
                print("1")
            if m == "06" or m == "07" and not check_save[1]:
                check_save[1] = True
                ntrim = np.empty((2, day_cnt, 432, 432, 2))
                ntrim[0] = nptrimestre[0][:day_cnt][:][:][:]
                ntrim[1] = nptrimestre[1][:day_cnt][:][:][:]
                np.save('.\\res\\np\\' + str(nb_lead_months) + 'm_2class\\' + str(y) + "_2" + '.npy', ntrim)
                nptrimestre = np.zeros((2, 92, 432, 432, 2))
                day_cnt = 0
                print("2")
            if m == "09" or m == "10" and not check_save[2]:
                check_save[2] = True
                ntrim = np.empty((2, day_cnt, 432, 432, 2))
                ntrim[0] = nptrimestre[0][:day_cnt][:][:][:]
                ntrim[1] = nptrimestre[1][:day_cnt][:][:][:]
                np.save('.\\res\\np\\' + str(nb_lead_months) + 'm_2class\\' + str(y) + "_3" + '.npy', ntrim)
                nptrimestre = np.zeros((2, 92, 432, 432, 2))
                day_cnt = 0
                print("3")

        ntrim = np.empty((2, day_cnt, 432, 432, 2))
        ntrim[0] = nptrimestre[0][:day_cnt][:][:][:]
        ntrim[1] = nptrimestre[1][:day_cnt][:][:][:]
        np.save('.\\res\\np\\' + str(nb_lead_months) + 'm_2class\\' + str(y) + "_4" + '.npy', ntrim)
        print("4")
        print(str(y) + " done")


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
        model1_df = np.load('./res/result_df/model_' + str(model1) + '_1_maxextent_correct.npy')

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
                model1_df[i, j] = result_icenet[i][j]/100

    if model2 != 0:
        compMode = True
        save_fn = str(model1) + "vs" + str(model2) + ".png"
        t += " comparison with the "
        if model2 == 1 or model2 == 2 or model2 == 3 or model2 == 4:
            model2_df = np.load('./res/result_df/model_' + str(model2) + '_1_maxextent_correct.npy')

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
                    model2_df[i, j] = result_icenet[i][j]/100

    x_labels = [1, 2, 3, 4, 5, 6]
    y_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if compMode:
        ax = sns.heatmap(model1_df[:, :]*100 - model2_df[:, :]*100, annot=True, vmin=-30, vmax=30, xticklabels=x_labels, yticklabels=y_labels, fmt='.3g', cmap='RdYlGn')
    else:  # simply model1 values
        ax = sns.heatmap(model1_df[:, :] * 100, annot=True, vmin=50, vmax=100, xticklabels=x_labels, yticklabels=y_labels, fmt='.3g', cmap='RdYlGn')

    ax.set_title(t, fontsize=14)

    ax.tick_params(axis='y', rotation=0)
    ax.set_xlabel('Lead Time (months)', fontweight='bold')
    ax.set_ylabel('Calendar month', fontweight='bold')
    cbar = ax.collections[0].colorbar
    cbar.set_label("Binary Accuracy (%)", fontsize=15)

    plt.savefig('./res/plots/' + save_fn, format='png')
    plt.show()


def create_heatpmap_df(model_number, classxy, metric):
    max_extent_months = np.load("L:\\Thesis\\res\\result_df\\max_extent_months.npy")  # shape (12, 432, 432)
    result_df = np.zeros((12, 6))
    for lead_time in range(1, 7):
        if model_number != 4:
            model = load_model('./res/models/' + str(model_number) + '_' + str(lead_time) + 'm_train' + str(classxy) + '_class' + str(classxy) + "_big")
        num_batch = 1
        for month in range(0, 12):
            if month == 3 or month == 6 or month == 9:
                num_batch += 1
            for year in range(2016, 2022):
                if year == 2021 and month in range(12-lead_time, 13):
                    continue
                days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                if (year % 4) == 0:
                    days[1] = 29
                if year == 2021:
                    days[1] = 26
                month_num = month % 3
                start = 0
                if month_num == 1:
                    start = days[month-1]
                elif month_num == 2:
                    start = days[month-2] + days[month-1]
                end = start + days[month]

                data = np.load('./res/np/' + str(lead_time) + 'm_2class/' + str(year) + '_' + str(num_batch) + '.npy')  # shape (2, #days, 432, 432, 2)

                X_test = data[0, start:end, :, :, classxy]
                y_test = data[1, start:end, :, :, classxy]

                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                test_ds = test_ds.batch(1)
                for batch_idx, (x_batch, y_batch) in enumerate(test_ds):
                    if model_number != 4:
                        y_pred = model(x_batch)
                    else:
                        y_pred = x_batch

                    metric.update_state(np.reshape(y_batch, (186624, 1)), np.reshape(y_pred, (186624, 1)), sample_weight=np.reshape(max_extent_months[(month+lead_time) % 12], 186624))
            result_df[(month+lead_time) % 12][lead_time-1] = metric.result()
            months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            print(f"metric for the month \"{months[(month+lead_time) % 12]}\" with a lead time of {lead_time}: {metric.result()}")
            metric.reset_state()

    print(result_df)
    np.save('.\\res\\result_df\\model_' + str(model_number) + '_' + str(classxy) + '_maxextent_correct.npy', result_df)


def create_ncdf_file(data_array, name):  # a 432x432 npy array containing sea ice values
    ref = Dataset('L:\\Thesis\\res\\ncdf\\ice_conc_nh_ease2-250_icdr-v2p0_201809061200.nc', mode='r')
    fn = 'L:\\Thesis\\res\\ncdf\\' + name
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


def get_max_extent(year_range):
    max_extent = np.zeros((432, 432))
    max_extent_months = np.zeros((12, 432, 432))
    for y in year_range:
        base_path = "L:\\ice_conc_data\\"

        for m in ["%.2d" % i for i in range(1, 13)]:
            days_files = os.listdir(base_path + str(y) + '\\' + m)

            for day in days_files:
                file = base_path + str(y) + '\\' + m + '\\' + day
                day_data = Dataset(file, mode='r')
                day_conc = day_data.variables['ice_conc'][0]
                day_conc = np.array(day_conc)

                for i in range(432):
                    for j in range(432):
                        if day_conc[i][j] > 0.15:
                            max_extent[i][j] = 1
                            max_extent_months[int(m)-1][i][j] = 1
                day_data.close()
        print(f"year {y} done")
    np.save(".\\res\\result_df\\max_extent_months.npy", max_extent_months)
    np.save(".\\res\\result_df\\max_extent.npy", max_extent)
    return max_extent


def epoch_comp():
    metric = BinaryAccuracy()
    result_tab = np.empty((5, 3))  # nb epoch, (train_acc, test_acc, difference)
    max_extent = np.load("L:\\Thesis\\res\\result_df\\max_extent.npy")  # shape (432, 432)
    it = 0
    for epoch in [10]:  # 1, 2, 3, 5,
        model = load_model('.\\res\\models\\1_6m_train1_class1_epoch' + str(epoch))
        for y in range(1976, 2016):  # TODO change after running
            for num_trim in range(1, 5):
                data = np.load('./res/np/6m_2class/' + str(y) + '_' + str(num_trim) + '.npy')  # shape (2, #days, 432, 432, 2)
                X_test = data[0, :, :, :, 1]
                y_test = data[1, :, :, :, 1]
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                test_ds = test_ds.batch(1)
                for batch_idx, (x_batch, y_batch) in enumerate(test_ds):
                    y_pred = model(x_batch)
                    metric.update_state(y_batch, y_pred)
            print(f"year {y} done")
        result_tab[it][0] = metric.result()

        metric.reset_state()

        for y in range(2016, 2022):
            for num_trim in range(1, 5):
                data = np.load('./res/np/6m_2class/' + str(y) + '_' + str(num_trim) + '.npy')  # shape (2, #days, 432, 432, 2)
                X_test = data[0, :, :, :, 1]
                y_test = data[1, :, :, :, 1]
                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                test_ds = test_ds.batch(1)
                for batch_idx, (x_batch, y_batch) in enumerate(test_ds):
                    y_pred = model(x_batch)
                    metric.update_state(y_batch, y_pred)
            print(f"year {y} done")
        result_tab[it][1] = metric.result()

        result_tab[it][2] = result_tab[it][0] - result_tab[it][1]
        print(f"results for epoch {epoch}:")
        print(result_tab[it])
        it += 1
    print("print all:")
    print(result_tab)
    np.save("./res/result_df/epoch_tab.npy", result_tab)


def CNN_model(loss=BinaryCrossentropy(), metric=BinaryAccuracy(), learning_rate=1e-4):  # to switch conc/class mse+mse&acc/bin+bin

    model = get_model(1)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    model.summary(line_length=120)
    training_loop(5, model, loss, metric, optimizer, 1, 1, 1)  # 3 lasts are x_class, y_class, lead_time

    return model
    