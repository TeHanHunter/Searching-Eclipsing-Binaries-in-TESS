import os
import sys
import pickle
import warnings
import tensorflow

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from wotan import flatten
from scipy.interpolate import interp1d
from multiprocessing import Pool, Array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input, Dense, Conv1D, AveragePooling1D, Flatten


def make_cnn(maxlen):
    input_local = Input(shape=(maxlen, 1))
    x = Conv1D(16, 5, strides=1)(input_local)
    # x = Conv1D(16, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)
    x = Conv1D(8, 5, strides=1)(x)
    # x = Conv1D(8, 5, strides=1)(x)
    x = AveragePooling1D(pool_size=5, strides=2)(x)

    xf = Flatten()(x)
    z = Dense(64, activation='relu')(xf)
    # z = Dropout(0.1)(z)
    z = Dense(32, activation='relu')(z)
    z = Dense(8, activation='relu')(z)

    output = Dense(1, activation='sigmoid', name='main_output')(z)
    model = Model(inputs=input_local, outputs=output)

    SGDsolver = SGD(lr=0.1, momentum=0.25, decay=0.0001, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=SGDsolver,
                  metrics=['accuracy'])
    return model


def period(source, stdv_threshold='0.0005'):
    first_trial = 200
    second_trial = 200
    period_range = 10  # days

    # produce periods
    a_0 = 0.1
    r = (period_range * 10) ** (1 / first_trial)
    length = first_trial
    geometric = [a_0 * r ** (n - 1) for n in range(1, length + 1)]

    a_sam = 0.09
    r_sam = (period_range * 11.2) ** 1e-5
    leng = int(1e5)
    sample_period = np.array([a_sam * r_sam ** (n - 1) for n in range(1, leng + 1)])

    std = np.zeros(len(sample_period))
    for i in range(len(sample_period)):
        p = sample_period[i]
        t_pf = source.time % p
        t_pf_sort = np.sort(t_pf)
        gap = np.diff(t_pf_sort)
        std[i] = np.std(gap / p)

    std_geo = np.zeros(len(geometric))
    for i in range(len(geometric)):
        p = geometric[i]
        t_pf = source.time % p
        t_pf_sort = np.sort(t_pf)
        gap = np.diff(t_pf_sort)
        std_geo[i] = np.std(gap / p)

    # first trial periods
    f = interp1d(sample_period[np.where(std < float(stdv_threshold))], std[np.where(std < float(stdv_threshold))],
                 kind='nearest')
    std_mod_p = f(geometric)
    mod_p = np.zeros(len(std_mod_p))
    for i in range(len(std_mod_p)):
        mod_p[i] = sample_period[np.where(std == std_mod_p[i])]

    # second trial periods
    mod_periods = np.zeros((first_trial, second_trial))
    for i in range(len(mod_p)):
        mod_periods[i] = np.linspace(mod_p[i] / np.sqrt(r), mod_p[i] * np.sqrt(r), second_trial)
        # TODO: linspace
    std_mod_periods = f(mod_periods.reshape((1, first_trial * second_trial))[0])
    mod_periods = np.zeros(len(std_mod_periods))
    for i in range(len(std_mod_periods)):
        mod_periods[i] = sample_period[np.where(std == std_mod_periods[i])]
    return mod_p, mod_periods.reshape((first_trial, second_trial))


def cnn_prediction(source, lightcurve, star_num=0, Sample_number=500, mod_p=None, mod_periods=None):
    time_raw = source.time
    flux_raw = lightcurve[star_num]
    # mean = np.mean(flux_raw)
    flux_1d = flatten(time_raw, flux_raw, break_tolerance=0.1, window_length=1, edge_cutoff=0.25,
                      return_trend=False)
    # remove nan in flux again(causes trouble for cnn)
    index = np.invert(np.isnan(flux_1d))
    flux_1d = flux_1d[index]
    time_1d = time_raw[index]
    # make CNN tests
    period = mod_p
    t_0 = np.linspace(-0.1, 0.1, 5)
    predict = np.zeros((len(period), len(t_0)))
    # cut = int(min(len(time_1d), length_sector[0]))
    cut = len(time_1d)
    argsort = flux_1d[0:cut].argsort()

    for j in range(len(period)):
        p = period[j]
        t_zero = np.median(time_1d[0:cut][argsort][0:20] % p) / p
        for k in range(len(t_0)):
            t_pf = np.array((time_1d[0:cut] + (0.5 - t_zero + t_0[k]) * p) % p)
            t = np.linspace(np.min(t_pf), np.max(t_pf), Sample_number)
            f = interp1d(t_pf, flux_1d[0:cut], kind='nearest')
            flux = f(t)
            # np.max(flux) - np.min(flux) np.percentile(flux, 100) - np.percentile(flux, 0)
            flux /= (np.max(flux) - np.min(flux)) / 4
            flux -= np.average(flux)
            predict[j][k] = np.array(cnn(flux.reshape((1, Sample_number, 1))))
        if np.max(predict) >= 0.99999:
            break
    idx = np.where(predict == np.max(predict))
    if np.max(predict) < 0.5:
        pass
    else:
        ### repeat in the region near the best result of first step for higher precision
        period_ = mod_periods[np.where(period == period[idx[0][0]])][0]

        t_0_ = np.linspace(-0.1, 0.1, 5)
        predict = np.zeros((len(period_), len(t_0_)))
        for j in range(len(period_)):
            p = period_[j]
            t_zero = np.mean(time_1d[argsort][0:5] % p) / p
            for k in range(len(t_0_)):
                t_pf = np.array((time_1d + (0.5 - t_zero + t_0_[k]) * p) % p)
                t = np.linspace(np.min(t_pf), np.max(t_pf), Sample_number)
                f = interp1d(t_pf, flux_1d, kind='nearest')
                flux = f(t)
                flux /= (np.max(flux) - np.min(flux)) / 4
                flux -= np.average(flux)
                predict[j][k] = np.array(cnn(flux.reshape((1, Sample_number, 1))))

        idx = np.where(predict == np.max(predict))
        p = period_[idx[0][0]]
        t_0 = t_0_[idx[1][0]]
        return (p, t_0, np.max(predict))


if __name__ == '__main__':
    with open('/mnt/c/Users/tehan/Documents/GitHub/Searching-Eclipsing-Binaries-in-TESS/source_NGC_7654_90.pkl',
              'rb') as input:
        source = pickle.load(input)
    lightcurve_1 = np.load('/mnt/c/users/tehan/desktop/epsf_lc_0_5000.npy')
    lightcurve_2 = np.load('/mnt/c/users/tehan/desktop/epsf_lc_5000_10000.npy')
    lightcurve_3 = np.load('/mnt/c/users/tehan/desktop/epsf_lc_10000_15000.npy')
    lightcurve_4 = np.load('/mnt/c/users/tehan/desktop/epsf_lc_15000_20000.npy')
    lightcurve = np.append(lightcurve_1, lightcurve_2, axis=0)
    lightcurve = np.append(lightcurve, lightcurve_3, axis=0)
    lightcurve = np.append(lightcurve, lightcurve_4, axis=0)
    Sample_number = 500
    tensorflow.get_logger().setLevel('ERROR')  ## ignore internal TensorFlow Warning message
    cnn = make_cnn(Sample_number)
    cnn_weights = '/mnt/c/users/tehan/documents/github/Searching-Eclipsing-Binaries-in-TESS/tess_cnn.h5'
    cnn.load_weights(str(cnn_weights))

    mod_p, mod_periods = period(source)


    def multi_cnn(star_num):
        r = cnn_prediction(source, lightcurve, star_num=star_num, Sample_number=500, mod_p=mod_p,
                           mod_periods=mod_periods)
        return r


    # p, t_0, predict = cnn_prediction(source, lightcurve, star_num=77, Sample_number=500, mod_p=mod_p,
    #                                  mod_periods=mod_periods)

    ### stars in the frame:
    # in_frame = []
    # for i in range(len(lightcurve)):
    #     if 0 <= source.gaia['Sector_17_x'][i] <= 89 and 0 <= source.gaia['Sector_17_y'][i] <= 89:
    #         in_frame.append(i)
    in_frame = np.where(np.min(lightcurve, axis=1) > 0)[0]

    # with Pool() as p:
    #     r = list(tqdm(p.imap(multi_cnn, in_frame), total=len(in_frame)))
    # np.save('/mnt/c/users/tehan/desktop/cnn_result_new', r)
    # r = np.load('/mnt/c/users/tehan/desktop/cnn_result_new.npy', allow_pickle=True)
    # result = []
    # for i in range(len(in_frame)):
    #     result.append(cnn_prediction(source, lightcurve, star_num=in_frame[i], Sample_number=500, mod_p=mod_p,
    #                                  mod_periods=mod_periods))
    #     print(str(i) + ' / ' + str(len(in_frame)))
    # np.save('/mnt/c/users/tehan/desktop/cnn_result_new', result)

    # for i, index in enumerate(in_frame):
    #     if r[i] is not None and r[i][-1] > 0.95:
    #         plt.plot(source.time, lightcurve[index])
    #         plt.title(r[i][0])
    #         plt.show()

    x = source.gaia['Sector_17_x']
    y = source.gaia['Sector_17_y']
    # dis = np.sqrt(
    #     (np.array(x) - source.gaia['Sector_17_x'][77]) ** 2 + (np.array(y) - source.gaia['Sector_17_y'][77]) ** 2)
    # arg = np.argsort(dis)
    # for i in range(10):
    #     plt.plot(lightcurve[arg[np.where(arg < 1000)][i]])
    #     plt.show()

    r = np.load('/mnt/c/users/tehan/desktop/cnn_result_new.npy', allow_pickle=True)
    eb = []
    per = []
    for i, index in enumerate(in_frame):
        if r[i] is not None and r[i][-1] > 0.999:
            per.append(r[i][0])
            eb.append(index)
    # a = np.stack((np.array(eb), np.array(per), np.array(x)[eb], np.array(y)[eb]))
    # np.savetxt('/mnt/c/users/tehan/desktop/eb_candidate_new.csv', a.T[np.argsort(a[1])], delimiter=",")

    plt.imshow(np.log10(source.flux[0]), vmin=np.min(np.log10(source.flux[0])),
               vmax=np.max(np.log10(source.flux[0])), origin='lower', cmap='gray')
    plt.scatter(x[eb], y[eb], s=3, c=per, cmap='tab20')
    plt.title('Periods predictions')
    cbar = plt.colorbar()
    cbar.labelbad = 55
    cbar.set_label('Periods (days)', rotation=270, labelpad=15)
    # plt.savefig('/mnt/c/users/tehan/desktop/per.png', dpi=300)
    plt.show()
