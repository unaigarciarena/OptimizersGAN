import numpy as np
from Generalization import decodify_descriptor
from tunable_gan_class_optimization import GAN
from gaussians import create_data, mmd, plt_center_assignation
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf

fs = [1, 2, 3, 4, 5, 7, 8, 9]
runs = range(0, 30)
sizes = [10, 784]
reps = range(5)


def self_reshape(data):

    d = [np.reshape(data[i], (-1, 2)) for i in np.arange(0, data.shape[0])]
    return np.concatenate(d, axis=0)


def load_f_gan(f, inst, size, rnd=False):
    gans = np.load("NewLossRes/ResultsPareto" + str(inst) + "_F" + str(f) + "_100_100_2_" + str(size) + ".npy")
    if rnd:
        best = np.random.randint(0, 100)
    else:
        best = np.argmin(gans[:, -2])

    best_gan = gans[best, :-3].astype("int")
    descriptor = decodify_descriptor(best_gan, 2)
    gan = GAN(descriptor)

    return gan


def test_gan(gan, seed):

    gan.training_definition(seed)
    samples = gan.separated_running("fixed", data, 150, 20000, 1000, 50001)
    samples *= data_max
    samples += data_min
    mmd_value, centers = mmd(samples, unscaled_data)
    print(mmd_value)

    return mmd_value, centers,  samples


def test_trans():

    collection = np.zeros((len(fs), len(runs), len(sizes), len(reps)))

    for fi, f in enumerate(fs):
        for run in runs:
            for isize, size in enumerate(sizes):
                best_gan = load_f_gan(f, run, size)
                for rep in reps:
                    mmd_val, centers, samples = test_gan(best_gan, rep)
                    plt_center_assignation(samples, centers, modes=n_centers, save=True, name="F" + str(f) + "_S" + str(size) + "_R" + str(run) + "_Rep" + str(rep) + ".pdf")
                    collection[fi, run - 1, isize, rep] = mmd_val

    np.save("Collection.npy", collection)


def test_random(seed):

    collection = np.zeros((100, 5))
    for i in range(100):
        gan = np.load("RandomGans.npy")[seed*100 + i, :-3].astype("int")
        descriptor = decodify_descriptor(gan, 2)
        gan = GAN(descriptor)
        for rep in reps:
            mmd_val, centers, samples = test_gan(gan, rep)
            collection[i, rep] = mmd_val

        np.save("Collection" + str(seed) + ".npy", collection)


def random_transfer():
    collection = np.load("Collection.npy")
    collection[collection < 0.0005] = 8
    for si, s in enumerate([10]):  # enumerate(sizes):
        for fi, f in enumerate(fs):
            mmds = np.log(collection[fi, :, si, :].flatten())

            sns.distplot(mmds, label="F" + str(f) + "_S" + str(s), kde=True, hist=False, bins=np.arange(-4.1, 3, 0.01), kde_kws={"bw":0.05}, hist_kws={"histtype": 'step', "stacked": True, "fill": False, "linewidth": 2})

        plt.legend()
        plt.show()


def cumulative():
    collection = np.load("Collection.npy")
    collection[collection < 0.0005] = 8
    rnd = np.log(np.load("RandomTrans.npy"))

    for si, s in enumerate([10, 784]):  # enumerate(sizes):
        lasti = 0
        fig, ax = plt.subplots(figsize=(8, 4))
        for fi, f in enumerate(fs):
            mmds = np.log(collection[fi, :, si, :].flatten())

            ax.hist(mmds, np.arange(-4.1, 2.1, 0.01), density=True, histtype='step', cumulative=True, label="F" + str(f) + "_S" + str(s))
            if fi == 7:
                ax.hist(rnd[lasti:lasti+30].flatten(), np.arange(-4.1, 2.1, 0.01), density=True, histtype='step', cumulative=True, label="Random", color="black")
            else:
                ax.hist(rnd[lasti:lasti + 30].flatten(), np.arange(-4.1, 2.1, 0.01), density=True, histtype='step', cumulative=True, color="black")
            lasti += 30

        plt.legend(loc=[0.03, 0.33])
        plt.xlabel("$log(MMD)$")
        plt.ylabel("Frequency")
        plt.title("Transfer learning")
        plt.show()


def random_trans():
    list = []
    for i in range(42):
        path = "RandomTrans/Collection" + str(i+1) + ".npy"
        if os.path.isfile(path):
            list += [np.load(path)]
        else:
            print(path)
    list = np.concatenate(list)
    np.save("RandomTrans.npy", list)


def min_transfer():
    collection = np.load("Collection.npy")
    collection[collection < 0.0005] = 288
    a = np.argmin(collection)
    a = np.unravel_index(a, collection.shape)
    best_gans = []
    for fi, f in enumerate(fs):
        for isize, size in enumerate(sizes):
            for run in runs:
                if np.sum(collection[fi, run-1, isize, :]<0.1) > 3:
                    gans = np.load("NewLossRes/ResultsPareto" + str(run) + "_F" + str(f) + "_100_100_2_" + str(size) + ".npy")
                    best = np.argmin(gans[:, -2])
                    best_gan = gans[best, :-4].astype("int")
                    best_gans += [[best_gan]]

    best_gans = np.concatenate(best_gans, )
    np.save("BestTransGans.npy", best_gans)


def plotting(data):
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(441)

    for k in range(1, 17):
        ax1 = plt.subplot(4, 4, k)
        auxC = data[392 * (k - 1):392 * k, :]
        plt.plot(auxC[:, 0], auxC[:, 1], 'k.')
    plt.show()


def set_transfer(single_run):
    path = "BestGans.npy"
    if not os.path.isfile(path):
        collection = np.zeros((len(fs), len(runs), len(sizes), 72))

        for fi, f in enumerate(fs):
            for run in runs:
                for isize, size in enumerate(sizes):
                    gans = np.load("NewLossRes/ResultsPareto" + str(run) + "_F" + str(f) + "_100_100_2_" + str(size) + ".npy")
                    best = np.argmin(gans[:, -2])

                    collection[fi, run-1, isize, :] = gans[best, :-4].astype("int")

        np.save("BestGans.npy", collection)
    else:
        collection = np.load(path)

    set_data = []

    for i in range(1000):
        aux_data = create_data(n_centers, 784//2, shuffle=False)
        if i<10:
            np.save('original_'+str(i),aux_data)
        aux_data = aux_data.flatten()
        set_data += [[aux_data]]

    set_data = np.concatenate(set_data)

    set_data_min = np.min(set_data)
    set_data -= set_data_min
    set_data_max = np.max(set_data)
    set_data /= set_data_max

    results = np.zeros((len(reps), 10, 10))

    #for fi, f in enumerate(fs):
        #print(fi,f,single_run,set_data_min,set_data_max)
        #for single_run in runs:
            #best_gan = collection[fi, single_run, 0, :].astype("int")
    best_gan = collection[1, 9, 0, :].astype("int")
    descriptor = decodify_descriptor(best_gan, 784)
    gan = GAN(descriptor)
    for x in [4]:
        for y in [1]:
            #for z in range(10):
                #for w in range(10):
                    for rep in range(5):
                        gan.training_definition(rep, x=x/10, y=y/10, z=z/10, w=w/10)
                        samples = gan.separated_running("fixed", set_data, 100, 15000, 50, 15001)
                        fig = plt.figure(figsize=(10, 10))
                        plt.subplot(451)

                        for isam in range(49):
                            ax1 = plt.subplot(7, 7, isam + 1)
                            plt.plot(set_data[isam, ::2], set_data[isam, 1::2], "o")
                            plt.plot(samples[isam, ::2], samples[isam, 1::2], "o")
                            plt.title(str(isam))
                            plt.xlim((0, 1))
                            plt.ylim((0, 1))
                        plt.show()
                        gan.sess.close()





if __name__ == "__main__":
    #min_transfer()
#if "a" == "b":
    parser = argparse.ArgumentParser()
    # f5_example()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs="+", help='ur momma')
    args = parser.parse_args()
    z = args.integers[0]
    w = args.integers[1]
    rep = args.integers[2]
    seed_ = 2
    n_centers = 8
    points = 1000
    np.random.seed(0)
    random.seed(0)
    data = create_data(n_centers, points)
    unscaled_data = np.copy(data)
    data_min = np.min(data)
    data -= data_min
    data_max = np.max(data)
    data /= data_max
    #test_random(seed_)
    #cumulative()
    #random_trans()
    set_transfer(seed_)


"""import numpy as np
import matplotlib.pyplot as plt
import os

n_centers = 8
fi = 1
rep = 0
single_run = 9

collection = np.load("BestGans.npy")

for i in range(0,9):
     for j in range(0,30):
         path = 'FTRL_RMS_100_samp_10000_un_samples_' + str(i) + '_' + str(j) + '.npy'
         if not os.path.isfile(path):
             continue
         fig= plt.figure(figsize=(10,10))
         plt.subplot(441)
         print(i,j)

         C = np.load(path)
         print(C.shape)

         for k in range(1,17):
             ax1=plt.subplot(4, 4, k)
             auxC = C[392*(k-1):392*k,:]
             plt.plot(auxC[:,0], auxC[:,1],'k.')

         best_gan = collection[i, j, 0, :].astype("int")
         descriptor = decodify_descriptor(best_gan, 784)
         plt.suptitle("F" + str(i) + ", R" + str(j) + ", Dl" + str(descriptor.Disc_network.number_loop_train) + ", Gl" + str(descriptor.Gen_network.number_loop_train) + "-" + descriptor.f_measure)
         plt.savefig("hue/F" + str(i) + ", R" + str(j) + ", Dl" + str(descriptor.Disc_network.number_loop_train) + ", Gl" + str(descriptor.Gen_network.number_loop_train) + "-" + descriptor.f_measure + ".jpg")
         plt.clf()

fig= plt.figure(figsize=(10,10))
plt.subplot(451)

for isam in range(49):
    ax1 = plt.subplot(7, 7, isam+1)
    plt.plot(set_data[isam, ::2], set_data[isam, 1::2], "o")
    plt.plot(samples[isam, ::2], samples[isam, 1::2], "o")
    plt.title(str(isam))
    plt.xlim((0, 1))
    plt.ylim((0, 1))
plt.show()"""