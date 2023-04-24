import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scikit_posthocs import posthoc_dunn
import os
import tabulate

methods = ['RMS','FTRL','Adagrad','Momentum','RMS_FTRL','FTRL_RMS']
model = 0
params = '_100_samp_10000_un_samples_'
fs = [0,1, 2, 3, 4, 5, 6, 7]
runs = range(0, 30)

data = np.load("DistanceResults.npy")
data_norm = np.load("DistanceResultsNormalized.npy")
data = data.transpose(1, 0, 4, 2, 3)
data_norm = data_norm.transpose(1, 0, 4, 2, 3)
print(data.shape)
print(data_norm.shape)

prueba = np.reshape(data, (8, 6, 2, 30 * 17))
prueba_norm = np.reshape(data_norm, (8, 6, 2, 30 * 17))

vars_prueba = np.var(prueba, axis=3)
means_prueba = np.mean(prueba, axis=3)

vars_prueba_norm = np.var(prueba_norm, axis=3)
means_prueba_norm = np.mean(prueba_norm, axis=3)


def heatmaps():
    sns.heatmap(means_prueba_norm[:, :, 1], annot=vars_prueba_norm[:, :, 1], cbar_kws={'label': 'Mean MMD'})

    plt.yticks(np.arange(0.5,8.5), ["F" + str(i) for i in [1, 2, 3, 4, 5, 7, 8, 9]], rotation=0)
    plt.xticks(np.arange(0.5,6.5), methods, rotation=15)
    plt.ylim((8, 0))
    plt.title("Mean KL divergence")
    plt.show()

    sns.heatmap(means_prueba[:, :, 1], annot=vars_prueba[:, :, 1], cbar_kws={'label': 'Mean MMD'})

    plt.yticks(np.arange(0.5,8.5), ["F" + str(i) for i in [1, 2, 3, 4, 5, 7, 8, 9]], rotation=0)
    plt.xticks(np.arange(0.5,6.5), methods, rotation=15)
    plt.ylim((8, 0))
    plt.title("Mean MMD")
    plt.show()


def scatter_heat(data, data1, title=""):  # Origin Function, Instance, Target Function, Repetition, Metric


    grid_x_offset = np.array([[i]*data.shape[0] for i in range(data.shape[1])]).flatten()
    grid_x = np.array([[i]*data.shape[0] for i in range(data.shape[1])]).flatten()
    grid_y = np.array([[i]*data.shape[1] for i in range(data.shape[0])]).transpose().flatten()

    means1 = np.mean(data1, axis=(2, 3))
    variances1 = np.var(data1, axis=(2, 3))
    means = np.mean(data, axis=(2, 3))
    variances = np.var(data, axis=(2, 3))
    # mean = np.mean(np.log(variances))

    f, ax = plt.subplots()

    temp = means.T.flatten().argsort()
    rank = np.empty_like(temp)
    rank[temp] = np.arange(len(means.T.flatten()))

    temp = means1.T.flatten().argsort()
    rank1 = np.empty_like(temp)
    rank1[temp] = np.arange(len(means1.T.flatten()))

    rank_dif = np.reshape(rank-rank1, (6,8)).T
    jj = ax.scatter(x=grid_x_offset, y=grid_y, c=np.log(variances.T.flatten()), s=means.T.flatten()*350, marker=8, cmap="viridis")
    jj = ax.scatter(x=grid_x_offset, y=grid_y, c=np.log(variances1.T.flatten()), s=means1.T.flatten()*350, marker=9, cmap="viridis")
    grid_x_offset = np.unique(grid_x_offset)
    plt.subplots_adjust(hspace=0, wspace=0, left=0.08, right=1, bottom=0.1, top=0.94)
    plt.title(title)

    test_path = "tests.npy"

    if not os.path.isfile(test_path):
        tests = np.zeros((len(methods), len(fs), 30, 2))
        for ifunction, function in enumerate(fs):
            for imethod, method in enumerate(methods):
                for run in range(30):
                    for imeth in range(imethod+1, len(methods)):
                        for norm, d in enumerate([data, data1]):
                            for norm1, d1 in enumerate([data, data1]):
                                if norm == 0 and norm1 == 1:
                                    continue
                                print(d[ifunction, imethod, run].shape)
                                test = posthoc_dunn([d[ifunction, imethod, run], d1[ifunction, imeth, run]])[1][2]
                                if test < 0.05:
                                    if np.mean(d[ifunction, imethod, run]) < np.mean(d1[ifunction, imeth, run]):
                                        tests[imethod, ifunction, run, norm] += 1
                                        tests[imeth, ifunction, run, norm1] -= 1
                                    else:
                                        tests[imethod, ifunction, run, norm] -= 1
                                        tests[imeth, ifunction, run, norm1] += 1

        np.save(test_path, tests)
    else:
        tests = np.load(test_path)

    tests = tests.sum(axis=2)
    tests = np.concatenate([np.array([[x[0], x[1]] for x in y]) for y in tests], axis=1)
    tests = np.concatenate((tests, np.mean(tests, axis=0).reshape((1, -1))), axis=0)
    print()
    tests = np.concatenate((tests, np.concatenate((np.mean(tests[:, 0::2], axis=1).reshape((-1, 1)), np.mean(tests[:, 1::2], axis=1).reshape((-1, 1))), axis=1)), axis=1)
    print(tabulate.tabulate(tests, tablefmt="latex", showindex="always"))
    for x, y in zip(grid_x, grid_y):
        plt.text(s=rank_dif[y, x], x=grid_x_offset[x] - (0 if np.log(variances[y, x]) < 1 else 0.1), y=y, horizontalalignment='center', verticalalignment='center', color="yellow" if np.log(variances[y, x]) < 1 else "black", fontsize=10)  # 2.5
        #plt.text(s=int(tests[x, y, 0]), x=grid_x_offset[x] - 0.25, y=y+0.3, horizontalalignment='center', verticalalignment='center', color="black", fontsize=10)  # 2.5
        #plt.text(s=int(tests[x, y, 1]), x=grid_x_offset[x] + 0.25, y=y + 0.3, horizontalalignment='center', verticalalignment='center', color="black", fontsize=10)  # 2.5
    plt.xticks(range(6), methods)
    plt.yticks([])
    plt.yticks(range(8), ["F" + str(i) for i in fs])
    plt.xlabel("Optimizer")
    plt.ylabel("Function")
    plt.title("MMD difference")
    print(plt.xlim(-0.4, 5.2))
    f.colorbar(mappable=jj, label=r"$log(var)$ values")

    plt.show()

#heatmaps()
scatter_heat(data[:, :, 1, :, :], data_norm[:, :, 1, :, :])