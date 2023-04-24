import tensorflow as tf
import numpy as np
from tunable_gan_class_optimization import xavier_init, GANDescriptor, GAN
from Class_F_Functions import FFunctions, pareto_frontier, igd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from scipy.stats import entropy
import random
import argparse
from scipy.stats import wilcoxon
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.lines import Line2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

lat_functions = np.array([np.random.uniform, np.random.normal])
lats = ["Uniform", "Normal"]
div_ms = ["Standard", "Forw.KL", "Rever.KL", "Pearson", "Sqr.Hel.", "Lst.Sqr.", "Modified", "Wass"]
divergence_measures = ["Standard_Divergence", "Forward_KL", "Reverse_KL", "Pearson_Chi_squared", "Squared_Hellinger",
                       "Least_squared", "Modified", "Wassertain"]
act_functions = np.array([None, tf.nn.relu, tf.nn.elu, tf.nn.softplus, tf.nn.softsign, tf.sigmoid, tf.nn.tanh])
acts = np.array(["Ident.", "ReLU", "eLU", "Softplus", "Softsign", "Sigmoid", "Tanh"])
max_layers = 10
init_functions = np.array([xavier_init, tf.random_uniform, tf.random_normal])
inits = ["Xavier", "Uniform", "Normal"]

general = ["Latent", "Loss"]
generator = ["Gen_N_hidden", "Gen_Loop"] + ["Gen_H_neurons_" + str(i) for i in range(11)] + ["Gen_Init_" + str(i) for i in range(11)] + ["Gen_Act_" + str(i) for i in range(11)]
discs = ["Dis_N_hidden", "Dis_Loop"] + ["Dis_H_neurons_" + str(i) for i in range(11)] + ["Dis_Init_" + str(i) for i in range(11)] + ["Dis_Act_" + str(i) for i in range(11)]
variables = general + generator + discs + ["Pareto", "IGD", "Time"]

x_dims = [10, 784]
functions = [1, 2, 3, 4, 5, 7, 8, 9]
# n_ind, n_gen = 50, 101


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)


def read_file(name):
    networks = np.load(name)
    objectives = networks[:, -3:]
    networks = networks[:, :-4]
    return networks, objectives


def decodify_descriptor(code, x_dim):

    latent = lat_functions[code[0]]
    divergence = divergence_measures[code[1]]
    gan_descriptor = GANDescriptor(x_dim, 30, latent, divergence)
    gen = code[2:37]
    gan_descriptor.gan_generator_initialization(n_hidden=gen[0], dim_list=gen[2:2+gen[0]], init_functions=init_functions[gen[13:14+gen[0]]], act_functions=act_functions[gen[24:25+gen[0]]], number_loop_train=gen[1])
    disc = code[37:]
    gan_descriptor.gan_discriminator_initialization(n_hidden=disc[0], dim_list=disc[2:2+disc[0]], init_functions=init_functions[disc[13:14+disc[0]]], act_functions=act_functions[disc[24:25+disc[0]]], number_loop_train=disc[1])
    return gan_descriptor


def example(function, x, gan_descriptor):
    mop_f = FFunctions(x, "F" + str(function))
    ps_all_x = mop_f.generate_ps_samples(1000)
    pf1, pf2 = mop_f.evaluate_mop_function(ps_all_x)
    ps_all_x[:, 1:] = (ps_all_x[:, 1:]+1)/2
    gan = GAN(gan_descriptor)
    gan.training_definition()
    samples = gan.separated_running("fixed", ps_all_x, 150, 1001,  1000, 1001)
    samples[:, 1:] = (samples[:, 1:]-0.5)*2
    nf1, nf2 = mop_f.evaluate_mop_function(samples)
    igd_val = igd((np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose()))
    # gd_val = igd((np.vstack((pf1, pf2)).transpose(), np.vstack((nf1, nf2)).transpose()))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$f(x_1)$')
    ax.set_ylabel('$f(x_2)$')
    plt.plot(pf1, pf2, 'b.')
    plt.plot(nf1, nf2, 'r.')
    plt.text(0, 0, "IGD: " + '%.3f' % igd_val[0], transform=ax.transAxes)
    # plt.text(0, 1, "GD:" + '%.3f' % gd_val[0], transform=ax.transAxes)
    plt.show()


def igd_gd(coded_gan, x_size, f):  # Returns (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f)

    mop_f = FFunctions(x_size, "F" + str(f))
    ps_all_x = mop_f.generate_ps_samples(1000)

    pf1, pf2 = mop_f.evaluate_mop_function(ps_all_x)

    gan_descriptor = decodify_descriptor(coded_gan, x_size)
    gan = GAN(gan_descriptor)
    gan.training_definition()
    samples = gan.separated_running("fixed", ps_all_x, 150, 1001, 1000, 1001, convergence=1002)

    nf1, nf2 = mop_f.evaluate_mop_function(samples)
    parameter_tuples = [(np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose()), (np.vstack((pf1, pf2)).transpose(), np.vstack((nf1, nf2)).transpose()), (ps_all_x, samples), (samples, ps_all_x)]
    results = []
    for params in parameter_tuples:
        results += [igd(params)]
    return results


def igd_gd_corr():

    data = pd.DataFrame(columns=["Instance", "Generation", "Function", "IGD_Value", "Time", "New_IGD", "New_IGDP", "New_GD", "New_GDP", "IGD_Ord", "Var_IGD", "Var_IGDP", "Var_GD", "Var_GDP"])

    # Checkpoint
    ds = 0
    ind = 0
    inst = 1

    while ds < len(x_dims):
        d = x_dims[ds]
        print("Dimension:", d)
        while ind < len(functions):
            f = functions[ind]
            print("\tFunction:", f)
            while inst < 31:
                reset_graph(inst)
                print("\t\tInstance", inst)
                path = "results/GAN_Evals_" + str(inst) + "_F" + str(f) + "_N_20_ngen_500_Sel_0_X_" + str(d) + ".npy"
                if not os.path.isfile(path):
                    inst += 1
                    continue
                nets, objs = read_file(path)

                i1, i2, par_ind = pareto_frontier(objs[:, 1], objs[:, 2])

                for i, gan in enumerate(par_ind):
                    net = nets[gan]
                    results = igd_gd(net.astype("int"), d, f)

                    data.loc[-1] = [inst, "250", f, objs[gan, 1], objs[gan, 2], results[0][0], results[0][1], results[1][0], results[1][1], i, results[2][0], results[2][1], results[3][0], results[3][1]]
                    data.index = data.index + 1
                data.to_csv("new_res/GD_IGDs_best_F" + str(f) + "_D" + str(d) + "_Inst" + str(inst) + ".csv")
                inst += 1
            ind += 1
            inst = 0
        ds += 1
        ind = 0


def test():
    path = "results/GAN_Evals_1_F2_N_20_ngen_500_Sel_0_X_10.npy"
    nets, objs = read_file(path)
    hue = np.argsort(objs[:, 1])
    nets = nets.astype("int")
    nets = nets[hue]
    example(2, 10, decodify_descriptor(nets[0], 10))
    example(2, 10, decodify_descriptor(nets[int(nets.shape[0]/2)], 10))
    example(2, 10, decodify_descriptor(nets[-1], 10))


def boxplots(objs, n_ind):

    gens = [1, 26, 51, 76, 101]

    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row")
    axs[0][0].labelsize = "xx-large"
    plt.subplots_adjust(hspace=0, wspace=0, left=0.09, right=1, bottom=0.1, top=1)
    for ds, d in enumerate(x_dims):
        best_igds_gen = pd.DataFrame(columns=["Instance", "Generation", "Function", "IGD Value", "Time"])
        for ind, f in enumerate(functions):
            for inst in range(1, 31):
                for gen in gens:
                    arg = np.argmin(objs[ds, ind, inst-1, :gen, :n_ind, 1])
                    best_igds_gen.loc[-1] = [inst, str(gen-1), "F" + str(f), objs[ds, ind, inst-1, :gen, :n_ind, 1].flatten()[arg], objs[ds, ind, inst-1, :gen, :n_ind, 2].flatten()[arg]]
                    best_igds_gen.index = best_igds_gen.index + 1

        sns.boxplot(x="Function", y="Time", hue="Generation", data=best_igds_gen, palette="CMRmap", ax=axs[1][ds], linewidth=.5, fliersize=1, hue_order=[str(x-1) for x in gens])
        sns.boxplot(x="Function", y="IGD Value", hue="Generation", data=best_igds_gen, palette="CMRmap", ax=axs[0][ds], linewidth=.5, fliersize=1, hue_order=[str(x-1) for x in gens])

    axs[0][1].xaxis.set_ticks_position('none')
    axs[0][0].xaxis.set_ticks_position('none')
    axs[0][1].axes.get_xaxis().set_visible(False)
    axs[0][0].axes.get_xaxis().set_visible(False)
    axs[0][1].yaxis.set_ticks_position('none')
    axs[1][1].yaxis.set_ticks_position('none')
    axs[1][1].yaxis.set_visible(False)
    axs[0][1].yaxis.set_visible(False)
    axs[1][0].set_xlabel("$n = 10$")
    axs[1][1].set_xlabel("$n = 784$")
    axs[0][1].get_legend().remove()
    axs[0][0].get_legend().remove()
    axs[1][1].get_legend().remove()
    axs[1][0].get_legend().remove()
    axs[1][0].legend(ncol=3, title="Generation")

    # fig.text(0.6, 0.01, 'Function', ha='center', va='center', size="large")

    plt.show()
    

def densities():

    igds = np.zeros((8, 30, 10020))

    for ds, d in enumerate(x_dims):
        for ind, f in enumerate(functions):
            for inst in range(1, 31):
                path = "results/GAN_Evals_" + str(inst) + "_F" + str(f) + "_N_20_ngen_500_Sel_0_X_" + str(d) + ".npy"
                if not os.path.isfile(path):
                    continue
                nets, objs = read_file(path)
                igds[ind, inst-1, :] = objs[:, 1]
                sns.kdeplot(igds[ind, inst-1, :])
    return igds


def load_info(force=False, reduced=False, n_gen=100, n_ind=100):

    #root = str(n_gen-1) + "Gen" + str(n_ind) + "Ind/"
    root = "NewLossRes/"
    orig_path = root + "Gens.npy"
    sel = [2, 2]

    if os.path.isfile(orig_path) and not force:
        if not reduced:
            res = np.load(orig_path)
            return res[:, :, :, :, :, :-3].astype("int"), res[:, :, :, :, :, -3:]
        else:
            return reduce_gens(orig_path, force)

    # Dims, Functions, Instances, Generations, Individuals, Variables
    generations = np.zeros((2, 8, 30, n_gen+1, n_ind+1, 75))

    for ds, d in enumerate(x_dims):
        print("Dim", d)
        for ind, f in enumerate(functions):
            print("\tFunction", f)
            for inst in range(1, 31):
                print("\t\tInstance", inst)
                #path = root + "GAN_Evals_" + str(inst) + "_F" + str(f) + "_N_" + str(n_ind) + "_ngen_" + str(n_gen-1) + "_Sel_" + str(sel[ds]) + "_X_" + str(d) + ".npy"

                path = root + "ResultsPareto" + str(inst) + "_F"+str(f)+"_" + str(n_ind) + "_" + str(n_gen-1) + "_" + str(sel[ds]) + "_" + str(d) + ".npy"

                if not os.path.isfile(path):
                    print(path)
                    continue
                nets, objs = read_file(path)
                for gen in range(n_gen):
                    generations[ds, ind, inst-1, gen, :n_ind, :72] = nets[gen*n_ind:(gen+1)*n_ind]
                    generations[ds, ind, inst-1, gen, :n_ind, 72:] = objs[gen*n_ind:(gen+1)*n_ind]
                    for var in range(72):
                        un, cnt = np.unique(generations[ds, ind, inst-1, gen, :n_ind, var], return_counts=True)
                        generations[ds, ind, inst-1, gen, n_ind, var] = entropy(cnt)/un.shape[0]
                    generations[ds, ind, inst-1, gen, n_ind, 72:] = np.var(generations[ds, ind, inst-1, gen, :n_ind, 72:], axis=0)

    np.save(orig_path, generations)
    if not reduced:
        return generations[:, :, :, :, :, :-3].astype("int"), generations[:, :, :, :, :, -3:]
    else:
        return reduce_gens(orig_path, force)


def show_entr(gen):
    sns.set_style("darkgrid")

    for idim, dim in enumerate(x_dims):
        fig, axs = plt.subplots(9, sharex="col")
        plt.subplots_adjust(hspace=0)
        for ifs, fs in enumerate(functions):
            for e, j in enumerate([0, 1, 2, 3, 37, 38, 72, 73, 74]):
                axs[e].plot(np.mean(gen[idim, ifs, :, :, 20, j], axis=0), label="Function " + str(fs))
                axs[e].set_title(variables[j] + (" Entropy" if j < 71 else " Variance"), loc="right", y=0.5)
        axs[-1].legend(loc=(0.8, -1))
        axs[-1].set_xlabel("Generation")
        axs[-2].set_ylim((0, 5))
        axs[0].set_title("Mean Entropy or Variance for 9 variables across 30 runs for each function of size " + str(dim))
        plt.show()

    # auxx = fig.add_axes([0,0,0,0])
    # tp = TextPath((0.2, 0.65), "{", size=1)
    # trans = mtrans.Affine2D().scale(0.1, 0.6) + \
    # mtrans.Affine2D().translate(0, 0) + fig.transFigure
    # pp = PathPatch(tp, lw=0, fc="k", transform=trans)
    # auxx.add_artist(pp)


def app_per_gen(gen):
    for fi, f in enumerate(functions):
        fig, axs = plt.subplots(14, 2, sharex="col")
        fig.suptitle(r"Mean values for $20individuals\times 30runs=600$ in F" + str(f))
        plt.subplots_adjust(hspace=0)
        for dimi, dim in enumerate(x_dims):
            axs[0, dimi].set_title("Size: " + str(dim))
            latent = np.concatenate([gen[dimi, fi, :, :, i, 0] for i in range(20)], axis=0)
            lat_un = [np.unique(latent[:, lt], return_counts=True) for lt in range(latent.shape[1])]
            aux = np.zeros((501, 2))
            for i in range(501):
                aux[i, lat_un[i][0].astype("int")] = lat_un[i][1]
            for i in range(len(lat_functions)):
                axs[0, dimi].plot(aux[:, i], label=lats[i])
            axs[0, dimi].legend(loc="lower center")
            axs[0, dimi].set_ylabel("Latent dist.", rotation=60)

            loss = np.concatenate([gen[dimi, fi, :, :, i, 1] for i in range(20)], axis=0)
            loss_un = [np.unique(loss[:, lt], return_counts=True) for lt in range(loss.shape[1])]
            aux = np.zeros((501, 6))
            for i in range(501):
                aux[i, loss_un[i][0].astype("int")] = loss_un[i][1]
            for i in range(len(divergence_measures)):
                axs[1, dimi].plot(aux[:, i], label=divergence_measures[i])
            axs[1, dimi].legend(loc="lower right")
            axs[1, dimi].set_ylabel("Loss f.", rotation=60)

            gen_n = np.concatenate([gen[dimi, fi, :, :, i, 2] for i in range(20)], axis=0)
            gen_n_un = [np.unique(gen_n[:, lt], return_counts=True) for lt in range(gen_n.shape[1])]
            aux = np.zeros((501, 11))
            for i in range(501):
                aux[i, gen_n_un[i][0].astype("int")] = gen_n_un[i][1]
            for i in range(1, 11):
                axs[2, dimi].plot(aux[:, i], label=str(i))
            axs[2, dimi].legend(loc="lower left")
            axs[2, dimi].set_ylabel("Gen. size", rotation=60)

            gen_l = np.concatenate([gen[dimi, fi, :, :, i, 3] for i in range(20)], axis=0)
            gen_l_un = [np.unique(gen_l[:, lt], return_counts=True) for lt in range(gen_l.shape[1])]
            aux = np.zeros((501, 6))
            for i in range(501):
                aux[i, gen_l_un[i][0].astype("int")] = gen_l_un[i][1]
            for i in range(1, 6):
                axs[3, dimi].plot(aux[:, i], label=str(i))
            axs[3, dimi].legend(loc="lower right")
            axs[3, dimi].set_ylabel("Gen. loop", rotation=60)

            gen_ns = gen[dimi, fi, :, :, :20, 4:15]
            gen_ns = np.reshape(gen_ns, gen_ns.shape[:-2] + (-1,))
            gen_ns = np.concatenate([gen_ns[i] for i in range(gen_ns.shape[0])], axis=1)
            gen_ns[gen_ns == -1] = 0
            means = []
            for i in range(gen_ns.shape[0]):
                means += [np.sum(gen_ns[i])/np.sum(gen_n[:, i])]
            axs[6, dimi].plot(means, label="Gen.")
            axs[6, dimi].set_ylabel("Gen. lay.", rotation=60)

            gen_init = np.concatenate([gen[dimi, fi, :, :, i, 15] for i in range(20)], axis=0)
            gen_init_un = [np.unique(gen_init[:, lt], return_counts=True) for lt in range(gen_init.shape[1])]
            aux = np.zeros((501, 4))
            for i in range(501):
                aux[i, gen_init_un[i][0].astype("int")] = gen_init_un[i][1]
            for i in range(3):
                axs[4, dimi].plot(aux[:, i], label=inits[i])
            axs[4, dimi].legend(loc="lower center")
            axs[4, dimi].set_ylabel("Gen. init.", rotation=60)

            gen_act = np.concatenate([gen[dimi, fi, :, :, i, 26] for i in range(20)], axis=0)
            gen_act_un = [np.unique(gen_act[:, lt], return_counts=True) for lt in range(gen_act.shape[1])]
            aux = np.zeros((501, 7))
            for i in range(501):
                aux[i, gen_act_un[i][0].astype("int")] = gen_act_un[i][1]
            for i in range(7):
                axs[5, dimi].plot(aux[:, i], label=acts[i])
            axs[5, dimi].legend(loc="lower right")
            axs[5, dimi].set_ylabel("Gen. act.", rotation=60)

            dis_n = np.concatenate([gen[dimi, fi, :, :, i, 37] for i in range(20)], axis=0)
            dis_n_un = [np.unique(dis_n[:, lt], return_counts=True) for lt in range(dis_n.shape[1])]
            aux = np.zeros((501, 11))
            for i in range(501):
                aux[i, dis_n_un[i][0].astype("int")] = dis_n_un[i][1]
            for i in range(1, 11):
                axs[7, dimi].plot(aux[:, i], label=str(i))
            axs[7, dimi].legend(loc="lower left")
            axs[7, dimi].set_ylabel("Dis. size", rotation=60)

            dis_l = np.concatenate([gen[dimi, fi, :, :, i, 38] for i in range(20)], axis=0)
            dis_l_un = [np.unique(dis_l[:, lt], return_counts=True) for lt in range(dis_l.shape[1])]
            aux = np.zeros((501, 6))
            for i in range(501):
                aux[i, dis_l_un[i][0].astype("int")] = dis_l_un[i][1]
            for i in range(1, 6):
                axs[8, dimi].plot(aux[:, i], label=str(i))
            axs[8, dimi].legend(loc="lower right")
            axs[8, dimi].set_ylabel("Dis. Loop", rotation=60)

            dis_ns = gen[dimi, fi, :, :, :20, 39:50]
            dis_ns = np.reshape(dis_ns, dis_ns.shape[:-2] + (-1,))
            dis_ns = np.concatenate([dis_ns[i] for i in range(dis_ns.shape[0])], axis=1)
            dis_ns[dis_ns == -1] = 0
            means = []
            for i in range(dis_ns.shape[0]):
                means += [np.sum(dis_ns[i])/np.sum(dis_n[:, i])]
            axs[6, dimi].plot(means, label="Disc.")
            axs[6, dimi].set_ylabel("Dis. lay.", rotation=60)
            axs[6, dimi].legend(loc="lower right")

            dis_act = np.concatenate([gen[dimi, fi, :, :, i, 61] for i in range(20)], axis=0)
            dis_act_un = [np.unique(dis_act[:, lt], return_counts=True) for lt in range(dis_act.shape[1])]
            aux = np.zeros((501, 7))
            for i in range(501):
                aux[i, dis_act_un[i][0].astype("int")] = dis_act_un[i][1]
            for i in range(7):
                axs[10, dimi].plot(aux[:, i], label=acts[i])
            axs[10, dimi].legend(loc="lower left")
            axs[10, dimi].set_ylabel("Dis. act.", rotation=60)

            dis_init = np.concatenate([gen[dimi, fi, :, :, i, 50] for i in range(20)], axis=0)
            dis_init_un = [np.unique(dis_init[:, lt], return_counts=True) for lt in range(dis_init.shape[1])]
            aux = np.zeros((501, 4))
            for i in range(501):
                aux[i, dis_init_un[i][0].astype("int")] = dis_init_un[i][1]
            for i in range(3):
                axs[9, dimi].plot(aux[:, i], label=inits[i])
            axs[9, dimi].legend(loc="lower right")
            axs[9, dimi].set_ylabel("Dis. init.", rotation=60)

            pareto = np.concatenate([gen[dimi, fi, :, :, i, 72] for i in range(20)], axis=0)
            pareto = np.mean(pareto, axis=0)
            axs[11, dimi].plot(pareto)
            axs[11, dimi].set_ylabel("Par. sz.", rotation=60)

            igds = np.concatenate([gen[dimi, fi, :, :, i, 73] for i in range(20)], axis=0)
            igds[igds > 100] = np.nan
            igds = np.nanmean(igds, axis=0)
            axs[12, dimi].plot(igds)
            axs[12, dimi].set_ylabel("IGD", rotation=60)

            time = np.concatenate([gen[dimi, fi, :, :, i, 74] for i in range(20)], axis=0)
            time = np.mean(time, axis=0)
            axs[13, dimi].plot(time)
            axs[13, dimi].set_ylabel("Time", rotation=60)
            axs[13, dimi].set_xlabel("Generations")

        plt.show()


def first_last(gen, n_ind=50):

    firsts = gen[:, :, :, 0, :, :]
    lasts = gen[:, :, :, -2, :, :]
    for fi, f in enumerate(functions[:-1]):
        fig, axs = plt.subplots(4, 6, sharey="row")
        fig.set_size_inches(16, 6)
        fig.suptitle("Appearances from each variable in first and las generations for F" + str(f), fontsize="xx-large")
        plt.subplots_adjust(hspace=0, wspace=0, left=0.06, right=0.99, bottom=0.1, top=0.88)

        for dimi, dim in enumerate(x_dims):

            fgn = firsts[dimi, fi, :, :n_ind, 2].flatten()
            lgn = lasts[dimi, fi, :, :n_ind, 2].flatten()
            sns.distplot(fgn, ax=axs[dimi*2, 0], kde=False, bins=range(1, np.unique(fgn).shape[0]+1), label="First Gen.")
            sns.distplot(lgn, ax=axs[dimi*2, 0], kde=False, bins=range(1, np.unique(fgn).shape[0]+1), label="Last Gen.")

            fgl = firsts[dimi, fi, :, :n_ind, 3].flatten()
            lgl = lasts[dimi, fi, :, :n_ind, 3].flatten()
            sns.distplot(fgl, ax=axs[dimi*2, 1], kde=False, bins=range(1, np.unique(fgl).shape[0]+1))
            sns.distplot(lgl, ax=axs[dimi*2, 1], kde=False, bins=range(1, np.unique(fgl).shape[0]+1))

            fgns = firsts[dimi, fi, :, :n_ind, 4:15].flatten()
            lgns = lasts[dimi, fi, :, :n_ind, 4:15].flatten()
            sns.distplot(fgns[fgns > 0], ax=axs[dimi*2, 2], kde=False, bins=range(1, np.unique(fgns).shape[0]))
            sns.distplot(lgns[lgns > 0], ax=axs[dimi*2, 2], kde=False, bins=range(1, np.unique(fgns).shape[0]))

            fgi = firsts[dimi, fi, :, :n_ind, 15].flatten()
            lgi = lasts[dimi, fi, :, :n_ind, 15].flatten()
            sns.distplot(fgi, ax=axs[dimi*2, 3], kde=False, bins=range(np.unique(fgi).shape[0]+1))
            sns.distplot(lgi, ax=axs[dimi*2, 3], kde=False, bins=range(np.unique(fgi).shape[0]+1))

            fga = firsts[dimi, fi, :, :n_ind, 26].flatten()
            lga = lasts[dimi, fi, :, :n_ind, 26].flatten()
            sns.distplot(fga, ax=axs[dimi*2, 4], kde=False, bins=range(np.unique(fga).shape[0]+1))
            sns.distplot(lga, ax=axs[dimi*2, 4], kde=False, bins=range(np.unique(fga).shape[0]+1))

            fds = firsts[dimi, fi, :, :n_ind, 37].flatten()
            lds = lasts[dimi, fi, :, :n_ind, 37].flatten()
            sns.distplot(fds, ax=axs[dimi*2+1, 0], kde=False, bins=range(1, np.unique(fds).shape[0]+1))
            sns.distplot(lds, ax=axs[dimi*2+1, 0], kde=False, bins=range(1, np.unique(fds).shape[0]+1))

            fdl = firsts[dimi, fi, :, :n_ind, 38].flatten()
            ldl = lasts[dimi, fi, :, :n_ind, 38].flatten()
            sns.distplot(fdl, ax=axs[dimi*2+1, 1], kde=False, bins=range(1, np.unique(fdl).shape[0]+1))
            sns.distplot(ldl, ax=axs[dimi*2+1, 1], kde=False, bins=range(1, np.unique(fdl).shape[0]+1))

            fdns = firsts[dimi, fi, :, :n_ind, 39:50].flatten()
            ldns = lasts[dimi, fi, :, :n_ind, 39:50].flatten()
            sns.distplot(fdns[fdns > 0], ax=axs[dimi*2+1, 2], kde=False, bins=range(1, np.unique(fdns).shape[0]))
            sns.distplot(ldns[ldns > 0], ax=axs[dimi*2+1, 2], kde=False, bins=range(1, np.unique(fdns).shape[0]))

            fdi = firsts[dimi, fi, :, :n_ind, 50].flatten()
            ldi = lasts[dimi, fi, :, :n_ind, 50].flatten()
            sns.distplot(fdi, ax=axs[dimi*2+1, 3], kde=False, bins=range(np.unique(fdi).shape[0]+1))
            sns.distplot(ldi, ax=axs[dimi*2+1, 3], kde=False, bins=range(np.unique(fdi).shape[0]+1))

            fda = firsts[dimi, fi, :, :n_ind, 61].flatten()
            lda = lasts[dimi, fi, :, :n_ind, 61].flatten()
            sns.distplot(fda, ax=axs[dimi*2+1, 4], kde=False, bins=range(np.unique(fda).shape[0]+1))
            sns.distplot(lda, ax=axs[dimi*2+1, 4], kde=False, bins=range(np.unique(fda).shape[0]+1))

            floss = firsts[dimi, fi, :, :n_ind, 1].flatten()
            lloss = lasts[dimi, fi, :, :n_ind, 1].flatten()
            sns.distplot(floss, ax=axs[dimi*2+1, 5], kde=False, bins=range(np.unique(floss).shape[0]+1))
            sns.distplot(lloss, ax=axs[dimi*2+1, 5], kde=False, bins=range(np.unique(floss).shape[0]+1))

            flat = firsts[dimi, fi, :, :n_ind, 0].flatten()
            llat = lasts[dimi, fi, :, :n_ind, 0].flatten()
            sns.distplot(flat, ax=axs[dimi*2, 5], label="First Gen.", kde=False, bins=range(np.unique(flat).shape[0]+1))
            sns.distplot(llat, ax=axs[dimi*2, 5], label="Last Gen.", kde=False, bins=range(np.unique(flat).shape[0]+1))

        axs[3, 0].set_yticks([0, 275])
        axs[3, 0].set_yticklabels(["0", "275"])
        axs[3, 0].set_ylim(0, 550)
        axs[2, 0].set_yticks([0, 275])
        axs[2, 0].set_yticklabels(["0", "275"])
        axs[2, 0].set_ylim(0, 550)
        axs[1, 0].set_yticks([0, 275])
        axs[1, 0].set_yticklabels(["0", "275"])
        axs[1, 0].set_ylim(0, 550)
        axs[0, 0].set_yticks([0, 275, 550])
        axs[0, 0].set_yticklabels(["0", "275", "550"])

        axs[0, 0].legend()
        axs[0, 0].set_title("N. Layers")
        axs[0, 1].set_title("Loop")
        axs[0, 2].set_title("Neurons")
        axs[0, 3].set_title("Initialization")
        axs[0, 4].set_title("Activation")
        axs[0, 5].set_title("Loss f./Lat.repr.")
        axs[0, 0].set_ylabel("Generator")
        axs[1, 0].set_ylabel("Discriminator")
        axs[2, 0].set_ylabel("Generator")
        axs[3, 0].set_ylabel("Discriminator")
        axs[2, 5].set_zorder(20)
        axs[1, 5].set_zorder(21)
        axs[0, 5].set_zorder(22)

        plt.text(-29.5, 1700, r"$n=10$", rotation=90, size="x-large")
        plt.text(-29.5, 700, r"$n=784$", rotation=90, size="x-large")

        for i in range(3):
            for j in range(6):
                axs[i, j].set_xticks([])
        for i in range(4):
            for j in range(1, 6):
                axs[i, j].yaxis.set_ticks_position('none')
        axs[3, 0].set_xticks(np.arange(16/11, 11, 10/11))
        axs[3, 0].set_xticklabels([str(i) for i in range(1, 11)])
        axs[3, 1].set_xticks(np.arange(7/5, 5, 4/5))
        axs[3, 1].set_xticklabels(["1", "2", "3", "4", "5"])
        axs[3, 2].set_xticks(np.arange(75/51, 51, 7*50/51))
        axs[3, 2].set_xticklabels([str(i) for i in np.arange(1, 51, 7)])
        axs[3, 3].set_xticks(np.arange(1/2, 3, 1))
        axs[3, 3].set_xticklabels(inits)
        axs[3, 4].set_xticks(np.arange(1/2, 7, 1))
        axs[3, 4].set_xticklabels(acts, rotation=30)
        axs[3, 5].set_xticks(np.arange(0.5, 6, 1))
        axs[3, 5].set_xticklabels(div_ms, rotation=30)
        axs[1, 5].set_xticks(np.arange(0.5, 6, 1))
        axs[1, 5].set_xticklabels(div_ms, rotation=30)
        axs[2, 5].set_xticks(np.arange(0.5, 2, 1))
        axs[2, 5].set_xticklabels(lats)
        axs[0, 5].set_xticks(np.arange(0.5, 2, 1))
        axs[0, 5].set_xticklabels(lats)

        plt.savefig('First_Last_F' + str(f) + '.pdf')
        # plt.show()


def char_per_func(gan, n_ind, char=("neurons",)):

    chars = {"layers": [2, 37], "loop": [3, 38], "neurons": [np.arange(4, 15), np.arange(39, 50)], "init": [np.arange(15, 26), np.arange(50, 61)], "act": [np.arange(26, 37), np.arange(61, 72)], "losslat": [0, 1]}
    fs = [1, 2, 3, 4, 5, 7, 8, 9]

    for ch in char:

        fig, axs = plt.subplots(2, len(fs), sharey="row")
        fig.set_size_inches(16, 6)
        plt.subplots_adjust(hspace=0, wspace=0, left=0.07, right=1, bottom=0.14, top=0.94)
        # Dims, Functions, Instances, Generations, Individuals, Variables

        for fi, fun in enumerate(fs):
            for dimi, dim in enumerate(x_dims):

                f = gan[dimi, fi, :, 0, :n_ind, chars[ch][0]].flatten()
                l = gan[dimi, fi, :, -2, :n_ind, chars[ch][0]].flatten()

                sns.distplot(l, ax=axs[0 if ch is "losslat" else dimi, fi], kde=False, bins=range(int(ch is "layers" or ch is "loop" or ch is "neurons"), np.unique(f).shape[0]+2*int(ch is "layers" or ch is "loop" or ch is "neurons")+int(ch is "losslat")), label="$n=10$", hist_kws={"histtype": 'step', "stacked": True, "fill": False, "color": "r", "linewidth": 2} if (dimi == 1 and ch is "losslat") else {})
                f = gan[dimi, fi, :, 0, :n_ind, chars[ch][1]].flatten()
                l = gan[dimi, fi, :, -2, :n_ind, chars[ch][1]].flatten()

                sns.distplot(l, ax=axs[1 if ch is "losslat" else dimi, fi], kde=False, bins=range(int(ch is "layers" or ch is "loop" or ch is "neurons"), np.unique(f).shape[0]+2*int(ch is "layers" or ch is "loop" or ch is "neurons")+int(ch is "losslat")), label="$n=784$", hist_kws={"histtype": 'step', "stacked": True, "fill": False, "color": "r", "linewidth": 2} if (dimi == 1 and ch is "losslat") or ch is not "losslat" else {})

        if ch == "losslat":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[0, j].xaxis.tick_top()
                axs[0, j].set_xticks(np.arange(1/2, 2, 1))
                axs[0, j].set_xticklabels(["Unif.", "Norm."])
                axs[0, j].tick_params(axis='both', which='major', labelsize=15)
                axs[1, j].tick_params(axis='both', which='major', labelsize=15)
                axs[1, j].set_xticks(np.arange(1/2, 8, 1))
                axs[1, j].set_xticklabels(["S", "F", "R", "P", "H", "L", "M", "W"])
                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)

            # axs[0, 0].set_yticks([0, 450, 900])
            # axs[0, 0].set_yticklabels(["0", "450", "900"])
            # axs[1, 0].set_yticks([0, 450, 900])
            # axs[1, 0].set_yticklabels(["0", "450", "900"])
            # axs[0, 0].legend()
            # fig.suptitle("Loss functions and Prior distributions used by fully evolved GANs", fontsize="xx-large")

        if ch == "layers":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[1, j].set_xticks(np.arange(3/2, 12, 3))
                axs[1, j].set_xticklabels([str(i) for i in range(1, 11, 3)])
                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)
                axs[0, j].tick_params(axis='both', which='major', labelsize=20)
                axs[1, j].tick_params(axis='both', which='major', labelsize=20)

            # axs[0, 0].set_yticks([0, 450, 900])
            # axs[0, 0].set_yticklabels(["0", "450", "900"])
            # axs[1, 0].set_yticks([0, 250, 500])
            # axs[1, 0].set_yticklabels(["0", "250", "500"])

            # fig.suptitle("Number of layers present in fully evolved GANs", fontsize="xx-large")

        if ch == "neurons":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[1, j].set_xticks(np.arange(1/2, 52, 10))
                axs[1, j].set_xticklabels([str(i) for i in range(0, 52, 10)])

                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)
                axs[0, j].tick_params(axis='both', which='major', labelsize=15)
                axs[1, j].tick_params(axis='both', which='major', labelsize=15)

        if ch == "init":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[1, j].set_xticks(np.arange(1 / 2, 3, 1))
                axs[1, j].set_xticklabels(["X", "U", "N"])
                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)
                axs[0, j].tick_params(axis='both', which='major', labelsize=20)
                axs[1, j].tick_params(axis='both', which='major', labelsize=20)

            # axs[0, 0].set_yticks([0, 450, 900])
            # axs[0, 0].set_yticklabels(["0", "450", "900"])
            # axs[1, 0].set_yticks([0, 450, 900])
            # axs[1, 0].set_yticklabels(["0", "450", "900"])
            axs[1, 0].set_ylim((0, 4700))
            axs[0, 1].legend(loc=(-0.5, 0.5), prop={'size': 25})

        if ch == "loop":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[1, j].set_xticks(np.arange(3/2, 7, 1))
                axs[1, j].set_xticklabels([str(i) for i in range(1, 6, 1)])
                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)
                axs[0, j].tick_params(axis='both', which='major', labelsize=20)
                axs[1, j].tick_params(axis='both', which='major', labelsize=20)

        if ch == "act":
            for j in range(len(fs)):
                for i in [0, 1]:
                    axs[i, j].set_xticks([])
                axs[1, j].set_xticks(np.arange(1/2, 7, 1))
                axs[1, j].set_xticklabels(["I", "R", "E", "Sp", "Ss", "Sg", "T"])
                axs[1, j].set_xlabel("F" + str(fs[j]), size=20)
                axs[0, j].tick_params(axis='both', which='major', labelsize=15)
                axs[1, j].tick_params(axis='both', which='major', labelsize=15)

        axs[1, 0].set_ylabel("Generator", size=20)
        axs[0, 0].set_ylabel("Discriminator", size=20)
        plt.show()


def polar(gan, n_ind, char=("init",)):
    chars = {"layers": [2, 37], "loop": [3, 38], "neurons": [np.arange(4, 15), np.arange(39, 50)], "init": [np.arange(15, 26), np.arange(50, 61)], "act": [np.arange(26, 37), np.arange(61, 72)], "losslat": [0, 1]}
    ticks = {"layers": [range(1, 10), range(1, 10)], "loop": [3, 38], "neurons": [np.arange(4, 15), np.arange(39, 50)], "init": [["Xavier", "Uniform", "Normal"], ["Xavier", "Uniform", "Normal"]], "act": [["Ident.", "ReLU", "eLU", "Splus", "Ssign", "Sigm", "Tanh"], ["Ident.", "ReLU", "eLU", "Splus", "Ssign", "Sigm", "Tanh"]], "losslat": [["Uniform", "Normal"], div_ms]}
    fs = [7, 8]
    fjs = [5, 6]

    for ch in char:
        if ch == "losslat":
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.3, 1]})
        else:
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
        fig.set_size_inches(16, 6)
        plt.subplots_adjust(left=0, right=1, bottom=0.22, top=1)
        # Dims, Functions, Instances, Generations, Individuals, Variables
        data = []
        for fi, fj in enumerate(fs):

            for dimi, dim in enumerate(x_dims):
                f = gan[dimi, fjs[fi], :, 0, :n_ind, chars[ch][0]].flatten()
                l = gan[dimi, fjs[fi], :, -2, :n_ind, chars[ch][0]].flatten()
                gen = np.histogram(l, bins=range(int(ch is "layers"), np.unique(f).shape[0]+int(ch is "layers" or ch is "losslat")))

                data += [gen[0]]
                f = gan[dimi, fjs[fi], :, 0, :n_ind, chars[ch][1]].flatten()
                l = gan[dimi, fjs[fi], :, -2, :n_ind, chars[ch][1]].flatten()
                disc = np.histogram(l, bins=range(int(ch is "layers"), np.unique(f).shape[0]+int(ch is "layers" or ch is "losslat")))

                data += [disc[0]]

        data = np.array(data)
        data1 = np.concatenate(data[np.arange(0, len(data), 2)]).reshape((4, -1))
        data2 = np.concatenate(data[np.arange(1, len(data)+1, 2)]).reshape((4, -1))

        sns.heatmap(data1, annot=True, annot_kws={"size": 21}, fmt="d", ax=axs[0], cmap="viridis", cbar=False, xticklabels=ticks[ch][0], yticklabels=[])  # , vmin=min(np.min(data1), np.min(data2)), vmax=max(np.max(data1), np.max(data2)))
        sns.heatmap(data2, annot=True, annot_kws={"size": 21}, fmt="d", ax=axs[1], cmap="viridis", cbar=False, xticklabels=ticks[ch][1], yticklabels=[])  # , vmin=min(np.min(data1), np.min(data2)), vmax=max(np.max(data1), np.max(data2)))
        axs[0].tick_params(axis="y", length=0)
        axs[1].tick_params(axis="y", length=0)
        axs[0].set_ylim((0, 4))
        axs[1].set_ylim((0, 4))
        axs[1].set_yticks([0.5, 1.5, 2.5, 3.5])
        axs[0].set_yticks([0.5, 1.5, 2.5, 3.5])
        axs[0].yaxis.tick_right()
        axs[1].tick_params("y", labelrotation=0, labelsize=20)
        axs[0].tick_params("x", labelsize=20)
        axs[1].tick_params("x", labelsize=20)
        if ch == "losslat":
            axs[1].set_yticklabels(["F" + str(fs[0]) + "    \n$n=10$  ", "F" + str(fs[0]) + "    \n$n=784$ ", "F" + str(fs[1]) + "    \n$n=10$  ", "F" + str(fs[1]) + "    \n$n=784$ "])
        else:
            axs[1].set_yticklabels([" Disc. \n$n=10$  ", " Disc. \n$n=784$ ", "Gen.  \n$n=10$  ", "Gen.  \n$n=784$ "])
            axs[0].set_xlabel("F" + str(fs[0]), size=20)
            axs[1].set_xlabel("F" + str(fs[1]), size=20)
        # cmap = mpl.cm.viridis
        # norm = mpl.colors.Normalize(vmin=min(np.min(data1), np.min(data2)), vmax=max(np.max(data1), np.max(data2)))

        # cb1 = mpl.colorbar.ColorbarBase(axs[2], cmap=cmap, norm=norm, orientation='vertical', ticklocation="left")
        plt.show()


def activations(gan, n_ind):

    for dimi, dim in enumerate(x_dims):
        for fi, f, in enumerate([1, 8]):
            data_dsc = [[]]*10
            data_gen = [[]]*10
            for run in range(30):
                for ind in range(n_ind):
                    dsc = gan[dimi, fi, run, -2, ind, np.arange(26, 37)]
                    print(dsc)
                    dsc = dsc[dsc > -1][:-1]
                    gen = gan[dimi, fi, run, -2, ind, np.arange(61, 72)]
                    gen = gen[gen > -1][:-1]
                    adsc = np.argwhere(dsc > 0)
                    agen = np.argwhere(gen > 0)
                    print(gen.shape[0])
                    print(data_gen[gen.shape[0]])
                    data_dsc[dsc.shape[0]] = data_dsc[dsc.shape[0]] + [adsc[:, 0].tolist()]
                    data_gen[gen.shape[0]] = data_gen[gen.shape[0]] + [agen[:, 0].tolist()]

            for lays in range(1, 10):
                if len(data_dsc[lays]) > 0:
                    aux = [x if len(x) > 0 else [-1] for x in data_dsc[lays]]
                    aux = np.concatenate(aux)
                    sns.distplot(aux, bins=range(-1, 10), kde=False, label=str(lays), norm_hist=True)
            plt.legend(title="Disc. size")
            plt.title("Location of activation functions in discriminators for F" + str(f) + ", n=" + str(dim))
            plt.xlabel("Layer")
            plt.xticks(np.arange(-1/2, 10, 1), ["n/a"] + [str(x) for x in range(0, 10)])
            plt.savefig("DiscF" + str(f) + ", n" + str(dim) + ".pdf")
            plt.clf()

            for lays in range(1, 10):
                if len(data_gen[lays]) > 0:
                    aux = [x if len(x) > 0 else [-1] for x in data_gen[lays]]
                    aux = np.concatenate(aux)
                    sns.distplot(aux, bins=range(-1, 10), kde=False, label=str(lays), norm_hist=True)
            plt.legend(title="Gen. size")
            plt.title("Location of activation functions in discriminators for F" + str(f) + ", n=" + str(dim))
            plt.xlabel("Layer")
            plt.xticks(np.arange(-1/2, 10, 1), ["n/a"] + [str(x) for x in range(0, 10)])
            plt.savefig("GenF" + str(f) + ", n" + str(dim) + ".pdf")
            plt.clf()


def test_robustness(gen, objs, st, nd, inst):

    print("Inst", inst)
    for dimi, dim in enumerate(x_dims):
        print("\tDim: " + str(dim))
        for fi, f in enumerate(functions):
            print("\t\tF: " + str(f))
            results = np.zeros((2, 2, 1, 8, nd-st, 9))  # Rand/Best, dim, inst, func, reps, Metrics (9, orig, news: (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f))
            rand_gan = gen[dimi, fi, inst, 0, :]
            best_gan = gen[dimi, fi, inst, 1, :]

            results[0, dimi, 0, fi, 0, 0] = objs[dimi, fi, inst, 0, 1]
            results[1, dimi, 0, fi, 0, 0] = objs[dimi, fi, inst, 1, 1]

            for ii, i in enumerate(range(st, nd)):
                print("\t\t\tRep: " + str(i))
                set_seed(i)
                reset_graph(i)
                rand_res = igd_gd(rand_gan, dim, f)  # Returns (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f)
                best_res = igd_gd(best_gan, dim, f)
                results[0, dimi, 0, fi, ii, [1, 2]], results[0, dimi, 0, fi, ii, [3, 4]] = rand_res[0], rand_res[1]
                results[0, dimi, 0, fi, ii, [5, 6]], results[0, dimi, 0, fi, ii, [7, 8]] = rand_res[2], rand_res[3]
                results[1, dimi, 0, fi, ii, [1, 2]], results[1, dimi, 0, fi, ii, [3, 4]] = best_res[0], best_res[1]
                results[1, dimi, 0, fi, ii, [5, 6]], results[1, dimi, 0, fi, ii, [7, 8]] = best_res[2], best_res[3]

            np.save("Robustness_" + str(inst) + "_" + str(st) + "-" + str(nd) + "_" + str(dim) + "_" + str(f) + ".npy", results)
            del results


def unify_rob(st, nd):
    # Rand/Best, dim, inst, reps, Metrics (9, orig, news: (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f))
    results = np.zeros((2, 2, 29, 8, nd-st, 9))
    for inst in range(29):
        for dimi, dim in enumerate(x_dims):
            for fi, f in enumerate([1, 2, 3, 4, 5, 7, 8, 9]):
                res = np.load("RobNewLoss/Robustness_" + str(inst+1) + "_" + str(st) + "-" + str(nd) + "_" + str(dim) + "_" + str(f) + ".npy")
                results[:, dimi, inst, fi, :, :] = res[:, dimi, 0, fi, :, :]
    np.save("Robustness.npy", results)


def robust_plot(metrics=("IGD_o",)):
    ms = {"IGD_o": 1, "GD_o": 2, "IGDp_o": 3, "GDp_o": 4, "IGD_f": 5, "GD_f": 6, "IGDp_f": 7, "GDp_f": 8}

    results = np.log(np.load("Robustness.npy"))
    # res1 = np.random.normal(loc=0.5, size=(1, 2, 30, 8, 5, 9))
    # res2 = np.random.normal(loc=0.8, size=(1, 2, 30, 8, 5, 9))
    # results = np.concatenate((res1, res2), axis=0)
    fig, axs = plt.subplots(len(functions), len(metrics)*len(x_dims), sharey="row")

    # fig.suptitle("Robustness", fontsize="xx-large")
    plt.subplots_adjust(hspace=0, wspace=0)

    limits = [3, 50]
    ranges = [0.15, 2]
    x_lims = [-7.5, -5.5]
    # ylims = [(0, 340), (0, 330), (0, 194), (0, 249), (0, 194), (0, 120), (0, 194), (0, 194)]

    for dimi, dim in enumerate(x_dims):
        for fi, f in enumerate(functions):
            for smi, sm in enumerate(metrics):
                m = ms[sm]
                # results[:, dimi, :, fi, :, m][results[:, dimi, :, fi, :, m] > limits[m-1]] = limits[m-1]
                print(results[0, dimi, :, fi, :, m].flatten())
                sns.distplot(results[0, dimi, :, fi, :, m].flatten(), kde=False, label="Random", ax=axs[fi, dimi*len(metrics)+smi], bins=np.arange(x_lims[dimi], limits[m-1]+ranges[m-1], ranges[m-1]))
                sns.distplot(results[1, dimi, :, fi, :, m].flatten(), kde=False, label="Best GAN", ax=axs[fi, dimi*len(metrics)+smi], bins=np.arange(x_lims[dimi], limits[m-1]+ranges[m-1], ranges[m-1]), hist_kws={"histtype": 'step', "stacked": True, "fill": False, "linewidth": 2})

                # print(wilcoxon(results[0, dimi, :, fi, :, m].flatten(), results[1, dimi, :, fi, :, m].flatten()))

    for i, ax in enumerate(axs[:, 1]):
        # ax.yaxis.tick_right()
        axs[i, 0].set_ylabel("F" + str(functions[i]), rotation=0, position=(0, 0.2), size="large", horizontalalignment="right")

        # print(list(axs[i, 0].yaxis.get_majorticklabels()))
        # if i%2==0:
        #     plt.setp(axs[i, 0].yaxis.get_majorticklabels(), rotation=0, va="bottom")
    #axs[4, 0].set_ylim((0, 190))
    axs[5, 0].set_ylim((0, 197))
    axs[7, 0].set_ylim((0, 97))
    axs[6, 0].set_ylim((0, 97))
    fig.align_ylabels(axs[:, 0])
    axs[7, 0].set_xlabel(r"$n=10$", size="large")
    axs[7, 1].set_xlabel(r"$n=784$", size="large")
    plt.text(-7.5, -100, "$log($IGD$)$", size=15)
    axs[6, 1].legend(loc=(-0, 0))

    plt.show()


def igd_vs_gd():

    #  Rand/Best, dim, inst, reps, Metrics (9, orig, news: (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f))
    results = np.load("Robustness.npy")

    markers = [".", "*", "p", "P", "+", "_", "1", "^"]
    legend1_elements = [Line2D([0], [0], marker='.', color='w', label='Random GAN', markerfacecolor='black', markersize=15), Line2D([0], [0], marker='*', color='w', label='Best GAN', markerfacecolor='black', markersize=10)]
    legend2_elements = []

    for fi, f in enumerate(functions):
        legend2_elements += [Line2D([0], [0], marker='s', color='w', label='F' + str(f), markerfacecolor=np.array(sns.color_palette()[-fi]), markersize=15)]

    for fi, f in enumerate(functions):
        for si, s in enumerate(["Solution", "Feature"]):
            fig, axs = plt.subplots(2, 2, sharex="all", sharey="all")
            for mi, m in enumerate(["Regular", "Modified"]):
                for dimi, dim in enumerate(x_dims):
                    sns.scatterplot(x=results[0, dimi, :, fi, :, si*4+mi*2+1].flatten(), y=results[0, dimi, :, fi, :, si*4+mi*2+2].flatten(), ax=axs[mi, dimi], marker=markers[0], c=np.reshape(np.array(sns.color_palette()[-fi]), (1, -1)), s=200)
                    sns.scatterplot(x=results[1, dimi, :, fi, :, si*4+mi*2+1].flatten(), y=results[1, dimi, :, fi, :, si*4+mi*2+2].flatten(), ax=axs[mi, dimi], marker=markers[1], c=np.reshape(np.array(sns.color_palette()[-fi]), (1, -1)), s=200)
            axs[1, 0].set_xlabel(r"$n=10$", size="large")
            axs[1, 1].set_xlabel(r"$n=784$", size="large")
            axs[0, 0].set_ylabel(r"$IGD vs GD$", size="large")
            axs[1, 0].set_ylabel(r"$IGD_p vs GD_p$", size="large")
            axs[0, 0].legend(handles=legend1_elements)
            axs[1, 0].legend(handles=legend2_elements)
            fig.suptitle(s + " Space")

        plt.show()


def scat3d(f=1):

    #  Rand/Best, dim, inst, func, reps, Metrics (9, orig, news: (IGD_o, GD_o), (IGDp_o, GDp_o), (IGD_f, GD_f), (IGDp_f, GDp_f))
    results = np.load("Robustness.npy")
    fig = plt.figure()
    ax = Axes3D(fig, rect=(-0.1, 0.04, 1.06, 0.95))

    ax.scatter(results[1, 0, :, f, :, 1], results[1, 0, :, f, :, 2], results[1, 0, :, f, :, 5], label="Best GAN")
    ax.scatter(results[0, 0, :, f, :, 1], results[0, 0, :, f, :, 2], results[0, 0, :, f, :, 5], label="Random GAN")
    ax.set_xlabel("\n$IGD_o$", size=20)

    ax.set_ylabel("\n$GD_o$", size=20)
    ax.set_zlabel("\n$IGD_x$", size=20)
    ax.legend(loc=(0.33, 0.15), prop={'size': 17})
    ax.set_title("F" + str(functions[f]), size=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    lims = [(3, 100, 2), (9, 300, 2), (6, 200, 1.5), (7, 200, 2), (5, 150, 2), (12, 400, 2), (28, 800, 2), (12, 150, 2)]
    steps = [20, 50, 50, 50, 30, 100, 200, 30]

    ax.set_zticks(np.arange(0, lims[f][2] + 0.1, 0.5))
    ax.set_zticklabels([str(i) for i in np.arange(0, lims[f][2] + 0.1, 0.5)])
    ax.set_yticks(np.arange(0, lims[f][1] + 1, steps[f]))
    ax.set_yticklabels([str(i) for i in np.arange(0, lims[f][1] + 1, steps[f])])

    ax.set_xlim((0, lims[f][0]))
    ax.set_ylim((0, lims[f][1]))
    ax.set_zlim((0, lims[f][2]))

    # ax.scatter(results[1, 0, :, 6, :, 0], results[1, 0, :, 6, :, 2], results[1, 0, :, 6, :, 5])
    # ax.scatter(results[0, 0, :, 6, :, 0], results[0, 0, :, 6, :, 2], results[0, 0, :, 6, :, 5])
    plt.show()


def reduce_gens(n_gen, n_ind, root="", force=False):

    if root == "":
        root = str(n_gen) + "Gen" + str(n_ind) + "Ind"
    path = "ReducedGenerations.npy"
    if os.path.isfile(path) and not force:
        res = np.load(path)
        return res[:, :, :, :, :-3].astype("int"), res[:, :, :, :, -3:]

    res = np.load(root + "/Gens.npy")
    gen, obj = res[:, :, :, :, :, :-3].astype("int"), res[:, :, :, :, :, -3:]

    reduced_gen = np.zeros((2, 8, 30, 2, 72))  # Dims, Function, Instance, Best/Rnd, Values
    reduced_obj = np.zeros((2, 8, 30, 2, 3))

    gen = gen[:, :, :, :, :n_ind, :]
    obj = obj[:, :, :, :, :n_ind, :]

    for dimi, dim in enumerate(x_dims):
        for fi, f in enumerate(functions):
            for inst in range(30):

                rand_ind = np.random.randint(n_ind)
                reduced_gen[dimi, fi, inst, 0, :] = gen[dimi, fi, inst, 0, rand_ind, :]
                reduced_obj[dimi, fi, inst, 0, :] = obj[dimi, fi, inst, 0, rand_ind, :]

                best = np.argmin(obj[dimi, fi, inst, :-1, :, 1])
                best_gen, best_ind = best // n_ind, best % n_ind
                reduced_gen[dimi, fi, inst, 1, :] = gen[dimi, fi, inst, best_gen, best_ind, :]
                reduced_obj[dimi, fi, inst, 1, :] = obj[dimi, fi, inst, best_gen, best_ind, :]

    tot = np.concatenate((reduced_gen, reduced_obj), axis=4)

    np.save(path, tot)

    return reduced_gen.astype("int"), reduced_obj


def fn_transferability(reduced, st, nd, inst):

    for dimi, dim in enumerate(x_dims):
        print("Dim:", dim)
        for ofi, of in enumerate(functions):
            print("\tOrigin Function:", of)
            results = np.zeros((1, 8, nd-st, 8))  # Dim, Origin Function, Instance, Target Function, Repetition, Metric
            gan = reduced[dimi, ofi, inst, 1, :]  # Dims, Function, Instance, Best/Rnd, Values

            for tfi, tarf in enumerate(functions):
                print("\t\t\tTarget Function:", tarf)
                for rep in range(st, nd):
                    print("\t\t\t\tRepetition:", rep)
                    set_seed(rep)
                    reset_graph(inst)
                    res = igd_gd(gan, dim, tarf)
                    results[0, tfi, rep, :] = np.array(res).flatten()

            np.save("Function_trans_" + str(inst) + "_" + str(st) + "-" + str(nd) + "_" + str(dim) + "_" + str(of) + ".npy", results)


def sz_transferability(reduced, st, nd, inst):

    for ofi, of in enumerate(functions):
        print("\tOrigin Function:", of)
        results = np.zeros((1, 8, nd-st, 8))  # Origin Function, Instance, Target Function, Repetition, Metric
        gan = reduced[0, ofi, inst, 1, :]  # Dims, Function, Instance, Best/Rnd, Values
        for tfi, tarf in enumerate(functions):
            print("\t\t\tTarget Function:", tarf)
            for rep in range(st, nd):
                print("\t\t\t\tRepetition:", rep)
                set_seed(rep)
                reset_graph(inst)
                res = igd_gd(gan, 784, tarf)
                results[0, tfi, rep, :] = np.array(res).flatten()

        np.save("Size_trans_" + str(inst) + "_" + str(st) + "-" + str(nd) + "_" + str(of) + ".npy", results)
        del results


def unify_fn(st, nd):
    # dim, inst, origin function, target function, repetition, metric
    results = np.zeros((2, 30, 8, 8, nd-st, 8))
    for inst in range(1, 30):
        for dimi, dim in enumerate(x_dims):
            for fi, f in enumerate(functions):
                path = "Trans_new_loss/Function_trans_" + str(inst) + "_" + str(st) + "-" + str(nd) + "_" + str(dim) + "_" + str(f) + ".npy"
                if not os.path.isfile(path):
                    print(path)
                    continue
                res = np.load(path)
                results[dimi, inst, fi, :, :, :] = res
    np.save("Func_trans.npy", results)


def unify_sz(st, nd):
    # inst, origin function, target function, repetition, metric
    results = np.zeros((30, 8, 8, nd-st, 8))
    for inst in range(30):
        for fi, f in enumerate(functions):
            path = "Size_new_loss/Size_trans_" + str(inst) + "_" + str(st) + "-" + str(nd) + "_" + str(f) + ".npy"
            if not os.path.isfile(path):
                print(path)
                continue
            res = np.load(path)

            results[inst, fi, :, :, :] = res
    np.save("Size_trans.npy", results)


def scatter_heat(data, metric, title=""):  # Origin Function, Instance, Target Function, Repetition, Metric

    metrics = {"IGD_o": 0, "GD_o": 1, "IGDp_o": 2, "GDp_o": 3, "IGD_f": 4, "GD_f": 5, "IGDp_f": 6, "GDp_f": 7}

    data = data[:, :, :, :, metrics[metric]]

    tests = np.ones((8, 8))
    for io, o in enumerate(functions):
        for it, t in enumerate(functions):
            if not io == it:
                tests[io, it] = wilcoxon(data[:, io, it, :].flatten(), data[:, io, io, :].flatten())[1] if np.mean(data[:, io, it, :]) < np.mean(data[:, io, io, :]) else 1

    grid_x = np.array([[i]*data.shape[1] for i in range(data.shape[2])]).flatten()
    grid_y = np.array([[i]*data.shape[2] for i in range(data.shape[1])]).transpose().flatten()

    means = np.mean(data, axis=(0, 3))
    variances = np.var(data, axis=(0, 3))
    # mean = np.mean(np.log(variances))

    f, ax = plt.subplots()

    jj = ax.scatter(x=grid_x, y=grid_y, c=np.log(variances.flatten()), s=means.flatten()*350+480, vmin=-5.5, vmax=3.5)
    for io, o in enumerate(functions):
        for it, t in enumerate(functions):
            if not io == it:
                tests[io, it] = wilcoxon(data[:, io, it, :].flatten(), data[:, io, io, :].flatten())[1] if means[io, it] < means[it, it] else 1
                if tests[io, it] < 1:
                    ax.add_artist(plt.Circle((io, it), 0.4, color='black', fill=False))

    plt.subplots_adjust(hspace=0, wspace=0, left=0.08, right=1, bottom=0.1, top=0.94)
    plt.title(title)
    for x, y in zip(grid_x, grid_y):
        print(np.log(variances[x, y]))
        plt.text(s="%.2f" % means[x, y], x=x, y=y, horizontalalignment='center', verticalalignment='center', color="yellow" if np.log(variances[x, y]) < 1 else "b")  # 2.5
    plt.xticks(range(8), ["F" + str(i) for i in functions])
    plt.yticks([])
    plt.yticks(range(8), ["F" + str(i) for i in functions])
    plt.xlabel("Origin Function")
    plt.ylabel("Target Function")
    f.colorbar(mappable=jj, label=r"$log(var)$ values")

    plt.show()


def f5_example():
    f = 1
    for i in range(1, 31):
        data = np.load("100Gen50Ind/GAN_Evals_" + str(i) + "_F" + str(f) + "_N_50_ngen_100_Sel_0_X_10.npy")
        # rnd = data[0, :-3].astype("int")
        bst = np.argmin(data[:, -2])
        best = data[bst, :-3].astype("int")
        # g = np.argmin(data[:500, -2])
        # g10 = data[g, :-3].astype("int")

        mop_f = FFunctions(10, "F" + str(5))
        ps_all_x = mop_f.generate_ps_samples(1000)

        pf1, pf2 = mop_f.evaluate_mop_function(ps_all_x)
        """
        gan_descriptor = decodify_descriptor(rnd, 10)
        reset_graph(0)
        gan = GAN(gan_descriptor)
        gan.training_definition()
        samples = gan.separated_running("fixed", ps_all_x, 150, 1001, 1000, 1001)

        nf1, nf2 = mop_f.evaluate_mop_function(samples)
        nf1, nf2, _ = pareto_frontier(nf1, nf2)
        plt.plot(pf1, pf2, "b.", label="Original PF")
        plt.plot(nf1, nf2, "r.", label="PF approximation")
        plt.title("Random")
        #plt.text(0, 0, s="IGD:" + str(igd((np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose()))))
        plt.show()
        plt.clf()
        gan_descriptor = decodify_descriptor(g10, 10)
        reset_graph(g)
        gan = GAN(gan_descriptor)
        gan.training_definition()
        samples = gan.separated_running("fixed", ps_all_x, 150, 1001, 1000, 1001)

        nf1, nf2 = mop_f.evaluate_mop_function(samples)
        nf1, nf2, _ = pareto_frontier(nf1, nf2)
        plt.plot(pf1, pf2, "b.", label="Original PF")
        plt.plot(nf1, nf2, "r.", label="PF approximation")
        plt.title("G10")
        #plt.text(0, 0, s="IGD:" + str(igd((np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose()))))
        plt.show()
        plt.clf()
        """
        gan_descriptor = decodify_descriptor(best, 10)

        reset_graph(bst[0])
        gan = GAN(gan_descriptor)
        gan.training_definition()
        samples = gan.separated_running("fixed", ps_all_x, 150, 1001, 1000, 1001)

        nf1, nf2 = mop_f.evaluate_mop_function(samples)
        nf1, nf2, _ = pareto_frontier(nf1, nf2)
        plt.plot(pf1, pf2, "b.", label="Original PF")
        plt.plot(nf1, nf2, "r.", label="PF approximation")
        plt.title("Best")
        plt.text(0, 0, s="IGD:" + str(igd((np.vstack((nf1, nf2)).transpose(), np.vstack((pf1, pf2)).transpose()))))
        plt.show()


def three_comps():
    mop_f = FFunctions(3, "F" + str(2))
    ps_all_x = mop_f.generate_ps_samples(10000)
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(ps_all_x[:, 0], ps_all_x[:, 1], ps_all_x[:, 2])
    ax.tick_params(labelsize=15)
    ax.set_xlabel("$x_1$", fontsize=20)
    ax.set_ylabel("$x_2$", fontsize=20)
    ax.set_zlabel("$x_3$", fontsize=20)
    plt.show()


def rand_sols():
    mop_f = FFunctions(10, "F" + str(5))
    ps_all_x = mop_f.generate_ps_samples(1000)
    rand = ps_all_x

    pf1, pf2 = mop_f.evaluate_mop_function(ps_all_x)
    rand[:, 1:] = np.random.uniform(0, 1, (1000, 9))
    r1, r2 = mop_f.evaluate_mop_function(rand)
    plt.scatter(pf1, pf2, label="PS solutions")
    plt.scatter(r1, r2, label="Random solutions")
    plt.xlabel("$F5_1$", fontsize=30)
    plt.ylabel("$F5_2$", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(prop={'size': 15})
    plt.show()


if __name__ == "__main__":
    #three_comps()
    #test()
    #rand_sols()
    parser = argparse.ArgumentParser()
    #f5_example()
    parser.add_argument('integers', metavar='int', type=int, choices=range(3000), nargs=3, help='ur momma')
    args = parser.parse_args()
    # Dims, Functions, Instances, Generations, Individuals, Variables
    gans, goals = load_info(force=False, n_gen=101, n_ind=100)
    #activations(gans, n_ind=100)
    #boxplots(goals, n_ind=50)
    #["layers", "loop", "neurons", "init", "act", "losslat"]
    char_per_func(gans, n_ind=100)
    #polar(gans, n_ind=100)
    #gans, goals = reduce_gens(n_gen=100, n_ind=50, force=False, root="NewLossRes")

    # igd_gd_corr()

    #first_last(gans)
    start = args.integers[0]
    end = args.integers[1]
    instance = args.integers[2]
    #unify_rob(0, 15)
    #unify_fn(0, 15)
    #unify_sz(0, 15)
    #fn_transferability(gans, start, end, instance)
    #sz_transferability(gans, start, end, instance)
    #whole_data = np.load("Func_trans.npy")

    #scatter_heat(whole_data[0], "IGD_o", r"Transferability in $n=10$, with $IGD_o$")
    #scatter_heat(whole_data[1], "IGD_o", r"Transferability in $n=784$, with $IGD_o$")
    #whole_data = np.load("Size_trans.npy")
    #scat3d()
    #scatter_heat(whole_data, "IGD_o", r"Transferability from $n=10$ to $n=784$, with $IGD_o$")

    #test_robustness(gans, goals, start, end, instance)
    #robust_plot()
    #igd_vs_gd()


"""
f = 1
inst = 3
x_dim = 10
nets, objs = read_file("results/GAN_Evals_" + str(inst) + "_F" + str(f) + "_N_20_ngen_500_Sel_0_X_" + str(x_dim) + ".txt.npy")
nets = nets.astype("int")
new_x_dim = 784
gan_desc = decodify_descriptor(nets[0], new_x_dim)
test(f, new_x_dim, gan_desc)
"""
