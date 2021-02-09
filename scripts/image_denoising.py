import json
import os
from os.path import join, pardir
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from ot_sparse_projection import ot_sparse_projection, l2, misc, noise as ns, adaptive_thresholding
from ot_sparse_projection.dictionaries import get_filter_handler

# This script reproduces the whole denoising experiment in the paper and produces the images and tables

img_folder = join(pardir, 'img', 'denoising')

if not os.path.exists(img_folder):
    os.makedirs(img_folder)
filter_type = 'db2'
gamma = .1
n = 256
lambdas = [.25, .5, .65, .8, 1, 1.5, 2, 3, 4.5, 6]

imName = 'racoon'
W_HARD = 'OT hard'
W_SOFT = 'OT soft'
HARD = 'hard'
SOFT = 'soft'
RECONS = 'recons'
SPARSITIES = 'sparsities'

NEW_THRESH = "newThresh"

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

adaptive_values = {"normalShrink": [adaptive_thresholding.NormalShrink, 'd', colors[4]],
                   "sureShrink": [adaptive_thresholding.SureShrink, 'o', colors[5]],
                   "bayesShrink": [adaptive_thresholding.BayesShrink, 'p', colors[6]],
                   "visuShrink": [adaptive_thresholding.VisuShrink, '*', 'r']}

plot_values = {W_SOFT: ['-x', 'OT soft thresholding', colors[3]], W_HARD: ['-o', 'OT hard thresholding', colors[1]],
               SOFT: ['-d', 'Euclidean soft thresholding', colors[0]],
               HARD: ['-s', 'Euclidean hard thresholding', colors[2]]}

original, scaling = misc.get_image(imName, n)

filter_handler = get_filter_handler(original, filter_type)

mask = filter_handler.inverse_dot(original)
mask = np.ones_like(mask)
mask = filter_handler.array_to_coeffs(mask)
mask[0] = np.zeros_like(mask[0])
mask = filter_handler.coeffs_to_array(mask)

noises = {ns.SALT_PEPPER: [.05,
                           .1, .15], ns.GAUSSIAN: [.2, .3, .4
                                                   ]}
noise_names = {ns.SALT_PEPPER: "Salt \\& pepper", ns.GAUSSIAN: "Gaussian"}

SSIM = "SSIM"
PSNR = "pSNR"
similarities = {SSIM: ns.ssim, PSNR: ns.psnr}


def initialize_method_outputs(*args):
    values = {}
    for key in args:
        v = {RECONS: [], SPARSITIES: []}
        for sim in similarities.keys():
            v[sim] = []
        values[key] = v
    return values


def add_values(method_outputs, key, Z):
    values = method_outputs[key]
    recons = filter_handler.dot(filter_handler.reshape_coeffs(Z)).reshape(im.shape)
    recons = recons + shift
    values[RECONS].append(recons)
    values[SPARSITIES].append(misc.get_sparsity(Z))
    for sim, fun in similarities.items():
        values[sim].append(fun(original, recons, scaling=scaling))


table_scores = {}

for noise_type, values in noises.items():
    folder = join(img_folder, noise_type)

    if not os.path.exists(folder):
        os.makedirs(folder)

    misc.save_image(original * scaling, '{}/{}.png'.format(folder, imName))
    table_scores[noise_type] = {}
    for noise in values:
        table_scores[noise_type][noise] = {}
        noisyImName = join(folder, 'denoising_{}_{}_noise_{}'.format(noise_type, imName, noise))
        # load image
        try:
            im, scaling2 = misc.get_image("{}.png".format(noisyImName), n)
            im *= scaling2 / scaling
        except IOError as e:
            # the noisy image doesn't exist, create and save it
            im = ns.add_noise(original, noise_type, noise, max_val=255 / scaling)
            misc.save_image(im * scaling, "{}.png".format(noisyImName))
            im, scaling2 = misc.get_image("{}.png".format(noisyImName), n)
            im *= scaling2 / scaling
        shift = im.ravel().min() - 1e-10
        if shift > 0:
            shift = 0

        sparsities = []
        method_outputs = initialize_method_outputs(W_SOFT, W_HARD, SOFT, HARD)

        # compute sparsity paths
        for lamb in lambdas:
            _, Z, obj = ot_sparse_projection. \
                wasserstein_image_filtering_invertible_dictionary(im - shift, filter_handler, gamma, lamb * mask)
            sparsity_pattern = np.not_equal(0, Z)
            _, Z_wasserstein_hard, obj_hard = ot_sparse_projection. \
                OtFilteringSpecificPattern(filter_handler, gamma, sparsity_pattern, ).projection(im - shift)
            add_values(method_outputs, W_SOFT, Z)
            add_values(method_outputs, W_HARD, Z_wasserstein_hard)
            sparsity = misc.get_sparsity(Z)
            sparsities.append(sparsity)
            Y_l2, Z_l2 = l2.sparse_projection(im, filter_handler, sparsity)
            Y_l2_hard, Z_l2_hard = l2.hard_thresholding(im, filter_handler, sparsity)
            add_values(method_outputs, HARD, Z_l2_hard)
            add_values(method_outputs, SOFT, Z_l2)

        # plot scores
        for key, sim in similarities.items():
            table_scores[noise_type][noise][key] = {}
            plt.figure(figsize=[9., 3.])
            plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
            plt.xlabel("Sparsity")
            plt.ylabel(key)
            min_val = 100000
            max_val = 0
            for method in plot_values.keys():
                values = method_outputs[method]
                plot_v = plot_values[method]
                score = np.array(values[key])
                best = score.argmax()
                table_scores[noise_type][noise][key][method] = score[best]
                misc.save_image(values[RECONS][best] * scaling, '{}_{}_{}_{}.png'.format(noisyImName, filter_type,
                                                                                         key, method))
                min_val = min(min_val, score.min())
                max_val = max(max_val, score.max())
                plt.plot(1 - np.array(values[SPARSITIES]), score, plot_v[0], color=plot_v[2], label=plot_v[1])

            for method, values in adaptive_values.items():
                shrinker = values[0](im, filter_handler)
                Y_adaptive, Z_adaptive = shrinker.denoise()
                score = sim(original, Y_adaptive, scaling=scaling)
                min_val = min(min_val, score)
                max_val = max(max_val, score)
                table_scores[noise_type][noise][key][method] = score
                sparsity_adaptive = 1 - misc.get_sparsity(Z_adaptive)
                plt.plot(sparsity_adaptive, score, values[1], color=values[2], label=method)
                misc.save_image(Y_adaptive * scaling, '{}_{}_{}.png'.format(noisyImName, filter_type,
                                                                            method))

            score = -10
            for alpha in np.logspace(-4, 4, 15):
                shrinker = adaptive_thresholding.NewThresh(im, filter_handler, alpha=alpha)
                Y_adaptive_new, Z_adaptive_new = shrinker.denoise()
                score_new = sim(original, Y_adaptive, scaling=scaling)
                sparsity_adaptive_new = 1 - misc.get_sparsity(Z_adaptive)
                print("{}: {} - {}".format(alpha, sparsity_adaptive_new, score_new))
                if score_new > score:
                    score = score_new

                    min_val = min(min_val, score)
                    max_val = max(max_val, score)
                    Y_adaptive = Y_adaptive_new
                    Z_adaptive = Z_adaptive_new
                    sparsity_adaptive = sparsity_adaptive_new
                    ALPHA = alpha

            method = NEW_THRESH + ", $\\alpha={:.1e}$".format(Decimal(ALPHA))
            plt.plot(sparsity_adaptive, score, 's', label=method, color=colors[7])
            method = NEW_THRESH
            table_scores[noise_type][noise][key][method] = score
            misc.save_image(Y_adaptive * scaling, '{}_{}_{}_{}.png'.format(noisyImName, filter_type, key,
                                                                           method))

            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.xlim([0, 1.02])

            margin = .15 * max_val

            plt.ylim([min_val - margin, max_val + margin])
            figname = "{}_{}_{}.eps".format(noisyImName, filter_type, key)
            print(figname)
            plt.savefig(figname, transparent=True)
            plt.close()


with open(join(img_folder, "{}_{}_table.json".format(imName, filter_type)), "w") as fp:
    json.dump(table_scores, fp)

# Done with computations, now print the latex table
with open(join(img_folder, "{}_{}_table.json".format(imName, filter_type)), "r") as fp:
    table_scores = json.load(fp)

LATEX = 'latex'
INIT = 'init'

ALL_METHODS = []
for method in adaptive_values.keys():
    ALL_METHODS.append(method)
ALL_METHODS.append(NEW_THRESH)
ALL_METHODS.append(W_HARD)
ALL_METHODS.append(W_SOFT)


def init_latex():
    col_align = "|c|c||"
    methods = "Noise&$\\sigma$\t"
    for method in ALL_METHODS:
        col_align = col_align + "c|"
        methods = methods + "&\t" + method
    table = "\\begin{{subtable}}[h]{{\\textwidth}}\n" \
            "\\begin{{tabular}}{{{}}}\n\\hline\n{}"
    return table.format(col_align, methods)


tables_latex = {SSIM: init_latex(), PSNR: init_latex()}

new_line = "\\\\\n"

for noise_type, values in noises.items():
    line_start = new_line + "\\hline\\multirow{{{}}}{{*}}{{{}}}".format(len(values), noise_names[noise_type])
    for noise in values:
        if noise_type == ns.SALT_PEPPER:
            noise_latex = "{}\\%".format(int(100 * noise))
        else:
            noise_latex = round(noise, 2)
        line_start = "{}&${}$".format(line_start, noise_latex)
        for key in similarities.keys():
            current_latex = tables_latex[key]
            current_latex = current_latex + line_start
            best_score = -1
            for method in ALL_METHODS:
                score = table_scores[noise_type][str(noise)][key][method]
                if score > best_score:
                    best_score = score
            for method in ALL_METHODS:
                score = table_scores[noise_type][str(noise)][key][method]
                format_string = "{:0.3f}".format(score)
                if score == best_score:
                    format_string = "\\mathbf{{{}}}".format(format_string)
                else:
                    format_string = "{}".format(format_string)
                current_latex = "{}&${}$".format(current_latex, format_string)
            tables_latex[key] = current_latex

        line_start = "{}\\cline{{2-{}}}".format(new_line, 2 + len(ALL_METHODS))

for key in tables_latex.keys():
    latex = tables_latex[key]
    latex = latex + "\\\\\\hline\n\\end{{tabular}}\n\\caption{{{}}}\n\\end{{subtable}}".format(key)
    tables_latex[key] = latex
    with open(join(img_folder,"{}_{}_{}_table.tex".format(imName, filter_type, key)), "w") as fp:
        fp.write(latex)

with open(join(img_folder,"{}_{}_table.tex".format(imName, filter_type)), "w") as fp:
    fp.write("\\begin{table*}\n")

    for key in tables_latex.keys():
        fp.write(tables_latex[key])
        fp.write("\n\n")
    fp.write("\\caption{Denoising scores for different wavelet thresholding methods}\n" +
             "\\label{tab:denoising}")
    fp.write("\\end{table*}")
