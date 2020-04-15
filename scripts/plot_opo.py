#!/usr/bin/env python
import argparse
import logging
import math
import os
import sys
from subprocess import DEVNULL, run

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import directed_hausdorff

matplotlib.rcParams["text.usetex"] = True

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.7)


# Configure logging for Hack
logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _match(gain_list, current_list):
    """Match gains and currents based on their reading-order appearence."""
    cols = ["Document", "Supply Current (uA)", "GBWP (kHz)"]
    matched = []

    # Walk down each list, making pairs of gains and currents
    i = 0
    j = 0
    while i < len(gain_list) and j < len(current_list):
        if gain_list[i][0] < current_list[j][0]:
            i += 1
        elif gain_list[i][0] > current_list[j][0]:
            j += 1
        else:
            # If the docs are the same
            matched.append([gain_list[i][0], current_list[j][1], gain_list[i][1]])
            j += 1
            i += 1

    return pd.DataFrame(matched, columns=cols)


def _plot(infile, gainfile, currentfile, outfile, scale, gb, cb):
    """Plotting logic."""
    dirname = os.path.dirname(__file__)
    # Mfr PartNumber,GBWP (kHz),Supply Current (uA),GBWP/uA,min_voltage,max_voltage
    opo_data = pd.read_csv(infile, skipinitialspace=True)

    gain_data = pd.read_csv(gainfile, skipinitialspace=True)
    current_data = pd.read_csv(currentfile, skipinitialspace=True)

    # Process our data based on two values of b
    gain_b = gb
    current_b = cb
    logger.info(f"gain_b = {gain_b}, current_b = {current_b}")

    filtered_gain = gain_data[gain_data.p >= gain_b][["Document", "GBWP (kHz)", "sent"]]
    filtered_gain = filtered_gain.sort_values(by=["Document", "sent", "GBWP (kHz)"])
    filtered_current = current_data[current_data.p >= current_b][
        ["Document", "Supply Current (uA)", "sent"]
    ]
    filtered_current = filtered_current.sort_values(
        by=["Document", "sent", "Supply Current (uA)"]
    )

    # Rather than a naive cross-product, which would be far too messy, we
    # instead match up values based on their reading-order appearance in the
    # datasheet.
    filtered_gain_list = filtered_gain.values.tolist()
    filtered_current_list = filtered_current.values.tolist()
    matched = _match(filtered_gain_list, filtered_current_list)

    matched.to_csv(os.path.join(dirname, "matched.csv"), sep=",")
    logger.info(f"len(matched): {len(matched)}")

    logger.info("Plotting from {}...".format(infile))

    opo_view = opo_data[["Supply Current (uA)", "GBWP (kHz)"]].copy()
    our_view = matched[["Supply Current (uA)", "GBWP (kHz)"]].copy()
    opo_view["Source"] = "Digi-Key"
    our_view["Source"] = "Our Approach"
    total = pd.concat([opo_view, our_view])

    # Calculate the number of exact matches
    opo_nd = opo_view[["Supply Current (uA)", "GBWP (kHz)"]].values
    our_nd = our_view[["Supply Current (uA)", "GBWP (kHz)"]].values
    logger.info(
        f"Total opo approach: {len(opo_nd)} ({len(np.unique(opo_nd, axis=0))} unique)"
    )
    logger.info(
        f"Total our approach: {len(our_nd)} ({len(np.unique(our_nd, axis=0))} unique)"
    )

    unique_our = np.unique(our_nd, axis=0)

    pred = set(map(tuple, unique_our))
    gold = set(map(tuple, np.unique(opo_nd, axis=0)))
    TP_set = pred.intersection(gold)
    FP_set = pred.difference(gold)
    FN_set = gold.difference(pred)

    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / (TP + FP) if TP + FP > 0 else float("nan")
    rec = TP / (TP + FN) if TP + FN > 0 else float("nan")
    f1 = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float("nan")
    logger.info(f"prec: {prec} rec: {rec} f1: {f1}")

    #  total_count = 0
    #  unique_count = 0
    #  for i in unique_our:
    #      seen = set()
    #      for j in opo_nd:
    #          j_tup = tuple(j)
    #          if np.array_equal(i, j):
    #              total_count += 1
    #
    #              if j_tup not in seen:
    #                  unique_count += 1
    #
    #          seen.add(j_tup)
    #
    #  logger.info(f"total exact matches: {total_count} ({unique_count} unique)")

    # Calculate the Hausdorff Distance on the normalized plots.
    #
    # Data points are normalized by first taking the log10 of all points to
    # match shapes with the log10-based plot, subtracting the min of each
    # dimension, and dividing by the max of each dimension to bring all points
    # to [0, 1].
    opo_nd = np.log10(opo_view[["Supply Current (uA)", "GBWP (kHz)"]].values)
    our_nd = np.log10(our_view[["Supply Current (uA)", "GBWP (kHz)"]].values)

    c_min = min(opo_nd.min(axis=0)[0], our_nd.min(axis=0)[0])
    gbw_min = min(opo_nd.min(axis=0)[1], our_nd.min(axis=0)[1])
    opo_nd[:, 0] -= c_min
    our_nd[:, 0] -= c_min
    opo_nd[:, 1] -= gbw_min
    our_nd[:, 1] -= gbw_min

    c_max = max(opo_nd.max(axis=0)[0], our_nd.max(axis=0)[0])
    gbw_max = max(opo_nd.max(axis=0)[1], our_nd.max(axis=0)[1])
    opo_nd[:, 0] /= c_max
    our_nd[:, 0] /= c_max
    opo_nd[:, 1] /= gbw_max
    our_nd[:, 1] /= gbw_max

    # Calculate distribution of minimum distances
    distances = []
    for our_point in our_nd:
        distances.append(
            directed_hausdorff(our_point.reshape(1, 2), opo_nd)[0] / math.sqrt(2)
        )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    plot = sns.distplot(
        distances,
        hist_kws={"cumulative": True, "rwidth": 0.80},
        norm_hist=True,
        bins=45,
        kde=False,
    )
    #  plot = sns.distplot(distances, norm_hist=True)

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Normalized Hausdorff Distance")
    plot.set(ylabel="Cumulative Probability")
    plot.set_xlim(0, 0.05)
    plot.set_ylim(0, 1)
    pp = PdfPages("cdf.pdf")
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    run(["pdfcrop", "cdf.pdf", "cdf.pdf"], stdout=DEVNULL, check=True)
    logger.info(f"Plot saved to cdf.pdf")

    temp = np.array(distances)
    logger.info(
        f"90p: {np.percentile(temp, 90)} "
        + f"95p: {np.percentile(temp, 95)} "
        + f"99p: {np.percentile(temp, 99)} "
        + f"max: {np.amax(temp)}"
    )

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set(xscale=scale, yscale=scale)

    # Build a dataframe with both values
    total.drop_duplicates(inplace=True)
    plot = sns.scatterplot(
        data=total,
        hue="Source",
        style="Source",
        y="GBWP (kHz)",
        x="Supply Current (uA)",
        markers=["x", "+"],
        ax=ax,
        palette=reversed(sns.color_palette("Paired")[:2]),
    )

    # Highlight the Opo selection, if present
    target = total[
        (total["GBWP (kHz)"] == 400.0)
        & (total["Supply Current (uA)"] == 4.6)
        & (total["Source"] == "Our Approach")
    ]
    if target.size:
        logger.info("Contained the target Opo selection.")
        plot.plot(4.6, 400.0, color="r", marker="o", markersize=12, fillstyle="none")

    # Remove the seaborn legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Quiescent Current (uA)")
    plot.set(ylabel="GBW (kHz)")
    plot.set_xlim(1e-2, 1e6)
    plot.set_ylim(1e-1, 1e8)
    pp = PdfPages(outfile)
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)
    logger.info(f"Plot saved to {outfile}")
    logger.info("Matched values saved to ./matched.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opo-file",
        type=str,
        default="../hack/opamps/data/opo_processed_opamps.csv",
        help="CSV file of original OPO data",
    )
    parser.add_argument(
        "--gain-file",
        type=str,
        default="../hack/opamps/output_gain.csv",
        help="CSV file name to plot",
    )
    parser.add_argument(
        "--current-file",
        type=str,
        default="../hack/opamps/output_current.csv",
        help="CSV file name to plot",
    )
    parser.add_argument("-gb", type=float, help="Gain threshold to use.", default=0.40)
    parser.add_argument(
        "-cb", type=float, help="Current threshold to use.", default=0.75
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./opo.pdf",
        help="Path where the PDF should be saved.",
    )
    parser.add_argument("-s", "--scale", type=str, default="log")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Output INFO level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        ch = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    _plot(
        args.opo_file,
        args.gain_file,
        args.current_file,
        args.output,
        args.scale,
        args.gb,
        args.cb,
    )
