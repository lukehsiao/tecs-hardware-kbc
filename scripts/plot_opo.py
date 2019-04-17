#!/usr/bin/env python
import argparse
import logging
import os
import sys
from subprocess import DEVNULL, run

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

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


def _plot_sweep(infile, gainfile, currentfile, scale):
    """Plotting logic."""
    dirname = os.path.dirname(__file__)

    # Mfr PartNumber,GBWP (kHz),Supply Current (uA),GBWP/uA,min_voltage,max_voltage
    opo_data = pd.read_csv(infile, skipinitialspace=True)
    gain_data = pd.read_csv(gainfile, skipinitialspace=True)
    current_data = pd.read_csv(currentfile, skipinitialspace=True)

    # Process our data based on two values of b
    for gain_b in np.linspace(0.1, 1, num=20):
        for current_b in np.linspace(0.1, 1, num=20):
            fig, ax = plt.subplots(figsize=(6, 4))

            logger.info(f"gain_b = {gain_b}, current_b = {current_b}")

            filtered_gain = gain_data[gain_data.p >= gain_b][
                ["Document", "GBWP (kHz)", "sent"]
            ]
            filtered_gain = filtered_gain.sort_values(
                by=["Document", "sent", "GBWP (kHz)"]
            )
            filtered_current = current_data[current_data.p >= current_b][
                ["Document", "Supply Current (uA)", "sent"]
            ]
            filtered_current = filtered_current.sort_values(
                by=["Document", "sent", "Supply Current (uA)"]
            )

            # Naive cross product doesn't work.
            #  cross_product = pd.merge(filtered_gain, filtered_current, on="Document")

            filtered_gain_list = filtered_gain.values.tolist()
            filtered_current_list = filtered_current.values.tolist()
            matched = _match(filtered_gain_list, filtered_current_list)

            matched.to_csv(os.path.join(dirname, "matched.csv"), sep=",")
            logger.info(f"len(matched): {len(matched)}")

            logger.info("Plotting from {}...".format(infile))
            ax.set(xscale=scale, yscale=scale)

            opo_view = opo_data[["Supply Current (uA)", "GBWP (kHz)"]]
            our_view = matched[["Supply Current (uA)", "GBWP (kHz)"]]
            opo_view["Source"] = "Digi-Key"
            our_view["Source"] = "Our Approach"
            total = pd.concat([opo_view, our_view])

            # Build a dataframe with both values
            plot = sns.scatterplot(
                data=total,
                hue="Source",
                style="Source",
                y="GBWP (kHz)",
                x="Supply Current (uA)",
                markers=["x", "+"],
                ax=ax,
            )

            # Remove the seaborn legend title
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])

            sns.despine(bottom=True, left=True)
            plot.set(xlabel="Quiescent Current (uA)")
            plot.set(ylabel="GBW (kHz)")

            outfile = f"opo/{gain_b:.3f}_{current_b:.3f}.pdf"
            pp = PdfPages(outfile)
            pp.savefig(plot.get_figure().tight_layout())
            pp.close()
            logger.info("Plot saved to {}".format(outfile))
            run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)


def _plot(infile, gainfile, currentfile, outfile, scale):
    """Plotting logic."""
    fig, ax = plt.subplots(figsize=(6, 4))

    dirname = os.path.dirname(__file__)
    # Mfr PartNumber,GBWP (kHz),Supply Current (uA),GBWP/uA,min_voltage,max_voltage
    opo_data = pd.read_csv(infile, skipinitialspace=True)

    gain_data = pd.read_csv(gainfile, skipinitialspace=True)
    current_data = pd.read_csv(currentfile, skipinitialspace=True)

    # Process our data based on two values of b
    gain_b = 0.750
    current_b = 0.750
    logger.info(f"gain_b = {gain_b}, current_b = {current_b}")

    filtered_gain = gain_data[gain_data.p >= gain_b][["Document", "GBWP (kHz)", "sent"]]
    filtered_gain = filtered_gain.sort_values(by=["Document", "sent", "GBWP (kHz)"])
    filtered_current = current_data[current_data.p >= current_b][
        ["Document", "Supply Current (uA)", "sent"]
    ]
    filtered_current = filtered_current.sort_values(
        by=["Document", "sent", "Supply Current (uA)"]
    )

    # Naive cross product doesn't work.
    #  cross_product = pd.merge(filtered_gain, filtered_current, on="Document")

    filtered_gain_list = filtered_gain.values.tolist()
    filtered_current_list = filtered_current.values.tolist()
    matched = _match(filtered_gain_list, filtered_current_list)

    matched.to_csv(os.path.join(dirname, "matched.csv"), sep=",")
    logger.info(f"len(matched): {len(matched)}")

    logger.info("Plotting from {}...".format(infile))
    ax.set(xscale=scale, yscale=scale)

    opo_view = opo_data[["Supply Current (uA)", "GBWP (kHz)"]]
    our_view = matched[["Supply Current (uA)", "GBWP (kHz)"]]
    opo_view["Source"] = "Digi-Key"
    our_view["Source"] = "Our Approach"
    total = pd.concat([opo_view, our_view])

    # Build a dataframe with both values
    plot = sns.scatterplot(
        data=total,
        hue="Source",
        style="Source",
        y="GBWP (kHz)",
        x="Supply Current (uA)",
        markers=["x", "+"],
        ax=ax,
    )

    # Remove the seaborn legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Quiescent Current (uA)")
    plot.set(ylabel="GBW (kHz)")
    pp = PdfPages(outfile)
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    logger.info("Plot saved to {}".format(outfile))
    run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="CSV file name to plot")
    parser.add_argument("gain_file", type=str, help="CSV file name to plot")
    parser.add_argument("current_file", type=str, help="CSV file name to plot")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../fig/opo.pdf",
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

    #  _plot_sweep(args.csv_file, args.gain_file, args.current_file, args.scale)
    _plot(args.csv_file, args.gain_file, args.current_file, args.output, args.scale)
