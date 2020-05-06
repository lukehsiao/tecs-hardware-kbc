#!/usr/bin/env python
import argparse
import logging
import sys
from subprocess import DEVNULL, run

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = [
    r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
    r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
    #  r'\usepackage{helvet}',    # set the normal font here
    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
]

sns.set(style="whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.25})


# Configure logging for Hack
logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

FIGSIZE = (5, 3)


def _plot(infile):
    """Plotting logic."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # type,b,f1,precision,recall
    joint_data = pd.read_csv(infile, skipinitialspace=True)

    # Plot precision
    plot = sns.lineplot(
        data=joint_data, hue="type", style="type", y="precision", x="b", ax=ax
    )

    # Remove the seaborn legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 1])

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Threshold")
    plot.set(ylabel="Precision")
    outfile = "joint_precision.pdf"
    pp = PdfPages(outfile)
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)
    logger.info(f"Plot saved to {outfile}")

    # Plot recall
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot = sns.lineplot(
        data=joint_data, hue="type", style="type", y="recall", x="b", ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 1])

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Threshold")
    plot.set(ylabel="Recall")
    outfile = "joint_recall.pdf"
    pp = PdfPages(outfile)
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)
    logger.info(f"Plot saved to {outfile}")

    # Plot f1
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plot = sns.lineplot(data=joint_data, hue="type", style="type", y="f1", x="b", ax=ax)

    # Remove the seaborn legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.set_ylim([0, 1])
    ax.set_xlim([0.5, 1])

    sns.despine(bottom=True, left=True)
    plot.set(xlabel="Threshold")
    plot.set(ylabel="F1 Score")
    outfile = "joint_f1.pdf"
    pp = PdfPages(outfile)
    pp.savefig(plot.get_figure().tight_layout())
    pp.close()
    run(["pdfcrop", outfile, outfile], stdout=DEVNULL, check=True)
    logger.info(f"Plot saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="joint.csv", help="CSV file of joint data"
    )
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

    _plot(args.data)
