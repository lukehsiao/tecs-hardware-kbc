import csv
import logging
import os
import pdb

from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
This script should read in the existing dev and test sets of documents (because
we already have our own "ground truth" gold labels for those datasets) and pull
out documents that we know that Digikey/Mouser has and put them into a separate
dir for comparison purposes.
"""


def get_dataset_filenames(goldfile):
    """Returns a set of all filenames found in a given dataset gold csv."""
    filenames = set()
    with open(goldfile, "r") as gold:
        reader = csv.reader(gold)
        for line in reader:
            filename = line[0]
            filenames.add(filename.upper())
    if len(filenames) != 0:
        return filenames
    else:
        logger.error(f"No filenames found in {goldfile}.")
        pdb.set_trace()


def get_valid_filenames(dirname, filenames):
    """Returns a set of filenames that are found both in the Digikey/Mouser gold
    CSV and in the target dataset directory."""
    valid_filenames = set()
    logger.info(f"Getting valid filenames from {dirname}")
    for filename in tqdm(os.listdir(dirname)):
        if filename.endswith(".pdf") or filename.endswith(".PDF"):
            if filename.upper().replace(".PDF", "") in filenames:
                valid_filenames.add(filename.replace(".pdf", "").replace(".PDF", ""))
    if len(valid_filenames) != 0:
        return valid_filenames
    else:
        logger.error(f"No valid filenames found in {dirname}.")
        pdb.set_trace()


def move_valid_files(pdf, html, valid_filenames, limit=100, out="analysis/"):
    """Moves all filenames found in valid_filenames into an analysis
    dataset directory for comparison."""
    endpath = os.path.join(os.path.dirname(__file__), out)
    # if len(valid_filenames) < limit:
    # logger.error(
    # f"{len(valid_filenames)} valid filenames is "
    # + f"not enough to satisfy limit of {limit}."
    # )
    # pdb.set_trace()
    # elif len(valid_filenames) != limit:
    # valid_filenames = set(list(valid_filenames)[:limit])
    logger.info(f"Moving {len(valid_filenames)} files into {out} from {pdf} and {html}")
    for i, filename in enumerate(tqdm(valid_filenames)):
        if filename + ".pdf" in os.listdir(pdf):
            os.rename(
                os.path.join(pdf, filename + ".pdf"),
                os.path.join(endpath + "pdf/", filename + ".pdf"),
            )
        elif filename + ".PDF" in os.listdir(pdf):
            os.rename(
                os.path.join(pdf, filename + ".PDF"),
                os.path.join(endpath + "pdf/", filename + ".pdf"),
            )
        else:
            logger.error(f"Filename {filename} not found in {pdf}")
            pdb.set_trace()
        if filename + ".html" in os.listdir(html):
            os.rename(
                os.path.join(html, filename + ".html"),
                os.path.join(endpath + "html/", filename + ".html"),
            )
        elif filename + ".HTML" in os.listdir(html):
            os.rename(
                os.path.join(html, filename + ".HTML"),
                os.path.join(endpath + "html/", filename + ".html"),
            )
        else:
            logger.error(f"Filename {filename} not found in {html}")
            pdb.set_trace()
        if i >= limit:
            return


if __name__ == "__main__":

    # Run extraction on dev set
    dirname = os.path.dirname(__file__)
    pdfdir = os.path.join(dirname, "../../dev/pdf/")
    htmldir = os.path.join(dirname, "../../dev/html")
    gold_file = os.path.join(dirname, "../../standard_digikey_gold.csv")

    dataset_filenames = get_dataset_filenames(gold_file)

    filenames = get_valid_filenames(pdfdir, dataset_filenames)
    move_valid_files(pdfdir, htmldir, filenames, out="../../analysis/")

    # Run extraction on test set
    pdfdir = os.path.join(dirname, "../../test/pdf/")
    htmldir = os.path.join(dirname, "../../test/html")

    dataset_filenames = get_dataset_filenames(gold_file)

    filenames = get_valid_filenames(pdfdir, dataset_filenames)
    move_valid_files(pdfdir, htmldir, filenames, out="../../analysis/")
