import csv
import logging
import os

from hack.transistors.data.utils.analysis.make_dataset_from_list import move_files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_filenames(filename_file):
    filenames = set()
    with open(filename_file, "r") as inputfile:
        reader = csv.reader(inputfile)
        for line in reader:
            filenames.add(line[0])
    return filenames


if __name__ == "__main__":

    # Get filenames to move back into dev and test
    dirname = os.path.dirname(__name__)
    filenames_file = os.path.join(dirname, "../../analysis/filenames.csv")
    dev_gold = os.path.join(dirname, "../../dev/dev_gold.csv")
    test_gold = os.path.join(dirname, "../../test/test_gold.csv")
    test_path = os.path.join(dirname, "../../test/")
    dev_path = os.path.join(dirname, "../../dev/")
    analysis_path = os.path.join(dirname, "../../analysis/")

    # Get a set of all filenames found in filenames_file
    filenames = get_filenames(filenames_file)
    dev_filenames = filenames.intersection(get_filenames(dev_gold))
    test_filenames = filenames.intersection(get_filenames(test_gold))

    # Move filenames from analysis/ back into corresponding dirs
    move_files(test_filenames, analysis_path, test_path)
    move_files(dev_filenames, analysis_path, dev_path)
