#!/usr/bin/env python

"""
This script takes in a single file of gold labels and parses them into the
respective `dev` and `test` sets based on previously defined filenames in old
`dev_gold.csv` and `test_gold.csv` gold label files.
"""

import csv
import pdb


def split_gold(combinedfile, devfile, testfile, devoutfile, testoutfile):
    with open(combinedfile, "r") as combined, open(devfile, "r") as dev, open(
        testfile, "r"
    ) as test, open(devoutfile, "w") as devout, open(testoutfile, "w") as testout:
        combinedreader = csv.reader(combined)
        devreader = csv.reader(dev)
        testreader = csv.reader(test)
        devwriter = csv.writer(devout)
        testwriter = csv.writer(testout)

        # Make a set for dev filenames
        devfilenames = set()
        for line in devreader:
            filename = line[0]
            devfilenames.add(filename)

        # Make a set for test filenames
        testfilenames = set()
        for line in testreader:
            filename = line[0]
            testfilenames.add(filename)

        # Read in the combined gold and split it into dev and test files
        line_num = 0
        for line in combinedreader:
            line_num += 1
            filename = line[0]
            if filename in devfilenames:
                devwriter.writerow(line)
            elif filename in testfilenames:
                testwriter.writerow(line)
            else:
                print(f"[ERROR]: Invalid filename {filename} on line {line_num}")
                pdb.set_trace()


if __name__ == "__main__":
    # Split our combined gold dataset into the respective dev and test sets

    # Change `combinedfile` to the absolute path of your combined gold file.
    combinedfile = "/home/nchiang/repos/hack/hack/transistors/data/formatted_gold.csv"

    # Change `devfile` and `testfile` to the absolute paths of your original
    # `dev` and `test` gold label files.
    devfile = "/home/nchiang/repos/hack/hack/transistors/data/dev/dev_gold.csv"
    testfile = "/home/nchiang/repos/hack/hack/transistors/data/test/test_gold.csv"

    # Change `devoutfile` and `testoutfile` to the absolute paths of where you
    # want the final outputs from `combinedfile` to be written.
    devoutfile = "/home/nchiang/repos/hack/hack/transistors/data/dev/new_dev_gold.csv"
    testoutfile = (
        "/home/nchiang/repos/hack/hack/transistors/data/test/new_test_gold.csv"
    )

    # Run split
    split_gold(combinedfile, devfile, testfile, devoutfile, testoutfile)
