#!/usr/bin/env python

"""
This script reads in the dev gold and test gold label files. It then writes the
combined contents of both files to a single CSV outfile so as to allow for the
easy comparison of local data to the source set of combined gold labels.
"""

import csv


def combine_csv(combined, dev, test):
    with open(combined, "w") as out, open(dev, "r") as in1, open(test, "r") as in2:
        reader1 = csv.reader(in1)
        reader2 = csv.reader(in2)
        writer = csv.writer(out)
        for line in reader1:
            writer.writerow(line)
        for line in reader2:
            writer.writerow(line)


if __name__ == "__main__":
    # Change `combined` to match the absolute path of where you want to write the
    # combined gold label output CSV
    combined = "/home/nchiang/repos/hack/hack/transistors/data/combined_gold.csv"

    # Change `dev` to match the absolute path of your local dev gold label CSV
    dev = "/home/nchiang/repos/hack/hack/transistors/data/sorted_dev_gold.csv"

    # Change `test` to match the absolute path of your local test gold label CSV
    test = "/home/nchiang/repos/hack/hack/transistors/data/sorted_test_gold.csv"

    # Run combination
    combine_csv(combined, dev, test)
