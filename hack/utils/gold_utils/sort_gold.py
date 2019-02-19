#!/usr/bin/env python

import csv
import operator

"""
This script should take in a single gold CSV file and sort it into the same order
as previously made gold files. (So as to maintain a clear git history)
"""

if __name__ == "__main__":
    input = "/home/nchiang/repos/hack/hack/transistors/data/dev/old_dev_gold.csv"
    output = "/home/nchiang/repos/hack/hack/transistors/data/dev/sorted_dev_gold.csv"
    data = csv.reader(open(input), delimiter=",")
    # 0 specifies according to first column we want to sort
    sortedlist = sorted(data, key=operator.itemgetter(0))
    # Write sorted data into output file
    with open(output, "w") as f:
        fileWriter = csv.writer(f, delimiter=",")
        for row in sortedlist:
            fileWriter.writerow(row)
