"""
This script takes in a gold CSV file, sorts it, and writes the sorted output to
the specified gold output CSV.
"""

import csv
import operator

if __name__ == "__main__":
    # Change `input` to the absolute path of the to-be-sorted raw CSV.
    input = "/home/nchiang/repos/hack/hack/transistors/data/unsorted_formatted_gold.csv"

    # Change `output` to the absolute path of where you want the sorted output to
    # be written.
    output = "/home/nchiang/repos/hack/hack/transistors/data/formatted_gold.csv"
    data = csv.reader(open(input), delimiter=",")

    # 0 specifies according to first column we want to sort (i.e. filename)
    sortedlist = sorted(data, key=operator.itemgetter(0))
    # Write sorted data into output file
    with open(output, "w") as f:
        fileWriter = csv.writer(f, delimiter=",")
        for row in sortedlist:
            fileWriter.writerow(row)
