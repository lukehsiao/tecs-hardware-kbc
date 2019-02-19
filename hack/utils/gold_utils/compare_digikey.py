#!/usr/bin/env python

"""
This script reads in two CSVs (one from our labels and one from Digikey) where:
(filename, part_num, attribute_name, value) = line
and confirms that every value listed in our CSV is the same in Digikey's CSV.
"""

import csv


"""
Takes in a gold CSV and parses it into a data structure of nested maps (see below)
{filename: {part_num1: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            part_num2: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            }
 filename: {part_num1: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            part_num2: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            }
}
"""


def make_map(gold_file):
    with open(gold_file, "r") as inputcsv:
        reader = csv.reader(inputcsv)
        map = {}
        for line in reader:
            (filename, part_num, attr, value) = line
            # If filename already exists in map, just append to existing values
            if filename in map:
                # Check if part_num already exists
                if part_num in map[filename]:
                    # Account for multiple values for the same attr
                    if attr in map[filename][part_num]:
                        map[filename][part_num][attr].append(value)
                # Add new attr key and value pair to existing part_num
                else:
                    map[filename][part_num] = {attr: [value]}
            # Add new filename to map with nested data structure
            map[filename] = {part_num: {attr: [value]}}
        return map


"""
Takes in two maps and logs any discrepancies (NOTE: filenames are not compared)
"""


def compare_maps(digikey_map, our_map):
    for filename in our_map:
        for part_num in filename:
            if part_num not in digikey_map:
                print(f"[WARNING]: Part number {part_num} not in Digikey map")
                continue
            if our_map[filename][part_num] != digikey_map[filename][part_num]:
                print(f"[ERROR]: Nested data at {part_num} does not match")
                continue


"""
Run comparison on transistor gold data
"""
if __name__ == "__main__":
    # Change `our_csv` to the absolute path of your formatted gold label CSV
    our_csv = "/home/nchiang/repos/hack/transistors/data/dev/dev_gold.csv"

    # Change `digikey_csv` to the absolute path of the formatted Digikey gold CSV
    # that you want to compare with `our_csv`
    digikey_csv = "/home/nchiang/repos/hack/transistors/data/digikey_gold.csv"

    # Run comparison
    print(f"[INFO]: Running comparison on {digikey_csv} and {our_csv}")
    digikey_map = make_map(digikey_csv)
    our_map = make_map(our_csv)
    compare_maps(digikey_map, our_map)
