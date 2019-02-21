#!/usr/bin/env python

"""
This script reads in two CSVs (one from our labels and one from Digikey) where:
(filename, manuf, part_num, attribute_name, value) = line
and confirms that every value listed in our CSV is the same in Digikey's CSV.
"""

import csv
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger(__name__)


"""
Takes in a gold CSV and parses it into a data structure of nested maps (see below)
{manuf: {part_num1: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            part_num2: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            }
 manuf: {part_num1: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            part_num2: {attr: [value1, value2]
                        attr: [value1, value2]
                        }
            }
}
"""
def make_map(gold_file):
    reader = csv.reader(open(gold_file))
    gold_map = {}
    for line in reader:
        (filename, manuf, part_num, attr, value) = line
        # If manuf already exists in map, just append to existing values
        if manuf in gold_map:
            # Check if part_num already exists
            if part_num in gold_map[manuf]:
                # Account for multiple values for the same attr
                if attr in gold_map[manuf][part_num]:
                    if value not in gold_map[manuf][part_num][attr]:
                        gold_map[manuf][part_num][attr].append(value)
            # Add new attr key and value pair to existing part_num
            else:
                gold_map[manuf][part_num] = {attr: [value]}
        else:
            # Add new manuf to map with nested data structure
            gold_map[manuf] = {part_num: {attr: [value]}}
    return gold_map


"""
Takes in two maps, logs any discrepancies, and generates sig figs.
"""
def compare_maps(digikey_map, our_map):
    doc_fig = 0  # Number of files in our gold data not in Digikey's
    manuf_fig = 0  # Number of manuf in our gold data not in Digikey's
    part_fig = 0  # Number of parts in our gold data not in Digikey's
    value_fig = 0  # Number of values in our gold data not in Digikey's
    for manuf in our_map:
        if manuf not in digikey_map:
            log.debug(f"Manufacturer {manuf} not in Digikey map.")
            manuf_fig += 1
            continue
        for part_num in our_map[manuf]:
            if part_num not in digikey_map[manuf]:
                log.debug(f"Part number {part_num} not in Digikey map.")
                part_fig += 1
                continue
            elif our_map[manuf][part_num] != digikey_map[manuf][part_num]:
                log.debug(f"Nested data at {manuf}:{part_num} does not match.")
                value_fig += 1
                continue
    return (doc_fig, manuf_fig, part_fig, value_fig)


"""
Helper function used in debugging to find the location of a given manuf and part
"""
def get_filename(file, manufacturer, part):
    reader = csv.reader(open(file))
    for line in reader:
        (filename, manuf, part_num, attr, value) = line
        if manuf == manufacturer and part_num == part:
            return filename


"""
Run comparison on transistor gold data
"""
if __name__ == "__main__":
    # Change `our_csv` to the absolute path of your formatted gold label CSV
    our_csv = "/home/nchiang/repos/hack/hack/transistors/data/dev/dev_gold.csv"
    our_file = our_csv.split("/")[-1]

    # Change `digikey_csv` to the absolute path of the formatted Digikey gold CSV
    # that you want to compare with `our_csv`
    digikey_csv = "/home/nchiang/repos/hack/hack/transistors/data/digikey_gold.csv"
    digikey_file = digikey_csv.split("/")[-1]

    # Run comparison
    log.info(f"Running comparison on {digikey_file} and {our_file}.")
    digikey_map = make_map(digikey_csv)
    our_map = make_map(our_csv)
    (doc_fig, manuf_fig, part_fig, value_fig) = compare_maps(digikey_map, our_map)

    # Log significant figure outcome
    log.info(f"{doc_fig} files were not found in Digikey's gold data.")
    log.info(f"{manuf_fig} manufacturers were not found in Digikey's gold data.")
    log.info(f"{part_fig} parts were not found in Digikey's gold data.")
    log.info(f"{value_fig} values were not found in Digikey's gold data.")
