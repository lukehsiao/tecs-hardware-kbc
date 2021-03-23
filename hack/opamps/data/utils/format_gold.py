"""
This script reads in our raw gold files and parses them into attributes and
their corresponding parts. It then passes those attributes through several
preprocessors and normalizers after which they are written to an out CSV where:
(filename, manuf, part_num, attribute_name, value) = line
"""

import csv
import os

from hack.utils.gold_utils.normalizers import (
    gain_bandwidth_normalizer,
    manuf_normalizer,
    opamp_part_normalizer,
    opamp_voltage_normalizer,
    part_family_normalizer,
    supply_current_normalizer,
    temperature_normalizer,
)


def format_gold(raw_gold_file, formatted_gold_file, seen, append=False):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        for line in reader:

            (
                doc_name,
                part_family,
                part_num,
                manufacturer,
                typ_gbp,
                typ_supply_current,
                min_op_supply_volt,
                max_op_supply_volt,
                min_op_temp,
                max_op_temp,
                notes,
                annotator,
            ) = line

            # Map each attribute to its corresponding normalizer
            name_attr_norm = [
                ("part_family", part_family, part_family_normalizer),
                ("typ_gbp", typ_gbp, gain_bandwidth_normalizer),
                ("typ_supply_current", typ_supply_current, supply_current_normalizer),
                ("min_op_supply_volt", min_op_supply_volt, opamp_voltage_normalizer),
                ("max_op_supply_volt", max_op_supply_volt, opamp_voltage_normalizer),
                ("min_op_temp", min_op_temp, temperature_normalizer),
                ("max_op_temp", max_op_temp, temperature_normalizer),
            ]

            manuf = manuf_normalizer(manufacturer)
            part_num = opamp_part_normalizer(part_num)

            # Output tuples of each normalized attribute
            for name, attr, normalizer in name_attr_norm:
                if "N/A" not in attr:
                    for a in attr.split(delim):
                        if len(a.strip()) > 0:
                            output = [
                                doc_name.replace(".pdf", ""),
                                manuf,
                                part_num,
                                name,
                                normalizer(a),
                            ]
                            if tuple(output) not in seen:
                                writer.writerow(output)
                                seen.add(tuple(output))


if __name__ == "__main__":
    seen = set()  # set for fast O(1) amortized lookup

    # Change `raw_gold1` and `raw_gold2` to the absolute paths of your raw gold
    # CSVs. NOTE: Make sure to change the `line = ()` in `format_gold()` to
    # match what a line actually is in your raw gold CSV.
    dirname = os.path.dirname(__name__)
    raw_gold = "../src/raw_mouser_gold.csv"
    out_gold = "../mouser/our_gold.csv"

    # Run formatting
    format_gold(raw_gold, out_gold, seen)
