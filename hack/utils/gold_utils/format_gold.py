#!/usr/bin/env python

"""
This script reads in our raw gold files and parses them into attributes and
their corresponding parts. It then passes those attributes through several
preprocessors and normalizers after which they are written to an out CSV where:
(filename, manuf, part_num, attribute_name, value) = line
"""

import csv
import os

from hack.utils.gold_utils.normalizers import (
    current_normalizer,
    dissipation_normalizer,
    doc_normalizer,
    gain_bandwidth_normalizer,
    gain_normalizer,
    manuf_normalizer,
    opamp_part_normalizer,
    opamp_voltage_normalizer,
    part_family_normalizer,
    polarity_normalizer,
    supply_current_normalizer,
    temperature_normalizer,
    transistor_part_normalizer,
    transistor_temp_normalizer,
    voltage_normalizer,
)


def format_gold(component, raw_gold_file, formatted_gold_file, seen, append=False):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        for line in reader:

            if component == "opamp":
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
                    (
                        "typ_supply_current",
                        typ_supply_current,
                        supply_current_normalizer,
                    ),
                    (
                        "min_op_supply_volt",
                        min_op_supply_volt,
                        opamp_voltage_normalizer,
                    ),
                    (
                        "max_op_supply_volt",
                        max_op_supply_volt,
                        opamp_voltage_normalizer,
                    ),
                    ("min_op_temp", min_op_temp, temperature_normalizer),
                    ("max_op_temp", max_op_temp, temperature_normalizer),
                ]

                manuf = manuf_normalizer(manufacturer)
                part_num = opamp_part_normalizer(part_num)

            elif component == "transistor":
                # This reads in our "Hardware Gold" raw gold CSV where a line
                # is:
                (
                    doc_name,
                    part_family,
                    part_num,
                    manufacturer,
                    polarity,
                    ce_v_max,  # Collector-Emitter Voltage (MAX) (Vceo)
                    cb_v_max,  # Collector-Base Voltage (MAX)
                    eb_v_max,  # Emitter-Base Voltage (MAX)
                    c_current_max,  # Collector Current Continuous (MAX)
                    dev_dissipation,  # Total Device Dissipation
                    stg_temp_min,
                    stg_temp_max,
                    dc_gain_min,  # DC Current Gain (MIN)
                    notes,
                    annotator,
                ) = line

                # Map each attribute to its corresponding normalizer
                name_attr_norm = [
                    ("part_family", part_family, part_family_normalizer),
                    ("polarity", polarity, polarity_normalizer),
                    ("ce_v_max", ce_v_max, voltage_normalizer),
                    ("cb_v_max", cb_v_max, voltage_normalizer),
                    ("eb_v_max", eb_v_max, voltage_normalizer),
                    ("c_current_max", c_current_max, current_normalizer),
                    ("dev_dissipation", dev_dissipation, dissipation_normalizer),
                    ("stg_temp_min", stg_temp_min, temperature_normalizer),
                    ("stg_temp_max", stg_temp_max, temperature_normalizer),
                    ("dc_gain_min", dc_gain_min, gain_normalizer),
                ]

                doc_name = doc_normalizer(doc_name)
                manuf = manuf_normalizer(manufacturer)
                part_num = transistor_part_normalizer(part_num)

                # Output tuples of each normalized attribute
                for name, attr, normalizer in name_attr_norm:
                    if "N/A" not in attr:
                        for a in attr.split(delim):
                            if len(a.strip()) > 0:
                                output = [
                                    doc_name,
                                    manuf,
                                    part_num,
                                    name,
                                    normalizer(a),
                                ]
                                if tuple(output) not in seen:
                                    writer.writerow(output)
                                    seen.add(tuple(output))

            elif component == "transistor2":
                # This reads in our "Small-signal Bipolar Transistors Gold Data"
                # raw gold CSV where a line is:
                delim = " "
                (
                    doc_name,
                    part_num,
                    manufacturer,
                    polarity,
                    pin_count,
                    ce_v_max,  # Collector-Emitter Voltage (MAX) (Vceo)
                    cb_v_max,  # Collector-Base Voltage (MAX)
                    eb_v_max,  # Emitter-Base Voltage (MAX)
                    c_current_max,  # Collector Current Continuous (MAX)
                    dev_dissipation,  # Total Device Dissipation
                    stg_temp_min,
                    stg_temp_max,
                    stg_temp_unit,
                    dc_gain_min,  # DC Current Gain (MIN)
                    max_freq,
                    output_resistance,
                    cb_capacitance,
                    base_resistance,
                    done,
                    _,
                ) = line

                part_family = "N/A"

                # Map each attribute to its corresponding normalizer
                name_attr_norm = [
                    ("part_family", part_family, part_family_normalizer),
                    ("polarity", polarity, polarity_normalizer),
                    ("ce_v_max", ce_v_max, voltage_normalizer),
                    ("cb_v_max", cb_v_max, voltage_normalizer),
                    ("eb_v_max", eb_v_max, voltage_normalizer),
                    ("c_current_max", c_current_max, current_normalizer),
                    ("dev_dissipation", dev_dissipation, dissipation_normalizer),
                    ("stg_temp_min", stg_temp_min, transistor_temp_normalizer),
                    ("stg_temp_max", stg_temp_max, transistor_temp_normalizer),
                    ("dc_gain_min", dc_gain_min, gain_normalizer),
                ]

                doc_name = doc_normalizer(doc_name)
                manuf = manuf_normalizer(manufacturer)
                part_num = transistor_part_normalizer(part_num)

                # Output tuples of each normalized attribute
                for name, attr, normalizer in name_attr_norm:
                    if "N/A" not in attr:
                        # Only dc_gain_min has multiple values for our
                        # "Small-signal Bipolar Transistors Gold Data" CSV
                        # (where we unwisely used a space as the delimiter)
                        if name == "dc_gain_min":
                            for a in attr.split(delim):
                                if len(a.strip()) > 0:
                                    output = [
                                        doc_name,
                                        manuf,
                                        part_num,
                                        name,
                                        normalizer(a),
                                    ]
                                    if tuple(output) not in seen:
                                        writer.writerow(output)
                                        seen.add(tuple(output))
                        else:
                            if len(attr.strip()) > 0:
                                output = [
                                    doc_name,
                                    manuf,
                                    part_num,
                                    name,
                                    normalizer(attr),
                                ]
                                if tuple(output) not in seen:
                                    writer.writerow(output)
                                    seen.add(tuple(output))

            else:
                print(f"[ERROR]: Invalid hardware component {component}")
                os.exit()


if __name__ == "__main__":
    seen = set()  # set for fast O(1) amortized lookup

    # Change `raw_gold1` and `raw_gold2` to the absolute paths of your raw gold
    # CSVs. NOTE: Make sure to change the `line = ()` in `format_gold()` to
    # match what a line actually is in your raw gold CSV.
    raw_gold1 = "/home/nchiang/repos/hack/hack/transistors/data/raw_gold1.csv"
    raw_gold2 = "/home/nchiang/repos/hack/hack/transistors/data/raw_gold2.csv"

    # Change `formatted_gold` to the absolute path of where you want your final
    # formatted output to be written.
    formatted_gold = (
        "/home/nchiang/repos/hack/hack/transistors/data/unsorted_formatted_gold.csv"
    )

    # Run formatting
    format_gold("transistor", raw_gold1, formatted_gold, seen)
    format_gold("transistor2", raw_gold2, formatted_gold, seen, True)
