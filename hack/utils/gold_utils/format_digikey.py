#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import os

from hack.utils.gold_utils.normalizers import (
    gain_bandwidth_normalizer,
    general_normalizer,
    opamp_part_normalizer,
    opamp_voltage_normalizer,
    part_family_normalizer,
    supply_current_normalizer,
    temperature_normalizer,
    transistor_part_normalizer,
)
from hack.utils.gold_utils.preprocessors import (
    preprocess_c_current_cutoff_max,
    preprocess_c_current_max,
    preprocess_ce_v_max,
    preprocess_dc_gain_min,
    preprocess_freq_transition,
    preprocess_gbp,
    preprocess_manuf,
    preprocess_operating_temp,
    preprocess_operating_voltage,
    preprocess_polarity,
    preprocess_pwr_max,
    preprocess_supply_current,
    preprocess_url,
    preprocess_vce_saturation_max,
)


def format_digikey_gold(
    component, raw_gold_file, formatted_gold_file, seen, append=False
):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        for line in reader:
            if component == "opamp":
                """
                TODO: Opamp formatting has not been vetted yet...
                """
                # TODO: Do we want to just skip adding the ones that have missing docs?
                doc_name = preprocess_url(line[0])
                part_family = "N/A"  # Digkey does not have part_family attribute
                part_num = line[3]
                manufacturer = line[4]  # TODO: These don't match
                typ_gbp = preprocess_gbp(line[18])
                typ_supply_current = preprocess_supply_current(line[22])
                (min_op_supply_volt, max_op_supply_volt) = preprocess_operating_voltage(
                    line[24]
                )
                (min_op_temp, max_op_temp) = preprocess_operating_temp(line[25])

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

                part_num = opamp_part_normalizer(part_num)

            elif component == "transistor":

                """
                A line of Digikey gold data (we only care about some of it):
                """

                (
                    url,
                    image,
                    digikey_part_num,
                    manuf_part_num,
                    manufacturer,
                    description,
                    avaliable,
                    stock,
                    price,
                    at_qty,
                    min_qty,
                    packaging,
                    series,
                    part_status,  # We don't care about anything above this comment
                    transistor_type,
                    collector_current_max,
                    ce_voltage_max,
                    vce_saturation_max,
                    collector_cutoff_current_max,
                    dc_current_gain_min,
                    power_max,
                    frequency_transition,
                    operating_temp,
                    mounting_type,
                    packaging_type,
                    supplier_packaging,
                ) = line

                """
                Extract all useful Digikey values (even if we don't have them):
                NOTE: Preprocessing returns each attribute's value, a space,
                and it's unit.
                NOTE: Normalization renders all CSVs the same (i.e. set units).
                """

                # Right now, we just use URLs for filenames
                doc_name = preprocess_url(url)  # Checks for invalid URLs
                manuf = preprocess_manuf(manufacturer)

                # Extract implied values from attribute conditions
                # TODO: maybe check if the implied_supply_current's match?
                (
                    dc_gain_min,
                    implied_supply_current,
                    implied_ce_v_max,
                ) = preprocess_dc_gain_min(dc_current_gain_min)
                (
                    vce_saturation_max,
                    implied_base_current,
                    implied_supply_current1,
                ) = preprocess_vce_saturation_max(vce_saturation_max)

                # Relevant data (i.e. attributes that appear in our gold labels)
                c_current_max = preprocess_c_current_max(collector_current_max)
                polarity = preprocess_polarity(transistor_type)
                ce_v_max = preprocess_ce_v_max(ce_voltage_max)
                """
                NOTE: Attributes we have that Digikey does not:
                dev_dissipation
                cb_v_max
                stg_temp_max
                stg_temp_min
                eb_v_max
                """

                # Other data (i.e. only appears in Digikey's CSVs, not in ours)
                (min_op_temp, max_op_temp) = preprocess_operating_temp(operating_temp)
                c_current_cutoff_max = preprocess_c_current_cutoff_max(
                    collector_cutoff_current_max
                )
                pwr_max = preprocess_pwr_max(power_max)
                freq_transition = preprocess_freq_transition(frequency_transition)

                # Map each attribute to its corresponding normalizer
                name_attr_norm = [  # TODO: Update normalizers
                    # Data that both Digikey and our labels have:
                    ("polarity", polarity, general_normalizer),
                    ("c_current_max", c_current_max, general_normalizer),
                    ("ce_v_max", ce_v_max, general_normalizer),
                    ("dc_gain_min", dc_gain_min, general_normalizer),
                    # Data that Digikey has that we do not:
                    ("min_op_temp", min_op_temp, general_normalizer),
                    ("max_op_temp", max_op_temp, general_normalizer),
                    ("c_current_cutoff_max", c_current_cutoff_max, general_normalizer),
                    ("pwr_max", pwr_max, general_normalizer),
                    ("freq_transition", freq_transition, general_normalizer),
                    # Data that was implied by the conditions of other data:
                    ("implied_ce_v_max", implied_ce_v_max, general_normalizer),
                    (
                        "implied_supply_current",
                        implied_supply_current,
                        general_normalizer,
                    ),
                    (
                        "implied_supply_current",
                        implied_supply_current1,
                        general_normalizer,
                    ),
                    ("implied_base_current", implied_base_current, general_normalizer),
                    ("vce_sat_max", vce_saturation_max, general_normalizer),
                ]

                part_num = transistor_part_normalizer(manuf_part_num)

            else:
                print("[ERROR]: Invalid hardware component")
                os.exit()

            # Output tuples of each normalized attribute
            for name, attr, normalizer in name_attr_norm:
                if "N/A" not in attr:
                    for a in attr.split(delim):
                        if len(a.strip()) > 0:
                            output = [doc_name, manuf, part_num, name, normalizer(a)]
                            if tuple(output) not in seen:
                                writer.writerow(output)
                                seen.add(tuple(output))


if __name__ == "__main__":
    # Transform transistor CSVs
    component = "transistor"
    digikey_csv_dir = "/home/nchiang/repos/hack/transistors/data/csv/"
    formatted_gold = "/home/nchiang/repos/hack/transistors/data/digikey_gold2.csv"
    seen = set()
    # Run transformation
    for i, filename in enumerate(sorted(os.listdir(digikey_csv_dir))):
        if filename.endswith(".csv"):
            raw_path = os.path.join(digikey_csv_dir, filename)
            print(f"[INFO]: Parsing {raw_path}")
            if i == 0:  # create file on first iteration
                format_digikey_gold(component, raw_path, formatted_gold, seen)
            else:  # then just append to that same file
                format_digikey_gold(
                    component, raw_path, formatted_gold, seen, append=True
                )
