"""
This script takes in a path containing Digikey's raw gold CSVs. It then reads in
each raw CSV, formats it's gold data where:
(filename, part_num, attribute_name, value) = line
and writes the combined output to the specified output CSV.
"""

import csv
import logging
import os

import requests

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
    preprocess_doc,
    preprocess_freq_transition,
    preprocess_gbp,
    preprocess_manuf,
    preprocess_operating_temp,
    preprocess_operating_voltage,
    preprocess_polarity,
    preprocess_pwr_max,
    preprocess_supply_current,
    preprocess_vce_saturation_max,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), f"format_digikey.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def format_digikey_gold(
    component, raw_gold_file, formatted_gold_file, seen, append=False, filenames="standard",
):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        session = requests.Session()  # Only initialize a requests.Session() once
        for line in reader:
            if component == "opamp":
                """
                TODO: Opamp formatting has not been vetted yet...
                """
                
                manuf = preprocess_manuf(line[4])  # TODO: These don't match
                part_num = opamp_part_normalizer(line[3])
		
		# Right now, we get the filename from the URLs headers
                if filenames == "url":
                    doc_name = preprocess_doc(url, session=session)
                elif filenames == "standard":
                    # Set the doc_name to be manuf_partnum.pdf (as digikeyscraper does)
                    doc_name = f"{manuf}_{part_num}"
                else:
                    logger.debug(f"Invalid filename format {filenames}, reverting to standard")
                    doc_name = f"{manuf}_{part_num}"

                part_family = "N/A"  # Digkey does not have part_family attribute
                part_num = line[3]
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

                manuf = preprocess_manuf(manufacturer)
                part_num = transistor_part_normalizer(manuf_part_num)

                # Right now, we get the filename from the URLs headers
                if filenames == "url":
                    doc_name = preprocess_doc(url, session=session)
                elif filenames == "standard":
                    # Set the doc_name to be manuf_partnum.pdf (as digikeyscraper does)
                    doc_name = f"{manuf}_{part_num}"
                else:
                    logger.debug(f"Invalid filename format {filenames}, reverting to standard")
                    doc_name = f"{manuf}_{part_num}"
		
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

            else:
                logger.error(f"Invalid hardware component {component}")
                os.exit()

            # Output tuples of each normalized attribute
            for name, attr, normalizer in name_attr_norm:
                if "N/A" not in attr:
                    for a in attr.split(delim):
                        if len(a.strip()) > 0:
                            source_file = str(raw_gold_file.split("/")[-1])
                            output = [doc_name, manuf, part_num, name, normalizer(a), source_file]
                            if tuple(output) not in seen:
                                writer.writerow(output)
                                seen.add(tuple(output))


if __name__ == "__main__":
    # Transform the transistor dataset
    component = "transistor"
    filenames = "url"

    # Change `digikey_csv_dir` to the absolute path where Digikey's raw CSVs are
    # located.
    digikey_csv_dir = "/home/nchiang/repos/hack/hack/transistors/data/csv/"

    # Change `formatted_gold` to the absolute path of where you want the
    # combined gold to be written.
    formatted_gold = (
       f"/home/nchiang/repos/hack/hack/transistors/data/{filenames}_digikey_gold.csv"
    )
    seen = set()
    # Run transformation
    for i, filename in enumerate(sorted(os.listdir(digikey_csv_dir))):
        if filename.endswith(".csv"):
            raw_path = os.path.join(digikey_csv_dir, filename)
            logger.info(f"[INFO]: Parsing {raw_path}")
            if i == 0:  # create file on first iteration
                format_digikey_gold(component, raw_path, formatted_gold, seen, filenames=filenames)
            else:  # then just append to that same file
                format_digikey_gold(
                    component, raw_path, formatted_gold, seen, append=True, filenames=filenames
                )
