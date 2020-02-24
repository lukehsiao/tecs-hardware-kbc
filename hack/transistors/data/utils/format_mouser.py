"""
This script takes in a path containing Mouser's raw gold CSVs. It then reads in
each raw CSV, formats it's gold data where:
(filename, manuf, part_num, attribute_name, value) = line
and writes the combined output to the specified output CSV.
"""

import csv
import logging
import os

from hack.transistors.data.utils.normalizers import (
    general_normalizer,
    transistor_part_normalizer,
)
from hack.transistors.data.utils.preprocessors import (
    preprocess_c_current_max,
    preprocess_ce_v_max,
    preprocess_individual_operating_temp,
    preprocess_manuf,
    preprocess_mouser_doc,
    preprocess_polarity,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def format_mouser_gold(
    raw_gold_file, formatted_gold_file, seen, append=False, filenames="standard"
):
    delim = ";"
    with open(raw_gold_file, "r") as csvinput, open(
        formatted_gold_file, "a"
    ) if append else open(formatted_gold_file, "w") as csvoutput:
        writer = csv.writer(csvoutput, lineterminator="\n")
        reader = csv.reader(csvinput)
        next(reader, None)  # Skip header row
        for line in reader:
            """
            A line of Mouser gold data (we only care about some of it):
            """

            (
                mouser_part_num,
                manuf_part_num,
                manufacturer,
                url,
                stock,
                price,
                rohs_compliance,
                lifecycle,
                product_detail,
                mounting_type,
                package_case,
                transistor_type,
                configuration,
                ce_voltage_max,
                collector_base_voltage,
                emitter_base_voltage,
                vce_saturation_max,
                collector_current_max,
                power_dissipation,
                gain_band_prod,
                min_op_temp,
                max_op_temp,
                series,
                packaging,
            ) = line

            """
            Extract all useful Mouser values (even if we don't have them):
            NOTE: Preprocessing returns each attribute's value, a space,
            and it's unit.
            NOTE: Normalization renders all CSVs the same (i.e. set units).
            """

            doc_name = preprocess_mouser_doc(manufacturer, manuf_part_num)
            manuf = preprocess_manuf(manufacturer)
            part_num = transistor_part_normalizer(manuf_part_num)

            # For analysis purposes, skip datasheets that `get_docs()`
            # cannot get a filename for (i.e. datasheets that are not
            # in the dev or test dataset)
            if manuf == "N/A" or manuf is None or doc_name == "N/A" or doc_name is None:
                logger.warning(
                    f"Doc not found for manuf {manufacturer} "
                    + f"and part {manuf_part_num}, skipping."
                )
                continue

            # Relevant data (i.e. attributes that appear in our gold labels)
            c_current_max = preprocess_c_current_max(collector_current_max)
            polarity = preprocess_polarity(transistor_type)
            ce_v_max = preprocess_ce_v_max(ce_voltage_max)
            """
            NOTE: Attributes we have that Mouser does not:
            dev_dissipation
            cb_v_max
            stg_temp_max
            stg_temp_min
            eb_v_max
            """

            # Other data (i.e. only appears in Mouser's CSVs, not in ours)
            min_op_temp = preprocess_individual_operating_temp(min_op_temp)
            max_op_temp = preprocess_individual_operating_temp(max_op_temp)

            # Map each attribute to its corresponding normalizer
            name_attr_norm = [  # TODO: Update normalizers
                # Data that both Mouser and our labels have:
                ("polarity", polarity, general_normalizer),
                ("c_current_max", c_current_max, general_normalizer),
                ("ce_v_max", ce_v_max, general_normalizer),
                # Data that Mouser has that we do not:
                ("min_op_temp", min_op_temp, general_normalizer),
                ("max_op_temp", max_op_temp, general_normalizer),
            ]

            # Output tuples of each normalized attribute
            for name, attr, normalizer in name_attr_norm:
                if "N/A" not in attr:
                    for a in attr.split(delim):
                        if len(a.strip()) > 0:
                            source_file = str(raw_gold_file.split("/")[-1])
                            output = [
                                doc_name,
                                manuf,
                                part_num,
                                name,
                                normalizer(a),
                                source_file,
                            ]
                            if tuple(output) not in seen:
                                writer.writerow(output)
                                seen.add(tuple(output))


if __name__ == "__main__":
    # Transform the transistor dataset
    filenames = "standard"

    # Change `mouser_csv_dir` to the absolute path where Mouser's raw CSVs are
    # located.
    mouser_csv_dir = "/home/nchiang/repos/hack/hack/transistors/data/src/mouser-csv/"

    # Change `formatted_gold` to the absolute path of where you want the
    # combined gold to be written.
    formatted_gold = (
        f"/home/nchiang/repos/hack/hack/transistors/data/{filenames}_mouser"
        + "_gold.csv"
    )
    seen = set()
    # Run transformation
    for i, filename in enumerate(sorted(os.listdir(mouser_csv_dir))):
        if filename.endswith(".csv"):
            raw_path = os.path.join(mouser_csv_dir, filename)
            logger.info(f"[INFO]: Parsing {raw_path}")
            if i == 0:  # create file on first iteration
                format_mouser_gold(raw_path, formatted_gold, seen, filenames=filenames)
            else:  # then just append to that same file
                format_mouser_gold(
                    raw_path, formatted_gold, seen, append=True, filenames=filenames
                )
