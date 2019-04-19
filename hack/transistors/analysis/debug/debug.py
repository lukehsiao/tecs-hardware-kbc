import csv
import logging
import os
import pdb

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_filenames_from_sheet(file):
    filenames = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            filenames.add(line[1].upper())
    return filenames


def get_discrepancy_dic(file):
    discrepancies = {}
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            (d_type, filename, d_part, d_val, d_notes, d_class, d_notes, _) = line
            if filename in discrepancies:
                discrepancies[filename].append(d_type)
            else:
                discrepancies[filename] = [d_type]
    return discrepancies


if __name__ == "__main__":
    dirname = os.path.dirname(__name__)

    old_file = os.path.join(dirname, "old_ce_v_max_gold.csv")
    old_filenames = get_filenames_from_sheet(old_file)
    logger.info(f"Old Google Sheet is {len(old_filenames)} filenames long.")

    new_file = os.path.join(dirname, "ce_v_max_gold.csv")
    new_filenames = get_filenames_from_sheet(new_file)
    logger.info(f"New Google Sheet is {len(new_filenames)} filenames long.")

    if len(old_filenames) != len(new_filenames):
        logger.info(
            f"New Google Sheet is missing {old_filenames.difference(new_filenames)}."
        )
        pdb.set_trace()

    old_discrepancies = get_discrepancy_dic(old_file)
    logger.info(f"Old Google Sheet has {len(old_discrepancies)} discrepancies.")

    new_discrepancies = get_discrepancy_dic(new_file)
    logger.info(f"New Google Sheet has {len(new_discrepancies)} discrepancies.")

    for key in old_discrepancies:
        try:
            if new_discrepancies[key] != old_discrepancies[key]:
                logger.info(
                    f"Sheets do not match at {key}. New has "
                    + f"{len(new_discrepancies[key])} discrepancies while "
                    + f"old has {len(old_discrepancies[key])} discrepancies."
                )
                pdb.set_trace()
        except KeyError:
            logger.warn(f"Filename {key} was not found in new sheet, skipping.")

        except Exception as e:
            logger.error(f"{e} while comparing dics.")
            pdb.set_trace()
