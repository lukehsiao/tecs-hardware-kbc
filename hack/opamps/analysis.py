"""
This script acts as a placeholder to set a threshold for cands >b
and compare those cands with our gold data.
"""
import csv
import logging
import os

import numpy as np
from quantiphy import Quantity
from tqdm import tqdm

from hack.opamps.data.utils.compare_gold import print_score
from hack.opamps.opamp_utils import (
    Score,
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_entity_set(file, b=0.0, is_gain=True):
    entities = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            try:
                (doc, val, score) = line
                if float(score) > b:
                    if is_gain:
                        val = f"{float(val)} kHz"
                    else:
                        val = f"{float(val)} uA"
                    entities.add((doc.upper(), Quantity(val)))
            except Exception as e:
                logger.error(f"{e} while getting entity set from {file}.")
    return entities


def get_filenames(entities):
    filenames = set()
    for (doc, val) in entities:
        filenames.add(doc)
    return filenames


def get_filenames_from_file(file):
    filenames = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            filenames.add(line[0].upper())
    return filenames


def filter_filenames(entities, filenames):
    result = set()
    for (doc, val) in entities:
        if doc in filenames:
            result.add((doc, val))
    if len(result) == 0:
        logger.error(
            f"Filtering for {len(get_filenames(entities))} "
            + "entity filenames turned up empty."
        )
    return result


def main(
    devfile="gain_dev_probs.csv",
    testfile="gain_test_probs.csv",
    outfile="gain_analysis_discrepancies.csv",
    is_gain=True,
):
    logger.info(f"Scoring analysis dataset for {outfile.split('_')[0]}...")

    # First, read in CSV and convert to entity set
    dirname = os.path.dirname(__name__)
    test_file = os.path.join(dirname, testfile)
    dev_file = os.path.join(dirname, devfile)
    filenames_file = os.path.join(dirname, "data/analysis/filenames.csv")
    discrepancy_file = os.path.join(dirname, outfile)
    logger.info(
        f"Analysis dataset is {len(get_filenames_from_file(filenames_file))}"
        + " filenames long."
    )

    # Only look at entities that occur in Digikey's gold as well
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    gold = filter_filenames(
        get_gold_set(gold=[gold_file], is_gain=is_gain),
        get_filenames_from_file(filenames_file),
    )
    logger.info(f"Original gold set is {len(get_filenames(gold))} filenames long.")

    logger.info(f"Determining best b...")
    best_score = Score(0, 0, 0, [], [], [])
    best_b = 0
    best_entities = set()
    # best_gold = set()

    for b in tqdm(np.linspace(0.0, 1, num=1000)):
        # Get implied parts as well from entities
        dev_entities = get_entity_set(dev_file, b=b, is_gain=is_gain)
        test_entities = get_entity_set(test_file, b=b, is_gain=is_gain)
        entities = filter_filenames(
            dev_entities.union(test_entities), get_filenames(gold)
        )

        # Trim gold to exactly the filenames in entities
        # trimmed_gold = filter_filenames(gold, get_filenames(entities))

        # Score entities against gold data and generate comparison CSV
        score = entity_level_scores(entities, is_gain=is_gain, metric=gold)
        # logger.info(f"b = {b}\n" + f"f1 = {score.f1}")
        if score.f1 > best_score.f1:
            best_score = score
            best_b = b
            best_entities = entities
            # best_gold = trimmed_gold

    logger.info(f"Entity set is {len(get_filenames(best_entities))} filenames long.")
    # logger.info(
    # f"Trimmed gold set is now {len(get_filenames(best_gold))} filenames long."
    # )
    print_score(best_score, entities=f"cands > {best_b}", metric="our gold labels")

    compare_entities(
        set(best_score.FP), is_gain=is_gain, type="FP", outfile=discrepancy_file
    )
    compare_entities(
        set(best_score.FN),
        is_gain=is_gain,
        type="FN",
        outfile=discrepancy_file,
        append=True,
        entity_dic=gold_set_to_dic(best_entities),
    )


if __name__ == "__main__":
    main()  # typ_gbp
    main(  # typ_supply_current
        devfile="current_dev_probs.csv",
        testfile="current_test_probs.csv",
        outfile="current_analysis_discrepancies.csv",
        is_gain=False,
    )
