"""
This script acts as a placeholder to set a threshold for cands >b
and compare those cands with our gold data.
"""

import csv
import logging
import os

import numpy as np
from tqdm import tqdm

from hack.transistors.data.utils.compare_gold import print_score
from hack.transistors.transistor_utils import (
    Score,
    compare_entities,
    entity_level_scores,
    get_gold_set,
    get_implied_parts,
    gold_set_to_dic,
)
from hack.transistors.transistors import Relation, load_parts_by_doc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_entity_set(file, parts_by_doc, b=0.0):
    entities = set()
    errors = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            try:
                (doc, part, val, score) = line
                if float(score) > b:
                    # Add implied parts as well
                    for p in get_implied_parts(part, doc, parts_by_doc):
                        entities.add((doc, p, val))
            except KeyError:
                if doc not in errors:
                    logger.warning(f"{doc} was not found in parts_by_doc.")
                errors.add(doc)
                continue
            except Exception as e:
                logger.error(f"{e} while getting entity set from {file}.")
    return entities


def get_filenames(entities):
    filenames = set()
    for (doc, part, val) in entities:
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
    for (doc, part, val) in entities:
        if doc in filenames:
            result.add((doc, part, val))
    if len(result) == 0:
        logger.error(
            f"Filtering for {len(get_filenames(entities))} "
            + "entity filenames turned up empty."
        )
    return result


def main(relation=Relation.CE_V_MAX, outfile="discrepancies.csv"):

    # First, read in CSV and convert to entity set
    dirname = os.path.dirname(__name__)
    test_file = os.path.join(dirname, "ce_v_max_test_set_probs.csv")
    dev_file = os.path.join(dirname, "ce_v_max_dev_set_probs.csv")
    filenames_file = os.path.join(dirname, "data/analysis/filenames.csv")
    discrepancy_file = os.path.join(dirname, outfile)
    logger.info(
        f"Analysis dataset is {len(get_filenames_from_file(filenames_file))}"
        + " filenames long."
    )

    # Only look at entities that occur in Digikey's gold as well
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    gold = filter_filenames(
        get_gold_set(gold_file, attribute=relation.value),
        get_filenames_from_file(filenames_file),
    )
    logger.info(f"Original gold set is {len(get_filenames(gold))} filenames long.")

    logger.info(f"Determining best b...")
    best_score = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in tqdm(np.linspace(0.5, 1, num=1000)):
        # Get implied parts as well from entities
        parts_by_doc = load_parts_by_doc()
        dev_entities = get_entity_set(dev_file, parts_by_doc, b=b)
        test_entities = get_entity_set(test_file, parts_by_doc, b=b)
        entities = filter_filenames(
            dev_entities.union(test_entities), get_filenames(gold)
        )
        entity_dic = gold_set_to_dic(entities)

        # Trim gold to exactly the filenames in entities
        gold = filter_filenames(gold, get_filenames(entities))

        # Score entities against gold data and generate comparison CSV
        score = entity_level_scores(entities, attribute=relation.value, metric=gold)
        logger.info(f"b = {b}\n" + f"f1 = {score.f1}")
        if score.f1 > best_score.f1:
            best_score = score
            best_b = b
            best_entities = entities
            best_gold = gold

    logger.info(f"Entity set is {len(get_filenames(best_entities))} filenames long.")
    logger.info(
        f"Trimmed gold set is now {len(get_filenames(best_gold))} filenames long."
    )
    print_score(best_score, entities=f"cands > {best_b}", metric="our gold labels")

    compare_entities(
        set(best_score.FP),
        attribute=relation.value,
        type="FP",
        outfile=discrepancy_file,
    )
    compare_entities(
        set(best_score.FN),
        attribute=relation.value,
        type="FN",
        outfile=discrepancy_file,
        append=True,
        entity_dic=entity_dic,
    )


if __name__ == "__main__":
    outfile = "data_discrepancies.csv"
    main(outfile=outfile)
