"""
This script acts as a placeholder to set a threshold for cands >b
and compare those cands with our gold data.
"""

import csv
import logging
import os
import pdb

from hack.transistors.data.utils.compare_gold import print_score
from hack.transistors.transistor_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)
from hack.transistors.transistors import Relation

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_entity_set(file, b=0.5):
    entities = set()
    with open(file, "r") as input:
        reader = csv.reader(input)
        for line in reader:
            try:
                (doc, part, val, score) = line
                if float(score) > b:
                    entities.add((doc, part, val))
            except Exception as e:
                logger.error(f"{e} while getting entity set from {file}.")
                pdb.set_trace()
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
    return result


def valid_entities(docs, dev_docs, test_docs, dirname=os.path.dirname(__name__)):
    # Check to ensure that docs is made up of dev_docs and test_docs
    expected_docs = set()
    for filename in dev_docs:
        expected_docs.add(filename)
    for filename in test_docs:
        expected_docs.add(filename)

    if sorted(list(docs)) != sorted(list(expected_docs)):
        logger.error(f"Full docs is not made up of solely dev and test docs.")
        pdb.set_trace()

    # Check that dev_docs and test_docs are in their respective dirs.
    dev_gold = os.path.join(dirname, "data/dev/dev_gold.csv")
    test_gold = os.path.join(dirname, "data/test/test_gold.csv")

    expected_dev = get_filenames(get_gold_set(dev_gold))
    expected_test = get_filenames(get_gold_set(test_gold))

    difference_dev = set()
    for filename in dev_docs:
        if filename not in expected_dev:
            difference_dev.add(filename)

    difference_test = set()
    for filename in test_docs:
        if filename not in expected_test:
            difference_test.add(filename)

    if len(difference_test) > 0 and len(difference_dev) > 0:
        logger.error(
            f"Both dev and test sets do not match with their respective gold data."
        )
        pdb.set_trace()
    elif len(difference_test) > 0:
        logger.error(f"Test set does not match with test gold.")
        pdb.set_trace()
    elif len(difference_dev) > 0:
        logger.error(f"Dev test set does not match with dev gold.")
        pdb.set_trace()
    else:
        return True


def main(relation=Relation.CE_V_MAX, b=0.5, outfile="discrepancies.csv"):

    # First, read in CSV and convert to entity set
    dirname = os.path.dirname(__name__)
    test_file = os.path.join(dirname, "ce_v_max_test_set_probs.csv")
    dev_file = os.path.join(dirname, "ce_v_max_dev_set_probs.csv")
    filenames_file = os.path.join(dirname, "data/analysis/filenames.csv")
    discrepancy_file = os.path.join(dirname, outfile)

    # Only look at entities that occur in Digikey's gold as well
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    gold = filter_filenames(
        get_gold_set(gold_file), get_filenames_from_file(filenames_file)
    )

    dev_entities = get_entity_set(dev_file, b=b)
    test_entities = get_entity_set(test_file, b=b)
    entities = filter_filenames(dev_entities.union(test_entities), gold)
    entity_dic = gold_set_to_dic(entities)

    # Score entities against gold data and generate comparison CSV
    score = entity_level_scores(entities, attribute=relation.value, metric=gold)
    print_score(score, entities=f"cands > {b}", metric="our gold labels")

    compare_entities(
        set(score.FP), attribute=relation.value, type="FP", outfile=discrepancy_file
    )
    compare_entities(
        set(score.FN),
        attribute=relation.value,
        type="FN",
        outfile=discrepancy_file,
        append=True,
        entity_dic=entity_dic,
    )


if __name__ == "__main__":
    outfile = "data_discrepancies.csv"
    main(outfile=outfile)
