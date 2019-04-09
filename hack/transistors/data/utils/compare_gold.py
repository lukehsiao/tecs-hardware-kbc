"""
This should read in two gold files, score them against eachother,
and write all discrepancies to an output CSV.
"""

import csv
import logging
import os

from hack.transistors.transistor_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)
from hack.transistors.transistors import Relation

logger = logging.getLogger(__name__)


def print_score(score, entities=None, metric=None):
    logger.info("===================================================")
    logger.info(f"Scoring on {entities} using {metric} as metric")
    logger.info("===================================================")
    logger.info(f"Corpus Precision {score.prec:.3f}")
    logger.info(f"Corpus Recall    {score.rec:.3f}")
    logger.info(f"Corpus F1        {score.f1:.3f}")
    logger.info("---------------------------------------------------")
    logger.info(
        f"TP: {len(score.TP)} " f"| FP: {len(score.FP)} " f"| FN: {len(score.FN)}"
    )
    logger.info("===================================================\n")


def normalize_digikey_gold(digikey_gold, dirname=os.path.dirname(__name__)):
    """Remove source CSV file from Digikey CSV lines (e.g. ffe00114_8.csv)."""
    gold = os.path.join(dirname, "../analysis/digikey_gold.csv")
    with open(gold, "w") as temp:
        reader = csv.reader(open(digikey_gold, "r"))
        writer = csv.writer(temp)
        for line in reader:
            (doc, manuf, part, attr, val, source) = line
            # Get rid of units
            val = val.split(" ")[0].strip()
            writer.writerow((doc, manuf, part, attr, val))
    return gold


if __name__ == "__main__":

    # Compare our gold with Digikey's gold for the analysis set of 66 docs
    # (docs that both we and Digikey have gold labels for)
    # NOTE: We use our gold as gold for this comparison against Digikey
    attribute = Relation.CE_V_MAX.value
    dirname = os.path.dirname(__name__)
    outfile = os.path.join(dirname, "../analysis/gold_discrepancies.csv")

    # Us
    our_gold = os.path.join(dirname, "../analysis/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], attribute=attribute)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Digikey
    digikey_gold = normalize_digikey_gold(
        os.path.join(dirname, "../analysis/digikey_gold.csv")
    )
    digikey_gold_set = get_gold_set(gold=[digikey_gold], attribute=attribute)
    digikey_gold_dic = gold_set_to_dic(digikey_gold_set)

    # Score Digikey using our gold as metric
    score = entity_level_scores(
        digikey_gold_set, metric=our_gold_set, attribute=attribute
    )
    print_score(score, entities=digikey_gold, metric=our_gold)

    # Run final comparison using FN and FP
    compare_entities(
        set(score.FN),
        entity_dic=digikey_gold_dic,
        type="FN",
        gold_dic=our_gold_dic,
        outfile=outfile,
    )
    compare_entities(
        set(score.FP), type="FP", append=True, gold_dic=our_gold_dic, outfile=outfile
    )
