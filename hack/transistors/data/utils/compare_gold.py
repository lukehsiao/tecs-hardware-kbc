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

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), f"comparison.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def scoring(entities, gold_set, attribute=Relation.CE_V_MAX.value):
    score = entity_level_scores(entities, attribute=attribute, gold_set=gold_set)

    logger.info("===================================================")
    logger.info(f"Scoring on Digikey gold labels against ours as metric")
    logger.info("===================================================")
    logger.info(f"Corpus Precision {score.prec:.3f}")
    logger.info(f"Corpus Recall    {score.rec:.3f}")
    logger.info(f"Corpus F1        {score.f1:.3f}")
    logger.info("---------------------------------------------------")
    logger.info(
        f"TP: {len(score.TP)} " f"| FP: {len(score.FP)} " f"| FN: {len(score.FN)}"
    )
    logger.info("===================================================\n")

    return score


def normalize_digikey_gold(digikey_gold, dirname=os.path.dirname(__name__)):
    # Remove source file from Digikey CSV (i.e. ffe00114_8.csv)
    temp_gold = os.path.join(dirname, "../analysis/temp_digikey_gold.csv")
    with open(temp_gold, "w") as temp:
        reader = csv.reader(open(digikey_gold, "r"))
        writer = csv.writer(temp)
        for line in reader:
            (doc, manuf, part, attr, val, source) = line
            # Get rid of units
            val = val.split(" ")[0]
            writer.writerow((doc, manuf, part, attr, val))
    return temp_gold


if __name__ == "__main__":
    # Compare our gold with Digikey's gold for the analysis set of 66 docs
    # (docs that both we and Digikey have gold labels for)
    # NOTE: We use our gold as gold for this comparison against Digikey

    attribute = Relation.CE_V_MAX.value

    dirname = os.path.dirname(__name__)
    outfile = os.path.join(dirname, "../analysis/gold_discrepancies.csv")

    # Use our gold labels as the gold metric (assume no errors)
    our_gold = os.path.join(dirname, "../analysis/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], attribute=attribute)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Score digikey's gold at entity level against our gold labels
    digikey_gold = normalize_digikey_gold(
        os.path.join(dirname, "../analysis/digikey_gold.csv")
    )
    entities = get_gold_set(gold=[digikey_gold], attribute=attribute)

    score = scoring(entities, gold_set=our_gold_set)

    # Run final comparison
    compare_entities(entities, gold_dic=our_gold_dic, outfile=outfile)

    # Remove temp formatted gold files
    # os.remove(digikey_gold)
