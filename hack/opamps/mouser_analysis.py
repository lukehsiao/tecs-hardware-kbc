"""
This should read in two gold files (namely `data/mouser/mouser_gold.csv` and
`data/mouser/our_gold.csv`), score them against eachother, and write all
discrepancies to an output CSV.
"""
import logging
import os

import numpy as np
from tqdm import tqdm

from hack.opamps.analysis import get_entity_set, print_score
from hack.opamps.opamp_utils import (
    Score,
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def compare_fonduer(is_gain=False, num=100):
    """
    Compare the candidates extracted by Fonduer with our ground truth labels and
    write any discrepancies to `analysis/{attribute}_fonduer_discrepancies.csv`
    for manual debugging.
    """
    if is_gain:
        attribute = "gain"
    else:
        attribute = "current"

    dirname = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(dirname, f"analysis/{attribute}_fonduer_discrepancies.csv")

    # Ground truth
    our_gold = os.path.join(dirname, "data/mouser/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], is_gain=is_gain)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Fonduer
    test_file = os.path.join(dirname, "current_test_probs.csv")

    best_score = Score(0, 0, 0, [], [], [])
    best_b = 0
    best_entities = set()

    for b in tqdm(np.linspace(0.0, 1, num=num)):
        entities = get_entity_set(test_file, b=b, is_gain=is_gain)
        score = entity_level_scores(entities, is_gain=is_gain, metric=our_gold_set)

        if score.f1 > best_score.f1:
            best_score = score
            best_b = b
            best_entities = entities

    print_score(
        best_score,
        description=f"Scoring on extracted cands > {best_b:.3f} "
        + f"against {our_gold.split('/')[-1]}.",
    )

    # Run the final comparison using FN and FP
    probs = get_entity_set(test_file, b=0.0, is_gain=is_gain)
    compare_entities(
        set(best_score.FN),
        type="FN",
        probs=probs,
        gold_dic=our_gold_dic,
        outfile=outfile,
        entity_dic=gold_set_to_dic(best_entities),
    )
    compare_entities(
        set(best_score.FP),
        type="FP",
        probs=probs,
        gold_dic=our_gold_dic,
        outfile=outfile,
        entity_dic=gold_set_to_dic(best_entities),
        append=True,
    )


def compare_mouser(is_gain=False):
    """
    Compare Mouser's gold labels with our ground truth gold labels and write any
    discrepancies to `analysis/{attribute}_mouser_discrepancies.csv` for manual
    debugging.
    """
    if is_gain:
        attribute = "gain"
    else:
        attribute = "current"

    dirname = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(dirname, f"analysis/{attribute}_mouser_discrepancies.csv")

    # Ground truth
    our_gold = os.path.join(dirname, "data/mouser/our_gold.csv")
    our_gold_set = get_gold_set(gold=[our_gold], is_gain=is_gain)
    our_gold_dic = gold_set_to_dic(our_gold_set)

    # Mouser
    mouser_gold = os.path.join(dirname, "data/mouser/mouser_gold.csv")
    mouser_gold_set = get_gold_set(gold=[mouser_gold], is_gain=is_gain)
    mouser_gold_dic = gold_set_to_dic(mouser_gold_set)

    # Score Mouser using our gold as metric
    score = entity_level_scores(mouser_gold_set, metric=our_gold_set, is_gain=is_gain)
    # logger.info(f"Scores for {attribute}...")
    print_score(
        score,
        description=f"Scoring on {mouser_gold.split('/')[-1]} "
        + f"against {our_gold.split('/')[-1]}.",
    )

    # Run final comparison using FN and FP
    compare_entities(
        set(score.FN),
        entity_dic=mouser_gold_dic,
        type="FN",
        gold_dic=our_gold_dic,
        outfile=outfile,
    )
    compare_entities(
        set(score.FP), type="FP", append=True, gold_dic=our_gold_dic, outfile=outfile
    )


if __name__ == "__main__":
    compare_mouser()
    compare_fonduer()
