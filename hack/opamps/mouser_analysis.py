"""
This should read in two gold files (namely `data/mouser/mouser_gold.csv` and
`data/mouser/our_gold.csv`), score them against eachother, and write all
discrepancies to an output CSV.
"""
import logging
import os

from hack.opamps.analysis import print_score
from hack.opamps.opamp_utils import (
    compare_entities,
    entity_level_scores,
    get_gold_set,
    gold_set_to_dic,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main(is_gain=False):
    if is_gain:
        attribute = "gain"
    else:
        attribute = "current"

    dirname = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(dirname, f"analysis/{attribute}_mouser_discrepancies.csv")

    # Us
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
    # NOTE: We only care about the entity_dic for FN as they are the ones
    # where we want to know what Mouser does have for manual evalutation.
    # We already know what we have (as that was the FN that Mouser missed)
    # so all we care about is what Mouser actually does have for a doc.


if __name__ == "__main__":
    main()
