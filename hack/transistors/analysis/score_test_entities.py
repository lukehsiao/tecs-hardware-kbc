import logging
import os

from hack.transistors.compare_cands import (
    filter_filenames,
    get_entity_set,
    get_filenames,
)
from hack.transistors.data.utils.compare_gold import print_score
from hack.transistors.transistor_utils import entity_level_scores, get_gold_set
from hack.transistors.transistors import Relation, load_parts_by_doc

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    relation = Relation.CE_V_MAX
    b = 0.99

    dirname = os.path.dirname(__name__)
    gold_file = os.path.join(dirname, "data/analysis/our_gold.csv")
    test_file = os.path.join(dirname, "ce_v_max_test_set_probs.csv")
    parts_by_doc = load_parts_by_doc()

    gold = get_gold_set(gold_file, attribute=relation.value)
    entities = filter_filenames(
        get_entity_set(test_file, parts_by_doc, b=b), get_filenames(gold)
    )
    gold = filter_filenames(gold, get_filenames(entities))

    score = entity_level_scores(entities, attribute=relation.value, metric=gold)
    print_score(score)
