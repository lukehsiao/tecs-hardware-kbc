"""
NOTE: This script was optimized solely for the transistor comparison
using the `Analysis` dataset.
"""

import csv
import logging
import os
import pdb
import pickle
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.learning import SparseLogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal import analysis
from metal.label_model import LabelModel

from hack.transistors.transistor_lfs import (
    TRUE,
    ce_v_max_lfs,
    op_temp_max_lfs,
    op_temp_min_lfs,
    polarity_lfs,
    stg_temp_max_lfs,
    stg_temp_min_lfs,
)
from hack.transistors.transistor_matchers import get_matcher
from hack.transistors.transistor_spaces import (
    MentionNgramsPart,
    MentionNgramsTemp,
    MentionNgramsVolt,
)
from hack.transistors.transistor_throttlers import (
    ce_v_max_filter,
    op_temp_filter,
    polarity_filter,
    stg_temp_filter,
)
from hack.transistors.transistor_utils import (
    Score,
    digikey_entity_level_scores,
    entity_level_scores,
    load_transistor_labels,
)
from hack.utils import parse_dataset

# Use the first set of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), f"transistors.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Enum for tracking
class Relation(Enum):
    STG_TEMP_MIN = "stg_temp_min"
    STG_TEMP_MAX = "stg_temp_max"
    OP_TEMP_MIN = "op_temp_min"
    OP_TEMP_MAX = "op_temp_max"
    POLARITY = "polarity"
    CE_V_MAX = "ce_v_max"


def parsing(session, first_time=True, parallel=1, max_docs=float("inf")):
    dirname = os.path.dirname(__file__)
    logger.debug(f"Starting parsing...")
    docs, train_docs, dev_docs, test_docs, analysis_docs = parse_dataset(
        session, dirname, first_time=first_time, parallel=parallel, max_docs=max_docs
    )
    logger.debug(f"Done")

    logger.info(f"# of train Documents: {len(train_docs)}")
    logger.info(f"# of dev Documents: {len(dev_docs)}")
    logger.info(f"# of test Documents: {len(test_docs)}")
    logger.info(f"# of analysis Documents: {len(analysis_docs)}")

    logger.info(f"Documents: {session.query(Document).count()}")
    logger.info(f"Sections: {session.query(Section).count()}")
    logger.info(f"Paragraphs: {session.query(Paragraph).count()}")
    logger.info(f"Sentences: {session.query(Sentence).count()}")
    logger.info(f"Figures: {session.query(Figure).count()}")

    return docs, train_docs, dev_docs, test_docs, analysis_docs


def mention_extraction(session, relation, docs, first_time=True, parallel=1):
    Part = mention_subclass("Part")
    part_matcher = get_matcher("part")
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    if relation == Relation.STG_TEMP_MIN:
        Attr = mention_subclass("StgTempMin")
        attr_matcher = get_matcher("stg_temp_min")
        attr_ngrams = MentionNgramsTemp(n_max=2)
    elif relation == Relation.STG_TEMP_MAX:
        Attr = mention_subclass("StgTempMax")
        attr_matcher = get_matcher("stg_temp_max")
        attr_ngrams = MentionNgramsTemp(n_max=2)
    elif relation == Relation.OP_TEMP_MIN:
        Attr = mention_subclass("OpTempMin")
        attr_matcher = get_matcher("op_temp_min")
        attr_ngrams = MentionNgramsTemp(n_max=2)
    elif relation == Relation.OP_TEMP_MAX:
        Attr = mention_subclass("OpTempMax")
        attr_matcher = get_matcher("op_temp_max")
        attr_ngrams = MentionNgramsTemp(n_max=2)
    elif relation == Relation.POLARITY:
        Attr = mention_subclass("Polarity")
        attr_matcher = get_matcher("polarity")
        attr_ngrams = MentionNgrams(n_max=1)
    elif relation == Relation.CE_V_MAX:
        Attr = mention_subclass("CeVMax")
        attr_matcher = get_matcher("ce_v_max")
        attr_ngrams = MentionNgramsVolt(n_max=1)
    else:
        raise ValueError(f"Invalid Relation: {relation}")

    mention_extractor = MentionExtractor(
        session, [Part, Attr], [part_ngrams, attr_ngrams], [part_matcher, attr_matcher]
    )

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")
    logger.info(f"Total Part: {session.query(Part).count()}")
    logger.info(f"Total Attr: {session.query(Attr).count()}")
    return Part, Attr


def candidate_extraction(
    session,
    relation,
    part,
    attr,
    train_docs,
    dev_docs,
    test_docs,
    analysis_docs,
    first_time=True,
    parallel=1,
):
    if relation == Relation.STG_TEMP_MIN:
        Cand = candidate_subclass("PartStgTempMin", [part, attr])
        throttler = stg_temp_filter
    elif relation == Relation.STG_TEMP_MAX:
        Cand = candidate_subclass("PartStgTempMax", [part, attr])
        throttler = stg_temp_filter
    elif relation == Relation.OP_TEMP_MIN:
        Cand = candidate_subclass("PartOpTempMin", [part, attr])
        throttler = op_temp_filter
    elif relation == Relation.OP_TEMP_MAX:
        Cand = candidate_subclass("PartOpTempMax", [part, attr])
        throttler = op_temp_filter
    elif relation == Relation.POLARITY:
        Cand = candidate_subclass("PartPolarity", [part, attr])
        throttler = polarity_filter
    elif relation == Relation.CE_V_MAX:
        Cand = candidate_subclass("PartCeVMax", [part, attr])
        throttler = ce_v_max_filter
    else:
        raise ValueError(f"Invalid Relation: {relation}")

    candidate_extractor = CandidateExtractor(session, [Cand], throttlers=[throttler])

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs, analysis_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)
            logger.info(
                f"Cand in split={i}: "
                f"{session.query(Cand).filter(Cand.split == i).count()}"
            )

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)
    analysis_cands = candidate_extractor.get_candidates(split=3)

    logger.info(f"Total train candidate: {len(train_cands[0])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0])}")
    logger.info(f"Total test candidate: {len(test_cands[0])}")
    logger.info(f"Total analysis candidate: {len(analysis_cands[0])}")

    return (Cand, train_cands, dev_cands, test_cands, analysis_cands)


def featurization(
    session,
    train_cands,
    dev_cands,
    test_cands,
    analysis_cands,
    Cand,
    first_time=True,
    parallel=1,
):
    dirname = os.path.dirname(__file__)
    featurizer = Featurizer(session, [Cand])
    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        featurizer.apply(split=3, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    # Serialize feature matrices on first run
    if first_time:
        F_train = featurizer.get_feature_matrices(train_cands)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        F_test = featurizer.get_feature_matrices(test_cands)
        F_analysis = featurizer.get_feature_matrices(analysis_cands)
        pickle.dump(F_train, open(os.path.join(dirname, "F_train.pkl"), "wb"))
        pickle.dump(F_dev, open(os.path.join(dirname, "F_dev.pkl"), "wb"))
        pickle.dump(F_test, open(os.path.join(dirname, "F_test.pkl"), "wb"))
        pickle.dump(F_analysis, open(os.path.join(dirname, "F_analysis.pkl"), "wb"))
    else:
        F_train = pickle.load(open(os.path.join(dirname, "F_train.pkl"), "rb"))
        F_dev = pickle.load(open(os.path.join(dirname, "F_dev.pkl"), "rb"))
        F_test = pickle.load(open(os.path.join(dirname, "F_test.pkl"), "rb"))
        F_analysis = pickle.load(open(os.path.join(dirname, "F_analysis.pkl"), "rb"))
    logger.info("Done.")

    logger.info(f"Train shape: {F_train[0].shape}")
    logger.info(f"Test shape: {F_test[0].shape}")
    logger.info(f"Dev shape: {F_dev[0].shape}")
    logger.info(f"Analysis shape: {F_analysis[0].shape}")

    return F_train, F_dev, F_test, F_analysis


def load_labels(session, relation, cand, first_time=True):
    if first_time:
        logger.info(f"Loading gold labels for {relation.value}")
        load_transistor_labels(session, [cand], [relation.value], annotator_name="gold")


def labeling(session, cands, cand, split=1, train=False, first_time=True, parallel=1):
    labeler = Labeler(session, [cand])
    if relation == Relation.STG_TEMP_MIN:
        lfs = stg_temp_min_lfs
    elif relation == Relation.STG_TEMP_MAX:
        lfs = stg_temp_max_lfs
    elif relation == Relation.OP_TEMP_MIN:
        lfs = op_temp_min_lfs
    elif relation == Relation.OP_TEMP_MAX:
        lfs = op_temp_max_lfs
    elif relation == Relation.POLARITY:
        lfs = polarity_lfs
    elif relation == Relation.CE_V_MAX:
        lfs = ce_v_max_lfs
    else:
        raise ValueError(f"Invalid Relation: {relation}")

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=split, lfs=[lfs], train=train, parallelism=parallel)
        logger.info("Done...")

    logger.info("Getting label matrices...")
    L_mat = labeler.get_label_matrices(cands)
    L_gold = labeler.get_gold_labels(cands, annotator="gold")
    logger.info("Done.")
    logger.info(f"L_mat shape: {L_mat[0].shape}")
    logger.info(f"L_gold shape: {L_gold[0].shape}")

    if train:
        try:
            df = analysis.lf_summary(
                L_mat[0],
                lf_names=labeler.get_keys(),
                Y=L_gold[0].todense().reshape(-1).tolist()[0],
            )
            logger.info(f"\n{df.to_string()}")
        except Exception:
            pdb.set_trace()

    return L_mat, L_gold


def generative_model(relation, L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info("Training generative model...")
    model.train_model(L_train[0], n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train[0])
    plt.hist(marginals[:, TRUE - 1], bins=20)
    plt.savefig(f"{relation.value}_marginals.pdf")
    return marginals


def discriminative_model(train_cands, F_train, marginals, n_epochs=50, lr=0.001):
    disc_model = SparseLogisticRegression()

    logger.info("Training discriminative model...")
    disc_model.train(
        (train_cands[0], F_train[0]),
        marginals,
        n_epochs=n_epochs,
        lr=lr,
        host_device="GPU",
    )
    logger.info("Done.")

    return disc_model


def load_parts_by_doc():
    dirname = os.path.dirname(__file__)
    pickle_file = os.path.join(dirname, "data/parts_by_doc_dict.pkl")
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


def scoring(
    relation, disc_model, test_cands, test_docs, F_test, parts_by_doc=None, num=100
):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Iterate over a range of `b` values in order to find the b with the
    # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
    Y_prob = disc_model.marginals((test_cands[0], F_test[0]))

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            test_score = np.array(
                [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
            )
            true_pred = [
                test_cands[0][_] for _ in np.nditer(np.where(test_score == TRUE))
            ]
            result = entity_level_scores(
                true_pred,
                attribute=relation.value,
                corpus=test_docs,
                parts_by_doc=parts_by_doc,
            )
            logger.info(f"b = {b}, f1 = {result.f1}")
            if result.f1 > best_result.f1:
                best_result = result
                best_b = b
                best_true_pred = true_pred
        except Exception as e:
            logger.debug(f"{e}, skipping.")
            break

    logger.info("===================================================")
    logger.info(f"Scoring on Entity-Level Gold Data with b={best_b}")
    logger.info("===================================================")
    logger.info(f"Corpus Precision {best_result.prec:.3f}")
    logger.info(f"Corpus Recall    {best_result.rec:.3f}")
    logger.info(f"Corpus F1        {best_result.f1:.3f}")
    logger.info("---------------------------------------------------")
    logger.info(
        f"TP: {len(best_result.TP)} "
        f"| FP: {len(best_result.FP)} "
        f"| FN: {len(best_result.FN)}"
    )
    logger.info("===================================================\n")
    return best_result, best_b, best_true_pred


"""
Digikey comparision:
"""


def digikey_scoring(
    relation,
    disc_model,
    analysis_cands,
    analysis_docs,
    F_analysis,
    gold="digikey",
    parts_by_doc=None,
    num=100,
    debug=False,
    outfile="data/digikey_discrepancies.csv",
):
    if gold == "digikey":  # Use Digikey as gold
        logger.info("Calculating the best Digikey based F1 score and threshold (b)...")

        # Iterate over a range of `b` values in order to find the b with the
        # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
        Y_prob = disc_model.marginals((analysis_cands[0], F_analysis[0]))

        # Get prediction for a particular b, store the full tuple to output
        # (b, pref, rec, f1, TP, FP, FN)
        best_result = Score(0, 0, 0, [], [], [])
        best_b = 0
        for b in np.linspace(0, 1, num=num):
            try:
                analysis_score = np.array(
                    [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
                )
                true_pred = [
                    analysis_cands[0][_]
                    for _ in np.nditer(np.where(analysis_score == TRUE))
                ]
                (_, result) = digikey_entity_level_scores(
                    true_pred,
                    attribute=relation.value,
                    corpus=analysis_docs,
                    parts_by_doc=parts_by_doc,
                )
                logger.info(f"b = {b}, f1 = {result.f1}")
                if result.f1 > best_result.f1:
                    best_result = result
                    best_b = b
                    best_true_pred = true_pred
            except Exception as e:
                logger.debug(f"{e}, skipping.")
                break

    elif gold == "ours":  # Use our gold labels for classification
        logger.info("Calcularing the best F1 score and threshold (b)...")

        # Iterate over a range of `b` values in order to find the b with the
        # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
        Y_prob = disc_model.marginals((analysis_cands[0], F_analysis[0]))

        # Get prediction for a particular b, store the full tuple to output
        # (b, pref, rec, f1, TP, FP, FN)
        best_result = Score(0, 0, 0, [], [], [])
        best_b = 0
        for b in np.linspace(0, 1, num=num):
            try:
                analysis_score = np.array(
                    [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
                )
                true_pred = [
                    analysis_cands[0][_]
                    for _ in np.nditer(np.where(analysis_score == TRUE))
                ]
                result = entity_level_scores(
                    true_pred,
                    attribute=relation.value,
                    corpus=analysis_docs,
                    parts_by_doc=parts_by_doc,
                )
                logger.info(f"b = {b}, f1 = {result.f1}")
                if result.f1 > best_result.f1:
                    best_result = result
                    best_b = b
                    best_true_pred = true_pred
            except Exception as e:
                logger.debug(f"{e}, skipping.")
                break
    else:
        logger.error("Invalid gold type {gold}")
        pdb.set_trace()

    # Compare results with Digikey gold
    logger.info("Comparing final results with Digikey...")
    (gold_dic, end_result) = digikey_entity_level_scores(
        best_true_pred,
        attribute=relation.value,
        corpus=analysis_docs,
        parts_by_doc=parts_by_doc,
    )

    logger.info("=================================================================")
    logger.info(f"Digikey based scoring on Entity-Level Gold Data with b={best_b}")
    logger.info("=================================================================")
    logger.info(f"Corpus Precision {end_result.prec:.3f}")
    logger.info(f"Corpus Recall    {end_result.rec:.3f}")
    logger.info(f"Corpus F1        {end_result.f1:.3f}")
    logger.info("-----------------------------------------------------------------")
    logger.info(
        f"TP: {len(end_result.TP)} "
        f"| FP: {len(end_result.FP)} "
        f"| FN: {len(end_result.FN)}"
    )
    logger.info("=================================================================\n")

    if debug:
        # Write discrepancies (false pos and false neg) to a CSV file
        # for manual debugging
        with open(outfile, "w") as out:
            writer = csv.writer(out)
            writer.writerow(("Type:", "Filename:", "Part:", "Our Vals:", "Notes:"))
            for (doc, part, val) in best_result.FP:
                if doc in gold_dic:
                    if part in gold_dic[doc]:
                        writer.writerow(
                            (
                                "FP",
                                doc,
                                part,
                                val,
                                "Digikey vals: " + str(gold_dic[doc][part]),
                            )
                        )
                    else:
                        writer.writerow(
                            (
                                "FP",
                                doc,
                                part,
                                val,
                                "Digikey parts: " + str(gold_dic[doc]),
                            )
                        )
                else:
                    writer.writerow(
                        ("FP", doc, part, val, "Digikey does not have doc.")
                    )

            for (doc, part, val) in best_result.FN:
                if doc in gold_dic:
                    if part in gold_dic[doc]:
                        writer.writerow(
                            (
                                "FN",
                                doc,
                                part,
                                val,
                                "Digikey vals: " + str(gold_dic[doc][part]),
                            )
                        )
                    else:
                        writer.writerow(
                            (
                                "FN",
                                doc,
                                part,
                                val,
                                "Digikey parts: " + str(gold_dic[doc]),
                            )
                        )
                else:
                    writer.writerow(
                        ("FN", doc, part, val, "Digikey does not have doc.")
                    )

    return best_result, best_b


def main(
    conn_string,
    max_docs=float("inf"),
    first_time=True,
    parallel=2,
    relation=Relation.STG_TEMP_MAX,
):
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs, analysis_docs = parsing(
        session, first_time=first_time, parallel=parallel, max_docs=max_docs
    )

    (Part, Attr) = mention_extraction(
        session, relation, docs, first_time=first_time, parallel=parallel
    )

    (Cand, train_cands, dev_cands, test_cands, analysis_cands) = candidate_extraction(
        session,
        relation,
        Part,
        Attr,
        train_docs,
        dev_docs,
        test_docs,
        analysis_docs,
        first_time=first_time,
        parallel=parallel,
    )
    F_train, F_dev, F_test, F_analysis = featurization(
        session,
        train_cands,
        dev_cands,
        test_cands,
        analysis_cands,
        Cand,
        first_time=first_time,
        parallel=parallel,
    )
    load_labels(session, relation, Cand, first_time=first_time)
    logger.info("Labeling training data...")
    L_train, L_gold_train = labeling(
        session,
        train_cands,
        Cand,
        split=0,
        train=True,
        parallel=parallel,
        first_time=first_time,
    )
    logger.info("Done.")

    marginals = generative_model(relation, L_train)

    # L_dev, L_gold_dev = labeling(
    # session,
    # dev_cands,
    # Cand,
    # split=1,
    # train=True,
    # parallel=parallel,
    # first_time=first_time,
    # )

    # L_test, L_gold_test = labeling(
    # session,
    # test_cands,
    # Cand,
    # split=2,
    # train=True,
    # parallel=parallel,
    # first_time=first_fime,
    # )

    disc_models = discriminative_model(train_cands, F_train, marginals, n_epochs=10)

    parts_by_doc = load_parts_by_doc()
    best_result, best_b, best_true_pred = scoring(
        relation,
        disc_models,
        test_cands,
        test_docs,
        F_test,
        parts_by_doc=parts_by_doc,
        num=100,
    )

    """
    Digikey comparison:
    """

    digikey_best_result, digikey_best_b = digikey_scoring(
        relation,
        disc_models,
        analysis_cands,
        analysis_docs,
        F_analysis,
        parts_by_doc=parts_by_doc,
        debug=True,
        num=66,  # NOTE: As of right now we only have 66 docs in the analysis dataset.
        # TODO: Annotate more of train so as to have at least
        # 100 docs in the analysis dataset.
    )


if __name__ == "__main__":
    # See https://docs.python.org/3/library/os.html#os.cpu_count
    parallel = 8  # Change parallel for watchog
    component = "our_gold_large_transistor_analysis"
    conn_string = f"postgresql://localhost:5432/{component}"
    first_time = False
    relation = Relation.CE_V_MAX
    logger.info(f"\n\n")
    logger.info(f"=" * 80)
    logger.info(f"Beginning {component}::{relation.value} with parallel: {parallel}")

    main(
        conn_string,
        max_docs=3000,
        relation=relation,
        first_time=first_time,
        parallel=parallel,
    )
