#!/usr/bin/env python
# coding: utf-8

import csv
import logging
import os
import pickle

import numpy as np
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.features import Featurizer
from fonduer.learning import SparseLogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal.label_model import LabelModel

from hack.transistors.transistor_lfs import (
    TRUE,
    ce_v_max_lfs,
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
    polarity_filter,
    stg_temp_filter,
)
from hack.transistors.transistor_utils import (
    Score,
    cand_to_entity,
    candidates_to_entities,
    entity_level_scores,
    load_transistor_labels,
)
from hack.utils import parse_dataset

# Use the first set of GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def parsing(session, first_time=True, parallel=1, max_docs=float("inf")):
    dirname = os.path.abspath("")
    logger.debug(f"Starting parsing...")
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=first_time, parallel=parallel, max_docs=max_docs
    )
    logger.debug(f"Done")

    logger.info(f"# of train Documents: {len(train_docs)}")
    logger.info(f"# of dev Documents: {len(dev_docs)}")
    logger.info(f"# of test Documents: {len(test_docs)}")

    logger.info(f"Documents: {session.query(Document).count()}")
    logger.info(f"Sections: {session.query(Section).count()}")
    logger.info(f"Paragraphs: {session.query(Paragraph).count()}")
    logger.info(f"Sentences: {session.query(Sentence).count()}")
    logger.info(f"Figures: {session.query(Figure).count()}")

    return docs, train_docs, dev_docs, test_docs


def mention_extraction(
    session,
    docs,
    first_time=True,
    part=True,
    stg_temp_min=True,
    stg_temp_max=True,
    polarity=True,
    ce_v_max=True,
    parallel=1,
):
    Part = mention_subclass("Part")
    part_matcher = get_matcher("part")
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)

    StgTempMin = mention_subclass("StgTempMin")
    stg_temp_min_matcher = get_matcher("stg_temp_min")
    stg_temp_min_ngrams = MentionNgramsTemp(n_max=2)

    StgTempMax = mention_subclass("StgTempMax")
    stg_temp_max_matcher = get_matcher("stg_temp_max")
    stg_temp_max_ngrams = MentionNgramsTemp(n_max=2)

    Polarity = mention_subclass("Polarity")
    polarity_matcher = get_matcher("polarity")
    polarity_ngrams = MentionNgrams(n_max=1)

    CeVMax = mention_subclass("CeVMax")
    ce_v_max_matcher = get_matcher("ce_v_max")
    ce_v_max_ngrams = MentionNgramsVolt(n_max=1)

    mentions = []
    ngrams = []
    matchers = []

    # Only do those that are incrementally enabled
    if part:
        mentions.append(Part)
        ngrams.append(part_ngrams)
        matchers.append(part_matcher)

    if stg_temp_min:
        mentions.append(StgTempMin)
        ngrams.append(stg_temp_min_ngrams)
        matchers.append(stg_temp_min_matcher)

    if stg_temp_max:
        mentions.append(StgTempMax)
        ngrams.append(stg_temp_max_ngrams)
        matchers.append(stg_temp_max_matcher)

    if polarity:
        mentions.append(Polarity)
        ngrams.append(polarity_ngrams)
        matchers.append(polarity_matcher)

    if ce_v_max:
        mentions.append(CeVMax)
        ngrams.append(ce_v_max_ngrams)
        matchers.append(ce_v_max_matcher)

    mention_extractor = MentionExtractor(session, mentions, ngrams, matchers)

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")
    logger.info(f"Total Part: {session.query(Part).count()}")
    logger.info(f"Total StgTempMin: {session.query(StgTempMin).count()}")
    logger.info(f"Total StgTempMax: {session.query(StgTempMax).count()}")
    logger.info(f"Total Polarity: {session.query(Polarity).count()}")
    logger.info(f"Total CeVMax: {session.query(CeVMax).count()}")
    return Part, StgTempMin, StgTempMax, Polarity, CeVMax


def candidate_extraction(
    session,
    Part,
    StgTempMin,
    StgTempMax,
    Polarity,
    CeVMax,
    train_docs,
    dev_docs,
    test_docs,
    stg_temp_min=True,
    stg_temp_max=True,
    polarity=True,
    ce_v_max=True,
    first_time=True,
    parallel=1,
):
    PartStgTempMin = candidate_subclass("PartStgTempMin", [Part, StgTempMin])
    stg_temp_min_throttler = stg_temp_filter

    PartStgTempMax = candidate_subclass("PartStgTempMax", [Part, StgTempMax])
    stg_temp_max_throttler = stg_temp_filter

    PartPolarity = candidate_subclass("PartPolarity", [Part, Polarity])
    polarity_throttler = polarity_filter

    PartCeVMax = candidate_subclass("PartCeVMax", [Part, CeVMax])
    ce_v_max_throttler = ce_v_max_filter

    cands = []
    throttlers = []
    if stg_temp_min:
        cands.append(PartStgTempMin)
        throttlers.append(stg_temp_min_throttler)

    if stg_temp_max:
        cands.append(PartStgTempMax)
        throttlers.append(stg_temp_max_throttler)

    if polarity:
        cands.append(PartPolarity)
        throttlers.append(polarity_throttler)

    if ce_v_max:
        cands.append(PartCeVMax)
        throttlers.append(ce_v_max_throttler)

    candidate_extractor = CandidateExtractor(session, cands, throttlers=throttlers)

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)
            num_cands = session.query(Candidate).filter(Candidate.split == i).count()
            logger.info(f"Candidates in split={i}: {num_cands}")

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)

    logger.info(f"Total train candidate: {len(train_cands[0])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0])}")
    logger.info(f"Total test candidate: {len(test_cands[0])}")

    return (
        PartStgTempMin,
        PartStgTempMax,
        PartPolarity,
        PartCeVMax,
        train_cands,
        dev_cands,
        test_cands,
    )


def featurization(
    session,
    train_cands,
    dev_cands,
    test_cands,
    PartStgTempMin,
    PartStgTempMax,
    PartPolarity,
    PartCeVMax,
    stg_temp_min=True,
    stg_temp_max=True,
    polarity=True,
    ce_v_max=True,
    first_time=True,
    parallel=1,
):
    dirname = os.path.abspath("")
    cands = []
    if stg_temp_min:
        cands.append(PartStgTempMin)

    if stg_temp_max:
        cands.append(PartStgTempMax)

    if polarity:
        cands.append(PartPolarity)

    if ce_v_max:
        cands.append(PartCeVMax)

    featurizer = Featurizer(session, cands)
    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    if first_time:
        F_train = featurizer.get_feature_matrices(train_cands)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        F_test = featurizer.get_feature_matrices(test_cands)
        pickle.dump(F_train, open(os.path.join(dirname, "F_train.pkl"), "wb"))
        pickle.dump(F_dev, open(os.path.join(dirname, "F_dev.pkl"), "wb"))
        pickle.dump(F_test, open(os.path.join(dirname, "F_test.pkl"), "wb"))
    else:
        F_train = pickle.load(open(os.path.join(dirname, "F_train.pkl"), "rb"))
        F_dev = pickle.load(open(os.path.join(dirname, "F_dev.pkl"), "rb"))
        F_test = pickle.load(open(os.path.join(dirname, "F_test.pkl"), "rb"))
    logger.info("Done.")

    for i, cand in enumerate(cands):
        logger.info(f"{cand} Train shape: {F_train[i].shape}")
        logger.info(f"{cand} Test shape: {F_test[i].shape}")
        logger.info(f"{cand} Dev shape: {F_dev[i].shape}")

    return F_train, F_dev, F_test


def load_labels(session, relation, cand, first_time=True):
    if first_time:
        logger.info(f"Loading gold labels for {relation.value}")
        load_transistor_labels(session, [cand], [relation.value], annotator_name="gold")


def labeling(
    session, cands, cand_classes, lfs, split=1, train=False, first_time=True, parallel=1
):
    labeler = Labeler(session, cand_classes)

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=split, lfs=lfs, train=train, parallelism=parallel)
        logger.info("Done...")

    logger.info("Getting label matrices...")
    L_mat = labeler.get_label_matrices(cands)
    L_gold = labeler.get_gold_labels(cands, annotator="gold")
    logger.info("Done.")
    logger.info(f"L_mat shape: {L_mat[0].shape}")
    logger.info(f"L_gold shape: {L_gold[0].shape}")

    return L_mat, L_gold


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info("Training generative model...")
    model.train_model(L_train, n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train)
    return marginals


def discriminative_model(train_cands, F_train, marginals, n_epochs=50, lr=0.001):
    disc_model = SparseLogisticRegression()

    logger.info("Training discriminative model...")
    disc_model.train(
        (train_cands, F_train), marginals, n_epochs=n_epochs, lr=lr, host_device="GPU"
    )
    logger.info("Done.")

    return disc_model


def scoring(relation, disc_model, test_cands, test_docs, F_test, parts_by_doc, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Iterate over a range of `b` values in order to find the b with the
    # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
    Y_prob = disc_model.marginals((test_cands, F_test))

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            test_score = np.array(
                [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
            )
            true_pred = [test_cands[_] for _ in np.nditer(np.where(test_score == TRUE))]
            result = entity_level_scores(
                candidates_to_entities(true_pred, parts_by_doc=parts_by_doc),
                attribute=relation,
                corpus=test_docs,
            )
            logger.info(f"b = {b}, f1 = {result.f1}")
            if result.f1 > best_result.f1:
                best_result = result
                best_b = b
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
    return best_result, best_b


def dump_candidates(cands, Y_prob, outfile):
    """Output the ce_v_max candidates and their probabilities for later analysis."""
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, outfile), "w") as csvfile:
        writer = csv.writer(csvfile)
        for i, c in enumerate(cands):
            (doc, part, val) = cand_to_entity(c)
            writer.writerow([doc, part, val, Y_prob[i][TRUE - 1]])


def main(conn_string, max_docs=float("inf"), parse=True, first_time=True, parallel=4):
    init_logging(log_dir="logs")
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs = parsing(
        session, first_time=parse, parallel=parallel, max_docs=max_docs
    )

    Part, StgTempMin, StgTempMax, Polarity, CeVMax = mention_extraction(
        session, docs, first_time=first_time, parallel=parallel
    )

    (
        PartStgTempMin,
        PartStgTempMax,
        PartPolarity,
        PartCeVMax,
        train_cands,
        dev_cands,
        test_cands,
    ) = candidate_extraction(
        session,
        Part,
        StgTempMin,
        StgTempMax,
        Polarity,
        CeVMax,
        train_docs,
        dev_docs,
        test_docs,
        first_time=first_time,
        parallel=parallel,
    )

    pickle_file = "data/parts_by_doc_new.pkl"
    with open(pickle_file, "rb") as f:
        parts_by_doc = pickle.load(f)

    # First, check total recall
    for i, name in enumerate(["stg_temp_min", "stg_temp_max", "polarity", "ce_v_max"]):
        logger.info(name)
        result = entity_level_scores(
            candidates_to_entities(dev_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=dev_docs,
        )
        logger.info(f"Gain Total Dev Recall: {result.rec:.3f}")
        result = entity_level_scores(
            candidates_to_entities(test_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=test_docs,
        )
        logger.info(f"Gain Total Test Recall: {result.rec:.3f}")

    F_train, F_dev, F_test = featurization(
        session,
        train_cands,
        dev_cands,
        test_cands,
        PartStgTempMin,
        PartStgTempMax,
        PartPolarity,
        PartCeVMax,
        first_time=first_time,
        parallel=parallel,
    )

    logger.info("Labeling training data...")
    L_train, L_gold_train = labeling(
        session,
        train_cands,
        [PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax],
        [stg_temp_min_lfs, stg_temp_max_lfs, polarity_lfs, ce_v_max_lfs],
        split=0,
        train=True,
        parallel=parallel,
        first_time=True,
    )
    logger.info("Done.")

    relation = "stg_temp_min"
    marginals_stg_temp_min = generative_model(L_train[0])
    disc_model_stg_temp_min = discriminative_model(
        train_cands[0], F_train[0], marginals_stg_temp_min, n_epochs=100
    )
    best_result, best_b = scoring(
        relation,
        disc_model_stg_temp_min,
        test_cands[0],
        test_docs,
        F_test[0],
        parts_by_doc,
        num=100,
    )

    relation = "stg_temp_max"
    marginals_stg_temp_max = generative_model(L_train[1])
    disc_model_stg_temp_max = discriminative_model(
        train_cands[1], F_train[1], marginals_stg_temp_max, n_epochs=100
    )
    best_result, best_b = scoring(
        relation,
        disc_model_stg_temp_max,
        test_cands[1],
        test_docs,
        F_test[1],
        parts_by_doc,
        num=100,
    )

    relation = "polarity"
    marginals_polarity = generative_model(L_train[2])
    disc_model_polarity = discriminative_model(
        train_cands[2], F_train[2], marginals_polarity, n_epochs=100
    )
    best_result, best_b = scoring(
        relation,
        disc_model_polarity,
        test_cands[2],
        test_docs,
        F_test[2],
        parts_by_doc,
        num=100,
    )

    relation = "ce_v_max"
    marginals_ce_v_max = generative_model(L_train[3])
    disc_model_ce_v_max = discriminative_model(
        train_cands[3], F_train[3], marginals_ce_v_max, n_epochs=100
    )
    best_result, best_b = scoring(
        relation,
        disc_model_ce_v_max,
        test_cands[3],
        test_docs,
        F_test[3],
        parts_by_doc,
        num=100,
    )

    # Dump CSV files for CE_V_MAX for digi-key analysis
    Y_prob = disc_model_ce_v_max.marginals((test_cands[3], F_test[3]))
    dump_candidates(test_cands[3], Y_prob, "ce_v_max_test_probs.csv")
    Y_prob = disc_model_ce_v_max.marginals((dev_cands[3], F_dev[3]))
    dump_candidates(dev_cands[3], Y_prob, "ce_v_max_dev_probs.csv")
