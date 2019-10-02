#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
from timeit import default_timer as timer

from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.features import Featurizer
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler

from hack.transistors.transistor_lfs import FALSE, TRUE, ce_v_max_lfs
from hack.transistors.transistor_matchers import get_matcher
from hack.transistors.transistor_spaces import MentionNgramsPart, MentionNgramsVolt
from hack.transistors.transistor_throttlers import ce_v_max_filter
from hack.transistors.transistor_utils import (
    cand_to_entity,
    candidates_to_entities,
    entity_level_scores,
    get_gold_set,
)
from hack.transistors.transistors import discriminative_model, generative_model, scoring
from hack.utils import parse_dataset

logger = logging.getLogger(__name__)


def main(
    conn_string,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    re_label=False,
    gpu=None,
    parallel=4,
    log_dir=None,
    verbose=False,
    human_annotations=False,
    weak_supervision=True,
):
    # Setup initial configuration
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    if not log_dir:
        log_dir = "logs"

    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    dirname = os.path.dirname(os.path.abspath(__file__))
    init_logging(log_dir=os.path.join(dirname, log_dir), level=level)

    session = Meta.init(conn_string).Session()

    # Parsing
    logger.info(f"Starting parsing...")
    start = timer()
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=parse, parallel=parallel, max_docs=max_docs
    )
    end = timer()
    logger.warning(f"Parse Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"# of train Documents: {len(train_docs)}")
    logger.info(f"# of dev Documents: {len(dev_docs)}")
    logger.info(f"# of test Documents: {len(test_docs)}")
    logger.info(f"Documents: {session.query(Document).count()}")
    logger.info(f"Sections: {session.query(Section).count()}")
    logger.info(f"Paragraphs: {session.query(Paragraph).count()}")
    logger.info(f"Sentences: {session.query(Sentence).count()}")
    logger.info(f"Figures: {session.query(Figure).count()}")

    # Mention Extraction
    start = timer()
    mentions = []
    ngrams = []
    matchers = []

    Part = mention_subclass("Part")
    part_matcher = get_matcher("part")
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)

    mentions.append(Part)
    ngrams.append(part_ngrams)
    matchers.append(part_matcher)

    CeVMax = mention_subclass("CeVMax")
    ce_v_max_matcher = get_matcher("ce_v_max")
    ce_v_max_ngrams = MentionNgramsVolt(n_max=1)

    mentions.append(CeVMax)
    ngrams.append(ce_v_max_ngrams)
    matchers.append(ce_v_max_matcher)

    mention_extractor = MentionExtractor(session, mentions, ngrams, matchers)

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")
    logger.info(f"Total Part: {session.query(Part).count()}")
    logger.info(f"Total CeVMax: {session.query(CeVMax).count()}")

    # Candidate Extraction
    cands = []
    throttlers = []

    PartCeVMax = candidate_subclass("PartCeVMax", [Part, CeVMax])
    ce_v_max_throttler = ce_v_max_filter

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

    end = timer()
    logger.warning(f"Candidate Extraction Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"Total train candidate: {sum(len(_) for _ in train_cands)}")
    logger.info(f"Total dev candidate: {sum(len(_) for _ in dev_cands)}")
    logger.info(f"Total test candidate: {sum(len(_) for _ in test_cands)}")

    pickle_file = os.path.join(dirname, "data/parts_by_doc_new.pkl")
    with open(pickle_file, "rb") as f:
        parts_by_doc = pickle.load(f)

    # Check total recall
    for i, name in enumerate(["ce_v_max"]):
        logger.info(name)
        result = entity_level_scores(
            candidates_to_entities(dev_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=dev_docs,
        )
        logger.info(f"{name} Total Dev Recall: {result.rec:.3f}")
        result = entity_level_scores(
            candidates_to_entities(test_cands[i], parts_by_doc=parts_by_doc),
            attribute=name,
            corpus=test_docs,
        )
        logger.info(f"{name} Total Test Recall: {result.rec:.3f}")

    # Featurization
    start = timer()
    cands = [PartCeVMax]

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
        end = timer()
        logger.warning(f"Featurization Time (min): {((end - start) / 60.0):.1f}")

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

    logger.info("Labeling training data...")

    # Labeling
    start = timer()
    lfs = [ce_v_max_lfs]

    labeler = Labeler(session, cands)

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=0, lfs=lfs, train=True, parallelism=parallel)
        logger.info("Done...")

        # Uncomment if debugging LFs
        #  load_transistor_labels(session, cands, ["ce_v_max"])
        #  labeler.apply(split=1, lfs=lfs, train=False, parallelism=parallel)
        #  labeler.apply(split=2, lfs=lfs, train=False, parallelism=parallel)

    elif re_label:
        logger.info("Updating LFs...")
        labeler.update(split=0, lfs=lfs, parallelism=parallel)
        logger.info("Done...")

        # Uncomment if debugging LFs
        #  labeler.apply(split=1, lfs=lfs, train=False, parallelism=parallel)
        #  labeler.apply(split=2, lfs=lfs, train=False, parallelism=parallel)

    logger.info("Getting label matrices...")

    L_train = labeler.get_label_matrices(train_cands)

    # Uncomment if debugging LFs
    #  L_dev = labeler.get_label_matrices(dev_cands)
    #  L_dev_gold = labeler.get_gold_labels(dev_cands, annotator="gold")
    #
    #  L_test = labeler.get_label_matrices(test_cands)
    #  L_test_gold = labeler.get_gold_labels(test_cands, annotator="gold")

    logger.info("Done.")

    end = timer()
    logger.warning(f"Supervision Time (min): {((end - start) / 60.0):.1f}")

    start = timer()

    relation = "ce_v_max"
    idx = 0

    # TODO: Combine human annotations with weak supervision for best results
    if human_annotations:
        # Evaluate using human annotations
        logger.info("================================================================")
        logger.info("Scoring collector emitter voltage max using human annotations...")
        dev_gold_entities = get_gold_set(
            attribute=relation, docs=[doc.name for doc in dev_docs]
        )
        L_dev_gt = []
        count = 0
        for c in dev_cands[idx]:
            flag = FALSE
            if cand_to_entity(c) in dev_gold_entities:
                count += 1
                flag = TRUE
            L_dev_gt.append(flag)

        marginals_stg_temp_min = generative_model(L_train[idx])
        disc_model_stg_temp_min = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals_stg_temp_min,
            X_dev=(dev_cands[idx], F_dev[idx]),
            Y_dev=L_dev_gt,
            n_epochs=100,  # TODO: What does this mean?
            gpu=gpu,
            annotations=True,
        )
    if weak_supervision:
        # Evaluate solely using weak supervision
        logger.info("===============================================================")
        logger.info("Scoring collector emitter voltage max using weak supervision...")
        marginals_stg_temp_min = generative_model(L_train[idx])
        disc_model_stg_temp_min = discriminative_model(
            train_cands[idx],
            F_train[idx],
            marginals_stg_temp_min,
            n_epochs=100,
            gpu=gpu,
        )
    if not human_annotations and not weak_supervision:
        logger.error("Labeler needs either weak_supervision or " + "human_annotations.")

    best_result, best_b = scoring(
        relation,
        disc_model_stg_temp_min,
        test_cands[idx],
        test_docs,
        F_test[idx],
        parts_by_doc,
        num=100,
    )

    end = timer()
    logger.warning(f"Classification Time (min): {((end - start) / 60.0):.1f}")


if __name__ == "__main__":
    main(
        "postgresql:///ce_v_max",
        max_docs=100,
        parse=True,
        first_time=True,
        re_label=True,
        gpu=None,
        parallel=10,
        log_dir="comparison_logs",
        verbose=False,
        human_annotations=False,
        weak_supervision=True,
    )
