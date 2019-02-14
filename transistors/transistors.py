import logging
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.learning import LogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal import analysis
from metal.label_model import LabelModel
from transistor_matchers import get_matcher
from transistor_throttlers import ce_v_max_filter, polarity_filter, stg_temp_filter
from transistor_utils import entity_level_f1, load_transistor_labels

from transistor_lfs import (
    TRUE,
    ce_v_max_lfs,
    polarity_lfs,
    stg_temp_max_lfs,
    stg_temp_min_lfs,
)
from transistor_spaces import MentionNgramsPart, MentionNgramsTemp, MentionNgramsVolt

try:
    sys.path.append("..")
    from utils import parse_dataset
except Exception:
    raise ValueError("Unable to import utils.")

# Configure logging for Fonduer
logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# See https://docs.python.org/3/library/os.html#os.cpu_count
PARALLEL = len(os.sched_getaffinity(0)) // 2
COMPONENT = "transistors"
conn_string = "postgresql://localhost:5432/" + COMPONENT
logger.info(f"PARALLEL: {PARALLEL}")
FIRST_TIME = True


def parsing(session, first_time=True, parallel=1, max_docs=float("inf")):
    logger.debug(f"Starting parsing...")
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, first_time=first_time, parallel=parallel, max_docs=25
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


def mention_extraction(session, docs, parallel=1):
    Part = mention_subclass("Part")
    StgTempMin = mention_subclass("StgTempMin")
    StgTempMax = mention_subclass("StgTempMax")
    Polarity = mention_subclass("Polarity")
    CeVMax = mention_subclass("CeVMax")

    stg_temp_min_matcher = get_matcher("stg_temp_min")
    stg_temp_max_matcher = get_matcher("stg_temp_max")
    polarity_matcher = get_matcher("polarity")
    ce_v_max_matcher = get_matcher("ce_v_max")
    part_matcher = get_matcher("part")

    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)
    temp_ngrams = MentionNgramsTemp(n_max=2)
    volt_ngrams = MentionNgramsVolt(n_max=1)
    polarity_ngrams = MentionNgrams(n_max=1)

    mention_extractor = MentionExtractor(
        session,
        [Part, StgTempMin, StgTempMax, Polarity, CeVMax],
        [part_ngrams, temp_ngrams, temp_ngrams, polarity_ngrams, volt_ngrams],
        [
            part_matcher,
            stg_temp_min_matcher,
            stg_temp_max_matcher,
            polarity_matcher,
            ce_v_max_matcher,
        ],
    )

    if FIRST_TIME:
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
    mention_classes,
    train_docs,
    dev_docs,
    test_docs,
    first_time=True,
    parallel=1,
):

    (Part, StgTempMin, StgTempMax, Polarity, CeVMax) = mention_classes

    PartStgTempMin = candidate_subclass("PartStgTempMin", [Part, StgTempMin])
    PartStgTempMax = candidate_subclass("PartStgTempMax", [Part, StgTempMax])
    PartPolarity = candidate_subclass("PartPolarity", [Part, Polarity])
    PartCeVMax = candidate_subclass("PartCeVMax", [Part, CeVMax])

    temp_throttler = stg_temp_filter
    polarity_throttler = polarity_filter
    ce_v_max_throttler = ce_v_max_filter

    candidate_extractor = CandidateExtractor(
        session,
        [PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax],
        throttlers=[
            temp_throttler,
            temp_throttler,
            polarity_throttler,
            ce_v_max_throttler,
        ],
    )

    if FIRST_TIME:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=PARALLEL)
            count = (
                session.query(PartStgTempMin).filter(PartStgTempMin.split == i).count()
            )
            logger.info(f"PartStgTempMin in split={i}: " f"{count}")
            count = (
                session.query(PartStgTempMax).filter(PartStgTempMax.split == i).count()
            )
            logger.info(f"PartStgTempMax in split={i}: " f"count")
            logger.info(
                f"PartPolarity in split={i}: "
                f"{session.query(PartPolarity).filter(PartPolarity.split == i).count()}"
            )
            logger.info(
                f"PartCeVMax in split={i}: "
                f"{session.query(PartCeVMax).filter(PartCeVMax.split == i).count()}"
            )

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)

    logger.info(f"Total train candidate: {len(train_cands[0])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0])}")
    logger.info(f"Total test candidate: {len(test_cands[0])}")

    return (
        (train_cands, dev_cands, test_cands),
        (PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax),
    )


def featurization(session, cands, cand_classes, first_time=True, parallel=1):
    (PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax) = cand_classes
    (train_cands, dev_cands, test_cands) = cands
    featurizer = Featurizer(
        session, [PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax]
    )
    if FIRST_TIME:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    F_train = featurizer.get_feature_matrices(train_cands)
    F_dev = featurizer.get_feature_matrices(dev_cands)
    F_test = featurizer.get_feature_matrices(test_cands)
    logger.info("Done.")

    logger.info(f"Train shape: {F_train[0].shape}")
    logger.info(f"Test shape: {F_test[0].shape}")
    logger.info(f"Dev shape: {F_dev[0].shape}")

    return F_train, F_dev, F_test


def load_labels(session, cand_classes, first_time=False):
    (PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax) = cand_classes
    if FIRST_TIME:
        load_transistor_labels(
            session,
            [PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax],
            ["stg_temp_min", "stg_temp_max", "polarity", "ce_v_max"],
            annotator_name="gold",
        )


def labeling(session, cands, cand_classes, split=1, train=False, parallelism=1):
    (PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax) = cand_classes
    labeler = Labeler(
        session, [PartStgTempMin, PartStgTempMax, PartPolarity, PartCeVMax]
    )
    if FIRST_TIME or True:
        labeler.apply(
            split=split,
            lfs=[stg_temp_min_lfs, stg_temp_max_lfs, polarity_lfs, ce_v_max_lfs],
            train=True,
            parallelism=PARALLEL,
        )

    L_mat = labeler.get_label_matrices(cands)
    L_gold = labeler.get_gold_labels(cands, annotator="gold")

    df = analysis.lf_summary(
        L_mat[0],
        lf_names=labeler.get_keys(),
        Y=L_gold[0].todense().reshape(-1).tolist()[0],
    )
    print(df.to_string())

    return L_mat, L_gold


def generative_model(L_train):
    stg_temp_min_model = LabelModel(k=2)
    stg_temp_max_model = LabelModel(k=2)
    polarity_model = LabelModel(k=2)
    ce_v_max_model = LabelModel(k=2)

    stg_temp_min_model.train_model(L_train[0], n_epochs=500, print_every=100)
    stg_temp_max_model.train_model(L_train[1], n_epochs=500, print_every=100)
    polarity_model.train_model(L_train[2], n_epochs=500, print_every=100)
    ce_v_max_model.train_model(L_train[3], n_epochs=500, print_every=100)

    stg_temp_min_marginals = stg_temp_min_model.predict_proba(L_train[0])
    plt.hist(stg_temp_min_marginals[:, TRUE - 1], bins=20)
    plt.savefig("stg_temp_min_marginals.pdf")
    stg_temp_max_marginals = stg_temp_max_model.predict_proba(L_train[1])
    polarity_marginals = polarity_model.predict_proba(L_train[2])
    ce_v_max_marginals = ce_v_max_model.predict_proba(L_train[3])
    return (
        stg_temp_min_marginals,
        stg_temp_max_marginals,
        polarity_marginals,
        ce_v_max_marginals,
    )


def discriminative_model(train_cands, F_train, marginals):
    (
        stg_temp_min_marginals,
        stg_temp_max_marginals,
        polarity_marginals,
        ce_v_max_marginals,
    ) = marginals
    stg_temp_min_disc_model = LogisticRegression()
    stg_temp_max_disc_model = LogisticRegression()
    polarity_disc_model = LogisticRegression()
    ce_v_max_disc_model = LogisticRegression()

    stg_temp_min_disc_model.train(
        (train_cands[0], F_train[0]), stg_temp_min_marginals, n_epochs=50, lr=0.001
    )
    stg_temp_max_disc_model.train(
        (train_cands[1], F_train[1]), stg_temp_max_marginals, n_epochs=50, lr=0.001
    )
    polarity_disc_model.train(
        (train_cands[2], F_train[2]), polarity_marginals, n_epochs=50, lr=0.001
    )
    ce_v_max_disc_model.train(
        (train_cands[3], F_train[3]), ce_v_max_marginals, n_epochs=50, lr=0.001
    )
    return (
        stg_temp_min_disc_model,
        stg_temp_max_disc_model,
        polarity_disc_model,
        ce_v_max_disc_model,
    )


def load_parts_by_doc():
    pickle_file = "data/parts_by_doc_dict.pkl"
    with open(pickle_file, "rb") as f:
        return pickle.load(f)


def scoring(disc_models, test_cands, test_docs, F_test, parts_by_doc):
    (
        stg_temp_min_disc_model,
        stg_temp_max_disc_model,
        polarity_disc_model,
        ce_v_max_disc_model,
    ) = disc_models

    stg_temp_min_test_score = stg_temp_min_disc_model.predict(
        (test_cands[0], F_test[0]), b=0.6, pos_label=TRUE
    )
    stg_temp_min_true_pred = [
        test_cands[0][_] for _ in np.nditer(np.where(stg_temp_min_test_score == TRUE))
    ]
    (TP, FP, FN) = entity_level_f1(
        stg_temp_min_true_pred, "stg_temp_min", test_docs, parts_by_doc=parts_by_doc
    )

    stg_temp_max_test_score = stg_temp_max_disc_model.predict(
        (test_cands[1], F_test[1]), b=0.6, pos_label=TRUE
    )
    stg_temp_max_true_pred = [
        test_cands[1][_] for _ in np.nditer(np.where(stg_temp_max_test_score == TRUE))
    ]
    (TP, FP, FN) = entity_level_f1(
        stg_temp_max_true_pred, "stg_temp_max", test_docs, parts_by_doc=parts_by_doc
    )

    polarity_test_score = polarity_disc_model.predict(
        (test_cands[2], F_test[2]), b=0.6, pos_label=TRUE
    )
    polarity_true_pred = [
        test_cands[2][_] for _ in np.nditer(np.where(polarity_test_score == TRUE))
    ]
    (TP, FP, FN) = entity_level_f1(
        polarity_true_pred, "polarity", test_docs, parts_by_doc=parts_by_doc
    )

    ce_v_max_test_score = ce_v_max_disc_model.predict(
        (test_cands[3], F_test[3]), b=0.6, pos_label=TRUE
    )
    ce_v_max_true_pred = [
        test_cands[3][_] for _ in np.nditer(np.where(ce_v_max_test_score == TRUE))
    ]
    (TP, FP, FN) = entity_level_f1(
        ce_v_max_true_pred, "ce_v_max", test_docs, parts_by_doc=parts_by_doc
    )


def main():
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs = parsing(
        session, first_time=FIRST_TIME, parallel=PARALLEL
    )
    mention_classes = mention_extraction(session, docs, parallel=PARALLEL)
    cands, cand_classes = candidate_extraction(
        session,
        mention_classes,
        train_docs,
        dev_docs,
        test_docs,
        first_time=FIRST_TIME,
        parallel=PARALLEL,
    )
    F_train, F_dev, F_test = featurization(
        session, cands, cand_classes, first_time=FIRST_TIME, parallel=PARALLEL
    )
    load_labels(session, cand_classes, first_time=FIRST_TIME)
    (train_cands, dev_cands, test_cands) = cands
    L_train, L_gold_train = labeling(
        session, train_cands, cand_classes, split=0, train=True, parallelism=PARALLEL
    )
    marginals = generative_model(L_train)
    L_dev, L_gold_dev = labeling(
        session, dev_cands, cand_classes, split=1, train=False, parallelism=PARALLEL
    )

    disc_models = discriminative_model(train_cands, F_train, marginals)

    parts_by_doc = load_parts_by_doc()
    scoring(disc_models, test_cands, test_docs, F_test, parts_by_doc)


if __name__ == "__main__":
    main()
