import logging
import os
import pdb
import pickle
from pprint import pformat

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

from hack.opamps.opamp_lfs import FALSE, TRUE, current_lfs, gain_lfs
from hack.opamps.opamp_matchers import get_gain_matcher, get_supply_current_matcher
from hack.opamps.opamp_spaces import MentionNgramsCurrent
from hack.opamps.opamp_utils import (
    Score,
    cand_to_entity,
    entity_level_scores,
    entity_to_candidates,
    get_gold_set,
    print_scores,
)
from hack.utils import parse_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure logging for Hack
logging.basicConfig(
    format="[%(asctime)s][%(levelname)s] %(name)s:%(lineno)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), f"opamps.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parsing(session, first_time=False, parallel=4, max_docs=float("inf")):
    dirname = os.path.dirname(__file__)
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


def mention_extraction(session, docs, first_time=True, parallel=1):
    Gain = mention_subclass("Gain")
    gain_matcher = get_gain_matcher()
    gain_ngrams = MentionNgrams(n_max=2)
    Current = mention_subclass("SupplyCurrent")
    current_matcher = get_supply_current_matcher()
    current_ngrams = MentionNgramsCurrent(n_max=2)

    mention_extractor = MentionExtractor(
        session,
        [Gain, Current],
        [gain_ngrams, current_ngrams],
        [gain_matcher, current_matcher],
    )

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")
    logger.info(f"Total Gain: {session.query(Gain).count()}")
    logger.info(f"Total Current: {session.query(Current).count()}")
    return Gain, Current


def candidate_extraction(session, mentions, docs, first_time=True, parallel=1):
    (gain, current) = mentions
    (train_docs, dev_docs, test_docs) = docs
    GainCand = candidate_subclass("GainCand", [gain])
    CurrentCand = candidate_subclass("CurrentCand", [current])

    candidate_extractor = CandidateExtractor(session, [GainCand, CurrentCand])

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)
            logger.info(
                f"GainCand in split={i}: "
                f"{session.query(GainCand).filter(GainCand.split == i).count()}"
            )
            logger.info(
                f"CurrentCand in split={i}: "
                f"{session.query(CurrentCand).filter(CurrentCand.split == i).count()}"
            )

    return (GainCand, CurrentCand), candidate_extractor


def featurization(session, cand_sets, cand_classes, first_time=True, parallel=1):
    GainCand, CurrentCand = cand_classes
    (train_cands, dev_cands, test_cands) = cand_sets
    featurizer = Featurizer(session, [GainCand, CurrentCand])
    dirname = os.path.dirname(__file__)
    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    # Serialize feature matrices on first run
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

    logger.info(f"Train shape 0: {F_train[0].shape}")
    logger.info(f"Train shape 1: {F_train[1].shape}")
    logger.info(f"Test shape 0: {F_test[0].shape}")
    logger.info(f"Test shape 1: {F_test[1].shape}")
    logger.info(f"Dev shape 0: {F_dev[0].shape}")
    logger.info(f"Dev shape 1: {F_dev[1].shape}")

    return F_train, F_dev, F_test


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info(f"Training generative model for...")
    model.train_model(L_train, n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train)
    plt.hist(marginals[:, TRUE - 1], bins=20)
    plt.savefig(os.path.join(os.path.dirname(__file__), f"opamps_marginals.pdf"))
    return marginals


def discriminative_model(train_cands, F_train, marginals, n_epochs=50, lr=0.001):
    disc_model = SparseLogisticRegression()

    logger.info("Training discriminative model...")
    disc_model.train(
        (train_cands, F_train), marginals, n_epochs=n_epochs, lr=lr, host_device="GPU"
    )
    logger.info("Done.")

    return disc_model


def labeling(
    labeler, cands, split=1, lfs=None, train=False, first_time=True, parallel=1
):
    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=split, lfs=lfs, train=train, parallelism=parallel)
        logger.info("Done...")

    logger.info("Getting label matrices...")
    L_mat = labeler.get_label_matrices(cands)
    logger.info("Done...")
    logger.info(f"L_mat shape: {L_mat[0].shape}")

    return L_mat


def scoring(disc_model, cands, docs, F_mat, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Iterate over a range of `b` values in order to find the b with the
    # highest F1 score. We are using cardinality==2. See fonduer/classifier.py.
    Y_prob = disc_model.marginals((cands, F_mat))

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            test_score = np.array(
                [TRUE if p[TRUE - 1] > b else 3 - TRUE for p in Y_prob]
            )
            true_pred = [cands[_] for _ in np.nditer(np.where(test_score == TRUE))]
        except Exception as e:
            logger.debug(f"{e}, skipping.")
            break
        result = entity_level_scores(true_pred, corpus=docs)
        logger.info(
            f"({b:.3f}), f1:{result.f1:.3f} p:{result.prec:.3f} r:{result.rec:.3f}"
        )

        if result.f1 > best_result.f1:
            best_result = result
            best_b = b

    return best_result, best_b


def main(conn_string, max_docs=float("inf"), parse=False, first_time=True, parallel=1):
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs = parsing(
        session, first_time=parse, parallel=parallel, max_docs=max_docs
    )

    logger.debug(f"Test Docs: {pformat(test_docs)}")
    logger.debug(f"Dev Docs: {pformat(dev_docs)}")

    (Gain, Current) = mention_extraction(
        session, docs, first_time=first_time, parallel=parallel
    )

    (GainCand, CurrentCand), candidate_extractor = candidate_extraction(
        session,
        (Gain, Current),
        (train_docs, dev_docs, test_docs),
        first_time=first_time,
        parallel=parallel,
    )
    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)
    logger.info(f"Total train candidate: {len(train_cands[0]) + len(train_cands[1])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0]) + len(dev_cands[1])}")
    logger.info(f"Total test candidate: {len(test_cands[0]) + len(test_cands[1])}")

    logger.info("Done w/ candidate extraction.")

    # First, check total recall
    #  result = entity_level_scores(dev_cands[0], corpus=dev_docs)
    #  logger.info(f"Gain Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(test_cands[0], corpus=test_docs)
    #  logger.info(f"Gain Total Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #
    #  result = entity_level_scores(dev_cands[1], corpus=dev_docs, is_gain=False)
    #  logger.info(f"Current Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(test_cands[1], corpus=test_docs, is_gain=False)
    #  logger.info(f"Current Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")


    F_train, F_dev, F_test = featurization(
        session,
        (train_cands, dev_cands, test_cands),
        (GainCand, CurrentCand),
        first_time=first_time,
        parallel=parallel,
    )

    logger.info("Labeling training data...")
    labeler = Labeler(session, [GainCand, CurrentCand])
    L_train = labeling(
        labeler,
        train_cands,
        split=0,
        lfs=[gain_lfs, current_lfs],
        train=True,
        parallel=parallel,
    )
    logger.info("Done.")

    logger.info("Labeling dev data...")
    L_dev = labeling(
        labeler,
        dev_cands,
        split=1,
        lfs=[gain_lfs, current_lfs],
        train=False,
        parallel=parallel,
    )

    # Evaluate LF accuracy
    dev_gold_entities = get_gold_set(is_gain=True)

    L_dev_gt = []
    for c in dev_cands[0]:
        flag = FALSE
        for entity in cand_to_entity(c, is_gain=True):
            if entity in dev_gold_entities:
                flag = TRUE
        L_dev_gt.append(flag)

    df = analysis.lf_summary(L_dev[0], lf_names=labeler.get_keys(), Y=np.array(L_dev_gt))
    logger.info(f"\n{df.to_string()}")

    marginals = generative_model(L_train[0])

    disc_models = discriminative_model(
        train_cands[0], F_train[0], marginals, n_epochs=10
    )
    best_result, best_b = scoring(
        disc_models, dev_cands[0], dev_docs, F_dev[0], num=30
    )

    print_scores(best_result, best_b)

    try:
        fp_cands == entity_to_candidates(best_result.FP[0], test_cands[0])
    except Exception:
        pass

    # End with an interactive prompt
    pdb.set_trace()

    marginals = generative_model(L_train[1])

    disc_models = discriminative_model(
        train_cands[1], F_train[1], marginals, n_epochs=10
    )
    best_result, best_b = scoring(
        disc_models, dev_cands[1], dev_docs, F_dev[1], num=30
    )

    print_scores(best_result, best_b)


    try:
        fp_cands == entity_to_candidates(best_result.FP[0], test_cands[1])
    except Exception:
        pass

    # End with an interactive prompt
    pdb.set_trace()


if __name__ == "__main__":
    parallel = 8
    component = "opamps"
    conn_string = f"postgresql:///{component}"
    first_time = False
    parse = False
    max_docs = 100
    logger.info(f"\n\n")
    logger.info(f"=" * 30)
    logger.info(
        f"Beginning {component} with parallel: {parallel}, max_docs: {max_docs}"
    )

    main(
        conn_string,
        max_docs=max_docs,
        parse=parse,
        first_time=first_time,
        parallel=parallel,
    )
