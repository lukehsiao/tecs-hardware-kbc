import logging
import os
import pdb
import pickle

import matplotlib.pyplot as plt
import numpy as np
from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.learning import SparseLogisticRegression
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from metal import analysis
from metal.label_model import LabelModel

from hack.opamps.opamp_lfs import TRUE, opamp_lfs
from hack.opamps.opamp_matchers import get_gain_matcher, get_supply_current_matcher
from hack.opamps.opamp_spaces import MentionNgramsOpamps
from hack.opamps.opamp_utils import entity_level_scores, entity_to_candidates, Score
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
    gain_ngrams = MentionNgramsOpamps(n_max=2)
    Current = mention_subclass("SupplyCurrent")
    current_matcher = get_supply_current_matcher()
    current_ngrams = MentionNgramsOpamps(n_max=2)

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


def candidate_extraction(
    session, mentions, train_docs, dev_docs, test_docs, first_time=True, parallel=1
):
    (gain, current) = mentions
    Cand = candidate_subclass("GainCurrent", [gain, current])
    #  throttler = stg_temp_filter

    candidate_extractor = CandidateExtractor(session, [Cand])

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)
            logger.info(
                f"Cand in split={i}: "
                f"{session.query(Cand).filter(Cand.split == i).count()}"
            )

    train_cands = candidate_extractor.get_candidates(split=0)
    dev_cands = candidate_extractor.get_candidates(split=1)
    test_cands = candidate_extractor.get_candidates(split=2)

    logger.info(f"Total train candidate: {len(train_cands[0])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0])}")
    logger.info(f"Total test candidate: {len(test_cands[0])}")

    return (Cand, train_cands, dev_cands, test_cands)


def featurization(
    session, train_cands, dev_cands, test_cands, Cand, first_time=True, parallel=1
):
    featurizer = Featurizer(session, [Cand])
    if first_time:
        logger.info("Starting featurizer...")
        featurizer.apply(split=0, train=True, parallelism=parallel)
        featurizer.apply(split=1, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    # Serialize feature matrices on first run
    F_train = featurizer.get_feature_matrices(train_cands)
    F_dev = featurizer.get_feature_matrices(dev_cands)
    F_test = featurizer.get_feature_matrices(test_cands)
    logger.info("Done.")

    logger.info(f"Train shape: {F_train[0].shape}")
    logger.info(f"Test shape: {F_test[0].shape}")
    logger.info(f"Dev shape: {F_dev[0].shape}")

    return F_train, F_dev, F_test


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel(k=2)

    logger.info("Training generative model...")
    model.train_model(L_train[0], n_epochs=n_epochs, print_every=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train[0])
    plt.hist(marginals[:, TRUE - 1], bins=20)
    plt.savefig(os.path.join(os.path.dirname(__file__), f"opamps_marginals.pdf"))
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


def labeling(session, cands, cand, split=1, train=False, first_time=True, parallel=1):
    labeler = Labeler(session, [cand])
    lfs = opamp_lfs

    if first_time:
        logger.info("Applying LFs...")
        labeler.apply(split=split, lfs=[lfs], train=train, parallelism=parallel)
        logger.info("Done...")

    logger.info("Getting label matrices...")
    L_mat = labeler.get_label_matrices(cands)
    logger.info("Done...")
    logger.info(f"L_mat shape: {L_mat[0].shape}")

    if train:
        try:
            df = analysis.lf_summary(L_mat[0], lf_names=labeler.get_keys())
            logger.info(f"\n{df.to_string()}")
        except Exception:
            import pdb

            pdb.set_trace()

    return L_mat


def scoring(disc_model, test_cands, test_docs, F_test, num=100):
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
            result = entity_level_scores(true_pred, corpus=test_docs)
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


def main(conn_string, max_docs=float("inf"), first_time=True, parallel=2):
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs = parsing(
        session, first_time=False, parallel=parallel, max_docs=max_docs
    )

    (Gain, Current) = mention_extraction(
        session, docs, first_time=first_time, parallel=parallel
    )

    (Cand, train_cands, dev_cands, test_cands) = candidate_extraction(
        session,
        (Gain, Current),
        train_docs,
        dev_docs,
        test_docs,
        first_time=first_time,
        parallel=parallel,
    )
    F_train, F_dev, F_test = featurization(
        session,
        train_cands,
        dev_cands,
        test_cands,
        Cand,
        first_time=first_time,
        parallel=parallel,
    )
    logger.info("Labeling training data...")
    L_train = labeling(
        session, train_cands, Cand, split=0, train=True, parallel=parallel
    )
    logger.info("Done.")

    marginals = generative_model(L_train)

    labeling(session, dev_cands, Cand, split=1, train=False, parallel=parallel)

    disc_models = discriminative_model(train_cands, F_train, marginals, n_epochs=10)

    best_result, best_b = scoring(disc_models, test_cands, test_docs, F_test, num=100)

    try:
        fp_cands = entity_to_candidates(best_result.FP[1], test_cands[0])
    except Exception:
        pass

    # End with an interactive prompt
    pdb.set_trace()


if __name__ == "__main__":
    # See https://docs.python.org/3/library/os.html#os.cpu_count
    parallel = 4
    component = "opamps"
    conn_string = f"postgresql:///{component}"
    first_time = True
    max_docs = 200
    logger.info(f"\n\n")
    logger.info(f"=" * 30)
    logger.info(
        f"Beginning {component} with parallel: {parallel}, max_docs: {max_docs}"
    )

    main(conn_string, max_docs=max_docs, first_time=first_time, parallel=parallel)
