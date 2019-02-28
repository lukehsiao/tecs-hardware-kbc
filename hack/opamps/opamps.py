import logging
import os
import pdb
import pickle

from fonduer import Meta
from fonduer.candidates import CandidateExtractor, MentionExtractor
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence

from hack.opamps.opamp_matchers import get_gain_matcher, get_supply_current_matcher
from hack.opamps.opamp_spaces import MentionNgramsOpamps
from hack.utils import parse_dataset

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
    dirname = os.path.dirname(__file__)
    featurizer = Featurizer(session, [Cand])
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

    logger.info(f"Train shape: {F_train[0].shape}")
    logger.info(f"Test shape: {F_test[0].shape}")
    logger.info(f"Dev shape: {F_dev[0].shape}")

    return F_train, F_dev, F_test


def main(conn_string, max_docs=float("inf"), first_time=True, parallel=2):
    session = Meta.init(conn_string).Session()
    docs, train_docs, dev_docs, test_docs = parsing(
        session, first_time=True, parallel=parallel, max_docs=max_docs
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
    #  logger.info("Labeling training data...")
    #  L_train, L_gold_train = labeling(
    #      session, train_cands, Cand, split=0, train=True, parallel=parallel
    #  )
    #  logger.info("Done.")
    #
    #  marginals = generative_model(relation, L_train)
    #
    #  L_dev, L_gold_dev = labeling(
    #      session, dev_cands, Cand, split=1, train=False, parallel=parallel
    #  )
    #
    #  disc_models = discriminative_model(train_cands, F_train, marginals, n_epochs=10)
    #
    #  parts_by_doc = load_parts_by_doc()
    #  best_result, best_b = scoring(
    #      relation, disc_models, test_cands, test_docs, F_test, parts_by_doc, num=100
    #  )
    #
    #  try:
    #      fp_cand = entity_to_candidates(best_result.FP[1], test_cands[0])
    #  except Exception:
    #      pass

    # End with an interactive prompt
    pdb.set_trace()


if __name__ == "__main__":
    # See https://docs.python.org/3/library/os.html#os.cpu_count
    parallel = 4
    component = "opamps"
    conn_string = f"postgresql:///{component}"
    first_time = True
    max_docs = 150
    logger.info(f"\n\n")
    logger.info(f"=" * 30)
    logger.info(
        f"Beginning {component} with parallel: {parallel}, max_docs: {max_docs}"
    )

    main(conn_string, max_docs=max_docs, first_time=first_time, parallel=parallel)
