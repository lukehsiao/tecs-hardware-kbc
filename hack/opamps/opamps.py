import csv
import logging
import os
import pickle
from timeit import default_timer as timer

import emmental
import numpy as np
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from emmental.modules.embedding_module import EmbeddingModule
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionNgrams
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.features import Featurizer
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task
from fonduer.learning.utils import collect_word_counter
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from tqdm import tqdm

from hack.opamps.opamp_lfs import TRUE  # , current_lfs, gain_lfs
from hack.opamps.opamp_matchers import get_gain_matcher, get_supply_current_matcher
from hack.opamps.opamp_spaces import MentionNgramsCurrent
from hack.opamps.opamp_utils import (
    Score,
    cand_to_entity,
    candidates_to_entities,
    entity_level_scores,
    get_gold_set,
)
from hack.utils import parse_dataset

# Configure logging for Hack
logger = logging.getLogger(__name__)


def dump_candidates(cands, Y_prob, outfile, is_gain=True):
    """Output the candidates and their probabilities for later analysis."""
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, outfile), "w") as csvfile:
        writer = csv.writer(csvfile, lineterminator="\n")
        for i, c in enumerate(tqdm(cands)):
            for (doc, val) in cand_to_entity(c, is_gain=is_gain):
                if is_gain:
                    writer.writerow([doc, val.real / 1e3, Y_prob[i]])
                else:
                    writer.writerow([doc, val.real * 1e6, Y_prob[i]])


def generative_model(L_train, n_epochs=500, print_every=100):
    return NotImplementedError


def discriminative_model(
    train_cands, F_train, marginals, X_dev=None, Y_dev=None, n_epochs=50, lr=0.001
):
    raise NotImplementedError


def output_csv(cands, Y_prob, is_gain=True, append=False):
    dirname = os.path.dirname(__file__)
    if is_gain:
        filename = "output_gain.csv"
    else:
        filename = "output_current.csv"
    filename = os.path.join(dirname, filename)

    if append:
        with open(filename, "a") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            for i, c in enumerate(tqdm(cands)):
                for entity in cand_to_entity(c, is_gain=is_gain):
                    if is_gain:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real / 1e3,
                                c[0].context.sentence.position,
                                Y_prob[i],
                            ]
                        )
                    else:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real * 1e6,
                                c[0].context.sentence.position,
                                Y_prob[i],
                            ]
                        )
    else:
        with open(filename, "w") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            if is_gain:
                writer.writerow(["Document", "GBWP (kHz)", "sent", "p"])
            else:
                writer.writerow(["Document", "Supply Current (uA)", "sent", "p"])

            for i, c in enumerate(tqdm(cands)):
                for entity in cand_to_entity(c, is_gain=is_gain):
                    if is_gain:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real / 1e3,
                                c[0].context.sentence.position,
                                Y_prob[i],
                            ]
                        )
                    else:
                        writer.writerow(
                            [
                                entity[0],
                                entity[1].real * 1e6,
                                c[0].context.sentence.position,
                                Y_prob[i],
                            ]
                        )


def scoring(test_preds, test_cands, test_docs, is_gain=True, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    if is_gain:
        relation = "gain"
    else:
        relation = "current"
    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            positive = np.where(np.array(test_preds["probs"][relation])[:, TRUE] > b)
            true_pred = [test_cands[_] for _ in positive[0]]
            result = entity_level_scores(
                candidates_to_entities(true_pred, is_gain=is_gain),
                corpus=test_docs,
                is_gain=is_gain,
            )
            logger.info(
                f"b:{b:.3f} f1:{result.f1:.3f} p:{result.prec:.3f} r:{result.rec:.3f}"
            )
            if result.f1 > best_result.f1:
                best_result = result
                best_b = b
        except Exception as e:
            logger.error(f"{e}, skipping.")
            break

    logger.warning("===================================================")
    logger.warning(f"Entity-Level Gold Data score for {relation}, b={best_b:.3f}")
    logger.warning("===================================================")
    logger.warning(f"Corpus Precision {best_result.prec:.3f}")
    logger.warning(f"Corpus Recall    {best_result.rec:.3f}")
    logger.warning(f"Corpus F1        {best_result.f1:.3f}")
    logger.warning("---------------------------------------------------")
    logger.warning(
        f"TP: {len(best_result.TP)} "
        f"| FP: {len(best_result.FP)} "
        f"| FN: {len(best_result.FN)}"
    )
    logger.warning("===================================================\n")
    return best_result, best_b


def main(
    conn_string,
    gain=False,
    current=False,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    re_label=False,
    parallel=8,
    log_dir="logs",
    verbose=False,
):
    # Setup initial configuration
    if not log_dir:
        log_dir = "logs"

    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    dirname = os.path.dirname(os.path.abspath(__file__))
    init_logging(log_dir=os.path.join(dirname, log_dir), level=level)

    rel_list = []
    if gain:
        rel_list.append("gain")

    if current:
        rel_list.append("current")

    logger.info(f"=" * 30)
    logger.info(f"Running with parallel: {parallel}, max_docs: {max_docs}")

    session = Meta.init(conn_string).Session()

    # Parsing
    start = timer()
    logger.info(f"Starting parsing...")
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=parse, parallel=parallel, max_docs=max_docs
    )
    logger.debug(f"Done")
    end = timer()
    logger.warning(f"Parse Time (min): {((end - start) / 60.0):.1f}")

    logger.info(f"# of Documents: {len(docs)}")
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

    # Only do those that are enabled
    if gain:
        Gain = mention_subclass("Gain")
        gain_matcher = get_gain_matcher()
        gain_ngrams = MentionNgrams(n_max=2)
        mentions.append(Gain)
        ngrams.append(gain_ngrams)
        matchers.append(gain_matcher)

    if current:
        Current = mention_subclass("SupplyCurrent")
        current_matcher = get_supply_current_matcher()
        current_ngrams = MentionNgramsCurrent(n_max=3)
        mentions.append(Current)
        ngrams.append(current_ngrams)
        matchers.append(current_matcher)

    mention_extractor = MentionExtractor(session, mentions, ngrams, matchers)

    if first_time:
        mention_extractor.apply(docs, parallelism=parallel)

    logger.info(f"Total Mentions: {session.query(Mention).count()}")

    if gain:
        logger.info(f"Total Gain: {session.query(Gain).count()}")

    if current:
        logger.info(f"Total Current: {session.query(Current).count()}")

    cand_classes = []
    if gain:
        GainCand = candidate_subclass("GainCand", [Gain])
        cand_classes.append(GainCand)
    if current:
        CurrentCand = candidate_subclass("CurrentCand", [Current])
        cand_classes.append(CurrentCand)

    candidate_extractor = CandidateExtractor(session, cand_classes)

    if first_time:
        for i, docs in enumerate([train_docs, dev_docs, test_docs]):
            candidate_extractor.apply(docs, split=i, parallelism=parallel)

    train_cands = candidate_extractor.get_candidates(split=0)
    # Sort the dev_cands (which are used as training) to ensure deterministic behavior.
    dev_cands = candidate_extractor.get_candidates(split=1, sort=True)
    test_cands = candidate_extractor.get_candidates(split=2)
    logger.info(f"Total train candidate: {len(train_cands[0]) + len(train_cands[1])}")
    logger.info(f"Total dev candidate: {len(dev_cands[0]) + len(dev_cands[1])}")
    logger.info(f"Total test candidate: {len(test_cands[0]) + len(test_cands[1])}")

    logger.info("Done w/ candidate extraction.")
    end = timer()
    logger.warning(f"CE Time (min): {((end - start) / 60.0):.1f}")

    # First, check total recall
    #  result = entity_level_scores(
    #      candidates_to_entities(dev_cands[0], is_gain=True),
    #      corpus=dev_docs,
    #      is_gain=True,
    #  )
    #  logger.info(f"Gain Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(
    #      candidates_to_entities(test_cands[0], is_gain=True),
    #      corpus=test_docs,
    #      is_gain=True,
    #  )
    #  logger.info(f"Gain Total Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #
    #  result = entity_level_scores(
    #      candidates_to_entities(dev_cands[1], is_gain=False),
    #      corpus=dev_docs,
    #      is_gain=False,
    #  )
    #  logger.info(f"Current Total Dev Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")
    #  result = entity_level_scores(
    #      candidates_to_entities(test_cands[1], is_gain=False),
    #      corpus=test_docs,
    #      is_gain=False,
    #  )
    #  logger.info(f"Current Test Recall: {result.rec:.3f}")
    #  logger.info(f"\n{pformat(result.FN)}")

    start = timer()
    featurizer = Featurizer(session, cand_classes)

    if first_time:
        logger.info("Starting featurizer...")
        # Set feature space based on dev set, which we use for training rather
        # than the large train set.
        featurizer.apply(split=1, train=True, parallelism=parallel)
        featurizer.apply(split=0, parallelism=parallel)
        featurizer.apply(split=2, parallelism=parallel)
        logger.info("Done")

    logger.info("Getting feature matrices...")
    # Serialize feature matrices on first run
    if first_time:
        F_train = featurizer.get_feature_matrices(train_cands)
        F_dev = featurizer.get_feature_matrices(dev_cands)
        F_test = featurizer.get_feature_matrices(test_cands)
        end = timer()
        logger.warning(f"Featurization Time (min): {((end - start) / 60.0):.1f}")

        F_train_dict = {}
        F_dev_dict = {}
        F_test_dict = {}
        for idx, relation in enumerate(rel_list):
            F_train_dict[relation] = F_train[idx]
            F_dev_dict[relation] = F_dev[idx]
            F_test_dict[relation] = F_test[idx]

        pickle.dump(F_train_dict, open(os.path.join(dirname, "F_train_dict.pkl"), "wb"))
        pickle.dump(F_dev_dict, open(os.path.join(dirname, "F_dev_dict.pkl"), "wb"))
        pickle.dump(F_test_dict, open(os.path.join(dirname, "F_test_dict.pkl"), "wb"))
    else:
        F_train_dict = pickle.load(
            open(os.path.join(dirname, "F_train_dict.pkl"), "rb")
        )
        F_dev_dict = pickle.load(open(os.path.join(dirname, "F_dev_dict.pkl"), "rb"))
        F_test_dict = pickle.load(open(os.path.join(dirname, "F_test_dict.pkl"), "rb"))

        F_train = []
        F_dev = []
        F_test = []
        for relation in rel_list:
            F_train.append(F_train_dict[relation])
            F_dev.append(F_dev_dict[relation])
            F_test.append(F_test_dict[relation])

    logger.info("Done.")

    start = timer()
    logger.info("Labeling training data...")
    #  labeler = Labeler(session, cand_classes)
    #  lfs = []
    #  if gain:
    #      lfs.append(gain_lfs)
    #
    #  if current:
    #      lfs.append(current_lfs)
    #
    #  if first_time:
    #      logger.info("Applying LFs...")
    #      labeler.apply(split=0, lfs=lfs, train=True, parallelism=parallel)
    #  elif re_label:
    #      logger.info("Re-applying LFs...")
    #      labeler.update(split=0, lfs=lfs, parallelism=parallel)
    #
    #  logger.info("Done...")

    #  logger.info("Getting label matrices...")
    #  L_train = labeler.get_label_matrices(train_cands)
    #  logger.info("Done...")

    if first_time:
        marginals_dict = {}
        for idx, relation in enumerate(rel_list):
            # Manually create marginals from human annotations
            marginal = []
            dev_gold_entities = get_gold_set(is_gain=(relation == "gain"))
            for c in dev_cands[idx]:
                flag = False
                for entity in cand_to_entity(c, is_gain=(relation == "gain")):
                    if entity in dev_gold_entities:
                        flag = True

                if flag:
                    marginal.append([0.0, 1.0])
                else:
                    marginal.append([1.0, 0.0])

            marginals_dict[relation] = np.array(marginal)

        pickle.dump(
            marginals_dict, open(os.path.join(dirname, "marginals_dict.pkl"), "wb")
        )
    else:
        marginals_dict = pickle.load(
            open(os.path.join(dirname, "marginals_dict.pkl"), "rb")
        )

    marginals = []
    for relation in rel_list:
        marginals.append(marginals_dict[relation])

    end = timer()
    logger.warning(f"Weak Supervision Time (min): {((end - start) / 60.0):.1f}")

    start = timer()

    word_counter = collect_word_counter(train_cands)

    # Training config
    config = {
        "meta_config": {"verbose": True, "seed": 30},
        "model_config": {"model_path": None, "device": 0, "dataparallel": False},
        "learner_config": {
            "n_epochs": 500,
            "optimizer_config": {"lr": 0.001, "l2": 0.005},
            "task_scheduler": "round_robin",
        },
        "logging_config": {
            "evaluation_freq": 1,
            "counter_unit": "epoch",
            "checkpointing": False,
            "checkpointer_config": {
                "checkpoint_metric": {"model/all/train/loss": "min"},
                "checkpoint_freq": 1,
                "checkpoint_runway": 2,
                "clear_intermediate_checkpoints": True,
                "clear_all_checkpoints": True,
            },
        },
    }

    emmental.init(log_dir=Meta.log_path, config=config)

    # Generate word embedding module
    arity = 2
    # Geneate special tokens
    specials = []
    for i in range(arity):
        specials += [f"~~[[{i}", f"{i}]]~~"]

    emb_layer = EmbeddingModule(
        word_counter=word_counter, word_dim=300, specials=specials
    )
    train_idxs = []
    train_dataloader = []
    for idx, relation in enumerate(rel_list):
        diffs = marginals[idx].max(axis=1) - marginals[idx].min(axis=1)
        train_idxs.append(np.where(diffs > 1e-6)[0])

        # only uses dev set as training data, with human annotations
        train_dataloader.append(
            EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation,
                    dev_cands[idx],
                    F_dev[idx],
                    emb_layer.word2id,
                    marginals[idx],
                    train_idxs[idx],
                ),
                split="train",
                batch_size=256,
                shuffle=True,
            )
        )

    num_feature_keys = len(featurizer.get_keys())

    model = EmmentalModel(name=f"opamp_tasks")

    # List relation names, arities, list of classes
    tasks = create_task(
        rel_list,
        [2] * len(rel_list),
        num_feature_keys,
        [2] * len(rel_list),
        emb_layer,
        model="LogisticRegression",
    )

    for task in tasks:
        model.add_task(task)

    emmental_learner = EmmentalLearner()

    # If given a list of multi, will train on multiple
    emmental_learner.learn(model, train_dataloader)

    # List of dataloader for each relation
    for idx, relation in enumerate(rel_list):
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={relation: "labels"},
            dataset=FonduerDataset(
                relation, test_cands[idx], F_test[idx], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=256,
            shuffle=False,
        )

        test_preds = model.predict(test_dataloader, return_preds=True)

        best_result, best_b = scoring(
            test_preds,
            test_cands[idx],
            test_docs,
            is_gain=(relation == "gain"),
            num=100,
        )

        # Dump CSV files for analysis
        if relation == "gain":
            train_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, train_cands[idx], F_train[idx], emb_layer.word2id, 2
                ),
                split="train",
                batch_size=256,
                shuffle=False,
            )

            train_preds = model.predict(train_dataloader, return_preds=True)
            Y_prob = np.array(train_preds["probs"][relation])[:, TRUE]
            output_csv(train_cands[idx], Y_prob, is_gain=True)

            Y_prob = np.array(test_preds["probs"][relation])[:, TRUE]
            output_csv(test_cands[idx], Y_prob, is_gain=True, append=True)
            dump_candidates(
                test_cands[idx], Y_prob, "gain_test_probs.csv", is_gain=True
            )

            dev_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, dev_cands[idx], F_dev[idx], emb_layer.word2id, 2
                ),
                split="dev",
                batch_size=256,
                shuffle=False,
            )

            dev_preds = model.predict(dev_dataloader, return_preds=True)

            Y_prob = np.array(dev_preds["probs"][relation])[:, TRUE]
            output_csv(dev_cands[idx], Y_prob, is_gain=True, append=True)
            dump_candidates(dev_cands[idx], Y_prob, "gain_dev_probs.csv", is_gain=True)

        if relation == "current":
            train_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, train_cands[idx], F_train[idx], emb_layer.word2id, 2
                ),
                split="train",
                batch_size=256,
                shuffle=False,
            )

            train_preds = model.predict(train_dataloader, return_preds=True)
            Y_prob = np.array(train_preds["probs"][relation])[:, TRUE]
            output_csv(train_cands[idx], Y_prob, is_gain=False)

            Y_prob = np.array(test_preds["probs"][relation])[:, TRUE]
            output_csv(test_cands[idx], Y_prob, is_gain=False, append=True)
            dump_candidates(
                test_cands[idx], Y_prob, "current_test_probs.csv", is_gain=False
            )

            dev_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, dev_cands[idx], F_dev[idx], emb_layer.word2id, 2
                ),
                split="dev",
                batch_size=256,
                shuffle=False,
            )

            dev_preds = model.predict(dev_dataloader, return_preds=True)

            Y_prob = np.array(dev_preds["probs"][relation])[:, TRUE]
            output_csv(dev_cands[idx], Y_prob, is_gain=False, append=True)
            dump_candidates(
                dev_cands[idx], Y_prob, "current_dev_probs.csv", is_gain=False
            )

    end = timer()
    logger.warning(
        f"Classification AND dump data Time (min): {((end - start) / 60.0):.1f}"
    )
