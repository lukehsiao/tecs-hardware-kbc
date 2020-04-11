#!/usr/bin/env python
# coding: utf-8

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
from fonduer.candidates.models import (
    Candidate,
    Mention,
    candidate_subclass,
    mention_subclass,
)
from fonduer.features import Featurizer
from fonduer.learning.dataset import FonduerDataset
from fonduer.learning.task import create_task
from fonduer.learning.utils import collect_word_counter
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from fonduer.supervision import Labeler
from snorkel.labeling.model import LabelModel

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

logger = logging.getLogger(__name__)


def load_labels(session, relation, cand, first_time=True):
    if first_time:
        logger.info(f"Loading gold labels for {relation}")
        load_transistor_labels(session, [cand], [relation], annotator_name="gold")


def generative_model(L_train, n_epochs=500, print_every=100):
    model = LabelModel()

    logger.info("Training generative model...")
    model.fit(L_train=L_train, n_epochs=n_epochs, log_freq=print_every)
    logger.info("Done.")

    marginals = model.predict_proba(L_train)
    return marginals


def discriminative_model(train_cands, F_train, marginals, n_epochs=50, lr=0.001):
    raise NotImplementedError


def scoring(relation, test_preds, test_cands, test_docs, F_test, parts_by_doc, num=100):
    logger.info("Calculating the best F1 score and threshold (b)...")

    # Get prediction for a particular b, store the full tuple to output
    # (b, pref, rec, f1, TP, FP, FN)
    best_result = Score(0, 0, 0, [], [], [])
    best_b = 0
    for b in np.linspace(0, 1, num=num):
        try:
            positive = np.where(np.array(test_preds["probs"][relation])[:, TRUE] > b)
            true_pred = [test_cands[_] for _ in positive[0]]
            result = entity_level_scores(
                candidates_to_entities(
                    true_pred, parts_by_doc=parts_by_doc, progress_bar=False
                ),
                attribute=relation,
                corpus=test_docs,
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


def dump_candidates(cands, Y_prob, outfile):
    """Output the candidates and their probabilities for later analysis."""
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, outfile), "w") as csvfile:
        writer = csv.writer(csvfile)
        for i, c in enumerate(cands):
            (doc, part, val) = cand_to_entity(c)
            writer.writerow([doc, part, val, Y_prob[i]])


def main(
    conn_string,
    stg_temp_min=False,
    stg_temp_max=False,
    polarity=False,
    ce_v_max=False,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    re_label=False,
    parallel=4,
    log_dir=None,
    verbose=False,
):
    if not log_dir:
        log_dir = "logs"

    if verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    dirname = os.path.dirname(os.path.abspath(__file__))
    init_logging(log_dir=os.path.join(dirname, log_dir), level=level)

    rel_list = []
    if stg_temp_min:
        rel_list.append("stg_temp_min")

    if stg_temp_max:
        rel_list.append("stg_temp_max")

    if polarity:
        rel_list.append("polarity")

    if ce_v_max:
        rel_list.append("ce_v_max")

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

    # Only do those that are enabled
    Part = mention_subclass("Part")
    part_matcher = get_matcher("part")
    part_ngrams = MentionNgramsPart(parts_by_doc=None, n_max=3)

    mentions.append(Part)
    ngrams.append(part_ngrams)
    matchers.append(part_matcher)

    if stg_temp_min:
        StgTempMin = mention_subclass("StgTempMin")
        stg_temp_min_matcher = get_matcher("stg_temp_min")
        stg_temp_min_ngrams = MentionNgramsTemp(n_max=2)

        mentions.append(StgTempMin)
        ngrams.append(stg_temp_min_ngrams)
        matchers.append(stg_temp_min_matcher)

    if stg_temp_max:
        StgTempMax = mention_subclass("StgTempMax")
        stg_temp_max_matcher = get_matcher("stg_temp_max")
        stg_temp_max_ngrams = MentionNgramsTemp(n_max=2)

        mentions.append(StgTempMax)
        ngrams.append(stg_temp_max_ngrams)
        matchers.append(stg_temp_max_matcher)

    if polarity:
        Polarity = mention_subclass("Polarity")
        polarity_matcher = get_matcher("polarity")
        polarity_ngrams = MentionNgrams(n_max=1)

        mentions.append(Polarity)
        ngrams.append(polarity_ngrams)
        matchers.append(polarity_matcher)

    if ce_v_max:
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
    if stg_temp_min:
        logger.info(f"Total StgTempMin: {session.query(StgTempMin).count()}")
    if stg_temp_max:
        logger.info(f"Total StgTempMax: {session.query(StgTempMax).count()}")
    if polarity:
        logger.info(f"Total Polarity: {session.query(Polarity).count()}")
    if ce_v_max:
        logger.info(f"Total CeVMax: {session.query(CeVMax).count()}")

    # Candidate Extraction
    cands = []
    throttlers = []
    if stg_temp_min:
        PartStgTempMin = candidate_subclass("PartStgTempMin", [Part, StgTempMin])
        stg_temp_min_throttler = stg_temp_filter

        cands.append(PartStgTempMin)
        throttlers.append(stg_temp_min_throttler)

    if stg_temp_max:
        PartStgTempMax = candidate_subclass("PartStgTempMax", [Part, StgTempMax])
        stg_temp_max_throttler = stg_temp_filter

        cands.append(PartStgTempMax)
        throttlers.append(stg_temp_max_throttler)

    if polarity:
        PartPolarity = candidate_subclass("PartPolarity", [Part, Polarity])
        polarity_throttler = polarity_filter

        cands.append(PartPolarity)
        throttlers.append(polarity_throttler)

    if ce_v_max:
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
    for i, name in enumerate(rel_list):
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

    for i, cand in enumerate(cands):
        logger.info(f"{cand} Train shape: {F_train[i].shape}")
        logger.info(f"{cand} Test shape: {F_test[i].shape}")
        logger.info(f"{cand} Dev shape: {F_dev[i].shape}")

    logger.info("Labeling training data...")

    # Labeling
    start = timer()
    lfs = []
    if stg_temp_min:
        lfs.append(stg_temp_min_lfs)

    if stg_temp_max:
        lfs.append(stg_temp_max_lfs)

    if polarity:
        lfs.append(polarity_lfs)

    if ce_v_max:
        lfs.append(ce_v_max_lfs)

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

    if first_time:
        marginals_dict = {}
        for idx, relation in enumerate(rel_list):
            marginals_dict[relation] = generative_model(L_train[idx])

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
    logger.warning(f"Supervision Time (min): {((end - start) / 60.0):.1f}")

    start = timer()

    word_counter = collect_word_counter(train_cands)

    emmental.init(Meta.log_path)

    # Training config
    config = {
        "meta_config": {"verbose": True, "seed": 0},
        "model_config": {"model_path": None, "device": 0, "dataparallel": True},
        "learner_config": {
            "n_epochs": 5,
            "optimizer_config": {"lr": 0.001, "l2": 0.0},
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
    emmental.Meta.update_config(config=config)

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

        train_dataloader.append(
            EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation,
                    train_cands[idx],
                    F_train[idx],
                    emb_layer.word2id,
                    marginals[idx],
                    train_idxs[idx],
                ),
                split="train",
                batch_size=100,
                shuffle=True,
            )
        )

    num_feature_keys = len(featurizer.get_keys())

    model = EmmentalModel(name=f"transistor_tasks")

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

    # List of dataloader for each rlation
    for idx, relation in enumerate(rel_list):
        test_dataloader = EmmentalDataLoader(
            task_to_label_dict={relation: "labels"},
            dataset=FonduerDataset(
                relation, test_cands[idx], F_test[idx], emb_layer.word2id, 2
            ),
            split="test",
            batch_size=100,
            shuffle=False,
        )

        test_preds = model.predict(test_dataloader, return_preds=True)

        best_result, best_b = scoring(
            relation,
            test_preds,
            test_cands[idx],
            test_docs,
            F_test[idx],
            parts_by_doc,
            num=100,
        )

        # Dump CSV files for CE_V_MAX for digi-key analysis
        if relation == "ce_v_max":
            dev_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, dev_cands[idx], F_dev[idx], emb_layer.word2id, 2
                ),
                split="dev",
                batch_size=100,
                shuffle=False,
            )

            dev_preds = model.predict(dev_dataloader, return_preds=True)

            Y_prob = np.array(test_preds["probs"][relation])[:, TRUE]
            dump_candidates(test_cands[idx], Y_prob, "ce_v_max_test_probs.csv")
            Y_prob = np.array(dev_preds["probs"][relation])[:, TRUE]
            dump_candidates(dev_cands[idx], Y_prob, "ce_v_max_dev_probs.csv")

        # Dump CSV files for POLARITY for digi-key analysis
        if relation == "polarity":
            dev_dataloader = EmmentalDataLoader(
                task_to_label_dict={relation: "labels"},
                dataset=FonduerDataset(
                    relation, dev_cands[idx], F_dev[idx], emb_layer.word2id, 2
                ),
                split="dev",
                batch_size=100,
                shuffle=False,
            )

            dev_preds = model.predict(dev_dataloader, return_preds=True)

            Y_prob = np.array(test_preds["probs"][relation])[:, TRUE]
            dump_candidates(test_cands[idx], Y_prob, "polarity_test_probs.csv")
            Y_prob = np.array(dev_preds["probs"][relation])[:, TRUE]
            dump_candidates(dev_cands[idx], Y_prob, "polarity_dev_probs.csv")

    end = timer()
    logger.warning(f"Classification Time (min): {((end - start) / 60.0):.1f}")
