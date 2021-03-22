#!/usr/bin/env python
# coding: utf-8

import logging
import os
from timeit import default_timer as timer

import emmental
import torch
from emmental.data import EmmentalDataLoader
from emmental.learner import EmmentalLearner
from emmental.model import EmmentalModel
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionFigures
from fonduer.candidates.matchers import _Matcher
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from PIL import Image

from hack.circular_connectors.augment_policy import Augmentation
from hack.circular_connectors.config import emmental_config
from hack.circular_connectors.scheduler import DauphinScheduler
from hack.circular_connectors.task import create_task
from hack.circular_connectors.thumbnail_dataset import ThumbnailDataset
from hack.utils import parse_dataset

# Configure logging for Fonduer
logger = logging.getLogger(__name__)

TRUE = 1
FALSE = 0

torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # type: ignore


def main(
    conn_string,
    max_docs=float("inf"),
    parse=False,
    first_time=False,
    gpu=None,
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

    session = Meta.init(conn_string).Session()

    # Parsing
    logger.info("Starting parsing...")
    start = timer()
    docs, train_docs, dev_docs, test_docs = parse_dataset(
        session, dirname, first_time=first_time, parallel=parallel, max_docs=max_docs
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

    start = timer()

    Thumbnails = mention_subclass("Thumbnails")

    thumbnails_img = MentionFigures()

    class HasFigures(_Matcher):
        def _f(self, m):
            file_path = ""
            for prefix in [
                f"{dirname}/data/train/html/",
                f"{dirname}/data/dev/html/",
                f"{dirname}/data/test/html/",
            ]:
                if os.path.exists(prefix + m.figure.url):
                    file_path = prefix + m.figure.url
            if file_path == "":
                return False
            img = Image.open(file_path)
            width, height = img.size
            min_value = min(width, height)
            return min_value > 50

    mention_extractor = MentionExtractor(
        session, [Thumbnails], [thumbnails_img], [HasFigures()], parallelism=parallel
    )

    if first_time:
        mention_extractor.apply(docs)

    logger.info("Total Mentions: {}".format(session.query(Mention).count()))

    ThumbnailLabel = candidate_subclass("ThumbnailLabel", [Thumbnails])

    candidate_extractor = CandidateExtractor(
        session, [ThumbnailLabel], throttlers=[None], parallelism=parallel
    )

    if first_time:
        candidate_extractor.apply(train_docs, split=0)
        candidate_extractor.apply(dev_docs, split=1)
        candidate_extractor.apply(test_docs, split=2)

    train_cands = candidate_extractor.get_candidates(split=0)
    # Sort the dev_cands, which are used for training, for deterministic behavior
    dev_cands = candidate_extractor.get_candidates(split=1, sort=True)
    test_cands = candidate_extractor.get_candidates(split=2)

    end = timer()
    logger.warning(f"Candidate Extraction Time (min): {((end - start) / 60.0):.1f}")

    logger.info("Total train candidate:\t{}".format(len(train_cands[0])))
    logger.info("Total dev candidate:\t{}".format(len(dev_cands[0])))
    logger.info("Total test candidate:\t{}".format(len(test_cands[0])))

    fin = open(f"{dirname}/data/ground_truth.txt", "r")
    gt = set()
    for line in fin:
        gt.add("::".join(line.lower().split()))
    fin.close()

    # Labeling
    start = timer()

    def LF_gt_label(c):
        doc_file_id = (
            f"{c[0].context.figure.document.name.lower()}.pdf::"
            f"{os.path.basename(c[0].context.figure.url.lower())}"
        )
        return TRUE if doc_file_id in gt else FALSE

    gt_dev = [LF_gt_label(cand) for cand in dev_cands[0]]
    gt_test = [LF_gt_label(cand) for cand in test_cands[0]]

    end = timer()
    logger.warning(f"Supervision Time (min): {((end - start) / 60.0):.1f}")

    batch_size = 64
    input_size = 224
    K = 2

    emmental.init(log_dir=Meta.log_path, config=emmental_config)

    emmental.Meta.config["learner_config"]["task_scheduler_config"][
        "task_scheduler"
    ] = DauphinScheduler(augment_k=K, enlarge=1)

    train_dataset = ThumbnailDataset(
        "Thumbnail",
        dev_cands[0],
        gt_dev,
        "train",
        prob_label=True,
        prefix=f"{dirname}/data/dev/html/",
        input_size=input_size,
        transform_cls=Augmentation(2),
        k=K,
    )

    val_dataset = ThumbnailDataset(
        "Thumbnail",
        dev_cands[0],
        gt_dev,
        "valid",
        prob_label=False,
        prefix=f"{dirname}/data/dev/html/",
        input_size=input_size,
        k=1,
    )

    test_dataset = ThumbnailDataset(
        "Thumbnail",
        test_cands[0],
        gt_test,
        "test",
        prob_label=False,
        prefix=f"{dirname}/data/test/html/",
        input_size=input_size,
        k=1,
    )

    dataloaders = []

    dataloaders.append(
        EmmentalDataLoader(
            task_to_label_dict={"Thumbnail": "labels"},
            dataset=train_dataset,
            split="train",
            shuffle=True,
            batch_size=batch_size,
            num_workers=1,
        )
    )

    dataloaders.append(
        EmmentalDataLoader(
            task_to_label_dict={"Thumbnail": "labels"},
            dataset=val_dataset,
            split="valid",
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
        )
    )

    dataloaders.append(
        EmmentalDataLoader(
            task_to_label_dict={"Thumbnail": "labels"},
            dataset=test_dataset,
            split="test",
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
        )
    )

    model = EmmentalModel(name="Thumbnail")
    model.add_task(
        create_task("Thumbnail", n_class=2, model="resnet18", pretrained=True)
    )

    emmental_learner = EmmentalLearner()
    emmental_learner.learn(model, dataloaders)

    scores = model.score(dataloaders)

    logger.warning("Model Score:")
    logger.warning(f"precision: {scores['Thumbnail/Thumbnail/test/precision']:.3f}")
    logger.warning(f"recall: {scores['Thumbnail/Thumbnail/test/recall']:.3f}")
    logger.warning(f"f1: {scores['Thumbnail/Thumbnail/test/f1']:.3f}")
