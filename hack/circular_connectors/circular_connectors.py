#!/usr/bin/env python
# coding: utf-8

import logging
import os

import torch
from fonduer import Meta, init_logging
from fonduer.candidates import CandidateExtractor, MentionExtractor, MentionFigures
from fonduer.candidates.matchers import _Matcher
from fonduer.candidates.models import Mention, candidate_subclass, mention_subclass
from fonduer.parser.models import Document, Figure, Paragraph, Section, Sentence
from metal import EndModel
from metal.tuners import RandomSearchTuner
from PIL import Image

from hack.circular_connectors.disc_model.torchnet import get_cnn
from hack.circular_connectors.utils import ImageList, transform
from hack.utils import parse_dataset

# Configure logging for Fonduer
logger = logging.getLogger(__name__)

TRUE = 1
FALSE = 2
ABSTAIN = 0

PARALLEL = 16  # assuming a quad-core machine
ATTRIBUTE = "circular_connectors_full"
conn_string = "postgresql:///" + ATTRIBUTE

# If you've run this before, set FIRST_TIME to False to save time
FIRST_TIME = True

log_config = {"log_dir": "./run_logs", "run_name": "image"}

tuner_config = {"max_search": 3}

em_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # Network
    # The first value is the output dim of the input module (or the sum of
    # the output dims of all the input modules if multitask=True and
    # multiple input modules are provided). The last value is the
    # output dim of the head layer (i.e., the cardinality of the
    # classification task). The remaining values are the output dims of
    # middle layers (if any). The number of middle layers will be inferred
    # from this list.
    #     "layer_out_dims": [10, 2],
    # Input layer configs
    "input_layer_config": {
        "input_relu": False,
        "input_batchnorm": False,
        "input_dropout": 0.0,
    },
    # Middle layer configs
    "middle_layer_config": {
        "middle_relu": False,
        "middle_batchnorm": False,
        "middle_dropout": 0.0,
    },
    # Can optionally skip the head layer completely, for e.g. running baseline
    # models...
    "skip_head": True,
    # GPU
    "device": "cuda",
    # MODEL CLASS
    "resnet18"
    # DATA CONFIG
    "src": "gm",
    # TRAINING
    "train_config": {
        # Display
        "print_every": 1,  # Print after this many epochs
        "disable_prog_bar": False,  # Disable progress bar each epoch
        # Dataloader
        "data_loader_config": {"batch_size": 32, "num_workers": 8, "sampler": None},
        # Loss weights
        "loss_weights": [0.5, 0.5],
        # Train Loop
        "n_epochs": 20,
        # 'grad_clip': 0.0,
        "l2": 0.0,
        # "lr": 0.01,
        "validation_metric": "accuracy",
        "validation_freq": 1,
        # Evaluate dev for during training every this many epochs
        # Optimizer
        "optimizer_config": {
            "optimizer": "adam",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
            # Optimizer - Adam
            "adam_config": {"betas": (0.9, 0.999)},
        },
        # Scheduler
        "scheduler_config": {
            "scheduler": "reduce_on_plateau",
            # ['constant', 'exponential', 'reduce_on_plateu']
            # Freeze learning rate initially this many epochs
            "lr_freeze": 0,
            # Scheduler - exponential
            "exponential_config": {"gamma": 0.9},  # decay rate
            # Scheduler - reduce_on_plateau
            "plateau_config": {
                "factor": 0.5,
                "patience": 1,
                "threshold": 0.0001,
                "min_lr": 1e-5,
            },
        },
        # Checkpointer
        "checkpoint": True,
        "checkpoint_config": {
            "checkpoint_min": -1,
            # The initial best score to beat to merit checkpointing
            "checkpoint_runway": 0,
            # Don't start taking checkpoints until after this many epochs
        },
    },
}

init_logging(log_dir="logs")
session = Meta.init(conn_string).Session()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
logger.info(f"CWD: {os.getcwd()}")
dirname = "."

docs, train_docs, dev_docs, test_docs = parse_dataset(
    session, dirname, first_time=FIRST_TIME, parallel=PARALLEL, max_docs=float("inf")
)
logger.info(f"# of train Documents: {len(train_docs)}")
logger.info(f"# of dev Documents: {len(dev_docs)}")
logger.info(f"# of test Documents: {len(test_docs)}")

logger.info(f"Documents: {session.query(Document).count()}")
logger.info(f"Sections: {session.query(Section).count()}")
logger.info(f"Paragraphs: {session.query(Paragraph).count()}")
logger.info(f"Sentences: {session.query(Sentence).count()}")
logger.info(f"Figures: {session.query(Figure).count()}")

Thumbnails = mention_subclass("Thumbnails")

thumbnails_img = MentionFigures()


class HasFigures(_Matcher):
    def _f(self, m):
        file_path = ""
        for prefix in ["data/train/html/", "data/dev/html/", "data/test/html/"]:
            if os.path.exists(prefix + m.figure.url):
                file_path = prefix + m.figure.url
        if file_path == "":
            return False
        img = Image.open(file_path)
        width, height = img.size
        min_value = min(width, height)
        return min_value > 50


mention_extractor = MentionExtractor(
    session, [Thumbnails], [thumbnails_img], [HasFigures()], parallelism=PARALLEL
)

if FIRST_TIME:
    mention_extractor.apply(docs)

logger.info("Total Mentions: {}".format(session.query(Mention).count()))

ThumbnailLabel = candidate_subclass("ThumbnailLabel", [Thumbnails])

candidate_extractor = CandidateExtractor(
    session, [ThumbnailLabel], throttlers=[None], parallelism=PARALLEL
)

if FIRST_TIME or True:
    candidate_extractor.apply(train_docs, split=0)
    candidate_extractor.apply(dev_docs, split=1)
    candidate_extractor.apply(test_docs, split=2)

train_cands = candidate_extractor.get_candidates(split=0)
dev_cands = candidate_extractor.get_candidates(split=1)
test_cands = candidate_extractor.get_candidates(split=2)

logger.info("Total train candidate:\t{}".format(len(train_cands[0])))
logger.info("Total dev candidate:\t{}".format(len(dev_cands[0])))
logger.info("Total test candidate:\t{}".format(len(test_cands[0])))

fin = open("data/ground_truth.txt", "r")
gt = set()
for line in fin:
    gt.add("::".join(line.lower().split()))
fin.close()


def LF_gt_label(c):
    doc_file_id = (
        f"{c[0].context.figure.document.name.lower()}.pdf::"
        f"{os.path.basename(c[0].context.figure.url.lower())}"
    )
    return TRUE if doc_file_id in gt else FALSE


ans = {0: 0, 1: 0, 2: 0}

gt_dev_pb = []
gt_dev = []
gt_test = []

for cand in dev_cands[0]:
    if LF_gt_label(cand) == 1:
        ans[1] += 1
        gt_dev_pb.append([1.0, 0.0])
        gt_dev.append(1.0)
    else:
        ans[2] += 1
        gt_dev_pb.append([0.0, 1.0])
        gt_dev.append(2.0)

ans = {0: 0, 1: 0, 2: 0}
for cand in test_cands[0]:
    gt_test.append(LF_gt_label(cand))
    ans[gt_test[-1]] += 1

batch_size = 64
input_size = 224

train_loader = torch.utils.data.DataLoader(
    ImageList(
        data=dev_cands[0],
        label=torch.Tensor(gt_dev_pb),
        transform=transform(input_size),
        prefix="data/dev/html/",
    ),
    batch_size=batch_size,
    shuffle=False,
)

dev_loader = torch.utils.data.DataLoader(
    ImageList(
        data=dev_cands[0],
        label=gt_dev,
        transform=transform(input_size),
        prefix="data/dev/html/",
    ),
    batch_size=batch_size,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    ImageList(
        data=test_cands[0],
        label=gt_test,
        transform=transform(input_size),
        prefix="data/test/html/",
    ),
    batch_size=100,
    shuffle=False,
)

search_space = {
    "l2": [0.001, 0.0001, 0.00001],  # linear range
    "lr": {"range": [0.0001, 0.1], "scale": "log"},  # log range
}

train_config = em_config["train_config"]

# Defining network parameters
num_classes = 2
fc_size = 2
hidden_size = 2
pretrained = True

# Set CUDA device
torch.cuda.set_device(1)
# os.environ['CUDA_VISIBLE_DEVICES']='0'

# Initializing input module
input_module = get_cnn("resnet18", pretrained=pretrained, num_classes=num_classes)

# Initializing model object
init_args = [[num_classes]]
init_kwargs = {"input_module": input_module}
init_kwargs.update(em_config)

# Searching model
searcher = RandomSearchTuner(EndModel, **log_config)

end_model = searcher.search(
    search_space,
    dev_loader,
    train_args=[train_loader],
    init_args=init_args,
    init_kwargs=init_kwargs,
    train_kwargs=train_config,
    max_search=tuner_config["max_search"],
)

# Evaluating model
scores = end_model.score(
    test_loader, metric=["accuracy", "precision", "recall", "f1"], break_ties="abstain"
)

# ```
# ============================================================
# [SUMMARY]
# Best model: [0]
# Best config: {'l2': 0.0001, 'lr': 0.0010025532524850966}
# Best score: 0.9955849889624724
# ============================================================
# Accuracy: 0.926
# Precision: 0.675
# Recall: 0.856
# F1: 0.755
# Roc-auc: 0.964
#         y=1    y=2
#  l=1    137    66
#  l=2    23     982
# ```

# labels, _, probs = end_model._get_predictions(
#     test_loader, return_probs=True, break_ties="abstain"
# )
#
# pred_dict = {}
#
# for c, y in zip(test_cands[0], probs[:, 0]):
#     doc_file_id = (
#         f"{c[0].context.figure.document.name.lower()}.pdf::"
#         f"{os.path.basename(c[0].context.figure.url.lower())}"
#     )
#     pred_dict[doc_file_id] = y
#
# all_test_fig_id = set()
#
# for doc in test_docs:
#     for fig in doc.figures:
#         doc_file_id = f"{doc.name.lower()}.pdf::{os.path.basename(fig.url.lower())}"
#         all_test_fig_id.add(doc_file_id)
#
# b = 0.7
#
# tp = 0
# fp = 0
# fn = 0
#
# for id in all_test_fig_id:
#     if id in gt:
#         p = True
#     else:
#         p = False
#     if id in pred_dict and pred_dict[id] >= b:
#         t = True
#     else:
#         t = False
#
#     if t and p:
#         tp += 1
#     if t and not p:
#         fp += 1
#     if not t and p:
#         fn += 1
#
# prec = tp / (tp + fp)
# rec = tp / (tp + fn)
# f1 = 2 * prec * rec / (prec + rec)
#
# tp, fp, fn, prec, rec, f1
