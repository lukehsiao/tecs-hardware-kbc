import logging
from functools import partial

import torch.nn.functional as F
from emmental.modules.identity_module import IdentityModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from torch import nn

from hack.circular_connectors.modules.soft_cross_entropy_loss import (
    SoftCrossEntropyLoss,
)
from hack.circular_connectors.modules.torchnet import get_cnn

logger = logging.getLogger(__name__)


SCE = SoftCrossEntropyLoss(reduction="none")


def sce_loss(module_name, intermediate_output_dict, Y, active):
    if len(Y.size()) == 1:
        label = intermediate_output_dict[module_name][0].new_zeros(
            intermediate_output_dict[module_name][0].size()
        )
        label.scatter_(1, Y.view(Y.size()[0], 1), 1.0)
    else:
        label = Y

    return SCE(intermediate_output_dict[module_name][0][active], label[active])


def output_classification(module_name, immediate_output_dict):
    return F.softmax(immediate_output_dict[module_name][0], dim=1)


def create_task(task_name, n_class=2, model="resnet18", pretrained=True):

    feature_extractor = get_cnn(model, pretrained, num_classes=n_class)

    loss = sce_loss
    output = output_classification

    logger.info(f"Built model: {feature_extractor}")

    return EmmentalTask(
        name=task_name,
        module_pool=nn.ModuleDict(
            {"feature": feature_extractor, f"{task_name}_pred_head": IdentityModule()}
        ),
        task_flow=[
            {"name": "feature", "module": "feature", "inputs": [("_input_", "image")]},
            {
                "name": f"{task_name}_pred_head",
                "module": f"{task_name}_pred_head",
                "inputs": [("feature", 0)],
            },
        ],
        loss_func=partial(loss, f"{task_name}_pred_head"),
        output_func=partial(output, f"{task_name}_pred_head"),
        scorer=Scorer(metrics=["precision", "recall", "f1"]),
    )
