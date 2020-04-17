import numpy as np
import torch
from emmental.data import EmmentalDataset
from emmental.utils.utils import pred_to_prob

from hack.circular_connectors.transforms.compose import Compose
from hack.circular_connectors.transforms.normalize import Normalize
from hack.circular_connectors.transforms.resize import Resize
from hack.circular_connectors.transforms.to_tensor import ToTensor
from hack.circular_connectors.utils import default_loader


class ThumbnailDataset(EmmentalDataset):
    """Dataset to load thumbnail dataset.
    """

    def __init__(
        self,
        name,
        dataset,
        labels,
        split="train",
        transform_cls=None,
        prefix="",
        prob_label=False,
        input_size=224,
        k=1,
    ):
        X_dict, Y_dict = {"image_name": []}, {"labels": []}
        for i, (x, y) in enumerate(zip(dataset, labels)):
            X_dict["image_name"].append(f"{prefix}{x[0].context.figure.url}")
            Y_dict["labels"].append(y)

        if prob_label:
            labels = pred_to_prob(np.array(Y_dict["labels"]), 2)
        else:
            labels = np.array(Y_dict["labels"])

        Y_dict["labels"] = torch.from_numpy(labels)

        self.transform_cls = transform_cls
        self.transforms = None

        self.defaults = [
            Resize(input_size),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]

        # How many augmented samples to augment for each sample
        self.k = k if k is not None else 1

        super().__init__(name, X_dict=X_dict, Y_dict=Y_dict, uid="image_name")

    def gen_transforms(self):
        if self.transform_cls is not None:
            return self.transform_cls()
        else:
            return []

    def __getitem__(self, index):
        r"""Get item by index.
        Args:
          index(index): The index of the item.
        Returns:
          Tuple[Dict[str, Any], Dict[str, Tensor]]: Tuple of x_dict and y_dict
        """
        x_dict = {name: feature[index] for name, feature in self.X_dict.items()}
        y_dict = {name: label[index] for name, label in self.Y_dict.items()}
        x_dict["image"] = default_loader(x_dict["image_name"])

        new_x_dict = {}
        new_y_dict = {"labels": []}

        for name, feature in x_dict.items():
            if name not in new_x_dict:
                new_x_dict[name] = []
            if name == self.uid:
                for i in range(self.k):
                    new_x_dict[name].append(f"{feature}_{i}")
            elif name == "image":
                for i in range(self.k):
                    if self.transform_cls is None:
                        self.transforms = self.defaults
                    else:
                        self.transforms = self.gen_transforms() + self.defaults
                    new_img, new_label = Compose(self.transforms)(
                        feature,
                        y_dict["labels"],
                        X_dict=self.X_dict,
                        Y_dict=self.Y_dict,
                        transforms=self.transforms,
                    )
                    new_x_dict[name].append(new_img)
                    new_y_dict["labels"].append(new_label)
            else:
                for i in range(self.k):
                    new_x_dict[name].append(feature)
        for name, feature in y_dict.items():
            if name not in new_y_dict:
                new_y_dict[name] = []
            if name != "labels":
                for i in range(self.k):
                    new_y_dict[name].append(feature)

        return new_x_dict, new_y_dict
