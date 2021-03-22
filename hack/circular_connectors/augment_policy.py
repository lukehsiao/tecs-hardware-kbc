import logging
import random

from hack.circular_connectors.transforms import ALL_TRANSFORMS

logger = logging.getLogger(__name__)

PRECISION = 3


def parse_sequence(x, type="int"):
    x = x.split(",")
    if len(x) == 1:
        return int(x[0]) if type == "int" else float(x[0])
    return tuple([int(_) if type == "int" else float(_) for _ in x])


def parse_transform(policy):
    parsed_transforms = []

    if policy is None:
        return parsed_transforms

    transforms = policy.split("@")
    for transform in transforms:
        name = transform.split("_")[0]
        settings = transform.split("_")[1:] if len(transform.split("_")) > 1 else []
        if name in ALL_TRANSFORMS:
            prob = random.random()
            level = random.randint(0, 10 ** PRECISION) / float(10 ** PRECISION)
            config = {"prob": prob, "level": level}
            for setting in settings:
                if setting.startswith("PD"):
                    config["padding"] = parse_sequence(setting[2:], type="int")
                elif setting.startswith("PIN"):
                    config["pad_if_needed"] = bool(setting[3:])
                elif setting.startswith("PM"):
                    config["padding_mode"] = str(setting[2:])
                elif setting.startswith("MP"):
                    config["max_pixel"] = int(setting[2:])
                elif setting.startswith("MD"):
                    config["max_degree"] = int(setting[2:])
                elif setting.startswith("P"):
                    config["prob"] = float(setting[1:])
                elif setting.startswith("L"):
                    config["level"] = float(setting[1:])
                elif setting.startswith("S"):
                    config["size"] = parse_sequence(setting[1:], type="int")
                elif setting.startswith("A"):
                    config["alpha"] = float(setting[1:])
                elif setting.startswith("R"):
                    config["same_class_ratio"] = float(setting[1:])
                elif setting.startswith("I"):
                    config["interpolation"] = int(setting[1:])
                elif setting.startswith("B"):
                    config["brightness"] = float(setting[1:])
                elif setting.startswith("C"):
                    config["contrast"] = float(setting[1:])
                elif setting.startswith("T"):
                    config["saturation"] = float(setting[1:])
            parsed_transforms.append(ALL_TRANSFORMS[name](**config))
        else:
            raise ValueError(f"Unrecognized transformation {transforms}")

    return parsed_transforms


class Augmentation(object):
    """Given the augment policy to generate the list of augmentation functions."""

    def __init__(self, num_comp=1):
        self.num_comp = num_comp

    def __call__(self):
        all_transforms = [
            "AutoContrast",
            "Brightness",
            "Color",
            "Contrast",
            "Cutout_MP100",
            "Equalize",
            "Invert",
            "Posterize",
            "Rotate",
            "Sharpness",
            "ShearX",
            "ShearY",
            "Solarize",
            "TranslateX",
            "TranslateY",
        ]

        transform_keys = "@".join(
            random.choices(all_transforms, k=self.num_comp)
            + [
                "RandomCrop_P1_S224_PD20_PMreflect",
                "HorizontalFlip_P0.5",
                "Cutout_P1_MP100",
            ]
        )
        transforms = parse_transform(transform_keys)

        return transforms
