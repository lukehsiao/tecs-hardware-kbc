from hack.circular_connectors.transforms.auto_contrast import AutoContrast
from hack.circular_connectors.transforms.brightness import Brightness
from hack.circular_connectors.transforms.color import Color
from hack.circular_connectors.transforms.contrast import Contrast
from hack.circular_connectors.transforms.cutout import Cutout
from hack.circular_connectors.transforms.equalize import Equalize
from hack.circular_connectors.transforms.horizontal_filp import HorizontalFlip
from hack.circular_connectors.transforms.identity import Identity
from hack.circular_connectors.transforms.invert import Invert
from hack.circular_connectors.transforms.posterize import Posterize
from hack.circular_connectors.transforms.random_crop import RandomCrop
from hack.circular_connectors.transforms.resize import Resize
from hack.circular_connectors.transforms.rotate import Rotate
from hack.circular_connectors.transforms.sharpness import Sharpness
from hack.circular_connectors.transforms.shear_x import ShearX
from hack.circular_connectors.transforms.shear_y import ShearY
from hack.circular_connectors.transforms.solarize import Solarize
from hack.circular_connectors.transforms.translate_x import TranslateX
from hack.circular_connectors.transforms.translate_y import TranslateY
from hack.circular_connectors.transforms.vertical_flip import VerticalFlip

ALL_TRANSFORMS = {
    "AutoContrast": AutoContrast,
    "Brightness": Brightness,
    "Color": Color,
    "Contrast": Contrast,
    "Cutout": Cutout,
    "Equalize": Equalize,
    "HorizontalFlip": HorizontalFlip,
    "Identity": Identity,
    "Invert": Invert,
    "Posterize": Posterize,
    "RandomCrop": RandomCrop,
    "Resize": Resize,
    "Rotate": Rotate,
    "Sharpness": Sharpness,
    "ShearX": ShearX,
    "ShearY": ShearY,
    "Solarize": Solarize,
    "TranslateX": TranslateX,
    "TranslateY": TranslateY,
    "VerticalFlip": VerticalFlip,
}
