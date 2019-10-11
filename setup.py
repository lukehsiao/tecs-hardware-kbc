from setuptools import find_packages, setup

setup(
    name="hack",
    version="0.1.2",
    description="Building a hardware component knowledge base.",
    install_requires=[
        "fonduer @ git+https://github.com/HazyResearch/fonduer.git@emmental",
        "matplotlib",
        "numpy",
        "pandas<0.24.0,>=0.23.4",
        "pillow",
        "quantiphy",
        "scipy",
        "seaborn",
        "torch",
        "tqdm",
        "torchvision",
    ],
    scripts=[
        "bin/transistors",
        "bin/opamps",
        "bin/circular_connectors",
        "bin/analysis",
        "bin/figure10",
    ],
    packages=find_packages(),
)
