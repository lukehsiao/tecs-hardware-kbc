from setuptools import find_packages, setup

setup(
    name="hack",
    version="0.1.2",
    description="Building a hardware component knowledge base.",
    install_requires=[
        "fonduer @ git+https://github.com/HazyResearch/fonduer.git@emmental",
        "matplotlib",
        "numpy",
        "pandas<0.26.0,>=0.25.0",
        "pillow",
        "quantiphy",
        "scipy",
        "seaborn",
        "statsmodels",
        "torch>=1.3.0,<2.0.0",
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
