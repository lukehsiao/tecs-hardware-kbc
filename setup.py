from setuptools import find_packages, setup

setup(
    name="hack",
    version="0.1.2",
    description="Building a hardware component knowledge base.",
    install_requires=[
        "fonduer>=0.8.0, <0.9.0",
        "matplotlib",
        "numpy",
        "pandas<0.26.0,>=0.25.0",
        "pillow",
        "quantiphy",
        "scipy",
        "seaborn",
        "snorkel>=0.9.5",
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
    ],
    packages=find_packages(),
)
