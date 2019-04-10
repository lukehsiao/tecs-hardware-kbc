from setuptools import find_packages, setup

setup(
    name="hack",
    version="0.1.0",
    description="Building a hardware component knowledge base.",
    install_requires=[
        "fonduer>=0.6.2,<0.7.0",
        "matplotlib",
        "pillow",
        "quantiphy",
        "torch",
        "tqdm",
        "torchvision",
    ],
    scripts=["bin/transistors.py"],
    packages=find_packages(),
)
