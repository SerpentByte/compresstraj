from setuptools import setup, find_packages

setup(
    name="compresstraj",
    version="2.0.0",
    author="SerpentBye",
    description="Molecular dynamics trajectory compression tools using autoencoders",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pytorch-lightning",
        "MDAnalysis",
        "matplotlib",
        "tqdm"
    ],
    python_requires=">=3.8",
)

