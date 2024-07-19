from setuptools import setup, find_packages

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.readlines()

setup(
    name="compressTraj",
    version="2024.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="SerpentByte",
    author_email="wasim.abdul.1995@gmail.com",
    description="Using AutoEncoders to compress molecular dynamics trajectories.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SerpentByte/compressTraj", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

