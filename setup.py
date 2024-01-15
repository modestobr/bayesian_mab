import setuptools
from typing import List
import os

PACKAGE_NAME = "bayesian_mab"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open("./bayesian_mab/VERSION") as fh:
    version = fh.read()

def get_requirements() -> List[str]:
    if os.path.exists("requirements.txt"):
        requirements_path = "requirements.txt"
    else:
        requirements_path = f"{PACKAGE_NAME}.egg-info/requires.txt"

    with open(requirements_path, encoding="utf8") as config_file:
        requirements = config_file.read().splitlines()
    return [requirement for requirement in requirements if not requirement.startswith("--")]


setuptools.setup(
    name=PACKAGE_NAME,
    version=version,
    author="Brandao, Iago M.",
    description="Bayesian Multi Armed Bandit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/modestobr/bayesian_multi_armed_bandit",
    packages=setuptools.find_packages(
        exclude=["docs"]
    ),
    data_files=["bayesian_mab/VERSION"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    dependency_links=[],
)