"""
This file configures the Python package with entrypoints used for future runs on Databricks.

Please follow the `entry_points` documentation for more details on how to configure the entrypoint:
* https://setuptools.pypa.io/en/latest/userguide/entry_point.html
"""

from setuptools import find_packages, setup
from lendingclub_scoring import __version__

PACKAGE_REQUIREMENTS = ["pyyaml"]

# packages for local development and unit testing
# please note that these packages are already available in DBR, there is no need to install them on DBR.
LOCAL_REQUIREMENTS = [
    "pyspark==3.2.1",
    "delta-spark==1.1.0",
    "scikit-learn",
    "pandas",
    "mlflow",
    "lightgbm",
    "hyperopt"
]

TEST_REQUIREMENTS = [
    # development & testing tools
    "pytest",
    "pytest-mock",
    "coverage[toml]",
    "pytest-cov",
    "dbx>=0.7,<0.8"
]

setup(
    name="lendingclub_scoring",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["setuptools","wheel"],
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require={"local": LOCAL_REQUIREMENTS, "test": TEST_REQUIREMENTS},
    entry_points = {
        "console_scripts": [
            "train = lendingclub_scoring.tasks.train:entrypoint",
            "abtest = lendingclub_scoring.tasks.abtest:entrypoint",
            "eval = lendingclub_scoring.tasks.eval:entrypoint",
            "score = lendingclub_scoring.tasks.score:entrypoint"
    ]},
    version=__version__,
    description="",
    author="",
)
