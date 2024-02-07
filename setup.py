from pathlib import Path

from setuptools import find_packages, setup

SETUP_DIRECTORY = Path(__file__).resolve().parent

with (SETUP_DIRECTORY / "README.md").open() as ifs:
    LONG_DESCRIPTION = ifs.read()

install_requires = (
    [
        "jax>=0.4.14",
        "numpyro>=0.12.0",
        "optuna>=3.0.6",
        "pandas>=2.0.3",
        "pydantic>=2.5.2",
    ],
)

__version__ = "0.0.1"

setup(
    name="mixician",
    version=__version__,
    author="Yin Cheng",
    author_email="yin.sjtu@gmail.com",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yinsn/Mixician",
    python_requires=">=3.7",
    description="An advanced hybrid ranking engine for recommendation systems, designed to automate the optimization of algorithms and parameters tailored to diverse business objectives.",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    include_package_data=True,
)
