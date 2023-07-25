from setuptools import find_namespace_packages, find_packages, setup

setup(
    name="modular-addition",
    version="0.0.1",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "dm-haiku",
        "optax",
        "jax[cuda12_pip]",
        "setuptools",
        "plotly",
        "pandas",
        "tqdm",
        "typing_extensions",
    ],
)
