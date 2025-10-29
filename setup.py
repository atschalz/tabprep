from __future__ import annotations
from setuptools import find_packages, setup

requirements = [
    "pyyaml",
    "pytest",
    "typing-extensions>=4.11,<5",  # used for `Self` type hint
    "huggingface-hub",
    'kditransform',
    'skrub',
]

extras_require = {
    "category_encoders": [
        "category_encoders",  
    ],
}

benchmark_requires = []
for extra_package in [
    "category_encoders",
]:
    benchmark_requires += extras_require[extra_package]
benchmark_requires = list(set(benchmark_requires))
extras_require["benchmark"] = benchmark_requires

# FIXME: For 2025 paper, cleanup after
extras_require["benchmark"] += [
    # "seaborn==0.13.2",
    # "matplotlib==3.9.2",
    # "autorank==1.2.1",
    # "fastparquet",  # FIXME: Without this, parquet loading is inconsistent for list columns
    # "tueplots",
]

setup(
    name="tabprep",
    version="0.0.1",
    packages=find_packages(exclude=("__pycache__", "AutoGluonModels", "experiments", 'figures', 'repos', 'results')),
    package_data={
        # "tabrepo": [
        #     "metrics/_roc_auc_cpp/compile.sh",
        #     "metrics/_roc_auc_cpp/cpp_auc.cpp",
        #     "nips2025_utils/metadata/task_metadata_tabarena51.csv",
        #     "nips2025_utils/metadata/task_metadata_tabarena60.csv",
        #     "nips2025_utils/metadata/task_metadata_tabarena61.csv",
        # ],
    },
    # url="https://github.com/autogluon/tabrepo",
    license="Apache-2.0",
    author="AutoGluon Community",
    install_requires=requirements,
    extras_require=extras_require,
    description="Tabular data preprocessing and evaluation utilities for AutoGluon",
)
