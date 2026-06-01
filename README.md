# TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks

## What This Repo Contains
This repository contains examples how to use the feature generators introduced in "TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks". The current version provides a scikit-learn wrapper around the TabPrep feature generation logic that is implemented in AutoGluon. The main example in this repo is `example_sklearn.py`, which compares raw features against TabPrep-transformed features using standard sklearn models. More examples, in particular how to integrate TabPrep within benchmarking workflows of the TabArena benchmark will follow soon.

## ⚡ Quickstart
In a Python 3.11–3.13 environment, run:
```bash
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh
git clone https://github.com/autogluon/tabarena.git
uv pip install --prerelease=allow -e "./tabarena/tabarena[benchmark]"
python example_sklearn.py
```


## TabPrep Preprocessors
TabPrep is built from a small set of feature generators that target common tabular patterns. The preprocessors are implemented in AutoGluon's features module:

- `GroupByFeatureGenerator`: creates aggregate features of numerical features relative to group reference values from categorical columns.
- `RandomSubsetFeatureCompressionGenerator`: compresses (numerical) feature subsets into single numerical features to capture higher-order value-based (pseudo-categorical) interactions.
- `ArithmeticFeatureGenerator`: Applies Ordered Arithmetic Feature Expansion adding arithmetic combinations of numerical features.
- `CategoricalInteractionFeatureGenerator`: Adds categorical cross-features to capture interaction effects.
- `OOFTargetEncodingFeatureGenerator`: encodes categoricals with out-of-fold target statistics.

These generators are combined into the `TabPrepFeatureGenerator` in [tabprep.py](./tabprep.py). 

## General recommendations to maximize predictive performance with TabPrep
- Avoid data leakage when using target-aware generators such as out-of-fold target encoding.
- Pair TabPrep with downstream models that can ignore useless features, especially when many generated features are added.
- Treat the generator flags as a search space: different datasets benefit from different combinations of preprocessors and seeds.
- TabPrep is most useful when the data contains grouped structure, categorical interactions, or simple numeric relationships that a model may not discover on its own.

## Citation
If you use TabPrep in a publication, please reference
TabPrep: Closing the Feature Engineering Gap in Tabular Benchmarks. Andrej Tschalzev, Nick Erickson, Yuyang Wang, Huzefa Rangwala, Stefan Lüdtke, Heiner Stuckenschmidt, Christian Bartelt. arXiv preprint, 2026. 

