# Experiment Scripts for SAT-Based Decision Tree Research

Modular utilities for running experiments with the DT1 classifier on PMLB datasets.

## Quick Start

```python
from scripts.experiments.sampling_study import run_sampling_experiment

df = run_sampling_experiment(
    dataset_name="cars",
    sampling_rates=[0.05, 0.10, 0.15],
    n_runs=3,
    timeout=60.0,
    output_path="results.csv",
)
```

## Modules

### `scripts.data` — Data Loading & Preprocessing

| Function | Description |
|----------|-------------|
| `load_dataset(name)` | Fetch raw PMLB dataset |
| `load_and_preprocess(name, **kwargs)` | Fetch + full preprocessing pipeline |
| `preprocess(features, labels, **kwargs)` | Apply preprocessing steps |
| `filter_by_frequency(features, min_freq=0.05)` | Remove rare/too-common features |
| `select_k_best(features, labels, k)` | Select top-k features (mutual info) |
| `make_consistent(features, labels)` | Remove inconsistent samples |

**Example:**
```python
from scripts.data import load_and_preprocess

X, y = load_and_preprocess(
    "cars",
    random_state=42,
    min_feature_freq=0.05,
    n_features=20,
)
```

### `scripts.training` — Training Utilities

| Function/Class | Description |
|----------------|-------------|
| `train_dt1(X_train, y_train, X_test, y_test, timeout=60)` | Train DT1, return DT1Result |
| `train_sklearn_dt(X_train, y_train, max_depth=None)` | Train sklearn DT, return SklearnDTResult |

**Result dataclasses:**
```python
@dataclass
class DT1Result:
    classifier: DT1Classifier
    train_accuracy: float
    test_accuracy: float | None
    elapsed: float
    n_nodes: int
    build_result: BuildResult  # via property
```

### `scripts.experiments` — Experiment Runners

#### `run_sampling_experiment`

Vary training set size and measure DT1 performance.

```python
from scripts.experiments import run_sampling_experiment

df = run_sampling_experiment(
    dataset_name="cars",
    sampling_rates=[0.05, 0.10, 0.15, 0.20],
    n_runs=3,           # runs per sampling rate
    train_size=0.8,     # train/test split
    timeout=60.0,       # per-tree timeout
    max_duration=300,   # total experiment limit
    output_path="sampling_results.csv",
)
```

Returns DataFrame with columns: `r, samples, features, tree_size, train_acc, test_acc, time_s`

#### `compare_dt1_vs_sklearn`

Compare DT1 with sklearn DecisionTreeClassifier.

```python
from scripts.experiments import compare_dt1_vs_sklearn

df = compare_dt1_vs_sklearn(
    dataset_name="cars",
    sampling_rates=[0.05, 0.10, 0.15],
    n_runs=3,
    timeout=60.0,
    sklearn_max_depth=5,
)
```

Returns DataFrame with comparison metrics.

## Preprocessing Options

The `preprocess()` function supports these kwargs:

| Argument | Default | Description |
|----------|---------|-------------|
| `encode_features` | `"onehot"` | `"onehot"` or `"none"` |
| `binarize_labels` | `"ovr"` | `"ovr"` (one-vs-rest) or `"none"` |
| `min_feature_freq` | `0.05` | Filter features with freq < min |
| `feature_selection` | `"kbest"` | `"kbest"` or `None` |
| `n_features` | `None` | Number of features if k-best |
| `ensure_consistent` | `True` | Remove inconsistent samples |

## Running from Command Line

```bash
uv run python -c "
from scripts.experiments.sampling_study import run_sampling_experiment

run_sampling_experiment(
    dataset_name='cars',
    sampling_rates=[0.05, 0.10, 0.15],
    n_runs=2,
    timeout=30.0,
    output_path='experiment_results.csv',
)
"
```
