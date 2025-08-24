# API Reference — Disability Job Matching System
_Last updated: 2025-08-23 23:44_

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" />
  <img src="https://img.shields.io/badge/language-EN-blue" />
  <img src="https://img.shields.io/badge/federated-learning-purple" />
  <img src="https://img.shields.io/badge/privacy-DP%20%2B%20Shamir-teal" />
  <img src="https://img.shields.io/badge/blockchain-anchoring-orange" />
</p>

This document defines the **public API surface** of the Disability Job Matching System.  
It covers configuration schema, module-level functions and classes, CLI entry points, data contracts, and operational behaviors.  
The reference applies to the repository layout:

```
Disability Job Matching System/
├── README.md, README_IT.md
├── config.yaml, requirements.txt, streamlit_app.py
├── data/{raw, processed}
├── scripts/{01..10}_*.py, blockchain_data_anchoring.py
├── utils/{feature_engineering.py, scoring.py, parallel_training.py, visualization.py,
           enhanced_shamir_privacy.py, federated_learning.py, federated_data_splitter.py}
├── results/, results_LightGBM_federated/, results_mlp_federated/, results_mlp_federated_privacy/
└── results_blockchain_demo/, visualizations_federated_comparison/, docs/
```

**Scope & Conventions**
- Python ≥ 3.10. Static types are descriptive; runtime enforcement is best-effort.
- File paths are relative to repo root unless noted.
- All examples assume an activated virtual environment and a populated `config.yaml`.
- The Streamlit UI uses these APIs indirectly; this reference targets developers and integrators.


## Table of Contents
1. [Versioning & Stability](#versioning--stability)
2. [Configuration Schema](#configuration-schema)
3. [Data Contracts](#data-contracts)
4. [Modules](#modules)
   - [utils.feature_engineering](#utilsfeature_engineering)
   - [utils.scoring](#utilsscoring)
   - [utils.parallel_training](#utilsparallel_training)
   - [utils.visualization](#utilsvisualization)
   - [utils.federated_data_splitter](#utilsfederated_data_splitter)
   - [utils.federated_learning](#utilsfederated_learning)
   - [utils.enhanced_shamir_privacy](#utilsenhanced_shamir_privacy)
   - [blockchain_data_anchoring](#blockchain_data_anchoring)
5. [CLI & Scripts](#cli--scripts)
6. [Streamlit Integration](#streamlit-integration)
7. [Errors & Exceptions](#errors--exceptions)
8. [Performance Notes](#performance-notes)
9. [Security & Privacy Notes](#security--privacy-notes)
10. [Backward Compatibility](#backward-compatibility)
11. [Changelog (API surface)](#changelog-api-surface)
12. [Appendix A — Legacy API Reference](#appendix-a--legacy-api-reference)


## Versioning & Stability

- **Semantic intent**: Changes that break function signatures, parameter semantics, or return types will be noted in the changelog.
- **Public API**: Only the items documented here are considered stable. Undocumented helpers may change without notice.
- **Config schema**: Backward-compatible additions (new optional keys) are allowed; removals or type changes are breaking.


## Configuration Schema

`config.yaml` keys and types. Values shown are recommended defaults.

```yaml
seed: 42                          # int — global PRNG seed
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies:  data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir:  data/processed
  training_csv:   data/processed/Enhanced_Training_Dataset.csv
  results_dir:    results
ui:
  distance_max_km: 30             # int — default search radius in UI
training:
  model_set: ["LightGBM_Optimized", "MLP"]  # list[str]
  optuna_trials: 50               # int ≥ 1
  calibration: "sigmoid"          # "sigmoid" | "isotonic"
federated:
  rounds: 10                      # int ≥ 1
  min_clients: 3                  # int ≥ 1
  aggregator: "fedavg"            # "fedavg" | "trimmed_mean" | "coordinate_median"
  batch_size: 256                 # int ≥ 1
privacy:
  enabled: true                   # bool
  dp:
    epsilon: 1.0                  # float > 0
    delta: 1e-6                   # float in (0, 1)
    max_grad_norm: 1.0            # float > 0
    accountant: "rdp"             # "rdp"
  secure_agg:
    scheme: "shamir"              # "shamir"
    threshold: "3-of-5"           # string "t-of-n" (t ≤ n)
    dropout_recovery: true        # bool
anchoring:
  enabled: true                   # bool
  backend: "merkle"               # "merkle"
  anchor_every_n: 1               # int ≥ 1
```

**Notes**
- Paths must exist prior to execution; writers will create intermediate directories when possible.
- `distance_max_km` is **30** by default (not 40).
- DP parameters define the per-experiment privacy budget; log outputs should include final ε, δ.


## Data Contracts

### Raw Inputs
- **Candidates** — `Dataset_Candidati_Aggiornato.csv`
- **Companies**  — `Dataset_Aziende_con_Stima_Assunzioni.csv`

Columns are institution-specific; keep naming consistent across regions and versions.

### Processed Artifacts
- **Extended Candidates** — `Dataset_Candidati_Aggiornato_Extended.csv`
- **Extended Companies**  — `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv`
- **Training Table (canonical)** — `Enhanced_Training_Dataset.csv`

**Merged Model Summary**
- `results/merged_model_summary.csv` — consolidated metrics across runs/models.


## Modules
Public classes and functions exported by each module. Helper functions not listed are internal.

> Import style:
```python
from utils.feature_engineering import build_training_table
from utils.scoring import compute_match_score
from utils.federated_learning import FederatedTrainer, Aggregator
from utils.enhanced_shamir_privacy import ShamirShares, dp_clip_and_noise, RDPAccountant
```


### utils.feature_engineering

**Purpose**: Data preparation, extension, and canonical training table generation.

#### Public API

```python
def load_raw_candidates(path: str) -> "pd.DataFrame"
def load_raw_companies(path: str) -> "pd.DataFrame"

def extend_candidates(df: "pd.DataFrame") -> "pd.DataFrame"
def extend_companies(df: "pd.DataFrame") -> "pd.DataFrame"

def build_training_table(
    candidates: "pd.DataFrame",
    companies: "pd.DataFrame",
    *,
    align_schema: bool = True,
    drop_na_threshold: float = 0.8
) -> "pd.DataFrame"
```

**Behavior**
- `extend_*` may add derived columns (geo-coordinates, readiness indicators).
- `build_training_table` aligns schemas, resolves categorical encodings, and outputs `Enhanced_Training_Dataset.csv`.

**Examples**
```python
cand = load_raw_candidates("data/raw/Dataset_Candidati_Aggiornato.csv")
comp = load_raw_companies("data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv")
train_df = build_training_table(extend_candidates(cand), extend_companies(comp))
train_df.to_csv("data/processed/Enhanced_Training_Dataset.csv", index=False)
```


### utils.scoring

**Purpose**: Matching score and utilities.

#### Public API

```python
from typing import TypedDict

class ScoreBreakdown(TypedDict):
    compatibility: float  # [0,1]
    distance_penalty: float  # [0,1]
    readiness: float  # [0,1]
    total: float  # [0,1]

def compute_match_score(
    candidate: "pd.Series",
    job: "pd.Series",
    *,
    w_compat: float = 0.4,
    w_distance: float = 0.2,
    w_readiness: float = 0.4,
    distance_max_km: int = 30
) -> ScoreBreakdown
```

**Notes**
- Total score is a convex combination of normalized sub-scores.
- Distance penalty saturates beyond `distance_max_km`.

**Example**
```python
s = compute_match_score(c_row, j_row, distance_max_km=30)
print(s["total"], s)
```


### utils.parallel_training

**Purpose**: Multi-threaded training and model orchestration.

#### Public API
```python
def train_centralized_models(
    train_csv: str,
    *,
    models: list[str],
    optuna_trials: int = 50,
    calibration: str = "sigmoid",
    seed: int = 42,
    out_dir: str = "results"
) -> dict
```

**Behavior**
- Loads `Enhanced_Training_Dataset.csv`, splits data, tunes models, persists `*.joblib` and metrics.
- Returns a dictionary with per-model metrics.

**Example**
```python
train_centralized_models(
    "data/processed/Enhanced_Training_Dataset.csv",
    models=["LightGBM_Optimized", "MLP"],
    out_dir="results"
)
```


### utils.visualization

**Purpose**: Reusable charts for datasets and experiments.

#### Public API
```python
def plot_learning_curves(history: "pd.DataFrame", out_dir: str) -> str
def plot_confusion(y_true: "np.ndarray", y_pred: "np.ndarray", out_path: str) -> str
def compare_federated_runs(summary_csv: str, out_dir: str) -> str
```

**Notes**
- Outputs PNGs under `results/learning_curves/` or a provided directory.


### utils.federated_data_splitter

**Purpose**: Discover and prepare regional datasets for FL.

#### Public API
```python
def discover_regional_csvs(root: str = "data/federated") -> list[str]
def stratified_split_by_region(df: "pd.DataFrame", *, test_size: float = 0.2, seed: int = 42) -> dict[str, "pd.DataFrame"]
```


### utils.federated_learning

**Purpose**: Federated orchestration with robust aggregators.

#### Public Types
```python
from enum import Enum
class Aggregator(str, Enum):
    fedavg = "fedavg"
    trimmed_mean = "trimmed_mean"
    coordinate_median = "coordinate_median"
```

#### Public API
```python
class FederatedTrainer:
    def __init__(
        self,
        clients: list["pd.DataFrame"] | list[str],  # dataframes or CSV paths
        model: str = "MLP",
        aggregator: Aggregator = Aggregator.fedavg,
        rounds: int = 10,
        batch_size: int = 256,
        seed: int = 42
    ): ...

    def run(self) -> dict:
        'Runs FL rounds; returns metrics summary.'

    def evaluate_global(self) -> dict:
        'Evaluates the aggregated model on a holdout set.'

    def save(self, out_dir: str) -> None:
        'Saves global model and metadata.'
```

**Notes**
- Deterministic seeding across rounds.
- Robust aggregators mitigate outliers or partial participation.


### utils.enhanced_shamir_privacy

**Purpose**: Secure aggregation (Shamir) and Differential Privacy helpers.

#### Public API
```python
class ShamirShares:
    def __init__(self, threshold: str = "3-of-5", seed: int = 42): ...
    def split(self, secret: "np.ndarray") -> list["np.ndarray"]
    def reconstruct(self, shares: list["np.ndarray"]) -> "np.ndarray"

def mask_updates(
    update: "np.ndarray",
    *,
    threshold: str = "3-of-5",
    seed: int = 42
) -> tuple["np.ndarray", dict]:
    'Applies pairwise masks derived from secret shares; returns masked update and metadata.'

def dp_clip_and_noise(
    grads: "np.ndarray",
    *,
    max_grad_norm: float,
    epsilon: float,
    delta: float,
    accountant: str = "rdp",
    seed: int = 42
) -> "np.ndarray"

class RDPAccountant:
    def __init__(self, delta: float): ...
    def compose(self, epsilons: list[float]) -> float
```

**Notes**
- Clip first, then single application of Gaussian noise per gradient tensor.
- RDP accountant composes per-round privacy losses.


### blockchain_data_anchoring

**Purpose**: Merkle commitments for results and models; proof utilities.

#### Public API
```python
class MerkleTree:
    def __init__(self, leaves: list[bytes]): ...
    def root(self) -> bytes
    def proof(self, index: int) -> list[tuple[bytes, str]]  # (sibling, 'L'|'R')

def build_manifest(paths: list[str]) -> dict
def commit_manifest(manifest: dict) -> tuple[bytes, "MerkleTree"]
def verify_inclusion(root: bytes, leaf: bytes, proof: list[tuple[bytes, str]]) -> bool
```

**Notes**
- Deterministic hashing; byte-order documented in code comments.


## CLI & Scripts

Each script exposes an argparse-based CLI. Run with `-h` to inspect all options. Typical usage:

```bash
# Data extension & visualization
python scripts/01_generate_dataset.py --config config.yaml
python scripts/02_visualize_dataset.py --config config.yaml

# Centralized training & analysis
python scripts/03_train_models.py --config config.yaml --models LightGBM_Optimized MLP
python scripts/04_analyze_results.py --input results/merged_model_summary.csv --out results/

# Federated pipelines
python scripts/05_LightGBM_federated_training.py --rounds 10 --aggregator fedavg
python scripts/06_LightGBM_federated_visualization.py --input results_LightGBM_federated/

python scripts/07_mlp_federated_training.py --rounds 10 --batch-size 256 --aggregator trimmed_mean
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
python scripts/09_mlp_federated_privacy_visualization.py --input results_mlp_federated_privacy/

# Anchoring & benchmarks
python scripts/blockchain_data_anchoring.py --manifest results/
python scripts/10_blockchain_anchoring_bench.py --scale 10000
```

**Exit codes**
- `0` — success; artifacts written.
- `1` — misconfiguration (path/schema/args).
- `2` — runtime error (training, FL round, or plotting failure).


## Streamlit Integration

The UI is a thin layer on top of the API:
- Loads configuration and datasets.
- Calls scoring functions to present ranked matches.
- Optionally visualizes FL vs centralized results and privacy budgets.

**Health endpoint**: `/_stcore/health` (ensure `curl` in Docker for health checks).


## Errors & Exceptions

Common exception types surfaced by the public API:
- `FileNotFoundError` — missing paths in `paths.*`.
- `ValueError` — invalid schema, empty datasets, or parameter bounds.
- `RuntimeError` — training/FL loop errors, DP accountant inconsistencies.
- `AssertionError` — internal invariant checks (e.g., shape alignment).


## Performance Notes

- Use `trimmed_mean` or `coordinate_median` for robustness with volatile clients.
- Smaller `batch_size` improves stability on CPU-only nodes.
- Cache extended datasets under `data/processed/` to cut preprocessing time.


## Security & Privacy Notes

- No raw data leaves regional nodes under FL; only model deltas are exchanged.
- Secure aggregation masks per-parameter updates; DP ensures bounded leakage.
- Anchoring provides integrity and non-repudiation for published artifacts.


## Backward Compatibility

- New optional config keys are backward-compatible.
- Deprecated options remain for at least one minor cycle with warnings.


## Changelog (API surface)

- **v2**: Added federated modules, privacy helpers (Shamir + DP), and anchoring API.
- **v1**: Initial centralized training and scoring APIs.


## Appendix A — Legacy API Reference

> The following section preserves the prior API reference for completeness.

# 📚 API Reference - Disability Job Matching System

**Complete Code Documentation for Developers and Researchers**

---

## 📋 Overview

This API reference provides comprehensive documentation for all classes, methods, and functions in the Disability Job Matching System. The codebase is organized into modular components with clear separation of concerns.

### Module Structure

```
utils/
├── scoring.py              # Core matching algorithms
├── parallel_training.py    # ML training pipeline  
├── feature_engineering.py  # Data preprocessing
└── visualization.py        # Chart generation

scripts/
├── 01_generate_dataset.py  # Data pipeline entry point
├── 03_train_models.py      # Training pipeline entry point
└── 04_analyze_results.py   # Analysis pipeline entry point

streamlit_app.py            # Production web interface
```

---

## 🎯 Core Classes

### `EnhancedScoringSystem`

**Location**: `utils/scoring.py`

**Purpose**: Implements the core candidate-company matching algorithm with Italian language support and geographic processing.

#### Class Definition
```python
class EnhancedScoringSystem:
    def __init__(self)
```

**Attributes**:
- `geolocator`: Nominatim geocoding instance
- `loc_cache`: Dictionary for caching geocoding results
- `thresholds`: Default matching thresholds

#### Methods

##### `geocode_with_cache(address: str) -> Tuple[float, float]`
Geocodes Italian addresses with caching for performance optimization.

**Parameters**:
- `address` (str): Italian address string (e.g., "Verona, Italy")

**Returns**:
- `Tuple[float, float]`: (latitude, longitude) or (NaN, NaN) if geocoding fails

**Example**:
```python
scoring_system = EnhancedScoringSystem()
lat, lon = scoring_system.geocode_with_cache("Villafranca di Verona, Italy")
# Returns: (45.3506, 10.8444)
```

##### `haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float`
Calculates precise geographic distance using Haversine formula.

**Parameters**:
- `lat1, lon1` (float): First location coordinates
- `lat2, lon2` (float): Second location coordinates

**Returns**:
- `float`: Distance in kilometers, or NaN if invalid coordinates

**Example**:
```python
distance = scoring_system.haversine(45.4408, 10.9916, 45.3506, 10.8444)
# Returns: 15.2 (km between Verona and Villafranca)
```

##### `compatibility_score(exclusions: str, company_text: str) -> float`
Analyzes semantic compatibility between candidate exclusions and company activities using Italian TF-IDF.

**Parameters**:
- `exclusions` (str): Comma-separated candidate work limitations
- `company_text` (str): Company activity/compatibility description

**Returns**:
- `float`: Compatibility score [0.0-1.0], where 1.0 = perfect compatibility

**Algorithm**:
- Uses Italian stop words and accented character tokenization
- Applies TF-IDF vectorization with cosine similarity
- Weighted scoring: 70% max similarity + 30% average similarity
- Returns 1.0 - similarity_score for compatibility interpretation

**Example**:
```python
exclusions = "Turni notturni, Lavori in quota"
company_text = "Lavoro d'ufficio con orario diurno flessibile"
score = scoring_system.compatibility_score(exclusions, company_text)
# Returns: 0.95 (high compatibility)
```

##### `generate_enhanced_training_data(df_candidates: pd.DataFrame, df_companies: pd.DataFrame) -> pd.DataFrame`
Generates synthetic training dataset using probabilistic matching rules.

**Parameters**:
- `df_candidates` (DataFrame): Extended candidate dataset
- `df_companies` (DataFrame): Extended company dataset

**Returns**:
- `DataFrame`: Training dataset with features and probabilistic outcomes

**Process**:
1. Geocodes all addresses if coordinates missing
2. For each candidate-company pair:
   - Calculates compatibility, distance, attitude factors
   - Applies weighted scoring formula
   - Generates probabilistic outcome with controlled randomness
3. Creates one-hot encoded features for categorical variables

**Example**:
```python
df_train = scoring_system.generate_enhanced_training_data(df_candidates, df_companies)
# Returns: DataFrame with ~500K rows, 50+ features
```

---

### `ParallelModelTrainer`

**Location**: `utils/parallel_training.py`

**Purpose**: High-performance ML training with parallel processing and hyperparameter optimization.

#### Class Definition
```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42)
```

**Attributes**:
- `random_state`: Reproducibility seed
- `monitor`: System resource monitoring instance
- `optimizer`: Hyperparameter optimization instance

#### Methods

##### `parallel_hyperparameter_optimization(X: np.ndarray, y: np.ndarray) -> Dict`
Optimizes hyperparameters for multiple model families in parallel using Optuna.

**Parameters**:
- `X` (ndarray): Training features
- `y` (ndarray): Training labels

**Returns**:
- `Dict`: Best parameters for each model family

**Implementation**:
- Uses ThreadPoolExecutor with 3 concurrent workers
- Optimizes RandomForest, XGBoost, and LightGBM simultaneously
- 50 trials per model with TPE algorithm
- 3-fold cross-validation for robust evaluation

**Example**:
```python
trainer = ParallelModelTrainer()
best_params = trainer.parallel_hyperparameter_optimization(X_train, y_train)
# Returns: {'random_forest': {...}, 'xgboost': {...}, 'lightgbm': {...}}
```

##### `create_optimized_models(best_params: Dict) -> List[Dict]`
Creates model configurations with optimized hyperparameters.

**Parameters**:
- `best_params` (Dict): Optimized parameters from Optuna

**Returns**:
- `List[Dict]`: Model configurations ready for training

**Models Created**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized
- ExtraTrees
- GradientBoosting
- HistGradientBoosting
- MLP_Optimized

##### `parallel_model_training(model_configs: List, X_train, y_train, X_test, y_test) -> Dict`
Trains multiple models concurrently with probability calibration.

**Parameters**:
- `model_configs` (List): Model configurations from create_optimized_models
- `X_train, y_train`: Training data
- `X_test, y_test`: Testing data

**Returns**:
- `Dict`: Training results with models, metrics, and metadata

**Process**:
1. Trains up to 6 models concurrently using ThreadPoolExecutor
2. Applies CalibratedClassifierCV for probability calibration
3. Calculates comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
4. Monitors system resources during training

**Example**:
```python
results = trainer.parallel_model_training(model_configs, X_train, y_train, X_test, y_test)
# Returns: {'LightGBM_Optimized': {'model': ..., 'metrics': {...}, 'status': 'success'}}
```

##### `create_ensemble_model(results: Dict, X_train, y_train) -> VotingClassifier`
Creates ensemble model from successfully trained individual models.

**Parameters**:
- `results` (Dict): Results from parallel_model_training
- `X_train, y_train`: Training data for ensemble fitting

**Returns**:
- `VotingClassifier`: Calibrated ensemble model with soft voting

##### `save_models(results: Dict, ensemble_model, save_dir: str = 'results')`
Saves all trained models and metrics to disk.

**Parameters**:
- `results` (Dict): Training results
- `ensemble_model`: Ensemble model instance
- `save_dir` (str): Output directory path

**Outputs**:
- Individual model files: `{model_name}.joblib`
- Ensemble model: `ensemble_model.joblib`
- Metrics summary: `metrics_summary.csv`

---

### `JobMatchingDemo`

**Location**: `streamlit_app.py`

**Purpose**: Production Streamlit interface with real-time matching capabilities.

#### Class Definition
```python
class JobMatchingDemo:
    def __init__(self)
```

**Initialization Process**:
1. Sets up geocoding with Italian focus
2. Loads candidate and company data (real or demo)
3. Loads trained ML models if available
4. Initializes caching systems

#### Methods

##### `load_data()`
Automatically detects and loads appropriate dataset (real vs demo).

**Logic**:
- Checks for real data files in `data/processed/`
- Falls back to demo data generation if real data unavailable
- Handles coordinate parsing for Italian addresses
- Creates demo data with realistic Italian geographic distribution

##### `find_matches(candidate_data: Dict, top_k: int = 5, distance_threshold: int = 30) -> List[Dict]`
Core real-time matching function with intelligent filtering.

**Parameters**:
- `candidate_data` (Dict): Candidate information
- `top_k` (int): Number of top recommendations to return
- `distance_threshold` (int): Maximum distance in kilometers

**Returns**:
- `List[Dict]`: Ranked company recommendations

**Process**:
1. **Global Filters**: Attitude threshold validation
2. **Geographic Filtering**: Distance-based company filtering  
3. **Compatibility Analysis**: TF-IDF semantic matching
4. **Multi-factor Scoring**: Weighted combination of factors
5. **Ranking**: Sort by final score, return top K

**Scoring Formula**:
```python
final_score = (
    0.35 × compatibility_score +
    0.25 × distance_factor + 
    0.20 × attitude_score +
    0.10 × retention_rate +
    0.05 × experience_bonus +
    0.05 × company_bonuses
)
```

**Example**:
```python
demo = JobMatchingDemo()
candidate = {
    'Area di Residenza': 'Verona, Italy',
    'Score Attitudine al Collocamento': 0.75,
    'Esclusioni': 'Turni notturni'
}
matches = demo.find_matches(candidate, top_k=5, distance_threshold=30)
# Returns: [{'Nome Azienda': 'Azienda_001', 'Score Finale': 89.5, ...}, ...]
```

---

## 🔧 Utility Functions

### Feature Engineering (`utils/feature_engineering.py`)

##### `extend_candidates_dataset(df: pd.DataFrame) -> pd.DataFrame`
Enhances candidate dataset with engineered features.

**Enhancements**:
- **Disability simulation**: Realistic distribution of Italian disability types
- **Employment history**: Experience years calculation from first employment
- **Unemployment duration**: Time-based calculations
- **Education mapping**: Italian education levels with disability correlations

**Example**:
```python
from utils.feature_engineering import extend_candidates_dataset
df_extended = extend_candidates_dataset(df_candidates)
# Adds: Years_of_Experience, Durata Disoccupazione, enhanced Tipo di Disabilità
```

##### `extend_companies_dataset(df: pd.DataFrame) -> pd.DataFrame`
Enhances company dataset with business intelligence features.

**Enhancements**:
- **Company size categorization**: Small/medium/large based on employee count
- **Certification flags**: Disability-friendly certification simulation
- **Remote work indicators**: Modern work arrangement flags
- **Retention rate calculation**: Success metrics from historical data

### Visualization (`utils/visualization.py`)

##### `visualize_distribution(df: pd.DataFrame)`
Generates distribution plots for dataset analysis.

**Plots Created**:
- Attitude score distribution
- Compatibility score distribution  
- Distance distribution
- Outcome balance

##### `visualize_correlations(df: pd.DataFrame)`
Creates correlation heatmap for feature analysis.

---

## 📊 Data Preprocessing Pipeline

### `prepare_data_for_training(df_train: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict`

**Location**: `utils/parallel_training.py`

**Purpose**: Comprehensive data preprocessing for ML training.

**Parameters**:
- `df_train` (DataFrame): Raw training dataset
- `test_size` (float): Test set proportion
- `random_state` (int): Reproducibility seed

**Returns**:
- `Dict`: Processed data ready for ML training

**Pipeline Steps**:
1. **Target Extraction**: Separate outcome variable
2. **Missing Value Handling**: Median imputation for numerical features
3. **Train-Test Split**: Stratified split preserving class balance
4. **Robust Scaling**: RobustScaler for outlier resistance
5. **Feature Selection**: SelectKBest with F-statistic (top 50 features)
6. **Class Balancing**: SMOTE oversampling for minority class

**Output Structure**:
```python
{
    "X_train": np.ndarray,     # Balanced training features
    "y_train": np.ndarray,     # Balanced training labels  
    "X_test": np.ndarray,      # Test features
    "y_test": np.ndarray,      # Test labels
    "scaler": RobustScaler,    # Fitted scaler for inference
    "selector": SelectKBest    # Fitted selector for inference
}
```

---

## 🔍 Configuration System

### Configuration Loading
```python
import yaml

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load system configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
```

### Default Configuration Structure
```yaml
matching_thresholds:
  attitude_min: 0.3           # float [0.0-1.0]
  compatibility_min: 0.5      # float [0.0-1.0]  
  distance_max_km: 30         # int [5-50]
  match_probability_cutoff: 0.6 # float [0.0-1.0]

model_training:
  random_state: 42            # int
  optuna_trials: 50           # int [10-100]
  n_jobs: 4                   # int [1-8]
  feature_selection_k: 50     # int [10-100]

geocoding:
  delay: 0.5                  # float [0.1-2.0]
  timeout: 10                 # int [5-30]
  user_agent: str             # Custom user agent string
  cache_file: str             # Path to cache file

italian_language:
  stop_words: List[str]       # Italian stop words
  token_pattern: str          # Regex for Italian tokens
```

---

## 📈 Performance Monitoring

### `SystemResourceMonitor`

**Location**: `utils/parallel_training.py`

**Purpose**: Real-time system resource tracking during ML operations.

#### Methods

##### `start()`
Begins background monitoring of CPU and memory usage.

##### `stop()`
Stops monitoring and finalizes statistics.

##### `stats() -> Dict`
Returns average resource utilization statistics.

**Example**:
```python
monitor = SystemResourceMonitor()
monitor.start()
# ... perform ML training ...
monitor.stop()
stats = monitor.stats()
# Returns: {'avg_cpu': 75.2, 'avg_mem': 68.5}
```

---

## 🔒 Error Handling and Validation

### Input Validation Patterns

```python
def validate_candidate_data(candidate: Dict) -> bool:
    """Validate candidate data structure and values"""
    required_fields = [
        'Area di Residenza', 'Score Attitudine al Collocamento',
        'Years_of_Experience', 'Durata Disoccupazione'
    ]
    
    # Check required fields
    if not all(field in candidate for field in required_fields):
        return False
    
    # Validate ranges
    if not 0.0 <= candidate['Score Attitudine al Collocamento'] <= 1.0:
        return False
    
    if candidate['Years_of_Experience'] < 0:
        return False
    
    return True
```

### Exception Handling

```python
try:
    result = scoring_system.compatibility_score(exclusions, company_text)
except Exception as e:
    logger.error(f"Compatibility scoring failed: {e}")
    result = 0.5  # Default neutral score
```

---

## 🚀 Usage Examples

### Complete Training Pipeline
```python
# 1. Load and extend datasets
from utils.feature_engineering import extend_candidates_dataset, extend_companies_dataset
from utils.scoring import EnhancedScoringSystem
from utils.parallel_training import ParallelModelTrainer, prepare_data_for_training

# Load raw data
df_candidates = pd.read_csv('data/raw/Dataset_Candidati_Aggiornato.csv')
df_companies = pd.read_csv('data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv')

# Extend with features
df_cand_ext = extend_candidates_dataset(df_candidates)
df_comp_ext = extend_companies_dataset(df_companies)

# Generate training data
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_comp_ext)

# Prepare for ML training
data = prepare_data_for_training(df_train)

# Train models
trainer = ParallelModelTrainer()
best_params = trainer.parallel_hyperparameter_optimization(data['X_train'], data['y_train'])
model_configs = trainer.create_optimized_models(best_params)
results = trainer.parallel_model_training(model_configs, data['X_train'], data['y_train'], 
                                         data['X_test'], data['y_test'])

# Save models
ensemble = trainer.create_ensemble_model(results, data['X_train'], data['y_train'])
trainer.save_models(results, ensemble)
```

### Real-time Matching
```python
# Initialize matching system
demo = JobMatchingDemo()

# Define candidate
candidate = {
    'Area di Residenza': 'Sommacampagna, Verona, Italy',
    'Score Attitudine al Collocamento': 0.80,
    'Years_of_Experience': 5,
    'Durata Disoccupazione': 12,
    'Titolo di Studio': 'Diploma',
    'Tipo di Disabilità': 'Motoria',
    'Esclusioni': 'Lavori in quota'
}

# Find matches
matches = demo.find_matches(candidate, top_k=5, distance_threshold=30)

# Process results
for i, match in enumerate(matches, 1):
    print(f"{i}. {match['Nome Azienda']}: {match['Score Finale']:.1f}% "
          f"({match['Distanza (km)']} km)")
```

---

## 📞 Support and Extension

### Adding New Features

**To add a new compatibility scoring method**:
```python
class EnhancedScoringSystem:
    def new_compatibility_method(self, exclusions: str, company_text: str) -> float:
        # Implement new logic
        return score
    
    def compatibility_score(self, exclusions: str, company_text: str) -> float:
        # Choose method based on configuration
        if self.config.get('use_new_method'):
            return self.new_compatibility_method(exclusions, company_text)
        else:
            return self.original_compatibility_score(exclusions, company_text)
```

**To add a new ML model**:
```python
def create_optimized_models(self, best_params: Dict) -> List[Dict]:
    models = [...]  # existing models
    
    # Add new model
    models.append({
        'name': 'NewModel_Optimized',
        'class': NewModelClass,
        'params': {**best_params.get('new_model', {}), 'random_state': self.random_state}
    })
    
    return models
```

### Custom Integration

For organization-specific integrations, extend the base classes:

```python
class CustomJobMatcher(JobMatchingDemo):
    def __init__(self, organization_config):
        super().__init__()
        self.org_config = organization_config
    
    def find_matches(self, candidate_data, **kwargs):
        # Apply organization-specific rules
        base_matches = super().find_matches(candidate_data, **kwargs)
        return self.apply_org_filters(base_matches)
    
    def apply_org_filters(self, matches):
        # Custom filtering logic
        return filtered_matches
```

---

*This API reference provides comprehensive documentation for integrating with and extending the Disability Job Matching System. For specific implementation questions or custom development needs, contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Target Audience**: Developers, Researchers, Integration Specialists
