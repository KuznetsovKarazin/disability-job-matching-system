# API Reference — Disability Job Matching System
_Last updated: 2025-08-24 15:30_

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" />
  <img src="https://img.shields.io/badge/language-EN-blue" />
  <img src="https://img.shields.io/badge/federated-learning-purple" />
  <img src="https://img.shields.io/badge/privacy-DP%20%2B%20Shamir-teal" />
  <img src="https://img.shields.io/badge/blockchain-anchoring-orange" />
</p>

This document defines the **public API surface** of the Disability Job Matching System.  
It covers configuration schema, module-level functions and classes, CLI entry points, data contracts, and operational behaviors.

**Scope & Conventions**
- Python ≥ 3.8. Static types are descriptive; runtime enforcement is best-effort.
- File paths are relative to repo root unless noted.
- All examples assume an activated virtual environment and a populated `config.yaml`.
- The Streamlit UI uses these APIs indirectly; this reference targets developers and integrators.

## Table of Contents
1. [Versioning & Stability](#versioning--stability)
2. [Configuration Schema](#configuration-schema)
3. [Data Contracts](#data-contracts)
4. [Modules](#modules)
5. [CLI & Scripts](#cli--scripts)
6. [Streamlit Integration](#streamlit-integration)
7. [Errors & Exceptions](#errors--exceptions)
8. [Performance Notes](#performance-notes)
9. [Security & Privacy Notes](#security--privacy-notes)
10. [Backward Compatibility](#backward-compatibility)

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
- `distance_max_km` is **30** by default.
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

### utils.feature_engineering

**Purpose**: Data preparation, extension, and canonical training table generation.

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

### utils.scoring

**Purpose**: Matching score and utilities.

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

### utils.parallel_training

**Purpose**: Multi-threaded training and model orchestration.

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

### utils.federated_learning

**Purpose**: Federated orchestration with robust aggregators.

```python
from enum import Enum

class Aggregator(str, Enum):
    fedavg = "fedavg"
    trimmed_mean = "trimmed_mean"
    coordinate_median = "coordinate_median"

class FederatedTrainer:
    def __init__(
        self,
        clients: list["pd.DataFrame"] | list[str],
        model: str = "MLP",
        aggregator: Aggregator = Aggregator.fedavg,
        rounds: int = 10,
        batch_size: int = 256,
        seed: int = 42
    ): ...

    def run(self) -> dict:
        """Runs FL rounds; returns metrics summary."""

    def evaluate_global(self) -> dict:
        """Evaluates the aggregated model on a holdout set."""

    def save(self, out_dir: str) -> None:
        """Saves global model and metadata."""
```

### utils.enhanced_shamir_privacy

**Purpose**: Secure aggregation (Shamir) and Differential Privacy helpers.

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
    """Applies pairwise masks derived from secret shares."""

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

### blockchain_data_anchoring

**Purpose**: Merkle commitments for results and models; proof utilities.

```python
class MerkleTree:
    def __init__(self, leaves: list[bytes]): ...
    def root(self) -> bytes
    def proof(self, index: int) -> list[tuple[bytes, str]]  # (sibling, 'L'|'R')

def build_manifest(paths: list[str]) -> dict
def commit_manifest(manifest: dict) -> tuple[bytes, "MerkleTree"]
def verify_inclusion(root: bytes, leaf: bytes, proof: list[tuple[bytes, str]]) -> bool
```

## CLI & Scripts

Each script exposes an argparse-based CLI. Run with `-h` to inspect all options.

```bash
# Data extension & visualization
python scripts/01_generate_dataset.py --config config.yaml
python scripts/02_visualize_dataset.py --config config.yaml

# Centralized training & analysis
python scripts/03_train_models.py --config config.yaml --models LightGBM_Optimized MLP
python scripts/04_analyze_results.py --input results/merged_model_summary.csv

# Federated pipelines
python scripts/05_LightGBM_federated_training.py --rounds 10 --aggregator fedavg
python scripts/07_mlp_federated_training.py --rounds 10 --aggregator trimmed_mean
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6

# Anchoring & benchmarks
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
- **Performance Reality**: Federated learning shows slight performance degradation (-0.06% F1 for LightGBM, -4% for MLP) compared to centralized training.

## Security & Privacy Notes

- No raw data leaves regional nodes under FL; only model deltas are exchanged.
- Secure aggregation masks per-parameter updates; DP ensures bounded leakage.
- Anchoring provides integrity and non-repudiation for published artifacts.
- **Privacy Cost**: Minimal performance impact with practical DP parameters (ε=1.0, δ=1e-06).

## Backward Compatibility

- New optional config keys are backward-compatible.
- Deprecated options remain for at least one minor cycle with warnings.

---

## Core Classes (Legacy Reference)

### `EnhancedScoringSystem`

**Location**: `utils/scoring.py`

**Purpose**: Implements the core candidate-company matching algorithm with Italian language support and geographic processing.

```python
class EnhancedScoringSystem:
    def __init__(self): ...
    
    def geocode_with_cache(self, address: str) -> tuple[float, float]:
        """Geocodes Italian addresses with caching."""
    
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates precise geographic distance using Haversine formula."""
    
    def compatibility_score(self, exclusions: str, company_text: str) -> float:
        """Analyzes semantic compatibility using Italian TF-IDF."""
    
    def generate_enhanced_training_data(self, df_candidates: "pd.DataFrame", df_companies: "pd.DataFrame") -> "pd.DataFrame":
        """Generates synthetic training dataset using probabilistic matching rules."""
```

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

### `ParallelModelTrainer`

**Location**: `utils/parallel_training.py`

**Purpose**: High-performance ML training with parallel processing and hyperparameter optimization.

```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42): ...
    
    def parallel_hyperparameter_optimization(self, X: "np.ndarray", y: "np.ndarray") -> dict:
        """Optimizes hyperparameters for multiple model families using Optuna."""
    
    def parallel_model_training(self, model_configs: list, X_train, y_train, X_test, y_test) -> dict:
        """Trains multiple models concurrently with probability calibration."""
    
    def create_ensemble_model(self, results: dict, X_train, y_train) -> "VotingClassifier":
        """Creates ensemble model from successfully trained individual models."""
```

**Models Trained**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized (Best: F1=0.901)
- HistGradientBoosting
- GradientBoosting
- MLP_Optimized
- ExtraTrees

### `JobMatchingDemo`

**Location**: `streamlit_app.py`

**Purpose**: Production Streamlit interface with real-time matching capabilities.

```python
class JobMatchingDemo:
    def __init__(self): ...
    
    def load_data(self):
        """Automatically detects and loads appropriate dataset (real vs demo)."""
    
    def find_matches(self, candidate_data: dict, top_k: int = 5, distance_threshold: int = 30) -> list[dict]:
        """Core real-time matching function with intelligent filtering."""
```

## Data Preprocessing Pipeline

```python
def prepare_data_for_training(df_train: "pd.DataFrame", test_size: float = 0.2, random_state: int = 42) -> dict:
    """Comprehensive data preprocessing for ML training."""
```

**Pipeline Steps**:
1. **Target Extraction**: Separate outcome variable
2. **Missing Value Handling**: Median imputation for numerical features
3. **Train-Test Split**: Stratified split preserving class balance
4. **Robust Scaling**: RobustScaler for outlier resistance
5. **Feature Selection**: SelectKBest with F-statistic (top 50 features)
6. **Class Balancing**: SMOTE oversampling for minority class

## Performance Results

**Centralized Learning (Baseline)**
- **Best Model**: LightGBM_Optimized (F1=0.901, ROC-AUC=0.708)

**LightGBM Federated Learning**
- **Centralized**: F1 ≈ 0.9012, ROC-AUC ≈ 0.716
- **Federated**: F1 ≈ 0.9007, ROC-AUC ≈ 0.687
- **Performance Gap**: -0.0005 F1-score

**MLP Federated Learning**
- **Centralized**: F1 ≈ 0.828, Accuracy ≈ 0.735
- **Federated**: F1 ≈ 0.788, Accuracy ≈ 0.695
- **Privacy-Preserving**: Minimal additional cost with ε=1.0, δ=1e-06

**Blockchain Anchoring**
- **Build times**: 100 records = 2.28s; 1k = 30.47s; 10k = 344.07s
- **Proof generation**: 1.11ms - 20.65ms (O(log n) scaling)
- **Verification**: 24.49ms average with 100% accuracy

## Usage Examples

### Complete Training Pipeline
```python
from utils.feature_engineering import extend_candidates_dataset, extend_companies_dataset
from utils.scoring import EnhancedScoringSystem
from utils.parallel_training import ParallelModelTrainer

# Load and extend data
df_candidates = pd.read_csv('data/raw/Dataset_Candidati_Aggiornato.csv')
df_companies = pd.read_csv('data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv')

df_cand_ext = extend_candidates_dataset(df_candidates)
df_comp_ext = extend_companies_dataset(df_companies)

# Generate training data
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_comp_ext)

# Train models
trainer = ParallelModelTrainer()
results = trainer.parallel_model_training(configs, X_train, y_train, X_test, y_test)
```

### Real-time Matching
```python
demo = JobMatchingDemo()
candidate = {
    'Area di Residenza': 'Verona, Italy',
    'Score Attitudine al Collocamento': 0.75,
    'Esclusioni': 'Turni notturni'
}
matches = demo.find_matches(candidate, top_k=5, distance_threshold=30)
```

---
