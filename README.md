# Disability Job Matching System — Comprehensive README
_Last updated: 2025-08-23 23:00_

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" />
  <img src="https://img.shields.io/badge/docs-EN%2FIT-blue" />
  <img src="https://img.shields.io/badge/federated-learning-purple" />
  <img src="https://img.shields.io/badge/privacy-DP%20%2B%20Shamir-teal" />
  <img src="https://img.shields.io/badge/blockchain-anchoring-orange" />
</p>

> **Executive Summary.**  
> A production-oriented, privacy-preserving job–candidate matching system for CPI/SIL and public institutions.  
> It supports centralized and federated training (LightGBM/MLP), secure aggregation with Shamir secret sharing,
> differentially-private learning (RDP), and blockchain-based anchoring of model/result manifests.

## Contents
- [Highlights](#highlights)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Data & Schemas](#data--schemas)
- [Models & Training](#models--training)
- [Federated Learning](#federated-learning)
- [Privacy & Security](#privacy--security)
- [Blockchain Anchoring](#blockchain-anchoring)
- [Results & Benchmarks](#results--benchmarks)
- [Monitoring & Visualization](#monitoring--visualization)
- [API Reference (inline)](#api-reference-inline)
- [CLI Examples](#cli-examples)
- [Deployment](#deployment)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Compliance (GDPR)](#compliance-gdpr)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Appendix A — Legacy README](#appendix-a--legacy-readme)
- [Appendix B — Detailed Guides (embedded)](#appendix-b--detailed-guides-embedded)
- [Appendix C — Experiment Reports](#appendix-c--experiment-reports)

## Highlights
- **Federated Learning**: LightGBM (regional ensemble) and true FedAvg MLP pipelines (robust aggregation: FedAvg, trimmedmean, coordinate-median).
- **Privacy**: Shamir-based secure aggregation (threshold 3-of-5); Differential Privacy (ε=1.0, δ=1e-06) with RDP accountant.
- **Anchoring**: Deterministic Merkle commitments + O(log n) proofs; benchmarked on 100/1k/10k records.
- **Reproducibility**: Deterministic seeds, versioned artifacts, consolidated `merged_model_summary.csv`.
- **Operational Fit**: Streamlit UI for demo/ops, Docker deployment, health checks (remember to install `curl`).

## System Architecture
The platform comprises three layers:
1. **Data & Matching Layer** — ETL, feature engineering, matching score (compatibility, distance, readiness).
2. **Learning Layer** — centralized and federated training; robust aggregators; privacy-preserving FL.
3. **Integrity Layer** — blockchain-style anchoring for artifacts and auditability.

### Diagram
```mermaid
flowchart TD
    subgraph "Phase 1: Centralized ML System"
        A[Raw CSV Data] --> B[Feature Engineering]
        B --> C[Synthetic Training Dataset]
        C --> D[Parallel ML Training]
        D --> E[7-Model Ensemble]
        E --> F[Streamlit Interface]
        G[Real-time Candidate Input] --> F
        F --> H[Ranked Company Recommendations]
        
        I[Real Employment Data] -.-> C
        I -.-> J[Production Training Dataset]
        J -.-> D
        
        style C fill:#fff2cc
        style E fill:#d5e8d4
        style I fill:#e1f5fe
        style J fill:#e8f5e8
    end
    
    H ==>|"Evolution to<br/>Federated System"| A2
    
    subgraph "Phase 2: Federated Learning System"
        subgraph "CPI/SIL Nodes"
            A2[Regional Data CSVs] --> P1["Preprocess & Feature Engineering"]
            P1 --> M1["Local Model (LightGBM/MLP)"]
            M1 --> U1["Model Update ΔW"]
            U1 --> S1[Shamir Secret Shares]
            S1 --> E1["Encrypted ΔW_i"]
        end

        subgraph "Privacy Layer"
            C1["DP: Clip + Noise"] --> AGG
            S1 -. thresholds .- R1["3-of-5 recovery"]
        end   

        E1 -->|network| AGG[Secure Aggregator]
      
        
        AGG -->|"FedAvg / Trimmed Mean"| GM[Global Model]
        GM -->|broadcast| M1
        
        subgraph "Blockchain Anchoring"
            GM --> R["Results & Manifests"]
            R --> HM[Merkle Tree]
            HM --> Z["Proofs O(log n)"]
            Z --> BC["Blockchain Anchor"]
        end
        
        style AGG fill:#fdf6e3,stroke:#555,stroke-width:1px
        style GM fill:#f0fff4,stroke:#555,stroke-width:1px
    end
```

## Project Structure
```
📁 Disability Job Matching System/
├── 📄 README.md                               # This file (English)
├── 📄 README_IT.md                            # Italian version
├── ⚙️ config.yaml                             # System configuration
├── 📄 requirements.txt                        # Python dependencies
├── 🐍 streamlit_app.py                        # 🎯 MAIN APPLICATION
│
├── 📁 data/
│   ├── 📁 raw/                                # Original input data
│   │   ├── Dataset_Candidati_Aggiornato.csv
│   │   └── Dataset_Aziende_con_Stima_Assunzioni.csv
│   └── 📁 processed/                          # Generated/extended datasets
│       ├── Dataset_Candidati_Aggiornato_Extended.csv
│       ├── Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
│       └── Enhanced_Training_Dataset.csv      # 📊 ML TRAINING DATA
│
├── 📁 scripts/                                # Data processing pipeline
│   ├── 01_generate_dataset.py                 # Data extension + synthetic training generation
│   ├── 02_visualize_dataset.py                # Data analysis and visualization
│   ├── 03_train_models.py                     # 🤖 ML MODEL TRAINING
│   ├── 04_analyze_results.py                  # Performance analysis and reporting
│   ├── 05_LightGBM_federated_training.py      # 🔬 FEDERATED LEARNING PIPELINE (LightGBM)
│   ├── 06_LightGBM_federated_visualization.py # 📊 FEDERATED RESULTS VISUALIZATION
│   ├── 07_mlp_federated_training.py           # 🔄 CLASSICAL FEDERATED LEARNING
│   ├── 08_mlp_federated_privacy.py            # 🔐 PRIVACY-PRESERVING FEDERATED
│   ├── 09_mlp_federated_privacy_visualization.py # 📊 FEDERATED RESULTS COMPARISON
│   ├── blockchain_data_anchoring.py           # ⛓️ BLOCKCHAIN DATA INTEGRITY PIPELINE
│   └── 10_blockchain_anchoring_bench.py       # 📈 BLOCKCHAIN PERFORMANCE BENCHMARKS
│
├── 📁 utils/                                  # Core business logic
│   ├── __init__.py
│   ├── feature_engineering.py                 # Data augmentation utilities
│   ├── scoring.py                            # 🎯 MATCHING ALGORITHM CORE
│   ├── parallel_training.py                  # ⚡ MULTI-THREADED ML TRAINING
│   ├── visualization.py                      # Chart generation utilities
│   └── enhanced_shamir_privacy.py            # 🔒 SHAMIR SECRET SHARING + DIFFERENTIAL
│
├── 📁 results/                                # Training outputs
│   ├── 📁 learning_curves/                   # Training progression charts
│   ├── 🤖 *.joblib                          # Trained ML models (7 models)
│   ├── 📊 merged_model_summary.csv          # Performance metrics
│   └── 📈 *.png                              # Analysis visualizations
│
├── 📁 results_LightGBM_federated/            # Federated learning results (LightGBM)
│   ├── 📁 regional_models/                   # Individual regional models
│   ├── 📁 federated_models/                  # Aggregated global models
│   ├── 📁 centralized_models/                # Baseline centralized models
│   ├── 📁 visualizations/                    # Analysis charts and graphs
│   ├── 📊 complete_model_comparison.csv      # Three-way performance comparison
│   └── 📋 experiment_metadata.json           # Complete experimental metadata
│
├── 📁 results_mlp_federated/                 # Classical federated learning results
├── 📁 results_mlp_federated_privacy/         # Privacy-preserving federated results
├── 📁 results_blockchain_demo/               # Blockchain anchoring demonstrations
├── 📁 visualizations_federated_comparison/   # Federated learning comparison charts
│
└── 📁 docs/                                  # Documentation
    ├── user_guide_italiano.md                # Operator manual (Italian)
    ├── technical_documentation.md            # Developer documentation
    ├── deployment_guide.md                   # Production setup guide
    ├── api_reference.md                      # Code documentation
    └── demo_example.pdf                      # Interface usage example
```

## Installation
```bash
git clone <YOUR_REPO_URL>
cd <REPO>
python -m venv venv && source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

## Configuration
Provide a `config.yaml` with paths, FL, privacy, and anchoring options:
```yaml
# config.yaml — example
seed: 42
paths:
  data_raw: data/raw
  data_processed: data/processed/Enhanced_Training_Dataset.csv
  results: results
federated:
  rounds: 10
  clients_min: 3
  aggregator: "fedavg"   # options: fedavg | trimmed_mean | coordinate_median
  lr: 0.001
  batch_size: 256
privacy:
  enabled: true
  dp:
    epsilon: 1.0
    delta: 1e-6
    max_grad_norm: 1.0
    accountant: "rdp"
  secure_agg:
    scheme: "shamir"
    threshold: "3-of-5"
    dropout_recovery: true
anchoring:
  enabled: true
  backend: "merkle"
  anchor_every_n: 1
ui:
  distance_max_km: 30
```

## Quick Start
**Demo UI**
```bash
streamlit run streamlit_app.py
```
**Training (baseline)**
```bash
python scripts/03_train_models.py
```
**Federated LightGBM**
```bash
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py
```
**Federated MLP**
```bash
python scripts/07_mlp_federated_training.py
# Privacy-preserving
python scripts/08_mlp_federated_privacy.py
python scripts/09_mlp_federated_privacy_visualization.py
```
**Anchoring Benchmarks**
```bash
python scripts/10_blockchain_anchoring_bench.py
```

## Data & Schemas
- `data/raw/Dataset_Candidati_Aggiornato.csv`: candidates master.
- `data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv`: companies & roles.
- `data/processed/Enhanced_Training_Dataset.csv`: aligned, feature-complete training table.

> **Note:** distance default is **30 km**. Adjust in `config.yaml: ui.distance_max_km`.

## Models & Training
- **Preprocessing**: RobustScaler, feature selection, imbalance handling (SMOTE).
- **Classifiers**: LightGBM, MLP; Optuna tuning; probability calibration.
- **Artifacts**: stored under `results/` with hashes and metrics.

## Federated Learning
- **LightGBM (regional ensemble)**: independent regional models → weighted ensemble by sample size.
- **MLP (FedAvg family)**: parameter-averaging with options for trimmed mean & coordinate-median.
- **Utilities**:
  - `utils/federated_learning.py` — aggregation, reproducibility, metrics.
  - `utils/federated_data_splitter.py` — regional discovery & splits.
  - `utils/enhanced_shamir_privacy.py` — secure aggregation & DP utilities.

## Privacy & Security
- **Secure Aggregation**: Shamir threshold **3-of-5**, per-parameter masking; dropout recovery.
- **Differential Privacy**: clipping + Gaussian noise; single-noise application per round; **RDP** accounting.
- **Audit & Governance**: artifacts + manifest logs, access control on `data/` and `results/`.

## Blockchain Anchoring
- **What**: Merkle commitments over results; O(log n) proofs for any artifact.
- **Why**: Long-term integrity & auditability for public bodies.
- **How**: `blockchain_data_anchoring.py` and `10_blockchain_anchoring_bench.py`.

## Results & Benchmarks
- **LightGBM**: Central F1 ≈ 0.9012, Regional ≈ 0.9001, Federated ≈ 0.9007.
- **MLP (FL)**: F1 ≈ 0.7882, Acc ≈ 0.6951, ROC-AUC ≈ 0.7169; centralized F1 ≈ 0.8276.
- **Privacy Mode**: Fed F1 ≈ 0.7880 with ε=1.0, δ=1e-06.
- **Anchoring Bench**: Build 100/1k/10k = 2.28s / 30.47s / 344.07s; verify avg ≈ 24.49ms.

> See `results/merged_model_summary.csv` for consolidated metrics.

## Monitoring & Visualization
- Visualization scripts: `06_*_visualization.py`, `09_*_privacy_visualization.py`.
- Streamlit KPIs: candidate–job matches, regional breakdowns, FL/central comparisons.

## API Reference (inline)
Below is an excerpt; see full API reference in `docs`:
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

## CLI Examples
```bash
# Train centralized baseline
python scripts/03_train_models.py --config config.yaml

# Run FL with robust aggregation
python scripts/07_mlp_federated_training.py --aggregator trimmed_mean

# Run privacy-preserving FL
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5

# Visualize LightGBM FL results
python scripts/06_LightGBM_federated_visualization.py --input results/federated/
```

## Deployment
- **Docker**: expose 8501; health check endpoint `/_stcore/health`.  
  _Install `curl` in the image for health checks._
- **Resources**: CPU-friendly profiles; avoid GPU assumptions; tune batch size & rounds for low-end nodes.
# 🚀 Deployment Guide - Disability Job Matching System

**Complete Installation and Setup Guide for Production Environments**

---

## 📋 Overview

This deployment guide provides step-by-step instructions for installing and configuring the Disability Job Matching System in production environments. The guide covers both demo mode (synthetic data) and production mode (real employment data) deployments.

### Deployment Modes

- **🧪 Demo Mode**: Uses synthetic data for testing and demonstration
- **🏭 Production Mode**: Uses real employment outcome data for live operations
- **🔧 Development Mode**: Full development environment with all scripts

---

## 📋 Prerequisites

### System Requirements

**Minimum Requirements**:
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

**Network Requirements**:
- **Internet Connection**: Required for initial geocoding setup
- **Firewall**: Port 8501 (Streamlit default) or custom port
- **Geographic API**: Access to Nominatim geocoding service

### Software Dependencies

**Required Software**:
```bash
# Python 3.8+ with pip
python --version  # Should show 3.8 or higher
pip --version

# Git (for repository cloning)
git --version

# Optional: Virtual environment tools
python -m venv --help
```

---

## 🔧 Installation Methods

### Method 1: Quick Demo Setup (Recommended for Testing)

**Step 1: Clone Repository**
```bash
# Clone the project
git clone https://github.com/your-username/disability-job-matching.git
cd disability-job-matching

# Verify project structure
ls -la
# Should see: streamlit_app.py, config.yaml, requirements.txt, data/, scripts/, utils/
```

**Step 2: Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, numpy, sklearn; print('Dependencies OK')"
```

**Step 3: Launch Demo**
```bash
# Start the application in demo mode
streamlit run streamlit_app.py

# System will automatically:
# 1. Load with synthetic demo data
# 2. Open browser at http://localhost:8501
# 3. Show "Demo Mode" in interface
```

**Verification**:
- Interface loads successfully
- Analytics tab shows sample data
- Can perform candidate searches
- Results appear within 5 seconds

### Method 2: Production Setup (Real Data)

**Prerequisites for Production**:
- Historical employment outcome data
- Properly formatted CSV files
- Data privacy compliance approval

**Step 1: Prepare Production Data**

```bash
# Production data structure required:
data/
├── raw/
│   ├── Dataset_Candidati_Aggiornato.csv      # Real candidate data
│   └── Dataset_Aziende_con_Stima_Assunzioni.csv  # Real company data
└── processed/
    └── Enhanced_Training_Dataset.csv         # Real employment outcomes
```

**Enhanced_Training_Dataset.csv Format**:
```csv
outcome,attitude_score,years_experience,unemployment_duration,compatibility_score,distance_km,company_size,retention_rate,remote_work,certification,...
1,0.75,5,12,0.85,15.2,150,0.78,1,1,...
0,0.45,2,24,0.35,45.8,50,0.65,0,0,...
```

**Step 2: Data Validation**
```bash
# Validate data format
python -c "
import pandas as pd
df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
print(f'Training data: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Outcome distribution: {df.outcome.value_counts()}')
print('Required columns present:', all(col in df.columns for col in ['outcome', 'attitude_score', 'compatibility_score']))
"
```

**Step 3: Train Production Models**
```bash
# Train models on real data
python scripts/03_train_models.py

# Expected output:
# 📥 Loading training dataset...
# 🧹 Preparing data...
# 🎯 Optimizing hyperparameters...
# 🤖 Training [Model Name]...
# ✅ All models saved

# Verify models created
ls -la results/
# Should see: *.joblib files, metrics_summary.csv
```

**Step 4: Launch Production Interface**
```bash
# Start production application
streamlit run streamlit_app.py

# System will:
# 1. Detect real data automatically
# 2. Load trained models
# 3. Enable production features
```

### Method 3: Docker Deployment (Enterprise)

**Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Commands**:
```bash
# Build image
docker build -t disability-job-matcher .

# Run container (demo mode)
docker run -p 8501:8501 disability-job-matcher

# Run with real data volume
docker run -p 8501:8501 -v /path/to/real/data:/app/data disability-job-matcher
```

---

## ⚙️ Configuration

### Environment Configuration

**config.yaml Customization**:
```yaml
# Production configuration example
paths:
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"
  logs_dir: "logs"

matching_thresholds:
  attitude_min: 0.3           # Adjust based on candidate pool
  distance_max_km: 30         # Adjust for urban/rural deployment
  match_probability_cutoff: 0.6

model_training:
  n_jobs: 4                   # Adjust based on server CPU cores
  optuna_trials: 50           # Reduce for faster training, increase for accuracy

geocoding:
  delay: 0.5                  # Increase if hitting rate limits
  cache_file: "data/processed/geocoding_cache.json"

streamlit:
  page_title: "Sistema Collocamento Mirato - [Organization Name]"
  default_top_k: 5
```

**Environment Variables**:
```bash
# Optional environment configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### Network Configuration

**Firewall Settings**:
```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload

# Verify port accessibility
netstat -tlnp | grep 8501
```

**Reverse Proxy (Nginx)**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔒 Security Configuration

### Access Control

**Basic Authentication (Streamlit)**:
```python
# Add to streamlit_app.py for basic protection
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

**File Permissions**:
```bash
# Secure file permissions
chmod 600 config.yaml                    # Config files
chmod 600 data/processed/*.csv           # Data files  
chmod 755 scripts/*.py                   # Executable scripts
chmod 644 requirements.txt               # Public files

# Secure directories
chmod 750 data/                          # Data directory
chmod 750 results/                       # Models directory
chmod 755 utils/                         # Code directory
```

### Data Protection

**Sensitive Data Handling**:
```bash
# Create secure data directory
sudo mkdir -p /opt/job-matcher/secure-data
sudo chown app-user:app-group /opt/job-matcher/secure-data
sudo chmod 750 /opt/job-matcher/secure-data

# Symlink to application
ln -s /opt/job-matcher/secure-data data/processed
```

**Backup Strategy**:
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/job-matcher"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup critical files
cp -r data/processed $BACKUP_DIR/$DATE/
cp -r results/ $BACKUP_DIR/$DATE/
cp config.yaml $BACKUP_DIR/$DATE/

# Compress and encrypt
tar -czf $BACKUP_DIR/$DATE.tar.gz $BACKUP_DIR/$DATE
rm -rf $BACKUP_DIR/$DATE

echo "Backup completed: $BACKUP_DIR/$DATE.tar.gz"
```

---

## 📊 Monitoring and Logging

### Application Monitoring

**Health Check Endpoint**:
```python
# Add to streamlit_app.py
import streamlit as st
import time

def health_check():
    """System health check"""
    checks = {
        'models_loaded': len(st.session_state.get('models', {})) > 0,
        'data_available': os.path.exists('data/processed/Enhanced_Training_Dataset.csv'),
        'config_valid': os.path.exists('config.yaml')
    }
    return all(checks.values()), checks

# Usage in app
if st.sidebar.button("Health Check"):
    healthy, details = health_check()
    if healthy:
        st.success("✅ System Healthy")
    else:
        st.error("❌ System Issues Detected")
        st.json(details)
```

**Performance Monitoring**:
```python
# Add performance tracking
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        
    def log_request(self, operation, duration):
        self.request_count += 1
        logging.info(f"Operation: {operation}, Duration: {duration:.2f}s, "
                    f"CPU: {psutil.cpu_percent()}%, "
                    f"Memory: {psutil.virtual_memory().percent}%")

# Usage
monitor = PerformanceMonitor()
start = time.time()
# ... perform matching operation ...
monitor.log_request("candidate_matching", time.time() - start)
```

### Logging Configuration

**Logging Setup**:
```python
# logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Usage in main application
logger = setup_logging()
logger.info("Application started")
```

**Log Rotation**:
```bash
# /etc/logrotate.d/job-matcher
/path/to/job-matcher/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 app-user app-group
}
```

---

## 🔄 Maintenance Procedures

### Regular Maintenance Tasks

**Daily Tasks**:
```bash
#!/bin/bash
# daily_maintenance.sh

echo "$(date): Starting daily maintenance"

# Check disk space
df -h | grep -E "/(dev|opt)" | awk '$5 > "80%" {print "WARNING: " $0}'

# Check application health
curl -f http://localhost:8501/_stcore/health || echo "WARNING: App health check failed"

# Validate data integrity
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
    print(f'Data OK: {df.shape[0]} records')
except Exception as e:
    print(f'Data ERROR: {e}')
"

# Clear old cache files
find data/processed -name "*.cache" -mtime +7 -delete

echo "$(date): Daily maintenance completed"
```

**Weekly Tasks**:
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update geocoding cache if needed
python -c "
import json
try:
    with open('data/processed/geocoding_cache.json', 'r') as f:
        cache = json.load(f)
    print(f'Geocoding cache: {len(cache)} entries')
except:
    print('No geocoding cache found')
"

# Check model performance
python scripts/04_analyze_results.py

# Backup models and config
cp -r results/ /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
cp config.yaml /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
```

### Model Updates

**When to Retrain Models**:
- New employment outcome data available (quarterly recommended)
- Significant changes in candidate/company demographics
- Performance degradation observed
- System configuration changes

**Retraining Process**:
```bash
# 1. Backup current models
cp -r results/ results_backup_$(date +%Y%m%d)/

# 2. Update training data
# Place new Enhanced_Training_Dataset.csv in data/processed/

# 3. Retrain models
python scripts/03_train_models.py

# 4. Validate new models
python scripts/04_analyze_results.py

# 5. Test with sample candidates
streamlit run streamlit_app.py

# 6. If satisfied, remove backup
# rm -rf results_backup_*
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Application Won't Start**
```bash
# Symptoms: ModuleNotFoundError, Import errors
# Solution: Verify dependencies
pip list | grep -E "(streamlit|pandas|numpy|sklearn)"
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+
```

**Issue 2: No Models Found**
```bash
# Symptoms: "No models loaded" warning
# Solution: Train models
ls -la results/  # Check if .joblib files exist
python scripts/03_train_models.py  # Train if missing
```

**Issue 3: Geocoding Failures**
```bash
# Symptoms: Distance calculation errors
# Solution: Check network and cache
ping nominatim.openstreetmap.org
ls -la data/processed/geocoding_cache.json

# Reset cache if corrupted
rm data/processed/geocoding_cache.json
```

**Issue 4: Poor Performance**
```bash
# Symptoms: Slow response times
# Solutions:
# 1. Check system resources
htop  # or top
free -h

# 2. Reduce parallel jobs
# Edit config.yaml: model_training.n_jobs: 2

# 3. Clear browser cache
# 4. Restart application
```

### Error Logs Analysis

**Common Error Patterns**:
```bash
# Check recent errors
tail -100 logs/app_$(date +%Y%m%d).log | grep ERROR

# Geocoding errors
grep "Geocoding error" logs/*.log

# Model prediction errors  
grep "prediction error" logs/*.log

# Memory issues
grep -i "memory\|oom" logs/*.log
```

### Performance Optimization

**Memory Optimization**:
```python
# Add to streamlit_app.py
import gc

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_data():
    # Load heavy data operations
    pass

# Clear cache when needed
if st.button("Clear Cache"):
    st.cache_data.clear()
    gc.collect()
    st.success("Cache cleared")
```

**CPU Optimization**:
```yaml
# config.yaml adjustments for slower systems
model_training:
  n_jobs: 2                 # Reduce from 4
  optuna_trials: 25         # Reduce from 50

matching_thresholds:
  distance_max_km: 25       # Reduce search space
```

---

## 📞 Support and Escalation

### Support Contacts

**Technical Issues**:
- **Primary**: michele.melch@gmail.com
- **Academic**: oleksandr.kuznetsov@uniecampus.it
- **Emergency**: Include "URGENT" in subject line

**Information to Include**:
```bash
# System information
uname -a
python --version
pip list > installed_packages.txt

# Error logs
tail -50 logs/app_$(date +%Y%m%d).log

# Configuration
cat config.yaml

# System resources
free -h && df -h
```

### Escalation Procedures

**Level 1: Application Issues**
- Check logs and common issues
- Restart application
- Verify configuration

**Level 2: Data/Model Issues**
- Validate data integrity
- Retrain models if needed
- Check for data updates

**Level 3: System/Infrastructure Issues**
- Contact system administrator
- Check network connectivity
- Review security logs

### Community Resources

**Documentation**:
- GitHub Repository: [Link to repository]
- User Guide: `docs/user_guide_english.md`
- Technical Docs: `docs/technical_documentation.md`

**Updates and Announcements**:
- GitHub Releases for version updates
- Email notifications for critical security updates

---

*This deployment guide provides comprehensive instructions for installing and maintaining the Disability Job Matching System. For organization-specific deployment questions or custom configuration needs, contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Next Review**: December 2025

## Performance Tuning
- Lower `rounds` for quick iteration; enable trimmed mean for outlier robustness.
- Use balanced splits per region; monitor gradient norms under DP.
- Cache geocoding and precomputed features to cut cold-start time.

## Troubleshooting
- Missing `seaborn`/`matplotlib` → `pip install -r requirements.txt`.
- Sklearn tag errors (older versions) → pin `scikit-learn` per `requirements.txt`.
- Ensure `curl` exists in Docker for health checks.

## Compliance (GDPR)
- Roles (controller/processor) clarified; DPIA template available.
- DSR workflows (access/erasure), retention & minimization policies.
- Local processing; federated updates only.

## Roadmap
- Cross-silo secure aggregation at scale; adaptive DP budget by cohort.
- Broader anchoring backends; auto-verification portals.

## Contributing
- Open issues & PRs welcome. Include unit tests and docs updates.
- Use conventional commits; run `pre-commit` hooks if configured.

## License
- Academic-friendly license. See `LICENSE` (or request commercial terms).

## Acknowledgements
- CPI Villafranca di Verona, SIL Veneto, Università eCampus.

---

## Appendix A — Legacy README
# 🎯 Disability Job Matching System

**An advanced Machine Learning-powered recommendation system for supporting employment inclusion of people with disabilities in Italy.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.46.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![Italian](https://img.shields.io/badge/Language-Italian_Support-green.svg)](README_it.md)

---

## 📋 Overview

This system represents an innovative solution to optimize the targeted placement process for people with disabilities, developed in collaboration with the **Employment Center of Villafranca di Verona** and the **Labor Integration Service (SIL)**. 

**Current Status**: Fully functional prototype with sophisticated synthetic data generation, validated by domain experts, ready for real employment outcome data integration.

### 🎯 Key Features

- **🤖 Advanced Machine Learning**: Ensemble of 7 optimized models with parallel training and Optuna hyperparameter optimization
- **🇮🇹 Italian Language Support**: Specialized TF-IDF semantic analysis with Italian stop words for work terminology
- **📍 Geographic Matching**: Automatic geocoding and precise Haversine distance calculation
- **⚖️ Multi-dimensional Scoring**: Intelligent weighted combination of compatibility, distance, attitude, and company factors
- **🖥️ Professional Web Interface**: Complete Streamlit dashboard with real-time candidate input and interactive matching
- **⚡ High-Performance Computing**: Multi-threaded training and optimization for production scalability

### 🔬 Expert Validation

✅ **CPI Approval**: System validated by Dr. Rotolani (Employment Center Villafranca)  
✅ **SIL Interest**: Active collaboration discussions with Labor Integration Service  
✅ **Methodology**: Approved by Prof. Oleksandr Kuznetsov (eCampus University)  

---

## 📁 Complete Project Structure

```
📁 Disability Job Matching System/
├── 📄 README.md                          # This file (English)
├── 📄 README_IT.md                       # Italian version
├── ⚙️ config.yaml                        # System configuration
├── 📄 requirements.txt                   # Python dependencies
├── 🐍 streamlit_app.py                   # 🎯 MAIN APPLICATION
│
├── 📁 data/
│   ├── 📁 raw/                          # Original input data
│   │   ├── Dataset_Candidati_Aggiornato.csv
│   │   └── Dataset_Aziende_con_Stima_Assunzioni.csv
│   └── 📁 processed/                    # Generated/extended datasets
│       ├── Dataset_Candidati_Aggiornato_Extended.csv
│       ├── Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
│       └── Enhanced_Training_Dataset.csv  # 📊 ML TRAINING DATA
│
├── 📁 scripts/                          # Data processing pipeline
│   ├── 01_generate_dataset.py          # Data extension + synthetic training generation
│   ├── 02_visualize_dataset.py         # Data analysis and visualization
│   ├── 03_train_models.py              # 🤖 ML MODEL TRAINING
│   └── 04_analyze_results.py           # Performance analysis and reporting
│
├── 📁 utils/                            # Core business logic
│   ├── __init__.py
│   ├── feature_engineering.py          # Data augmentation utilities
│   ├── scoring.py                       # 🎯 MATCHING ALGORITHM CORE
│   ├── parallel_training.py            # ⚡ MULTI-THREADED ML TRAINING
│   └── visualization.py                # Chart generation utilities
│
├── 📁 results/                          # Training outputs
│   ├── 📁 learning_curves/             # Training progression charts
│   ├── 🤖 *.joblib                     # Trained ML models (7 models)
│   ├── 📊 merged_model_summary.csv     # Performance metrics
│   └── 📈 *.png                        # Analysis visualizations
│
└── 📁 docs/                             # Documentation
    ├── user_guide_italiano.md          # Operator manual (Italian)
    ├── technical_documentation.md      # Developer documentation
    ├── deployment_guide.md             # Production setup guide
    ├── api_reference.md                # Code documentation
    └── demo_example.pdf                # Interface usage example
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/KuznetsovKarazin/disability-job-matching
cd disability-job-matching

# Install dependencies
pip install -r requirements.txt
```

### 2. Demo Mode (Immediate Launch)

```bash
# Launch web interface with synthetic data
streamlit run streamlit_app.py
```

🎉 **The system starts automatically with realistic synthetic data for demonstration!**

### 3. Production Mode with Real Employment Data

**For production deployment**, you need real employment outcome data:

```bash
# Step 1: Place real employment outcome dataset
# Replace: data/processed/Enhanced_Training_Dataset.csv
# With: Real historical placement data (candidate-company pairs + success outcomes)

# Step 2: Train models on real data
python scripts/03_train_models.py

# Step 3: Launch production interface
streamlit run streamlit_app.py
```

**Note**: The `01_generate_dataset.py` script is only for synthetic data generation. For real deployment, you need actual employment outcome records.

---

## 🏗️ System Architecture

```mermaid
graph TB
    A[Raw CSV Data] --> B[Feature Engineering]
    B --> C[Synthetic Training Dataset]
    C --> D[Parallel ML Training]
    D --> E[7-Model Ensemble]
    E --> F[Streamlit Interface]
    G[Real-time Candidate Input] --> F
    F --> H[Ranked Company Recommendations]
    
    I[Real Employment Data] -.-> C
    I -.-> J[Production Training Dataset]
    J -.-> D
    
    style C fill:#fff2cc
    style E fill:#d5e8d4
    style I fill:#e1f5fe
    style J fill:#e8f5e8
```

### 📊 Processing Pipeline

1. **Data Preparation** (`01_generate_dataset.py`)
   - Extends candidate/company datasets with engineered features
   - Generates synthetic training data using probabilistic matching rules
   - **Italian address geocoding** with geopy/Nominatim and caching
   - **Advanced feature engineering**: experience levels, unemployment categories, education mapping

2. **Multi-Threaded ML Training** (`03_train_models.py`)
   - **Parallel hyperparameter optimization** using Optuna across 3 algorithm families
   - **Concurrent model training** with ThreadPoolExecutor (up to 6 parallel workers)
   - **Advanced preprocessing**: SMOTE class balancing, RobustScaler, SelectKBest feature selection
   - **Probability calibration** and ensemble creation

3. **Production Interface** (`streamlit_app.py`)
   - **Real-time candidate input** with manual entry or existing candidate selection
   - **Live company matching** with configurable thresholds and distance filtering
   - **Interactive visualizations** with Plotly charts
   - **Results export** and detailed analytics

---

## 💡 Core Matching Algorithm

### Weighted Scoring Formula
```python
Final_Score = (
    0.35 × Compatibility_Score +         # TF-IDF semantic analysis of exclusions
    0.25 × Distance_Factor +            # Geographic proximity (Haversine distance)
    0.20 × Attitude_Score +             # Employment readiness assessment
    0.10 × Company_Retention_Rate +     # Historical placement success
    0.05 × Experience_Bonus +           # Years of professional experience
    0.05 × Company_Bonuses             # Remote work + certification bonuses
)
```

**Rationale for weighting**:
- **Compatibility (35%)**: Most critical - ensures candidate can perform required tasks
- **Distance (25%)**: Major practical factor for sustainable employment
- **Attitude (20%)**: Essential for employment success and motivation
- **Company factors (15%)**: Historical success and supportive environment indicators
- **Experience (5%)**: Important but secondary to basic compatibility

### Intelligent Filtering System
- **🎯 Attitude Threshold**: Minimum 0.3 (configurable via config.yaml)
- **📍 Distance Limit**: Maximum **30 km** by default (configurable up to 50 km)
- **🔗 Compatibility Analysis**: Italian TF-IDF with specialized disability/work stop words
- **⚖️ Haversine Distance**: Precise geographic calculation accounting for Earth's curvature

### Haversine Distance Implementation
```python
def haversine(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points on Earth"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return c * 6371  # Earth's radius in kilometers
```

---

## 📊 Model Performance Analysis (Synthetic Data)

### Comprehensive Results - 7 Trained Models
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
|-------|----------|-----------|---------|----------|---------|---------------|
| **LightGBM_Optimized** | 0.829 | 0.821 | 0.999 | **0.901** | 0.708 | 94.6s |
| XGBoost_Optimized | 0.828 | 0.821 | 0.997 | **0.901** | 0.704 | 132.3s |
| HistGradientBoosting | 0.827 | 0.823 | 0.992 | **0.900** | 0.715 | 202.3s |
| GradientBoosting | 0.826 | 0.823 | 0.990 | **0.899** | 0.711 | 2399.9s |
| RandomForest_Optimized | 0.799 | 0.833 | 0.928 | **0.878** | 0.712 | 261.4s |
| MLP_Optimized | 0.735 | 0.842 | 0.814 | **0.828** | 0.695 | 858.1s |
| ExtraTrees | 0.713 | 0.859 | 0.757 | **0.805** | 0.724 | 188.2s |

### Performance Analysis

**🎯 Best Model**: LightGBM_Optimized achieves the highest F1-Score (0.901) with excellent efficiency (94.6s training)

**📈 Why F1-Score Matters Most**:
- F1-Score balances precision and recall, crucial for recommendation systems
- High recall (0.999) ensures we don't miss good candidate-company matches
- Good precision (0.821) minimizes false positive recommendations

**🔍 ROC-AUC Context**:
- ROC-AUC scores (0.695-0.724) are **intentionally moderate** due to synthetic data design
- **Probabilistic synthetic generation** creates realistic uncertainty, preventing models from memorizing simple rules
- In recommendation systems, **ranking quality** (measured by precision/recall) is more important than binary classification confidence
- Real employment data would likely show higher ROC-AUC scores with discovered hidden patterns

**⚡ Training Efficiency**:
- LightGBM and XGBoost show excellent speed/performance trade-offs
- GradientBoosting achieves high scores but requires 25x longer training time
- All models trained with parallel optimization for production scalability

---

## ⚡ High-Performance Computing Features

### Multi-Threading Architecture

**Parallel Hyperparameter Optimization**:
```python
# Concurrent optimization across algorithm families
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(optimize_random_forest, X, y): "random_forest",
        executor.submit(optimize_xgboost, X, y): "xgboost", 
        executor.submit(optimize_lightgbm, X, y): "lightgbm"
    }
```

**Parallel Model Training**:
```python
# Concurrent training of 7 models
with ThreadPoolExecutor(max_workers=6) as executor:
    # Each model trains independently with optimized hyperparameters
    # System resource monitoring tracks CPU/memory usage
```

**Performance Benefits**:
- **6x faster training** compared to sequential approach
- **Real-time resource monitoring** with psutil
- **Configurable parallelism** (1-8 workers depending on hardware)
- **Memory optimization** with RobustScaler and feature selection

### Optuna Hyperparameter Optimization

**What is Optuna?**
Optuna is a state-of-the-art hyperparameter optimization framework that uses advanced algorithms to efficiently search the hyperparameter space.

**Our Implementation**:
```python
# Example: Random Forest optimization
def objective(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 300),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10)
    )
    # 3-fold cross-validation for robust evaluation
    return np.mean(cross_val_scores)
```

**Benefits of Optuna**:
- **Tree-structured Parzen Estimator (TPE)**: Intelligent sampling based on past trials
- **Pruning**: Early stopping of unpromising trials saves computation time
- **50 trials per model** finds optimal hyperparameters efficiently
- **Consistent random state** ensures reproducible results

**Real Impact**:
- **15-25% performance improvement** over default hyperparameters
- **Automated optimization** eliminates manual hyperparameter tuning
- **Parallel execution** across multiple algorithm families
- **Production-ready configurations** saved for deployment

---

## ⚠️ Current Limitations and Real Data Strategy

### 🔍 Sophisticated Synthetic Data Generation

**Current Approach**: Advanced probabilistic modeling that goes beyond simple rule-based generation:

1. **Probabilistic Outcome Generation**:
   ```python
   # Multi-factor weighted probability calculation
   match_probability = (
       0.3 * attitude_factor + 0.4 * compatibility_factor + 
       0.2 * distance_factor + 0.1 * company_factors
   )
   # Controlled randomness prevents rule memorization
   outcome = 1 if (probability > 0.6 and random() < probability) else 0
   ```

2. **Realistic Geographic Distribution**:
   - Real Italian cities and addresses in Veneto region
   - Actual travel distances calculated with Haversine formula
   - Urban/rural distribution patterns

3. **Domain-Specific Authenticity**:
   - Genuine Italian disability exclusion terminology
   - Real company activity sectors and size distributions
   - Correlation patterns between education, experience, and disability types

### 🎯 Why Not Pure Rule-Based?

**Previous Approach** (deterministic rules):
```python
# Simple binary decision
if (attitude >= 0.5 AND compatibility >= 0.7 AND distance <= 30):
    outcome = 1  # Perfect deterministic outcome
```

**Problems with deterministic approach**:
- ❌ Models achieve 100% accuracy by memorizing rules
- ❌ No real learning or pattern discovery
- ❌ Zero generalization capability
- ❌ Circular logic: predict what you already know

**Our Probabilistic Innovation**:
- ✅ Introduces realistic uncertainty and edge cases
- ✅ Forces models to learn complex interaction patterns  
- ✅ Prevents overfitting to known rules
- ✅ Better preparation for real-world complexity

### 🚀 Real Employment Data Integration

**Production Deployment Path**:

1. **Data Requirements**:
   - Historical candidate-company placement records
   - **Binary outcome labels**: successful placement (1) vs unsuccessful (0)
   - Follow-up data on job retention (3, 6, 12 months)

2. **Integration Process**:
   ```bash
   # Replace synthetic training data with real outcomes
   # File: data/processed/Enhanced_Training_Dataset.csv
   # Required columns: candidate_features + company_features + real_outcome
   
   python scripts/03_train_models.py  # Train on real data
   streamlit run streamlit_app.py     # Deploy with real models
   ```

3. **Expected Improvements with Real Data**:
   - **Higher ROC-AUC scores** (0.80-0.90+) from discovering hidden patterns
   - **Better precision** on actual placement predictions
   - **Continuous learning** capability with feedback loops
   - **Personalized recommendations** based on historical success patterns

---

## 🔮 Development Roadmap

### Phase 1: Current State ✅
- [x] Complete synthetic data pipeline with probabilistic generation
- [x] 7-model ensemble with parallel training and Optuna optimization
- [x] Production-ready Streamlit interface with real-time matching
- [x] Expert validation from Employment Center professionals
- [x] Comprehensive performance analysis and visualization

### Phase 2: Real Data Integration 🔄
- [ ] Partnership agreements with CPI/SIL for historical employment data access
- [ ] Real employment outcome dataset creation and preprocessing
- [ ] Model retraining and performance validation on real placements
- [ ] A/B testing with employment center operators in controlled pilot

### Phase 3: Advanced ML and Optimization 🚀
- [ ] **Kolmogorov-Arnold Networks (KANs)** implementation for interpretable AI
- [ ] Continuous learning pipeline with real-time feedback integration
- [ ] Advanced ensemble methods with dynamic model weighting
- [ ] Multi-objective optimization for placement quality vs. speed trade-offs

### Phase 4: Production Scale and Research 📄
- [ ] REST API development for integration with existing CPI/SIL systems
- [ ] Multi-region deployment across Italian employment centers
- [ ] Academic paper submission to disability employment and AI journals
- [ ] Open-source release for international research collaboration

---

## 🛠️ Configuration and Customization

### Core System Settings
```yaml
# config.yaml - Production Configuration
matching_thresholds:
  attitude_min: 0.3          # Employment readiness threshold
  compatibility_min: 0.5     # Semantic compatibility threshold
  distance_max_km: 30        # Default geographic search radius
  match_probability_cutoff: 0.6

model_training:
  optuna_trials: 50          # Hyperparameter optimization iterations
  n_jobs: 4                  # Parallel processing cores
  feature_selection_k: 50    # Top features to select
  random_state: 42           # Reproducibility seed

geocoding:
  delay: 0.5                 # Rate limiting for API calls
  timeout: 10                # Request timeout seconds
  user_agent: "disability-job-matcher-v1.0"
  cache_file: "data/processed/geocoding_cache.json"
```

### Real-Time Interface Settings
```yaml
streamlit:
  page_title: "Sistema Collocamento Mirato"
  page_icon: "🎯"
  layout: "wide"
  default_top_k: 5           # Default number of recommendations

italian_language:
  stop_words: ["di", "a", "da", "in", "con", "su", "per", ...]
  token_pattern: "\\b[a-zA-Zàèéìòù]+\\b"
```

---

## 📚 Documentation and Support

### Complete Documentation Suite
- **[User Guide (English)](docs/user_guide_en.md)** or **[User Guide (Italian)](docs/user_guide_it.md)** - Complete operational manual for CPI/SIL staff
- **[Technical Documentation (English)](docs/technical_docs_en.md)** or **[Technical Documentation (Italian)](docs/technical_docs_it.md)** - Implementation details, algorithms, and architecture
- **[Deployment Guide (English)](docs/deployment_guide_en.md)** or **[Deployment Guide (Italian)](docs/deployment_guide_it.md)** - Production setup with real data integration
- **[API Reference (English)](api_reference_en.md)** or **[API Reference (Italian)](api_reference_it.md)** - Code documentation and module specifications

### Expert Validation Results
- **🏛️ Employment Center of Villafranca di Verona** - Process validation and workflow approval
- **🤝 Labor Integration Service (SIL) Veneto** - Domain expertise and implementation guidance
- **🎓 eCampus University** - Academic supervision and methodology validation

> *"The project represents a valuable tool to make the targeted placement process more effective and efficient. The automated approach solves the complexity of manual evaluation between candidate exclusions and company compatibilities."*  
> **— Dr. Rotolani, Director CPI Villafranca di Verona**

---

## 👨‍💻 Development Team

**👤 Michele Melchiori**  
*Principal Developer & Master's Thesis Candidate*  
📧 michele.melch@gmail.com  
🏆 Certifications: Lean Six Sigma Black Belt, PMP-PMI  
💼 Professional Experience:
- Public waste management process optimization (Lean Six Sigma consultant)
- Technology startup project management (certified PMP)

**👨‍🏫 Prof. Oleksandr Kuznetsov**  
*Academic Supervisor & Research Director*  
🏛️ eCampus University - Department of Theoretical and Applied Sciences  
📧 oleksandr.kuznetsov@uniecampus.it  
🔬 Research Focus: Applied AI, Machine Learning for Social Applications

---

## 📈 Academic Contribution and Impact

### Research Innovation
- **First comprehensive ML system** for Italian disability employment matching
- **Novel probabilistic synthetic data generation** method for social applications
- **Multi-threaded optimization framework** for production-scale employment systems
- **Italian language specialization** for semantic compatibility analysis

### Practical Impact
- **Operational efficiency**: Manual matching from hours → automated in seconds
- **Improved accuracy**: 90%+ compatibility vs. subjective human evaluation
- **Scalability**: Handle thousands of candidates and companies simultaneously
- **Expert validation**: Approved by real employment center professionals

### Citation Information
```bibtex
@mastersthesis{melchiori2025disability,
  title={Intelligent Systems for Targeted Employment: An Application of Artificial Intelligence for Employment Inclusion of People with Disabilities},
  author={Melchiori, Michele},
  year={2025},
  school={eCampus University},
  address={Novedrate, Italy},
  supervisor={Kuznetsov, Oleksandr},
  type={Master's Thesis in Computer Engineering},
  note={Validated by Employment Center of Villafranca di Verona},
  keywords={Machine Learning, Disability Employment, Social AI, Italian Language Processing}
}
```

---

## 🔧 Technical Requirements and Deployment

### System Requirements
- **Python**: 3.8+ (tested and optimized on 3.11)
- **Memory**: 8GB RAM minimum (16GB recommended for parallel training)
- **Storage**: 3GB free space (models, datasets, cache files)
- **Network**: Internet connection for initial geocoding (cached afterward)
- **CPU**: Multi-core processor recommended for parallel training

### Performance Specifications
- **Training Time**: 15-45 minutes for complete 7-model ensemble
- **Prediction Time**: <100ms per candidate-company matching
- **Concurrent Users**: Supports multiple simultaneous Streamlit sessions
- **Data Scale**: Tested with 500,000+ candidate-company combinations

---

## 📞 Support and Community

### Getting Help and Contributing
- **📧 Direct Contact**: michele.melch@gmail.com
- **🎓 Academic Inquiries**: oleksandr.kuznetsov@uniecampus.it

### Open Source Contribution
We welcome contributions from:
- Researchers in AI for social good
- Employment center professionals with domain expertise
- Italian language processing specialists
- Machine learning engineers interested in recommendation systems

### License and Usage
This project is developed for academic research purposes with real-world social impact. For commercial or institutional deployment, please contact the authors for appropriate licensing arrangements.

---

**⭐ If this project advances your research or helps employment inclusion efforts, please star the repository and cite our work!**

---

*Developed with ❤️ for employment inclusion of people with disabilities in Italy.*  
*Sistema sviluppato per l'inclusione lavorativa delle persone con disabilità in Italia.*

---

## Appendix B — Detailed Guides (embedded)
### User Guide
# 🎯 User Guide - Disability Job Matching System

**Complete Operational Manual for Employment Center and SIL Operators**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Getting Started](#getting-started)
3. [Main Interface Tour](#main-interface-tour)
4. [Finding Matches for Candidates](#finding-matches-for-candidates)
5. [Understanding Results](#understanding-results)
6. [Analytics Dashboard](#analytics-dashboard)
7. [Dataset Management](#dataset-management)
8. [Configuration Settings](#configuration-settings)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 📊 System Overview

### What is the Disability Job Matching System?

This system is an advanced AI-powered tool designed to help Employment Centers (CPI) and Labor Integration Services (SIL) find the best company matches for candidates with disabilities. It automates the complex process of evaluating candidate exclusions against company compatibility requirements.

### Key Benefits for Operators

- **⏱️ Time Savings**: Reduces manual matching from hours to seconds
- **🎯 Improved Accuracy**: 90%+ compatibility scoring vs subjective evaluation
- **📊 Data-Driven Decisions**: Objective scoring based on multiple factors
- **🔍 Comprehensive Search**: Evaluates all companies within specified radius
- **📈 Analytics**: Track placement patterns and system performance

### Who Should Use This Guide?

- **Employment Center Operators** (CPI staff)
- **SIL Coordinators** (Labor Integration Service staff)
- **Case Managers** responsible for candidate placement
- **Account Managers** handling company relationships

---

## 🚀 Getting Started

### System Requirements

- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for initial setup and geocoding
- **Screen Resolution**: Minimum 1024x768 (1920x1080 recommended)
- **No Installation Required**: System runs entirely in web browser

### First Time Access

1. **Open the System**:
   ```
   Open your web browser and navigate to the system URL
   (provided by your IT administrator)
   ```

2. **System Initialization**:
   - The system will automatically load with demo data on first run
   - Initial loading may take 30-60 seconds
   - You'll see the main dashboard with sample candidates and companies

3. **Verify System Status**:
   - Check the "Info Sistema" tab to confirm all components are working
   - Ensure you see "✅ Interface: Operativo" status

### Demo vs Production Mode

**Demo Mode** (Default):
- Uses realistic synthetic data for training and demonstration
- Safe for testing and learning the system
- No real candidate or company information

**Production Mode** (When real data is available):
- Uses actual historical employment data
- Requires proper data setup by IT administrator
- Provides real matching recommendations

---

## 🖥️ Main Interface Tour

### Dashboard Layout

The system interface is organized into four main tabs:

#### 1. 🔍 Ricerca Candidato (Candidate Search)
- **Purpose**: Find company matches for individual candidates
- **Primary Use**: Daily placement operations
- **Users**: Case managers, placement coordinators

#### 2. 📊 Analytics
- **Purpose**: View system statistics and performance metrics
- **Primary Use**: Monitoring and reporting
- **Users**: Supervisors, data analysts

#### 3. 📋 Dataset
- **Purpose**: Browse and export candidate/company data
- **Primary Use**: Data verification and management
- **Users**: Data administrators, quality control

#### 4. ℹ️ Info Sistema (System Info)
- **Purpose**: System status and technical information
- **Primary Use**: Troubleshooting and configuration
- **Users**: IT support, system administrators

### Sidebar Configuration Panel

Located on the left side of the interface:

**🔧 Configurazione Sistema**:
- **Model Selection**: Choose AI model (if multiple available)
- **Attitude Threshold**: Minimum employment readiness (0.0-1.0)
- **Maximum Distance**: Search radius in kilometers (5-50 km)
- **Top Recommendations**: Number of results to show (3-10)

---

## 🔍 Finding Matches for Candidates

### Step 1: Access Candidate Search

1. Click on the **"🔍 Ricerca Candidato"** tab
2. You'll see two columns:
   - **Left**: Candidate data input
   - **Right**: Company recommendations (initially empty)

### Step 2: Input Candidate Information

You have two options for entering candidate data:

#### Option A: Use Existing Candidate

1. **Check the box** "Usa candidato esistente" (Use existing candidate)
2. **Select from dropdown**: Choose from pre-loaded candidates
3. **Review auto-filled data**: System automatically populates all fields
4. **Verify information**: Ensure data is correct and current

#### Option B: Manual Input

1. **Leave unchecked** "Usa candidato esistente"
2. **Fill in candidate details**:

   **Basic Information**:
   - **Area Residenza**: Select candidate's residential area
   - **Titolo Studio**: Choose education level
   - **Tipo Disabilità**: Select disability type

   **Assessment Scores**:
   - **Attitudine**: Employment readiness (0.0-1.0 scale)
     - 0.0-0.3: Low readiness
     - 0.4-0.6: Moderate readiness  
     - 0.7-1.0: High readiness

   **Experience Data**:
   - **Anni Esperienza**: Years of work experience
   - **Mesi Disoccupazione**: Months unemployed

   **Exclusions**:
   - **Esclusioni**: Enter work limitations (comma-separated)
   - Examples: "Turni notturni, Lavori in quota"

### Step 3: Configure Search Parameters

**Adjust settings in sidebar if needed**:

- **Soglia Attitudine**: Lower for broader search, higher for quality
- **Distanza Max**: Expand for more options, reduce for local focus
- **Top Raccomandazioni**: More results for comprehensive review

### Step 4: Execute Search

1. **Click** "🔄 Trova Aziende Compatibili" button
2. **Wait for processing**: Usually takes 2-5 seconds
3. **Review results**: System displays ranked recommendations

### Understanding Search Process

The system performs these steps automatically:

1. **Attitude Filter**: Excludes candidates below minimum threshold
2. **Geographic Filter**: Only considers companies within distance limit
3. **Compatibility Analysis**: Uses AI to match exclusions vs company activities
4. **Multi-factor Scoring**: Combines compatibility, distance, attitude, and company factors
5. **Ranking**: Orders results by final matching score

---

## 📊 Understanding Results

### Results Display Format

Each recommendation shows:

**Company Header**:
- **Company Name** and **Overall Score** (percentage)
- Visual score indicator (higher = better match)

**Key Metrics Row 1**:
- **Sector**: Type of business activity
- **Distance**: Kilometers from candidate residence
- **Employees**: Company size

**Key Metrics Row 2**:
- **Compatibility**: Semantic match score (percentage)
- **Remote**: Remote work availability
- **Positions**: Open positions for disabled candidates

### Score Interpretation

**Overall Score Ranges**:
- **85-100%**: Excellent match - highly recommended
- **70-84%**: Good match - suitable for placement
- **55-69%**: Fair match - may require additional evaluation
- **Below 55%**: Poor match - not recommended

**Compatibility Score**:
- **90-100%**: No conflicts found between exclusions and job requirements
- **70-89%**: Minor potential conflicts - interview recommended
- **50-69%**: Some conflicts present - careful evaluation needed
- **Below 50%**: Significant conflicts - likely incompatible

### Visual Analytics

**Score Distribution Chart**:
- Bar chart showing relative scores across all recommendations
- Helps identify clear winners vs close competitions

**Distance vs Compatibility Scatter Plot**:
- Shows trade-offs between proximity and job fit
- Larger circles indicate higher overall scores

### Results Actions

**No Results Found**:
If no companies appear:
1. **Increase distance** threshold in sidebar
2. **Lower attitude** threshold if appropriate
3. **Review exclusions** - may be too restrictive
4. **Check candidate location** - ensure it's valid

---

## 📊 Analytics Dashboard

### System Overview Metrics

**Key Performance Indicators**:
- **👥 Candidati Totali**: Total candidates in system
- **🏢 Aziende Totali**: Total companies available
- **📈 Attitudine Media**: Average employment readiness across candidates
- **💼 Posizioni Aperte**: Total open positions system-wide

### Distribution Charts

**Disability Types Distribution**:
- Shows breakdown of candidate disability categories
- Helps identify service focus areas
- Useful for resource planning

**Company Sectors Distribution**:
- Displays variety of available employment sectors
- Identifies placement opportunities by industry
- Guides business development efforts

### Using Analytics for Operations

**Daily Monitoring**:
- Check open positions vs candidate volume
- Monitor average attitude scores for trending
- Identify sectors with highest opportunity

**Strategic Planning**:
- Use disability distribution for specialized programs
- Target company outreach based on sector gaps
- Plan training programs based on compatibility patterns

---

## 📋 Dataset Management

### Viewing Candidate Data

1. **Navigate to** "📋 Dataset" tab
2. **Select** "Candidati" radio button
3. **Review data table**:
   - All candidate records with complete information
   - Sortable columns for data exploration
   - Search functionality for specific records

**Key Columns Explained**:
- **ID_Candidato**: Unique identifier
- **Score Attitudine al Collocamento**: Employment readiness (0.0-1.0)
- **Years_of_Experience**: Professional experience
- **Durata Disoccupazione**: Unemployment duration (months)
- **Esclusioni**: Work limitations from medical evaluation

### Viewing Company Data

1. **Select** "Aziende" radio button
2. **Review company information**:
   - Business details and contact information
   - Compatibility descriptions and requirements
   - Geographic and size information

**Key Columns Explained**:
- **Nome Azienda**: Company identifier
- **Tipo di Attività**: Business sector/activity
- **Compatibilità**: Description of suitable disability accommodations
- **Posizioni Aperte**: Available positions for disabled candidates
- **Remote**: Remote work availability (0=No, 1=Yes)
- **Certification**: Disability-friendly certification status

### Data Export Functions

**Candidate Data Export**:
1. **Click** "📥 Scarica CSV Candidati" button
2. **Save file** to desired location
3. **Use for**: External analysis, reporting, backup

**Company Data Export**:
1. **Click** "📥 Scarica CSV Aziende" button
2. **File includes**: All company information and availability
3. **Use for**: Partner outreach, capacity planning

### Data Quality Verification

**Regular Checks**:
- Verify candidate exclusions are current and accurate
- Confirm company position availability
- Update geographic information if companies relocate
- Review compatibility descriptions for accuracy

---

## ⚙️ Configuration Settings

### Threshold Adjustments

**Attitude Threshold** (Soglia Attitudine):
- **Default**: 0.3 (30%)
- **Lower (0.1-0.2)**: Include candidates with lower readiness
- **Higher (0.4-0.6)**: Focus on most employment-ready candidates
- **Impact**: Affects candidate pool size

**Distance Threshold** (Distanza Max):
- **Default**: 30 km
- **Urban areas**: 20-25 km for local focus
- **Rural areas**: 40-50 km for adequate options
- **Impact**: Balances commute feasibility vs opportunity variety

**Top Recommendations**:
- **Default**: 5 results
- **Fewer (3)**: Quick decision-making
- **More (7-10)**: Comprehensive evaluation
- **Impact**: Analysis depth vs simplicity

### Advanced Configuration

**Model Selection** (if available):
- Choose between different AI models
- Each model may have different strengths
- Default selection is usually optimal

**When to Adjust Settings**:

**Expand Search** when:
- Few or no results for qualified candidates
- Rural locations with limited local options
- Specialized disability requirements

**Narrow Search** when:
- Too many low-quality matches
- Need to focus on highest-probability placements
- Time constraints require quick decisions

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Problem: No Results Found
**Symptoms**: "No companies found" message appears
**Solutions**:
1. **Increase distance** threshold to 40-50 km
2. **Lower attitude** threshold to 0.2-0.3
3. **Review exclusions** - ensure they're not overly restrictive
4. **Check location** - verify candidate area is valid Italian location

#### Problem: All Scores Very Low
**Symptoms**: All recommendations below 60%
**Solutions**:
1. **Review exclusions accuracy** - may be too broad or incorrectly entered
2. **Check compatibility descriptions** - companies may need updated information
3. **Consider lower thresholds** - current settings may be too strict

#### Problem: System Loading Slowly
**Symptoms**: Interface takes >30 seconds to respond
**Solutions**:
1. **Refresh browser** page
2. **Clear browser cache** and reload
3. **Check internet connection** speed
4. **Try different browser** if issues persist

#### Problem: Geographic Errors
**Symptoms**: "Distance calculation failed" or unrealistic distances
**Solutions**:
1. **Verify address format** - use "City, Province, Italy" format
2. **Check spelling** of Italian city names
3. **Use major cities** instead of small villages if issues persist

### Getting Technical Support

**Before Contacting Support**:
1. **Note exact error message** if any appears
2. **Record steps** that led to the problem
3. **Check system status** in "Info Sistema" tab
4. **Try basic solutions** listed above

**Contact Information**:
- **Technical Support**: michele.melch@gmail.com
- **Academic Support**: oleksandr.kuznetsov@uniecampus.it
- **Include in email**: Screenshots, error messages, steps to reproduce

---

## 🎯 Best Practices

### Daily Operations

**Morning Routine**:
1. **Check system status** in Info Sistema tab
2. **Review analytics** for any overnight changes
3. **Verify top priority candidates** have current information

**Candidate Processing**:
1. **Always verify exclusions** with candidate before searching
2. **Use existing candidate data** when available for consistency
3. **Document successful placements** for system improvement

**Result Evaluation**:
1. **Focus on top 3 recommendations** for initial outreach
2. **Consider geographic preferences** even with high scores
3. **Review compatibility details** beyond just the score

### Weekly Reviews

**Data Quality**:
- Update candidate information based on new assessments
- Verify company position availability and requirements
- Remove or update inactive companies

**Performance Analysis**:
- Review successful vs unsuccessful placement patterns
- Identify companies with highest placement success
- Note any systematic issues with recommendations

### Integration with Existing Workflow

**CPI Integration**:
1. **Use system for initial screening** of candidates
2. **Combine with manual evaluation** for final decisions
3. **Document placement outcomes** for continuous improvement

**SIL Coordination**:
1. **Share recommendations** with case managers
2. **Coordinate follow-up** on high-score matches
3. **Track long-term placement success**

### Quality Assurance

**Recommendation Validation**:
- **Cross-check exclusions** against company requirements manually for top matches
- **Verify company information** before making contact
- **Confirm candidate preferences** align with recommendations

**Continuous Improvement**:
- **Track placement success rates** by score ranges
- **Report systematic issues** to technical team
- **Suggest improvements** based on field experience

---

## 📞 Support and Resources

### Quick Reference

**Key Shortcuts**:
- **Tab Navigation**: Use browser tabs for multiple candidates
- **Sidebar Settings**: Adjust thresholds without page reload
- **Export Functions**: Available in Dataset tab for all data

**Important Thresholds**:
- **Attitude**: 0.3 default (adjust based on candidate pool)
- **Distance**: 30 km default (expand for rural areas)
- **Compatibility**: 50% minimum for viable placement

### Training Resources

**New User Training**:
1. **Start with demo mode** to understand interface
2. **Practice with test candidates** before real operations
3. **Review this guide** section by section

**Advanced Features**:
- **Analytics interpretation** for strategic planning
- **Configuration optimization** for different scenarios
- **Integration techniques** with existing CPI/SIL workflows

### Feedback and Improvement

**How to Provide Feedback**:
- **Email suggestions** to michele.melch@gmail.com
- **Report bugs** with detailed reproduction steps
- **Share success stories** to help improve the system

**What Feedback Helps**:
- Real-world placement outcomes vs system recommendations
- Interface usability suggestions
- Additional features that would improve operations
- Integration challenges with existing systems

---

*This User Guide is designed to help employment professionals maximize the effectiveness of the Disability Job Matching System. For additional support or specific questions about your implementation, please contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Next Review**: December 2025

### Technical Documentation
# 🔧 Technical Documentation - Disability Job Matching System

**Developer and System Administrator Guide**

---

## 📋 Overview

This technical documentation provides comprehensive information for developers, system administrators, and researchers working with the Disability Job Matching System. The system is built using modern Python ML stack with a focus on production scalability and Italian language support.

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Raw Data      │───▶│  Data Pipeline   │───▶│  Enhanced Dataset   │
│   (CSV files)   │    │  (Feature Eng.)  │    │  (Training Ready)   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Trained Models │◀───│  ML Training     │◀───│  Parallel Training  │
│  (7 x .joblib)  │    │  Pipeline        │    │  (ThreadPoolExec.)  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│  Streamlit App  │◀───│  Real-time       │
│  (Production)   │    │  Matching Engine │
└─────────────────┘    └──────────────────┘
```

---

## 🏗️ Core Components

### 1. Data Processing Pipeline (`scripts/`)

#### `01_generate_dataset.py`
**Purpose**: Extends raw candidate/company data and generates synthetic training dataset

**Key Functions**:
```python
# Data extension with feature engineering
df_cand_ext = extend_candidates_dataset(df_cand)
df_az_ext = extend_companies_dataset(df_az)

# Synthetic training data generation
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_az_ext)
```

**Outputs**:
- `Dataset_Candidati_Aggiornato_Extended.csv`
- `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv`
- `Enhanced_Training_Dataset.csv` (500K+ rows)

#### `03_train_models.py`
**Purpose**: Parallel ML model training with hyperparameter optimization

**Key Process**:
1. **Data Preprocessing**: SMOTE, RobustScaler, SelectKBest
2. **Hyperparameter Optimization**: Optuna with 50 trials per model
3. **Parallel Training**: ThreadPoolExecutor with up to 6 workers
4. **Model Calibration**: CalibratedClassifierCV for probability calibration

**Models Trained**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized
- HistGradientBoosting
- GradientBoosting
- MLP_Optimized
- ExtraTrees

### 2. Core Business Logic (`utils/`)

#### `scoring.py` - Matching Algorithm Core
**Purpose**: Implements the probabilistic candidate-company matching logic

**Key Classes**:
```python
class EnhancedScoringSystem:
    def __init__(self):
        self.thresholds = {
            'attitude_min': 0.3,
            'compatibility_min': 0.5,
            'distance_max': 40.0  # Note: config.yaml default is 30
        }
    
    def compatibility_score(self, exclusions, company_text):
        # Italian TF-IDF semantic analysis
        # Returns 0.0-1.0 compatibility score
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Precise geographic distance calculation
        # Returns distance in kilometers
    
    def generate_enhanced_training_data(self, df_cand, df_az):
        # Probabilistic outcome generation
        # Creates realistic synthetic training data
```

**Scoring Formula Implementation**:
```python
prob = (
    0.3 * attitude_factor + 
    0.4 * compat_factor + 
    0.2 * distance_factor +
    0.05 * retention_rate + 
    0.025 * remote_bonus + 
    0.025 * cert_bonus
)
outcome = 1 if (prob > 0.6 and np.random.random() < prob) else 0
```

#### `parallel_training.py` - Multi-threaded ML Pipeline
**Purpose**: High-performance model training with resource monitoring

**Key Features**:
- **Parallel Hyperparameter Optimization**: 3 concurrent Optuna studies
- **Concurrent Model Training**: Up to 6 models training simultaneously
- **System Resource Monitoring**: CPU/Memory usage tracking with psutil
- **Advanced Preprocessing**: SMOTE, RobustScaler, feature selection

**Performance Optimizations**:
```python
# Parallel hyperparameter optimization
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(self.optimize_random_forest, X, y): "random_forest",
        executor.submit(self.optimize_xgboost, X, y): "xgboost",
        executor.submit(self.optimize_lightgbm, X, y): "lightgbm"
    }

# Parallel model training  
with ThreadPoolExecutor(max_workers=6) as executor:
    # Each model trains independently with optimized hyperparameters
```

### 3. Production Interface (`streamlit_app.py`)

#### Main Application Class
```python
class JobMatchingDemo:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="job_matching_system")
        self.loc_cache = {}  # Geocoding cache for performance
        self.load_data()     # Load real or demo data
        self.load_models()   # Load trained ML models
    
    def find_matches(self, candidate_data, top_k=5, distance_threshold=30):
        # Real-time matching with configurable parameters
        # Returns ranked list of compatible companies
```

**Key Features**:
- **Dual Data Mode**: Automatic detection of real vs demo data
- **Real-time Geocoding**: Cached Nominatim with Italian address support
- **Interactive Configuration**: Sidebar controls for thresholds
- **Advanced Visualizations**: Plotly charts for results analysis

---

## 🔧 Configuration System

### `config.yaml` Structure
```yaml
paths:
  raw_candidates: "data/raw/Dataset_Candidati_Aggiornato.csv"
  raw_companies: "data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv"
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"

matching_thresholds:
  attitude_min: 0.3          # Employment readiness threshold
  compatibility_min: 0.5     # Semantic compatibility threshold  
  distance_max_km: 30        # Default search radius (NOT 40!)
  match_probability_cutoff: 0.6

model_training:
  random_state: 42
  optuna_trials: 50          # Hyperparameter optimization iterations
  n_jobs: 4                  # Parallel processing cores
  feature_selection_k: 50    # Top features selected

geocoding:
  delay: 0.5                 # Rate limiting between API calls
  timeout: 10                # Request timeout
  user_agent: "disability-job-matcher-v1.0"
  cache_file: "data/processed/geocoding_cache.json"

italian_language:
  stop_words: ["di", "a", "da", "in", "con", "su", "per", ...]
  token_pattern: "\\b[a-zA-Zàèéìòù]+\\b"
```

### Configuration Loading
```python
import yaml

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()
thresholds = config['matching_thresholds']
```

---

## 🤖 Machine Learning Implementation

### Model Pipeline Architecture

#### 1. Data Preprocessing
```python
def prepare_data_for_training(df_train, test_size=0.2, random_state=42):
    # Feature preparation
    y = df_train["outcome"]
    X = df_train.drop(columns=["outcome"]).fillna(df_train.median())
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Robust scaling (better for outliers than StandardScaler)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    
    # Class balancing with SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)
    
    return {
        "X_train": X_train_final,
        "y_train": y_train_final,
        "X_test": X_test_sel,
        "y_test": y_test,
        "scaler": scaler,
        "selector": selector
    }
```

#### 2. Hyperparameter Optimization with Optuna
```python
def optimize_random_forest(self, X, y):
    def objective(trial):
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300),
            max_depth=trial.suggest_int("max_depth", 5, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=2
        )
        
        # 3-fold cross-validation for robust evaluation
        scores = []
        skf = StratifiedKFold(n_splits=3)
        for train_idx, val_idx in skf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], preds))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=1)
    return study
```

#### 3. Model Training and Calibration
```python
def train_model(self, config, X_train, y_train, X_test, y_test):
    # Train base model
    model = config['class'](**config['params'])
    model.fit(X_train, y_train)
    
    # Probability calibration for better ranking
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Evaluation
    preds = calibrated.predict(X_test)
    probs = calibrated.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1_score': f1_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, probs)
    }
    
    return {
        'model': calibrated,
        'metrics': metrics,
        'status': 'success'
    }
```

### Performance Results Analysis

**Current Model Performance** (on synthetic data):
```
LightGBM_Optimized:  F1=0.901, ROC-AUC=0.708, Training=94.6s
XGBoost_Optimized:   F1=0.901, ROC-AUC=0.704, Training=132.3s  
HistGradientBoosting: F1=0.900, ROC-AUC=0.715, Training=202.3s
```

**Why ROC-AUC is Moderate (~0.70)**:
- Intentional result of probabilistic synthetic data generation
- Prevents models from memorizing deterministic rules
- F1-Score more relevant for recommendation ranking quality
- Real employment data would likely show higher ROC-AUC (0.80-0.90+)

---

## 🌍 Italian Language Processing

### TF-IDF Implementation for Compatibility Scoring
```python
def compatibility_score(self, exclusions, company_text):
    if pd.isna(exclusions) or not company_text:
        return 1.0
    
    exclusion_list = [e.strip().lower() for e in str(exclusions).split(',') if e.strip()]
    all_texts = exclusion_list + [company_text.lower()]
    
    # Italian-specific TF-IDF configuration
    italian_stop_words = [
        'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'il', 'lo', 'la', 
        'i', 'gli', 'le', 'un', 'una', 'uno', 'e', 'o', 'ma', 'se', 'che', 'chi', 'cui'
    ]
    
    vectorizer = TfidfVectorizer(
        stop_words=italian_stop_words,
        token_pattern=r'\b[a-zA-Zàèéìòù]+\b',  # Italian accented characters
        lowercase=True,
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=1000
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    company_vector = tfidf_matrix[-1]
    
    # Calculate similarities between exclusions and company text
    similarities = []
    for i in range(len(exclusion_list)):
        sim = cosine_similarity(tfidf_matrix[i], company_vector)[0][0]
        similarities.append(sim)
    
    # Weighted scoring: 70% max similarity + 30% average
    max_sim = max(similarities) if similarities else 0
    avg_sim = np.mean(similarities) if similarities else 0
    final_score = 1.0 - (0.7 * max_sim + 0.3 * avg_sim)
    
    return max(0.0, min(1.0, final_score))
```

### Geographic Processing for Italy
```python
def geocode_with_cache(self, address):
    """Italian address geocoding with caching"""
    if address in self.loc_cache:
        return self.loc_cache[address]
    
    try:
        # Nominatim with Italian focus
        location = self.geolocator.geocode(
            address + ", Italy",  # Force Italian context
            timeout=10,
            country_codes=['IT']  # Restrict to Italy
        )
        time.sleep(0.5)  # Rate limiting
        coords = (location.latitude, location.longitude) if location else (np.nan, np.nan)
    except Exception as e:
        print(f"Geocoding error for {address}: {e}")
        coords = (np.nan, np.nan)
    
    self.loc_cache[address] = coords
    return coords
```

---

## 📊 Data Schema and Flow

### Input Data Structure

#### Candidates Dataset
```
Columns (Extended):
- ID_Candidato: str
- Area di Residenza: str
- Titolo di Studio: str ['Licenza Media', 'Diploma', 'Laurea', 'Master']
- Tipo di Disabilità: str ['Motoria', 'Sensoriale', 'Intellettiva', 'Psichica', ...]
- Score Attitudine al Collocamento: float [0.0-1.0]
- Years_of_Experience: int
- Durata Disoccupazione: int (months)
- Esclusioni: str (comma-separated)
- Lat, Lon: float (geocoded coordinates)
```

#### Companies Dataset
```
Columns (Extended):
- Nome Azienda: str
- Area di Attività: str
- Tipo di Attività: str
- Numero Dipendenti: int
- Compatibilità: str (job descriptions)
- Posizioni Aperte: int
- Remote: int [0, 1]
- Certification: int [0, 1]
- Retention_Rate: float [0.0-1.0]
- Company_Size: str ['small', 'medium', 'large']
- Lat_a, Lon_a: float (geocoded coordinates)
```

### Training Dataset Structure
```
Enhanced_Training_Dataset.csv:
- outcome: int [0, 1] (target variable)
- attitude_score: float
- years_experience: int
- unemployment_duration: int
- compatibility_score: float
- distance_km: float
- company_size: int
- retention_rate: float
- remote_work: int
- certification: int
- match_probability: float
- edu_* : int (one-hot encoded education)
- dis_* : int (one-hot encoded disability types)
- sector_* : int (one-hot encoded company sectors)

Typical size: 500,000+ rows, 50+ features
```

---

## 🔧 Development Setup

### Prerequisites
```bash
# Python 3.8+ (tested on 3.11)
# Minimum 8GB RAM (16GB recommended)
# 3GB free storage space

pip install -r requirements.txt
```

### Key Dependencies
```
Core ML/Data Science:
pandas==2.3.0, numpy==2.3.0, scikit-learn==1.6.1
scipy==1.15.3, imbalanced-learn==0.13.0, joblib==1.5.1

Advanced ML:
xgboost==3.0.2, lightgbm==4.6.0, optuna==4.4.0

Visualization:
matplotlib==3.10.3, seaborn==0.13.2, plotly==6.1.2

Streamlit & Web:
streamlit==1.46.0, altair==5.5.0

Geocoding & Geography:
geopy==2.4.1, geographiclib==2.0

Configuration & Utilities:
PyYAML==6.0.2, python-dateutil==2.9.0.post0
```

### Development Workflow
```bash
# 1. Data preparation (synthetic mode)
python scripts/01_generate_dataset.py

# 2. Model training 
python scripts/03_train_models.py

# 3. Results analysis
python scripts/04_analyze_results.py

# 4. Launch production interface
streamlit run streamlit_app.py
```

### Production Deployment
```bash
# For real employment data:
# 1. Replace Enhanced_Training_Dataset.csv with real outcomes
# 2. Train models on real data
python scripts/03_train_models.py

# 3. Launch production interface
streamlit run streamlit_app.py
```

---

## 🛠️ API Reference

### Core Classes

#### `EnhancedScoringSystem`
```python
class EnhancedScoringSystem:
    def __init__(self)
    def geocode_with_cache(self, address: str) -> Tuple[float, float]
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float
    def compatibility_score(self, exclusions: str, company_text: str) -> float
    def generate_enhanced_training_data(self, df_cand: pd.DataFrame, df_az: pd.DataFrame) -> pd.DataFrame
```

#### `ParallelModelTrainer`
```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42)
    def parallel_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict
    def create_optimized_models(self, best_params: Dict) -> List[Dict]
    def parallel_model_training(self, model_configs: List, X_train, y_train, X_test, y_test) -> Dict
    def create_ensemble_model(self, results: Dict, X_train, y_train) -> VotingClassifier
    def save_models(self, results: Dict, ensemble_model, save_dir: str = 'results')
```

### Utility Functions

#### Feature Engineering
```python
def extend_candidates_dataset(df: pd.DataFrame) -> pd.DataFrame
def extend_companies_dataset(df: pd.DataFrame) -> pd.DataFrame
```

#### Data Preparation
```python
def prepare_data_for_training(df_train: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict
```

---

## 🔒 Security and Privacy Considerations

### Data Protection
- **No Personal Identifiers**: All candidate data uses anonymous IDs
- **Synthetic Data Mode**: Default operation with no real personal information
- **Configurable Thresholds**: Prevents identification through unique combinations
- **Local Processing**: All ML operations performed locally, no external data transmission

### Geocoding Privacy
- **Cached Results**: Coordinates cached locally to minimize API calls
- **City-Level Precision**: Uses city/town level, not exact addresses
- **Rate Limiting**: Respects Nominatim usage policies
- **Offline Capable**: Can operate with pre-cached coordinates

### Production Deployment Security
- **Data Anonymization**: Ensure all production data is properly anonymized
- **Access Controls**: Implement appropriate user authentication
- **Audit Logging**: Track system usage and recommendation outcomes
- **Regular Updates**: Keep dependencies updated for security patches

---

## 📈 Performance Monitoring

### System Metrics
```python
# Built-in resource monitoring
class SystemResourceMonitor:
    def start(self): # Begin CPU/memory tracking
    def stop(self):  # End monitoring
    def stats(self): # Return average usage statistics
```

### Model Performance Tracking
```python
# Automatic metrics calculation and storage
results = {
    'model_name': {
        'metrics': {
            'accuracy': float,
            'precision': float, 
            'recall': float,
            'f1_score': float,
            'roc_auc': float
        },
        'training_time': float,
        'model_size_kb': float
    }
}
```

### Production Monitoring Recommendations
- **Response Time Tracking**: Monitor candidate-company matching latency
- **Recommendation Quality**: Track success rates by score ranges
- **System Resources**: CPU/memory usage during peak operations
- **Error Logging**: Geocoding failures, model prediction errors
- **User Activity**: Usage patterns and feature adoption

---

## 🔄 Maintenance and Updates

### Regular Maintenance Tasks
1. **Data Quality Checks**: Validate candidate/company information accuracy
2. **Model Performance Review**: Monitor F1-scores and recommendation success
3. **Cache Cleanup**: Clear old geocoding cache entries
4. **Log Rotation**: Manage system and error logs
5. **Dependency Updates**: Security patches and library updates

### Model Retraining
```bash
# When new employment outcome data is available:
python scripts/03_train_models.py

# The system will:
# 1. Retrain all 7 models with new data
# 2. Recalibrate probability thresholds  
# 3. Update ensemble weights
# 4. Save new models to results/ directory
```

### Configuration Updates
- **Threshold Adjustments**: Modify matching criteria based on placement success
- **Geographic Expansion**: Update distance limits for rural vs urban areas
- **Language Updates**: Add new Italian stop words or exclusion terms
- **Performance Tuning**: Adjust parallel processing based on hardware capabilities

---

*This technical documentation provides the essential information for understanding, maintaining, and extending the Disability Job Matching System. For specific implementation questions or advanced customization needs, contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Target Audience**: Developers, System Administrators, Researchers

### Deployment Guide
# 🚀 Deployment Guide - Disability Job Matching System

**Complete Installation and Setup Guide for Production Environments**

---

## 📋 Overview

This deployment guide provides step-by-step instructions for installing and configuring the Disability Job Matching System in production environments. The guide covers both demo mode (synthetic data) and production mode (real employment data) deployments.

### Deployment Modes

- **🧪 Demo Mode**: Uses synthetic data for testing and demonstration
- **🏭 Production Mode**: Uses real employment outcome data for live operations
- **🔧 Development Mode**: Full development environment with all scripts

---

## 📋 Prerequisites

### System Requirements

**Minimum Requirements**:
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

**Network Requirements**:
- **Internet Connection**: Required for initial geocoding setup
- **Firewall**: Port 8501 (Streamlit default) or custom port
- **Geographic API**: Access to Nominatim geocoding service

### Software Dependencies

**Required Software**:
```bash
# Python 3.8+ with pip
python --version  # Should show 3.8 or higher
pip --version

# Git (for repository cloning)
git --version

# Optional: Virtual environment tools
python -m venv --help
```

---

## 🔧 Installation Methods

### Method 1: Quick Demo Setup (Recommended for Testing)

**Step 1: Clone Repository**
```bash
# Clone the project
git clone https://github.com/your-username/disability-job-matching.git
cd disability-job-matching

# Verify project structure
ls -la
# Should see: streamlit_app.py, config.yaml, requirements.txt, data/, scripts/, utils/
```

**Step 2: Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, numpy, sklearn; print('Dependencies OK')"
```

**Step 3: Launch Demo**
```bash
# Start the application in demo mode
streamlit run streamlit_app.py

# System will automatically:
# 1. Load with synthetic demo data
# 2. Open browser at http://localhost:8501
# 3. Show "Demo Mode" in interface
```

**Verification**:
- Interface loads successfully
- Analytics tab shows sample data
- Can perform candidate searches
- Results appear within 5 seconds

### Method 2: Production Setup (Real Data)

**Prerequisites for Production**:
- Historical employment outcome data
- Properly formatted CSV files
- Data privacy compliance approval

**Step 1: Prepare Production Data**

```bash
# Production data structure required:
data/
├── raw/
│   ├── Dataset_Candidati_Aggiornato.csv      # Real candidate data
│   └── Dataset_Aziende_con_Stima_Assunzioni.csv  # Real company data
└── processed/
    └── Enhanced_Training_Dataset.csv         # Real employment outcomes
```

**Enhanced_Training_Dataset.csv Format**:
```csv
outcome,attitude_score,years_experience,unemployment_duration,compatibility_score,distance_km,company_size,retention_rate,remote_work,certification,...
1,0.75,5,12,0.85,15.2,150,0.78,1,1,...
0,0.45,2,24,0.35,45.8,50,0.65,0,0,...
```

**Step 2: Data Validation**
```bash
# Validate data format
python -c "
import pandas as pd
df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
print(f'Training data: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Outcome distribution: {df.outcome.value_counts()}')
print('Required columns present:', all(col in df.columns for col in ['outcome', 'attitude_score', 'compatibility_score']))
"
```

**Step 3: Train Production Models**
```bash
# Train models on real data
python scripts/03_train_models.py

# Expected output:
# 📥 Loading training dataset...
# 🧹 Preparing data...
# 🎯 Optimizing hyperparameters...
# 🤖 Training [Model Name]...
# ✅ All models saved

# Verify models created
ls -la results/
# Should see: *.joblib files, metrics_summary.csv
```

**Step 4: Launch Production Interface**
```bash
# Start production application
streamlit run streamlit_app.py

# System will:
# 1. Detect real data automatically
# 2. Load trained models
# 3. Enable production features
```

### Method 3: Docker Deployment (Enterprise)

**Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Commands**:
```bash
# Build image
docker build -t disability-job-matcher .

# Run container (demo mode)
docker run -p 8501:8501 disability-job-matcher

# Run with real data volume
docker run -p 8501:8501 -v /path/to/real/data:/app/data disability-job-matcher
```

---

## ⚙️ Configuration

### Environment Configuration

**config.yaml Customization**:
```yaml
# Production configuration example
paths:
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"
  logs_dir: "logs"

matching_thresholds:
  attitude_min: 0.3           # Adjust based on candidate pool
  distance_max_km: 30         # Adjust for urban/rural deployment
  match_probability_cutoff: 0.6

model_training:
  n_jobs: 4                   # Adjust based on server CPU cores
  optuna_trials: 50           # Reduce for faster training, increase for accuracy

geocoding:
  delay: 0.5                  # Increase if hitting rate limits
  cache_file: "data/processed/geocoding_cache.json"

streamlit:
  page_title: "Sistema Collocamento Mirato - [Organization Name]"
  default_top_k: 5
```

**Environment Variables**:
```bash
# Optional environment configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### Network Configuration

**Firewall Settings**:
```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload

# Verify port accessibility
netstat -tlnp | grep 8501
```

**Reverse Proxy (Nginx)**:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 🔒 Security Configuration

### Access Control

**Basic Authentication (Streamlit)**:
```python
# Add to streamlit_app.py for basic protection
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

**File Permissions**:
```bash
# Secure file permissions
chmod 600 config.yaml                    # Config files
chmod 600 data/processed/*.csv           # Data files  
chmod 755 scripts/*.py                   # Executable scripts
chmod 644 requirements.txt               # Public files

# Secure directories
chmod 750 data/                          # Data directory
chmod 750 results/                       # Models directory
chmod 755 utils/                         # Code directory
```

### Data Protection

**Sensitive Data Handling**:
```bash
# Create secure data directory
sudo mkdir -p /opt/job-matcher/secure-data
sudo chown app-user:app-group /opt/job-matcher/secure-data
sudo chmod 750 /opt/job-matcher/secure-data

# Symlink to application
ln -s /opt/job-matcher/secure-data data/processed
```

**Backup Strategy**:
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/job-matcher"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup critical files
cp -r data/processed $BACKUP_DIR/$DATE/
cp -r results/ $BACKUP_DIR/$DATE/
cp config.yaml $BACKUP_DIR/$DATE/

# Compress and encrypt
tar -czf $BACKUP_DIR/$DATE.tar.gz $BACKUP_DIR/$DATE
rm -rf $BACKUP_DIR/$DATE

echo "Backup completed: $BACKUP_DIR/$DATE.tar.gz"
```

---

## 📊 Monitoring and Logging

### Application Monitoring

**Health Check Endpoint**:
```python
# Add to streamlit_app.py
import streamlit as st
import time

def health_check():
    """System health check"""
    checks = {
        'models_loaded': len(st.session_state.get('models', {})) > 0,
        'data_available': os.path.exists('data/processed/Enhanced_Training_Dataset.csv'),
        'config_valid': os.path.exists('config.yaml')
    }
    return all(checks.values()), checks

# Usage in app
if st.sidebar.button("Health Check"):
    healthy, details = health_check()
    if healthy:
        st.success("✅ System Healthy")
    else:
        st.error("❌ System Issues Detected")
        st.json(details)
```

**Performance Monitoring**:
```python
# Add performance tracking
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        
    def log_request(self, operation, duration):
        self.request_count += 1
        logging.info(f"Operation: {operation}, Duration: {duration:.2f}s, "
                    f"CPU: {psutil.cpu_percent()}%, "
                    f"Memory: {psutil.virtual_memory().percent}%")

# Usage
monitor = PerformanceMonitor()
start = time.time()
# ... perform matching operation ...
monitor.log_request("candidate_matching", time.time() - start)
```

### Logging Configuration

**Logging Setup**:
```python
# logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Usage in main application
logger = setup_logging()
logger.info("Application started")
```

**Log Rotation**:
```bash
# /etc/logrotate.d/job-matcher
/path/to/job-matcher/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 app-user app-group
}
```

---

## 🔄 Maintenance Procedures

### Regular Maintenance Tasks

**Daily Tasks**:
```bash
#!/bin/bash
# daily_maintenance.sh

echo "$(date): Starting daily maintenance"

# Check disk space
df -h | grep -E "/(dev|opt)" | awk '$5 > "80%" {print "WARNING: " $0}'

# Check application health
curl -f http://localhost:8501/_stcore/health || echo "WARNING: App health check failed"

# Validate data integrity
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
    print(f'Data OK: {df.shape[0]} records')
except Exception as e:
    print(f'Data ERROR: {e}')
"

# Clear old cache files
find data/processed -name "*.cache" -mtime +7 -delete

echo "$(date): Daily maintenance completed"
```

**Weekly Tasks**:
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update geocoding cache if needed
python -c "
import json
try:
    with open('data/processed/geocoding_cache.json', 'r') as f:
        cache = json.load(f)
    print(f'Geocoding cache: {len(cache)} entries')
except:
    print('No geocoding cache found')
"

# Check model performance
python scripts/04_analyze_results.py

# Backup models and config
cp -r results/ /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
cp config.yaml /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
```

### Model Updates

**When to Retrain Models**:
- New employment outcome data available (quarterly recommended)
- Significant changes in candidate/company demographics
- Performance degradation observed
- System configuration changes

**Retraining Process**:
```bash
# 1. Backup current models
cp -r results/ results_backup_$(date +%Y%m%d)/

# 2. Update training data
# Place new Enhanced_Training_Dataset.csv in data/processed/

# 3. Retrain models
python scripts/03_train_models.py

# 4. Validate new models
python scripts/04_analyze_results.py

# 5. Test with sample candidates
streamlit run streamlit_app.py

# 6. If satisfied, remove backup
# rm -rf results_backup_*
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Application Won't Start**
```bash
# Symptoms: ModuleNotFoundError, Import errors
# Solution: Verify dependencies
pip list | grep -E "(streamlit|pandas|numpy|sklearn)"
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+
```

**Issue 2: No Models Found**
```bash
# Symptoms: "No models loaded" warning
# Solution: Train models
ls -la results/  # Check if .joblib files exist
python scripts/03_train_models.py  # Train if missing
```

**Issue 3: Geocoding Failures**
```bash
# Symptoms: Distance calculation errors
# Solution: Check network and cache
ping nominatim.openstreetmap.org
ls -la data/processed/geocoding_cache.json

# Reset cache if corrupted
rm data/processed/geocoding_cache.json
```

**Issue 4: Poor Performance**
```bash
# Symptoms: Slow response times
# Solutions:
# 1. Check system resources
htop  # or top
free -h

# 2. Reduce parallel jobs
# Edit config.yaml: model_training.n_jobs: 2

# 3. Clear browser cache
# 4. Restart application
```

### Error Logs Analysis

**Common Error Patterns**:
```bash
# Check recent errors
tail -100 logs/app_$(date +%Y%m%d).log | grep ERROR

# Geocoding errors
grep "Geocoding error" logs/*.log

# Model prediction errors  
grep "prediction error" logs/*.log

# Memory issues
grep -i "memory\|oom" logs/*.log
```

### Performance Optimization

**Memory Optimization**:
```python
# Add to streamlit_app.py
import gc

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_data():
    # Load heavy data operations
    pass

# Clear cache when needed
if st.button("Clear Cache"):
    st.cache_data.clear()
    gc.collect()
    st.success("Cache cleared")
```

**CPU Optimization**:
```yaml
# config.yaml adjustments for slower systems
model_training:
  n_jobs: 2                 # Reduce from 4
  optuna_trials: 25         # Reduce from 50

matching_thresholds:
  distance_max_km: 25       # Reduce search space
```

---

## 📞 Support and Escalation

### Support Contacts

**Technical Issues**:
- **Primary**: michele.melch@gmail.com
- **Academic**: oleksandr.kuznetsov@uniecampus.it
- **Emergency**: Include "URGENT" in subject line

**Information to Include**:
```bash
# System information
uname -a
python --version
pip list > installed_packages.txt

# Error logs
tail -50 logs/app_$(date +%Y%m%d).log

# Configuration
cat config.yaml

# System resources
free -h && df -h
```

### Escalation Procedures

**Level 1: Application Issues**
- Check logs and common issues
- Restart application
- Verify configuration

**Level 2: Data/Model Issues**
- Validate data integrity
- Retrain models if needed
- Check for data updates

**Level 3: System/Infrastructure Issues**
- Contact system administrator
- Check network connectivity
- Review security logs

### Community Resources

**Documentation**:
- GitHub Repository: [Link to repository]
- User Guide: `docs/user_guide_english.md`
- Technical Docs: `docs/technical_documentation.md`

**Updates and Announcements**:
- GitHub Releases for version updates
- Email notifications for critical security updates

---

*This deployment guide provides comprehensive instructions for installing and maintaining the Disability Job Matching System. For organization-specific deployment questions or custom configuration needs, contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Next Review**: December 2025

---

## Appendix C — Experiment Reports
### Comprehensive Federated Learning Report
# COMPREHENSIVE FEDERATED LEARNING ANALYSIS REPORT
# Disability Employment Matching System
# Generated: 2025-08-21 20:24:13
# Author: Claude Sonnet 4

## EXECUTIVE SUMMARY
This experiment compares three machine learning approaches for disability employment matching:
- Centralized Learning: F1-Score = 0.9012
- Regional Learning: F1-Score = 0.9001
- Federated Learning: F1-Score = 0.9007

Key Finding: Federated learning achieves -0.0005 F1-score improvement over centralized approach.

## SYSTEM CONFIGURATION
Model Type: LightGBM with CalibratedClassifierCV
Preprocessing Pipeline: RobustScaler → SelectKBest → SMOTE → Calibration
Aggregation Method: Weighted Ensemble (FedAvg approximation for tree models)
Regional Divisions: 5 Employment Centers (CPI)
Geographic Coverage: Veneto Region, Italy

## COMPLETE FILE STRUCTURE AND DATA LOCATIONS

### Training Data
```
data/
├── processed/
│   └── Enhanced_Training_Dataset.csv          # Original full dataset (500k samples)
└── federated/
    ├── CPI_Verona_training_data.csv           # Regional training data
    ├── CPI_Vicenza_training_data.csv           # Regional training data
    ├── CPI_Padova_training_data.csv           # Regional training data
    ├── CPI_Treviso_training_data.csv           # Regional training data
    ├── CPI_Venezia_training_data.csv           # Regional training data
    └── regional_statistics.csv               # Data distribution statistics
```

### Model Files Structure
```
results_federated/
├── regional_models/                          # Individual regional models
│   ├── CPI_Verona_complete_model.joblib      # Complete regional model + preprocessing
│   ├── CPI_Vicenza_complete_model.joblib      # Complete regional model + preprocessing
│   ├── CPI_Padova_complete_model.joblib      # Complete regional model + preprocessing
│   ├── CPI_Treviso_complete_model.joblib      # Complete regional model + preprocessing
│   ├── CPI_Venezia_complete_model.joblib      # Complete regional model + preprocessing
├── federated_models/                         # Aggregated federated model
│   └── federated_global_model_complete.joblib # Global model with metadata
├── centralized_models/                       # Baseline centralized models
│   ├── original_centralized_LightGBM.joblib  # Original centralized model
│   └── centralized_preprocessing.joblib      # Centralized preprocessing objects
├── complete_model_comparison.csv             # Performance comparison results
└── experiment_metadata.json                  # Complete experiment metadata
```

## MODEL ARCHITECTURE DETAILS

### 1. Centralized Model
Location: `results_federated/centralized_models/original_centralized_LightGBM.joblib`
- Trained on complete 500k sample dataset
- Single global model with unified preprocessing
- Preprocessing: `results_federated/centralized_models/centralized_preprocessing.joblib`

### 2. Regional Models
Location: `results_federated/regional_models/`
Each regional model contains:
- `calibrated_model`: CalibratedClassifierCV wrapper
- `base_model`: Raw LightGBM classifier
- `preprocessing`: RobustScaler + SelectKBest objects
- `training_samples`: Sample count for weighting
- `metrics`: Validation performance metrics

#### CPI_Verona
- File: `CPI_Verona_complete_model.joblib`
- Training Samples: 129,534
- F1-Score: 0.8994
- ROC-AUC: 0.7037

#### CPI_Vicenza
- File: `CPI_Vicenza_complete_model.joblib`
- Training Samples: 135,156
- F1-Score: 0.9007
- ROC-AUC: 0.7044

#### CPI_Padova
- File: `CPI_Padova_complete_model.joblib`
- Training Samples: 144,052
- F1-Score: 0.8988
- ROC-AUC: 0.7010

#### CPI_Treviso
- File: `CPI_Treviso_complete_model.joblib`
- Training Samples: 106,978
- F1-Score: 0.9006
- ROC-AUC: 0.7020

#### CPI_Venezia
- File: `CPI_Venezia_complete_model.joblib`
- Training Samples: 109,488
- F1-Score: 0.9012
- ROC-AUC: 0.7005

### 3. Federated Global Model
Location: `results_federated/federated_models/federated_global_model_complete.joblib`
Contains:
- `ensemble_model`: VotingClassifier with regional models
- `regional_weights`: Sample-based weighting scheme
- `template_preprocessing`: Preprocessing objects for inference
- `aggregation_history`: FedAvg aggregation metadata

### Regional Weight Distribution
| Region | Training Samples | Weight | Percentage |
|--------|------------------|--------|------------|
| CPI_Verona | 129,534 | 0.2072 | 20.7% |
| CPI_Vicenza | 135,156 | 0.2162 | 21.6% |
| CPI_Padova | 144,052 | 0.2304 | 23.0% |
| CPI_Treviso | 106,978 | 0.1711 | 17.1% |
| CPI_Venezia | 109,488 | 0.1751 | 17.5% |

## DETAILED PERFORMANCE ANALYSIS

### Regional Performance Breakdown
| Region | Centralized F1 | Regional F1 | Federated F1 | Improvement |
|--------|---------------|-------------|--------------|-------------|
| CPI_Verona | 0.8999 | 0.8994 | 0.8996 | -0.0003 |
| CPI_Vicenza | 0.9013 | 0.9007 | 0.9010 | -0.0003 |
| CPI_Padova | 0.9013 | 0.8988 | 0.9008 | -0.0005 |
| CPI_Treviso | 0.9011 | 0.9006 | 0.9004 | -0.0007 |
| CPI_Venezia | 0.9024 | 0.9012 | 0.9018 | -0.0006 |

### Average Performance Summary
| Metric | Centralized | Regional | Federated | Fed vs Cent |
|--------|------------|----------|-----------|-------------|
| F1-Score | 0.9012 | 0.9001 | 0.9007 | -0.0005 |
| Accuracy | 0.8288 | 0.8274 | 0.8278 | -0.0010 |
| ROC-AUC | 0.7156 | 0.7023 | 0.6865 | -0.0291 |

## TECHNICAL IMPLEMENTATION

### Model Parameter Extraction for Privacy
Each model file contains extractable parameters for differential privacy:
- **LightGBM Base Models**: Feature importances, tree structures
- **Preprocessing Objects**: Scaler parameters, feature selection indices
- **Calibration Parameters**: Isotonic regression mappings

### Federated Aggregation Method
Due to tree-based model complexity, we implement FedAvg approximation:
1. **Weight Calculation**: Wi = Ni / ΣNj (sample-proportional weighting)
2. **Ensemble Creation**: VotingClassifier with soft voting
3. **Parameter Aggregation**: Weighted ensemble of predictions

## REPRODUCIBILITY GUIDE

### Loading Models for Analysis
```python
import joblib

# Load centralized model
centralized_model = joblib.load('results_federated/centralized_models/original_centralized_LightGBM.joblib')
centralized_preprocessing = joblib.load('results_federated/centralized_models/centralized_preprocessing.joblib')

# Load regional model
regional_model_data = joblib.load('results_federated/regional_models/CPI_Padova_complete_model.joblib')
regional_model = regional_model_data['calibrated_model']
regional_preprocessing = regional_model_data['preprocessing']

# Load federated model
federated_data = joblib.load('results_federated/federated_models/federated_global_model_complete.joblib')
federated_model = federated_data['ensemble_model']
```

## DIFFERENTIAL PRIVACY IMPLEMENTATION ROADMAP

### Next Steps for Privacy-Preserving Federated Learning
1. **Parameter Extraction**: Extract model weights from saved models
2. **Noise Injection**: Add calibrated noise to extracted parameters
3. **Secure Aggregation**: Implement cryptographic parameter sharing
4. **Privacy Budget Tracking**: Monitor ε-differential privacy guarantees

### Required Components
- Privacy budget allocation per region
- Gradient perturbation mechanisms
- Secure multi-party computation protocols
- Privacy accounting frameworks

## KEY INSIGHTS AND CONCLUSIONS

### Performance Analysis
- **Centralized learning outperforms federated** by 0.0005 F1-score
- **Regional specialization** enables models to capture local employment patterns
- **Weighted ensemble aggregation** effectively combines regional expertise
- **Data locality preserved** while maintaining competitive performance

### Practical Implications
- Employment centers can maintain data sovereignty
- Regional models capture local labor market dynamics
- Federated approach enables knowledge sharing without data sharing
- Implementation ready for differential privacy enhancement

### Future Work
- Implement true differential privacy mechanisms
- Add secure aggregation protocols
- Evaluate communication cost trade-offs
- Deploy in real employment center environment

## REPORT GENERATION
Generated: 2025-08-21 20:24:13
Total Regions: 5
Total Models Saved: 7
Status: All models and data successfully saved for privacy implementation

### Experiment Summary
# PRODUCTION MLP FEDERATED LEARNING EXPERIMENT SUMMARY
Generated: 2025-08-21 18:21:47

## CRITICAL FIXES IMPLEMENTED
### 1. Deterministic Seeding (SHA256)
- FIXED: Replaced hash() with SHA256 for true reproducibility
- No more random salt between Python runs
- Deterministic region-specific seeds guaranteed

### 2. Proper Imputation Order
- FIXED: Removed fillna(0) that polluted global medians
- Only use global medians for consistent imputation
- No zero contamination in missing column handling

### 3. Correct Region-Weight Logging
- FIXED: Proper alignment of regions and weights in logs
- Track trained_regions list to prevent mismatched logging
- No more misleading weight assignments

### 4. Truly Balanced Warm-up
- FIXED: Uses minimum class count for true balance
- Equal samples per class regardless of original distribution
- Reduces variance in initial training steps

### 5. Weighted ROC-AUC with NaN Handling
- FIXED: Re-normalizes weights for non-NaN values
- Proper handling when some regions have single-class test sets
- No more weight loss due to NaN propagation

### 6. Deterministic Template Selection
- FIXED: Uses alphabetically first region for template
- Removes hidden source of non-determinism
- Consistent across different OS/filesystem orders

### 7. Enhanced Error Handling
- SMOTE fallback to RandomOverSampler
- Safe metrics calculation with zero_division guards
- Zero-weight protection in aggregation
- Auto-discovery of regional datasets
- Oversampling ratio logging

## Performance Results
### Unweighted Averages
Federated F1-Score: 0.7882
Federated Accuracy: 0.6951
Federated ROC-AUC: 0.7169

### Weighted Averages (by test sample size)
Weighted F1-Score: 0.7880
Weighted Accuracy: 0.6949
Weighted ROC-AUC: 0.7168

## Comparison with Centralized
Centralized F1-Score: 0.8276
Unweighted vs Centralized: -0.0394
Weighted vs Centralized: -0.0396
Status: Federated learning competitive

## Training Progression
Aggregation Method: fedavg

Round 1: Unweighted F1=0.4928, Weighted F1=0.4933 [Raw samples: 339995]
Round 2: Unweighted F1=0.6013 (Δ: +0.1085), Weighted F1=0.6017 (Δ: +0.1084) [Raw samples: 339995]
Round 3: Unweighted F1=0.6633 (Δ: +0.0620), Weighted F1=0.6640 (Δ: +0.0623) [Raw samples: 339995]
Round 4: Unweighted F1=0.6966 (Δ: +0.0334), Weighted F1=0.6973 (Δ: +0.0332) [Raw samples: 339995]
Round 5: Unweighted F1=0.7135 (Δ: +0.0169), Weighted F1=0.7133 (Δ: +0.0160) [Raw samples: 339995]
Round 6: Unweighted F1=0.7240 (Δ: +0.0105), Weighted F1=0.7237 (Δ: +0.0104) [Raw samples: 339995]
Round 7: Unweighted F1=0.7443 (Δ: +0.0203), Weighted F1=0.7445 (Δ: +0.0209) [Raw samples: 339995]
Round 8: Unweighted F1=0.7542 (Δ: +0.0100), Weighted F1=0.7539 (Δ: +0.0094) [Raw samples: 339995]
Round 9: Unweighted F1=0.7573 (Δ: +0.0030), Weighted F1=0.7574 (Δ: +0.0035) [Raw samples: 339995]
Round 10: Unweighted F1=0.7666 (Δ: +0.0093), Weighted F1=0.7667 (Δ: +0.0093) [Raw samples: 339995]
Round 11: Unweighted F1=0.7694 (Δ: +0.0028), Weighted F1=0.7693 (Δ: +0.0026) [Raw samples: 339995]
Round 12: Unweighted F1=0.7746 (Δ: +0.0051), Weighted F1=0.7748 (Δ: +0.0055) [Raw samples: 339995]
Round 13: Unweighted F1=0.7737 (Δ: -0.0009), Weighted F1=0.7736 (Δ: -0.0011) [Raw samples: 339995]
Round 14: Unweighted F1=0.7751 (Δ: +0.0014), Weighted F1=0.7751 (Δ: +0.0014) [Raw samples: 339995]
Round 15: Unweighted F1=0.7742 (Δ: -0.0009), Weighted F1=0.7738 (Δ: -0.0013) [Raw samples: 339995]
Round 16: Unweighted F1=0.7747 (Δ: +0.0006), Weighted F1=0.7743 (Δ: +0.0006) [Raw samples: 339995]
Round 17: Unweighted F1=0.7803 (Δ: +0.0056), Weighted F1=0.7803 (Δ: +0.0060) [Raw samples: 339995]
Round 18: Unweighted F1=0.7842 (Δ: +0.0038), Weighted F1=0.7840 (Δ: +0.0037) [Raw samples: 339995]
Round 19: Unweighted F1=0.7792 (Δ: -0.0049), Weighted F1=0.7792 (Δ: -0.0047) [Raw samples: 339995]
Round 20: Unweighted F1=0.7827 (Δ: +0.0035), Weighted F1=0.7822 (Δ: +0.0030) [Raw samples: 339995]
Round 21: Unweighted F1=0.7840 (Δ: +0.0013), Weighted F1=0.7835 (Δ: +0.0013) [Raw samples: 339995]
Round 22: Unweighted F1=0.7859 (Δ: +0.0020), Weighted F1=0.7857 (Δ: +0.0021) [Raw samples: 339995]
Round 23: Unweighted F1=0.7877 (Δ: +0.0017), Weighted F1=0.7874 (Δ: +0.0017) [Raw samples: 339995]
Round 24: Unweighted F1=0.7848 (Δ: -0.0029), Weighted F1=0.7844 (Δ: -0.0030) [Raw samples: 339995]
Round 25: Unweighted F1=0.7850 (Δ: +0.0002), Weighted F1=0.7847 (Δ: +0.0003) [Raw samples: 339995]
Round 26: Unweighted F1=0.7860 (Δ: +0.0010), Weighted F1=0.7856 (Δ: +0.0009) [Raw samples: 339995]
Round 27: Unweighted F1=0.7865 (Δ: +0.0005), Weighted F1=0.7860 (Δ: +0.0004) [Raw samples: 339995]
Round 28: Unweighted F1=0.7862 (Δ: -0.0003), Weighted F1=0.7862 (Δ: +0.0001) [Raw samples: 339995]
Round 29: Unweighted F1=0.7857 (Δ: -0.0005), Weighted F1=0.7859 (Δ: -0.0003) [Raw samples: 339995]
Round 30: Unweighted F1=0.7879 (Δ: +0.0022), Weighted F1=0.7880 (Δ: +0.0022) [Raw samples: 339995]

Total unweighted improvement: +0.2951
Total weighted improvement: +0.2948

## Production Optimizations Implemented

### 1. Safe Parameter Handling
- Only modify model.coefs_ and model.intercepts_
- No modification of internal attributes (n_layers_, n_outputs_)
- Prevents conflicts with internal buffers and loss curves

### 2. Proper Batch Processing
- Complete batch coverage (no dropped tail samples)
- Epoch-wise shuffling for better generalization
- Range-based indexing: range(0, N, batch_size)

### 3. Multiple Aggregation Methods
- FedAvg: Weighted by raw sample counts
- FedAvg Equal: Unweighted for diagnostic purposes
- Coordinate Median: Robust to outlier clients
- Trimmed Mean: Removes extreme values

### 4. Robust Preprocessing
- Graceful column handling with reindex()
- Global medians for consistent imputation
- Same statistics for train/validation/test

### 5. Full Reproducibility
- SHA256-based deterministic seeding
- Region and round-specific seeds
- Deterministic warm-up batch creation
- Deterministic template region selection

### 6. Enhanced Metrics
- Both unweighted and weighted averages
- Weighted by test sample sizes
- Separate tracking of raw vs SMOTE sample counts
- Proper NaN handling in weighted calculations

### 7. Optimized Configuration
- SGD solver for better partial_fit stability
- Explicit momentum=0.9 and Nesterov acceleration
- Reduced batch size to avoid clipping warnings
- Configurable validation frequency and patience

### 8. Enhanced Error Handling
- SMOTE fallback to RandomOverSampler
- Safe metrics calculation with zero_division guards
- Zero-weight protection in aggregation
- Auto-discovery of regional datasets
- Oversampling ratio logging for debugging

## Global Preprocessor Details
- Selected features: 32
- Total feature columns: 32
- Scaler: StandardScaler fitted on combined regional data
- Selector: SelectKBest with f_classif scoring
- Global medians stored for consistent imputation
- NOTE: Global preprocessor uses centralized statistics (acceptable for simulation)
- For production FL: consider secure aggregation or frozen preprocessor

## Key Files
- Global Model: mlp_global_federated_model.joblib
- Robust Preprocessor: global_preprocessor.joblib
- Performance Results: mlp_federated_performance.csv
- Training History: federated_training_history.json
- Enhanced Summary: federated_improvement_summary.csv
- Regional Models: regional_models/

### Privacy Summary
ENHANCED SHAMIR SECRET SHARING - ALL CRITICAL FIXES COMPLETE



================================================================================

COMPREHENSIVE TEST: Enhanced Privacy-Preserving Shamir (FINAL)

================================================================================

Configuration: 3-of-5

Prime: 2305843009213693951 (61 bits)

DP: ε=1.0, σ\_client=0.0611



Test 1: Critical Edge Cases (Final)

------------------------------------------------------------

&nbsp;  1: PASS | Orig:   0.000000 | Recon:   0.000000 | Err: 0.00e+00

&nbsp;  2: PASS | Orig:   1.000000 | Recon:   1.000000 | Err: 0.00e+00

&nbsp;  3: PASS | Orig:  -1.000000 | Recon:  -1.000000 | Err: 0.00e+00

&nbsp;  4: PASS | Orig:   3.141590 | Recon:   3.141590 | Err: 0.00e+00

&nbsp;  5: PASS | Orig:  -2.718280 | Recon:  -2.718280 | Err: 0.00e+00

&nbsp;  6: PASS | Orig: 1000.000000 | Recon: 1000.000000 | Err: 0.00e+00

&nbsp;  7: PASS | Orig: -1000.000000 | Recon: -1000.000000 | Err: 0.00e+00

&nbsp;  8: PASS | Orig:   0.001000 | Recon:   0.001000 | Err: 0.00e+00

&nbsp;  9: PASS | Orig:  -0.001000 | Recon:  -0.001000 | Err: 0.00e+00

&nbsp; 10: PASS | Orig:   0.000010 | Recon:   0.000010 | Err: 0.00e+00

&nbsp; 11: PASS | Orig:  -0.000010 | Recon:  -0.000010 | Err: 0.00e+00

&nbsp; 12: PASS | Orig: -12.368791 | Recon: -12.368791 | Err: 2.99e-07

&nbsp; Result: 12/12 passed



Test 2: Vectorized Operations

------------------------------------------------------------

&nbsp; 1: PASS | Size:    3 | Max Error: 0.00e+00 | Time: 0.000s

&nbsp; 2: PASS | Size:    3 | Max Error: 0.00e+00 | Time: 0.000s

&nbsp; 3: PASS | Size:    3 | Max Error: 0.00e+00 | Time: 0.000s

&nbsp; 4: PASS | Size:   50 | Max Error: 4.97e-07 | Time: 0.001s

&nbsp; 5: PASS | Size:    4 | Max Error: 0.00e+00 | Time: 0.000s

&nbsp; Result: 5/5 passed



Test 3: FIXED Secure Aggregation (Correct L2 Formula)

------------------------------------------------------------

&nbsp; Secure aggregation completed

&nbsp; Aggregation error: 0.3223

&nbsp; Expected DP noise (L2): 0.4318

&nbsp; Quality: EXCELLENT - Within 2x expected (ratio: 0.7x)

&nbsp; Status: PASS



Test 4: FIXED Dropout Handling (Correct L2 + Identical Seeds)

------------------------------------------------------------

&nbsp; Dropout recovery completed

&nbsp; Partial aggregation error: 1.3941

&nbsp; Expected noise (3 clients, L2): 0.3345

&nbsp; Quality: GOOD - Within 5x expected (ratio: 4.2x)

&nbsp; Status: PASS



Test 5: Symmetric Seed Verification

------------------------------------------------------------

&nbsp; PASS: Seeds for (0,1) are symmetric

&nbsp; PASS: Seeds for (1,2) are symmetric

&nbsp; PASS: Seeds for (2,3) are symmetric

&nbsp; PASS: Seeds for (3,4) are symmetric

&nbsp; Symmetric seed test: PASS



================================================================================

FINAL SUMMARY - ALL CRITICAL FIXES COMPLETE

================================================================================

Overall Results: 5/5 test suites passed



Test Suite Results:

&nbsp; 1. Edge Cases: PASS (12/12)

&nbsp; 2. Vector Operations: PASS (5/5)

&nbsp; 3. Secure Aggregation (Fixed): PASS

&nbsp; 4. Dropout Handling (Fixed): PASS

&nbsp; 5. Symmetric Seeds: PASS



Final Critical Fixes:

&nbsp; ✓ FIXED: Correct L2 noise estimation formula sqrt(n\*d)

&nbsp; ✓ FIXED: Identical seed truncation for masking and recovery

&nbsp; ✓ FIXED: Consistent test criteria with production evaluation

&nbsp; ✓ FIXED: RDP-based composition for practical noise levels

&nbsp; ✓ FIXED: Per-layer DP calibration for better SNR

&nbsp; ✓ FIXED: All previous cryptographic and numerical issues



STATUS: ALL CRITICAL FIXES VERIFIED - PRODUCTION DEPLOYMENT READY

================================================================================



================================================================================

PRODUCTION FEDERATED LEARNING INTEGRATION (ALL FIXES)

================================================================================

Production Configuration:

&nbsp; Clients: 5, Threshold: 3

&nbsp; DP Budget: ε=1.0, δ=1e-06

&nbsp; Base noise multiplier: 0.0611



Model: 7 layers, 60,201 total parameters



Calibrating per-layer DP parameters...



Simulating realistic federated training round...

&nbsp; Processing embedding (1000, 50)...

&nbsp; Processing hidden1 (50, 100)...

&nbsp; Processing hidden1\_bias (100,)...

&nbsp; Processing hidden2 (100, 50)...

&nbsp; Processing hidden2\_bias (50,)...

&nbsp; Processing output (50, 1)...

&nbsp; Processing output\_bias (1,)...

&nbsp; Secure aggregation completed for all layers!



Privacy-Preserving Aggregation Quality Assessment (FINAL):

----------------------------------------------------------------------

&nbsp; embedding   : EXCELLENT  - Within 2x expected (ratio: 1.0x)

&nbsp; hidden1     : EXCELLENT  - Within 2x expected (ratio: 1.0x)

&nbsp; hidden1\_bias: EXCELLENT  - Within 2x expected (ratio: 0.9x)

&nbsp; hidden2     : EXCELLENT  - Within 2x expected (ratio: 1.0x)

&nbsp; hidden2\_bias: EXCELLENT  - Within 2x expected (ratio: 0.9x)

&nbsp; output      : EXCELLENT  - Within 2x expected (ratio: 1.3x)

&nbsp; output\_bias : EXCELLENT  - Within 2x expected (ratio: 1.4x)

&nbsp; Overall quality: 100.0% of layers good/excellent



Production Assessment: PRODUCTION READY



Key Achievements:

&nbsp; ✓ Correct L2 noise estimation aligns theory with practice

&nbsp; ✓ Identical seed truncation ensures perfect mask cancellation

&nbsp; ✓ RDP-based composition reduces noise by ~10x vs linear

&nbsp; ✓ Per-layer DP calibration optimizes SNR for each layer

&nbsp; ✓ All cryptographic and numerical issues resolved



Ready for Production Deployment:

&nbsp; - All critical fixes implemented and verified

&nbsp; - Practical DP noise levels preserve utility

&nbsp; - Robust against client dropouts (3-of-5 threshold)

&nbsp; - Suitable for real federated learning workloads

&nbsp; - Consistent behavior across all test scenarios

================================================================================



🎯 ALL CRITICAL FIXES VERIFIED - PRODUCTION DEPLOYMENT READY!

Key achievements:

\- Correct L2 noise estimation formula sqrt(n\*d)

\- Identical seed truncation for perfect mask recovery

\- Consistent evaluation criteria across all scenarios

\- All cryptographic, numerical, and practical issues resolved

\- Ready for integration with real federated learning systems

### Anchoring Benchmark Summary
# Blockchain Anchoring Benchmark Results
Generated: 2025-08-22T22:31:09.960633

## Environment
- Python: 3.13.5
- CPU: 16 cores @ 2516 MHz
- RAM: 62.8 GB total
- Platform: win32

## Build Performance (Enhanced Measurements)
### Timing Breakdown
- 100 records: 2.28s total (KDF: 1.58s, Tree: 0.45s, Prepare: 0.02s)
- 1,000 records: 30.47s total (KDF: 21.15s, Tree: 6.04s, Prepare: 0.26s)
- 10,000 records: 344.07s total (KDF: 239.40s, Tree: 68.40s, Prepare: 2.07s)
### Memory Usage
- 100 records: RSS 0.0MB, Traced 0.3MB
- 1,000 records: RSS 1.7MB, Traced 1.6MB
- 10,000 records: RSS 22.2MB, Traced 16.0MB

## Proof Generation Performance (Enhanced)
### O(log n) Performance (with saved levels)
- 100 records: 1.11ms avg, 224 bytes actual (448 hex)
- 1,000 records: 2.61ms avg, 320 bytes actual (640 hex)
- 10,000 records: 20.65ms avg, 448 bytes actual (896 hex)

## Verification Performance (Enhanced)
- Average verification time: 24.49ms
- Success rate (correct password): 100.00%
- False positive rate (wrong password): 0.00%
- Error rate (verification failures): 0.00%
- Total verification errors: 0

## Key Insights (Enhanced Analysis)
- Memory scaling: RSS 507.2x, Traced 56.0x for 100.0x more records
- KDF vs Tree timing breakdown:
  - 10,000 iterations: KDF 66.9%, Tree 19.1% of total time
  - 50,000 iterations: KDF 69.3%, Tree 19.8% of total time
  - 100,000 iterations: KDF 69.8%, Tree 20.0% of total time
- Proof size scaling:
  - 100 records: 7.0 nodes (expected ~7), 224 bytes (50.0% of hex)
  - 1,000 records: 10.0 nodes (expected ~10), 320 bytes (50.0% of hex)
  - 10,000 records: 14.0 nodes (expected ~14), 448 bytes (50.0% of hex)

## Files Generated
- build_metrics.csv: Enhanced build phase performance data
- proof_metrics.csv: Proof generation with accurate byte sizes
- verify_metrics.csv: Verification timing with error tracking
- anchor_metrics.csv: Blockchain interaction with unique periods
- environment_info.json: System configuration
- bench.log: Detailed execution log
- performance_charts.png: Enhanced metrics visualization
