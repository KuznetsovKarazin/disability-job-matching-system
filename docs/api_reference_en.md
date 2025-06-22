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