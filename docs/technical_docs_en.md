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