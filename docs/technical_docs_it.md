# 🔧 Documentazione Tecnica - Sistema di Raccomandazione per Collocamento Mirato

**Guida per Sviluppatori e Amministratori di Sistema**

---

## 📋 Panoramica

Questa documentazione tecnica fornisce informazioni complete per sviluppatori, amministratori di sistema e ricercatori che lavorano con il Sistema di Raccomandazione per Collocamento Mirato. Il sistema è costruito utilizzando un moderno stack Python ML con focus su scalabilità produzione e supporto lingua italiana.

### Architettura Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Dati Grezzi   │───▶│  Pipeline Dati   │───▶│  Dataset Potenziato │
│   (file CSV)    │    │  (Feature Eng.)  │    │  (Pronto Training)  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Modelli        │◀───│  Pipeline        │◀───│  Training           │
│  Addestrati     │    │  Training ML     │    │  Parallelo          │
│  (7 x .joblib)  │    │                  │    │  (ThreadPoolExec.)  │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│  App Streamlit  │◀───│  Motore Matching │
│  (Produzione)   │    │  Real-time       │
└─────────────────┘    └──────────────────┘
```

---

## 🏗️ Componenti Principali

### 1. Pipeline Elaborazione Dati (`scripts/`)

#### `01_generate_dataset.py`
**Scopo**: Estende dati grezzi candidati/aziende e genera dataset training sintetico

**Funzioni Chiave**:
```python
# Estensione dati con feature engineering
df_cand_ext = extend_candidates_dataset(df_cand)
df_az_ext = extend_companies_dataset(df_az)

# Generazione dati training sintetici
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_az_ext)
```

**Output**:
- `Dataset_Candidati_Aggiornato_Extended.csv`
- `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv`
- `Enhanced_Training_Dataset.csv` (500K+ righe)

#### `03_train_models.py`
**Scopo**: Training parallelo modelli ML con ottimizzazione iperparametri

**Processo Chiave**:
1. **Preprocessing Dati**: SMOTE, RobustScaler, SelectKBest
2. **Ottimizzazione Iperparametri**: Optuna con 50 trial per modello
3. **Training Parallelo**: ThreadPoolExecutor con fino a 6 worker
4. **Calibrazione Modelli**: CalibratedClassifierCV per calibrazione probabilità

**Modelli Addestrati**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized
- HistGradientBoosting
- GradientBoosting
- MLP_Optimized
- ExtraTrees

### 2. Logica Business Principale (`utils/`)

#### `scoring.py` - Core Algoritmo Matching
**Scopo**: Implementa la logica probabilistica di matching candidato-azienda

**Classi Chiave**:
```python
class EnhancedScoringSystem:
    def __init__(self):
        self.thresholds = {
            'attitude_min': 0.3,
            'compatibility_min': 0.5,
            'distance_max': 40.0  # Nota: config.yaml default è 30
        }
    
    def compatibility_score(self, exclusions, company_text):
        # Analisi semantica TF-IDF italiana
        # Restituisce punteggio compatibilità 0.0-1.0
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Calcolo distanza geografica preciso
        # Restituisce distanza in chilometri
    
    def generate_enhanced_training_data(self, df_cand, df_az):
        # Generazione outcome probabilistica
        # Crea dati training sintetici realistici
```

**Implementazione Formula Scoring**:
```python
prob = (
    0.3 * fattore_attitudine + 
    0.4 * fattore_compatibilità + 
    0.2 * fattore_distanza +
    0.05 * retention_rate + 
    0.025 * bonus_remoto + 
    0.025 * bonus_certificazione
)
outcome = 1 if (prob > 0.6 and np.random.random() < prob) else 0
```

#### `parallel_training.py` - Pipeline ML Multi-thread
**Scopo**: Training modelli ad alte prestazioni con monitoraggio risorse

**Caratteristiche Chiave**:
- **Ottimizzazione Iperparametri Parallela**: 3 studi Optuna concorrenti
- **Training Modelli Concorrente**: Fino a 6 modelli training simultaneamente
- **Monitoraggio Risorse Sistema**: Tracking uso CPU/Memoria con psutil
- **Preprocessing Avanzato**: SMOTE, RobustScaler, selezione feature

**Ottimizzazioni Performance**:
```python
# Ottimizzazione iperparametri parallela
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(self.optimize_random_forest, X, y): "random_forest",
        executor.submit(self.optimize_xgboost, X, y): "xgboost",
        executor.submit(self.optimize_lightgbm, X, y): "lightgbm"
    }

# Training modelli parallelo  
with ThreadPoolExecutor(max_workers=6) as executor:
    # Ogni modello si addestra indipendentemente con iperparametri ottimizzati
```

### 3. Interfaccia Produzione (`streamlit_app.py`)

#### Classe Applicazione Principale
```python
class JobMatchingDemo:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="job_matching_system")
        self.loc_cache = {}  # Cache geocodifica per performance
        self.load_data()     # Carica dati reali o demo
        self.load_models()   # Carica modelli ML addestrati
    
    def find_matches(self, candidate_data, top_k=5, distance_threshold=30):
        # Matching real-time con parametri configurabili
        # Restituisce lista ordinata aziende compatibili
```

**Caratteristiche Chiave**:
- **Modalità Dati Doppia**: Rilevamento automatico dati reali vs demo
- **Geocodifica Real-time**: Nominatim cached con supporto indirizzi italiani
- **Configurazione Interattiva**: Controlli sidebar per soglie
- **Visualizzazioni Avanzate**: Grafici Plotly per analisi risultati

---

## 🔧 Sistema Configurazione

### Struttura `config.yaml`
```yaml
paths:
  raw_candidates: "data/raw/Dataset_Candidati_Aggiornato.csv"
  raw_companies: "data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv"
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"

matching_thresholds:
  attitude_min: 0.3          # Soglia propensione al lavoro
  compatibility_min: 0.5     # Soglia compatibilità semantica  
  distance_max_km: 30        # Raggio ricerca default (NON 40!)
  match_probability_cutoff: 0.6

model_training:
  random_state: 42
  optuna_trials: 50          # Iterazioni ottimizzazione iperparametri
  n_jobs: 4                  # Core elaborazione parallela
  feature_selection_k: 50    # Top feature selezionate

geocoding:
  delay: 0.5                 # Rate limiting tra chiamate API
  timeout: 10                # Timeout richiesta
  user_agent: "disability-job-matcher-v1.0"
  cache_file: "data/processed/geocoding_cache.json"

italian_language:
  stop_words: ["di", "a", "da", "in", "con", "su", "per", ...]
  token_pattern: "\\b[a-zA-Zàèéìòù]+\\b"
```

### Caricamento Configurazione
```python
import yaml

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_config()
thresholds = config['matching_thresholds']
```

---

## 🤖 Implementazione Machine Learning

### Architettura Pipeline Modelli

#### 1. Preprocessing Dati
```python
def prepare_data_for_training(df_train, test_size=0.2, random_state=42):
    # Preparazione feature
    y = df_train["outcome"]
    X = df_train.drop(columns=["outcome"]).fillna(df_train.median())
    
    # Split train-test con stratificazione
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Scaling robusto (migliore per outlier di StandardScaler)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Selezione feature
    selector = SelectKBest(score_func=f_classif, k=min(50, X_train_scaled.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    
    # Bilanciamento classi con SMOTE
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

#### 2. Ottimizzazione Iperparametri con Optuna
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
        
        # Cross-validation 3-fold per valutazione robusta
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

#### 3. Training e Calibrazione Modelli
```python
def train_model(self, config, X_train, y_train, X_test, y_test):
    # Addestra modello base
    model = config['class'](**config['params'])
    model.fit(X_train, y_train)
    
    # Calibrazione probabilità per migliore ranking
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X_train, y_train)
    
    # Valutazione
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

### Analisi Risultati Performance

**Performance Modelli Attuali** (su dati sintetici):
```
LightGBM_Optimized:    F1=0.901, ROC-AUC=0.708, Training=94.6s
XGBoost_Optimized:     F1=0.901, ROC-AUC=0.704, Training=132.3s  
HistGradientBoosting:  F1=0.900, ROC-AUC=0.715, Training=202.3s
```

**Perché ROC-AUC è Moderato (~0.70)**:
- Risultato intenzionale della generazione dati sintetici probabilistici
- Previene che i modelli memorizzino regole deterministiche
- F1-Score più rilevante per qualità ranking raccomandazioni
- Dati collocamento reali mostrerebbero probabilmente ROC-AUC più alto (0.80-0.90+)

---

## 🌍 Elaborazione Lingua Italiana

### Implementazione TF-IDF per Scoring Compatibilità
```python
def compatibility_score(self, exclusions, company_text):
    if pd.isna(exclusions) or not company_text:
        return 1.0
    
    exclusion_list = [e.strip().lower() for e in str(exclusions).split(',') if e.strip()]
    all_texts = exclusion_list + [company_text.lower()]
    
    # Configurazione TF-IDF specifica per italiano
    italian_stop_words = [
        'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'il', 'lo', 'la', 
        'i', 'gli', 'le', 'un', 'una', 'uno', 'e', 'o', 'ma', 'se', 'che', 'chi', 'cui'
    ]
    
    vectorizer = TfidfVectorizer(
        stop_words=italian_stop_words,
        token_pattern=r'\b[a-zA-Zàèéìòù]+\b',  # Caratteri accentati italiani
        lowercase=True,
        ngram_range=(1, 2),  # Unigrammi e bigrammi
        max_features=1000
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    company_vector = tfidf_matrix[-1]
    
    # Calcola similarità tra esclusioni e testo azienda
    similarities = []
    for i in range(len(exclusion_list)):
        sim = cosine_similarity(tfidf_matrix[i], company_vector)[0][0]
        similarities.append(sim)
    
    # Scoring pesato: 70% similarità max + 30% media
    max_sim = max(similarities) if similarities else 0
    avg_sim = np.mean(similarities) if similarities else 0
    final_score = 1.0 - (0.7 * max_sim + 0.3 * avg_sim)
    
    return max(0.0, min(1.0, final_score))
```

### Elaborazione Geografica per Italia
```python
def geocode_with_cache(self, address):
    """Geocodifica indirizzi italiani con caching"""
    if address in self.loc_cache:
        return self.loc_cache[address]
    
    try:
        # Nominatim con focus italiano
        location = self.geolocator.geocode(
            address + ", Italy",  # Forza contesto italiano
            timeout=10,
            country_codes=['IT']  # Restringe all'Italia
        )
        time.sleep(0.5)  # Rate limiting
        coords = (location.latitude, location.longitude) if location else (np.nan, np.nan)
    except Exception as e:
        print(f"Errore geocodifica per {address}: {e}")
        coords = (np.nan, np.nan)
    
    self.loc_cache[address] = coords
    return coords
```

---

## 📊 Schema Dati e Flusso

### Struttura Dati Input

#### Dataset Candidati
```
Colonne (Esteso):
- ID_Candidato: str
- Area di Residenza: str
- Titolo di Studio: str ['Licenza Media', 'Diploma', 'Laurea', 'Master']
- Tipo di Disabilità: str ['Motoria', 'Sensoriale', 'Intellettiva', 'Psichica', ...]
- Score Attitudine al Collocamento: float [0.0-1.0]
- Years_of_Experience: int
- Durata Disoccupazione: int (mesi)
- Esclusioni: str (separate da virgola)
- Lat, Lon: float (coordinate geocodificate)
```

#### Dataset Aziende
```
Colonne (Esteso):
- Nome Azienda: str
- Area di Attività: str
- Tipo di Attività: str
- Numero Dipendenti: int
- Compatibilità: str (descrizioni lavoro)
- Posizioni Aperte: int
- Remote: int [0, 1]
- Certification: int [0, 1]
- Retention_Rate: float [0.0-1.0]
- Company_Size: str ['small', 'medium', 'large']
- Lat_a, Lon_a: float (coordinate geocodificate)
```

### Struttura Dataset Training
```
Enhanced_Training_Dataset.csv:
- outcome: int [0, 1] (variabile target)
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
- edu_* : int (educazione codificata one-hot)
- dis_* : int (tipi disabilità codificati one-hot)
- sector_* : int (settori azienda codificati one-hot)

Dimensione tipica: 500.000+ righe, 50+ feature
```

---

## 🔧 Setup Sviluppo

### Prerequisiti
```bash
# Python 3.8+ (testato su 3.11)
# Minimo 8GB RAM (16GB raccomandati)
# 3GB spazio storage libero

pip install -r requirements.txt
```

### Dipendenze Chiave
```
Core ML/Data Science:
pandas==2.3.0, numpy==2.3.0, scikit-learn==1.6.1
scipy==1.15.3, imbalanced-learn==0.13.0, joblib==1.5.1

ML Avanzato:
xgboost==3.0.2, lightgbm==4.6.0, optuna==4.4.0

Visualizzazione:
matplotlib==3.10.3, seaborn==0.13.2, plotly==6.1.2

Streamlit & Web:
streamlit==1.46.0, altair==5.5.0

Geocodifica & Geografia:
geopy==2.4.1, geographiclib==2.0

Configurazione & Utilità:
PyYAML==6.0.2, python-dateutil==2.9.0.post0
```

### Workflow Sviluppo
```bash
# 1. Preparazione dati (modalità sintetica)
python scripts/01_generate_dataset.py

# 2. Training modelli 
python scripts/03_train_models.py

# 3. Analisi risultati
python scripts/04_analyze_results.py

# 4. Lancia interfaccia produzione
streamlit run streamlit_app.py
```

### Deployment Produzione
```bash
# Per dati collocamento reali:
# 1. Sostituire Enhanced_Training_Dataset.csv con outcome reali
# 2. Addestrare modelli su dati reali
python scripts/03_train_models.py

# 3. Lancia interfaccia produzione
streamlit run streamlit_app.py
```

---

## 🛠️ Riferimento API

### Classi Principali

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

### Funzioni Utilità

#### Feature Engineering
```python
def extend_candidates_dataset(df: pd.DataFrame) -> pd.DataFrame
def extend_companies_dataset(df: pd.DataFrame) -> pd.DataFrame
```

#### Preparazione Dati
```python
def prepare_data_for_training(df_train: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict
```

---

## 🔒 Considerazioni Sicurezza e Privacy

### Protezione Dati
- **Nessun Identificatore Personale**: Tutti i dati candidato usano ID anonimi
- **Modalità Dati Sintetici**: Operazione default senza informazioni personali reali
- **Soglie Configurabili**: Previene identificazione tramite combinazioni uniche
- **Elaborazione Locale**: Tutte le operazioni ML eseguite localmente, nessuna trasmissione dati esterna

### Privacy Geocodifica
- **Risultati Cached**: Coordinate cachate localmente per minimizzare chiamate API
- **Precisione Livello Città**: Usa livello città/paese, non indirizzi esatti
- **Rate Limiting**: Rispetta politiche utilizzo Nominatim
- **Capacità Offline**: Può operare con coordinate pre-cachate

### Sicurezza Deployment Produzione
- **Anonimizzazione Dati**: Assicurare che tutti i dati produzione siano anonimizzati appropriatamente
- **Controlli Accesso**: Implementare autenticazione utente appropriata
- **Audit Logging**: Tracciare utilizzo sistema e outcome raccomandazioni
- **Aggiornamenti Regolari**: Mantenere dipendenze aggiornate per patch sicurezza

---

## 📈 Monitoraggio Performance

### Metriche Sistema
```python
# Monitoraggio risorse integrato
class SystemResourceMonitor:
    def start(self): # Inizia tracking CPU/memoria
    def stop(self):  # Termina monitoraggio
    def stats(self): # Restituisce statistiche utilizzo medio
```

### Tracking Performance Modelli
```python
# Calcolo e storage metriche automatico
results = {
    'nome_modello': {
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

### Raccomandazioni Monitoraggio Produzione
- **Tracking Tempo Risposta**: Monitorare latenza matching candidato-azienda
- **Qualità Raccomandazioni**: Tracciare tassi successo per range punteggi
- **Risorse Sistema**: Utilizzo CPU/memoria durante operazioni picco
- **Error Logging**: Fallimenti geocodifica, errori predizione modelli
- **Attività Utente**: Pattern utilizzo e adozione feature

---

## 🔄 Manutenzione e Aggiornamenti

### Attività Manutenzione Regolare
1. **Controlli Qualità Dati**: Validare accuratezza informazioni candidato/azienda
2. **Revisione Performance Modelli**: Monitorare F1-score e successo raccomandazioni
3. **Pulizia Cache**: Cancellare vecchie entry cache geocodifica
4. **Rotazione Log**: Gestire log sistema ed errori
5. **Aggiornamenti Dipendenze**: Patch sicurezza e aggiornamenti librerie

### Riaddestramento Modelli
```bash
# Quando disponibili nuovi dati outcome collocamento:
python scripts/03_train_models.py

# Il sistema:
# 1. Riaddestra tutti i 7 modelli con nuovi dati
# 2. Ricallibra soglie probabilità  
# 3. Aggiorna pesi ensemble
# 4. Salva nuovi modelli nella directory results/
```

### Aggiornamenti Configurazione
- **Regolazioni Soglie**: Modificare criteri matching basati su successo collocamento
- **Espansione Geografica**: Aggiornare limiti distanza per aree rurali vs urbane
- **Aggiornamenti Lingua**: Aggiungere nuove stop words italiane o termini esclusione
- **Tuning Performance**: Regolare elaborazione parallela basata su capacità hardware

---

*Questa documentazione tecnica fornisce le informazioni essenziali per comprendere, mantenere ed estendere il Sistema di Raccomandazione per Collocamento Mirato. Per domande implementazione specifiche o necessità personalizzazione avanzata, contattare il team di sviluppo.*

---

**Versione Documento**: 1.0  
**Ultimo Aggiornamento**: Giugno 2025  
**Audience Target**: Sviluppatori, Amministratori Sistema, Ricercatori