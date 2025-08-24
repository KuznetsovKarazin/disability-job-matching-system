# Riferimento API — Sistema di Collocamento Mirato
_Ultimo aggiornamento: 2025-08-23 23:44_

<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" />
  <img src="https://img.shields.io/badge/language-IT-green" />
  <img src="https://img.shields.io/badge/federated-learning-purple" />
  <img src="https://img.shields.io/badge/privacy-DP%20%2B%20Shamir-teal" />
  <img src="https://img.shields.io/badge/blockchain-anchoring-orange" />
</p>

Questo documento definisce la **superficie API pubblica** del Sistema di Collocamento Mirato.  
Copre lo schema di configurazione, funzioni e classi a livello di modulo, entrypoint CLI, contratti dati e comportamenti operativi.  
Si applica al layout del repository indicato nella README:

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


## Indice
1. [Versioning & Stabilità](#versioning--stabilità)
2. [Schema di Configurazione](#schema-di-configurazione)
3. [Contratti Dati](#contratti-dati)
4. [Moduli](#moduli)
   - [utils.feature_engineering](#utilsfeature_engineering-1)
   - [utils.scoring](#utilsscoring-1)
   - [utils.parallel_training](#utilsparallel_training-1)
   - [utils.visualization](#utilsvisualization-1)
   - [utils.federated_data_splitter](#utilsfederated_data_splitter-1)
   - [utils.federated_learning](#utilsfederated_learning-1)
   - [utils.enhanced_shamir_privacy](#utilsenhanced_shamir_privacy-1)
   - [blockchain_data_anchoring](#blockchain_data_anchoring-1)
5. [CLI & Script](#cli--script)
6. [Integrazione Streamlit](#integrazione-streamlit)
7. [Errori & Eccezioni](#errori--eccezioni)
8. [Note di Performance](#note-di-performance)
9. [Sicurezza & Privacy](#sicurezza--privacy)
10. [Compatibilità all'Indietro](#compatibilità-allindietro)
11. [Changelog (API)](#changelog-api)
12. [Appendice A — API Legacy](#appendice-a--api-legacy)


## Versioning & Stabilità

- **Intento semantico**: le modifiche che rompono firme o semantica dei parametri saranno annotate nel changelog.
- **API Pubblica**: solo gli elementi documentati qui sono considerati stabili.
- **Schema config**: aggiunte compatibili (chiavi opzionali) sono ammesse; rimozioni o cambi tipo sono breaking.


## Schema di Configurazione

Chiavi di `config.yaml` e tipi. I valori mostrati sono consigliati.

```yaml
seed: 42
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies:  data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir:  data/processed
  training_csv:   data/processed/Enhanced_Training_Dataset.csv
  results_dir:    results
ui:
  distance_max_km: 30
training:
  model_set: ["LightGBM_Optimized", "MLP"]
  optuna_trials: 50
  calibration: "sigmoid"
federated:
  rounds: 10
  min_clients: 3
  aggregator: "fedavg"
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
```

**Note**
- I percorsi devono esistere; i writer creano directory intermedie quando possibile.
- Il raggio di default è **30 km**.
- I parametri DP definiscono il budget privacy per esperimento.


## Contratti Dati

### Input grezzi
- **Candidati** — `Dataset_Candidati_Aggiornato.csv`
- **Aziende**   — `Dataset_Aziende_con_Stima_Assunzioni.csv`

### Artefatti processati
- **Candidati Estesi** — `Dataset_Candidati_Aggiornato_Extended.csv`
- **Aziende Estese**   — `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv`
- **Tabella di Training (canonica)** — `Enhanced_Training_Dataset.csv`

**Riepilogo modelli**
- `results/merged_model_summary.csv` — metriche consolidate.


## Moduli
Classi e funzioni pubbliche esposte da ciascun modulo. Gli helper non elencati sono interni.


### utils.feature_engineering

**Scopo**: preparazione dati, estensione e generazione della tabella di training.

#### API Pubblica

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

**Esempio**
```python
c = load_raw_candidates("data/raw/Dataset_Candidati_Aggiornato.csv")
a = load_raw_companies("data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv")
df = build_training_table(extend_candidates(c), extend_companies(a))
df.to_csv("data/processed/Enhanced_Training_Dataset.csv", index=False)
```


### utils.scoring

**Scopo**: punteggio di matching e utilità collegate.

#### API Pubblica

```python
from typing import TypedDict

class ScoreBreakdown(TypedDict):
    compatibility: float
    distance_penalty: float
    readiness: float
    total: float

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

**Note**
- Somma pesata di sotto-scores normalizzati; penalità distanza satura oltre `distance_max_km`.


### utils.parallel_training

**Scopo**: training multi-thread e orchestrazione modelli.

#### API Pubblica
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


### utils.visualization

**Scopo**: grafici riutilizzabili per dataset ed esperimenti.

#### API Pubblica
```python
def plot_learning_curves(history: "pd.DataFrame", out_dir: str) -> str
def plot_confusion(y_true: "np.ndarray", y_pred: "np.ndarray", out_path: str) -> str
def compare_federated_runs(summary_csv: str, out_dir: str) -> str
```


### utils.federated_data_splitter

**Scopo**: discovery e preparazione dataset regionali per FL.

#### API Pubblica
```python
def discover_regional_csvs(root: str = "data/federated") -> list[str]
def stratified_split_by_region(df: "pd.DataFrame", *, test_size: float = 0.2, seed: int = 42) -> dict[str, "pd.DataFrame"]
```


### utils.federated_learning

**Scopo**: orchestrazione federata con aggregatori robusti.

#### Tipi Pubblici
```python
from enum import Enum
class Aggregator(str, Enum):
    fedavg = "fedavg"
    trimmed_mean = "trimmed_mean"
    coordinate_median = "coordinate_median"
```

#### API Pubblica
```python
class FederatedTrainer:
    def __init__(self, clients, model: str = "MLP", aggregator: Aggregator = Aggregator.fedavg,
                 rounds: int = 10, batch_size: int = 256, seed: int = 42): ...
    def run(self) -> dict:
        'Esegue i round FL e restituisce le metriche.'
    def evaluate_global(self) -> dict:
        'Valuta il modello aggregato su un holdout.'
    def save(self, out_dir: str) -> None:
        'Salva il modello globale e i metadati.'
```


### utils.enhanced_shamir_privacy

**Scopo**: secure aggregation (Shamir) e utilità per Privacy Differenziale.

#### API Pubblica
```python
class ShamirShares:
    def __init__(self, threshold: str = "3-of-5", seed: int = 42): ...
    def split(self, secret: "np.ndarray") -> list["np.ndarray"]
    def reconstruct(self, shares: list["np.ndarray"]) -> "np.ndarray"

def mask_updates(update: "np.ndarray", *, threshold: str = "3-of-5", seed: int = 42) -> tuple["np.ndarray", dict]
def dp_clip_and_noise(grads: "np.ndarray", *, max_grad_norm: float, epsilon: float, delta: float, accountant: str = "rdp", seed: int = 42) -> "np.ndarray"

class RDPAccountant:
    def __init__(self, delta: float): ...
    def compose(self, epsilons: list[float]) -> float
```


### blockchain_data_anchoring

**Scopo**: commit Merkle per risultati e modelli; prove di inclusione.

#### API Pubblica
```python
class MerkleTree:
    def __init__(self, leaves: list[bytes]): ...
    def root(self) -> bytes
    def proof(self, index: int) -> list[tuple[bytes, str]]

def build_manifest(paths: list[str]) -> dict
def commit_manifest(manifest: dict) -> tuple[bytes, "MerkleTree"]
def verify_inclusion(root: bytes, leaf: bytes, proof: list[tuple[bytes, str]]) -> bool
```


## CLI & Script

Ogni script espone una CLI basata su argparse. Usare `-h` per vedere tutte le opzioni.

```bash
python scripts/01_generate_dataset.py --config config.yaml
python scripts/02_visualize_dataset.py --config config.yaml

python scripts/03_train_models.py --config config.yaml --models LightGBM_Optimized MLP
python scripts/04_analyze_results.py --input results/merged_model_summary.csv --out results/

python scripts/05_LightGBM_federated_training.py --rounds 10 --aggregator fedavg
python scripts/06_LightGBM_federated_visualization.py --input results_LightGBM_federated/

python scripts/07_mlp_federated_training.py --rounds 10 --batch-size 256 --aggregator trimmed_mean
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
python scripts/09_mlp_federated_privacy_visualization.py --input results_mlp_federated_privacy/

python scripts/blockchain_data_anchoring.py --manifest results/
python scripts/10_blockchain_anchoring_bench.py --scale 10000
```


## Integrazione Streamlit

- Carica configurazione e dataset.
- Calcola e mostra ranking di matching.
- Visualizza confronti FL/centralizzato e budget privacy.

**Endpoint salute**: `/_stcore/health`.


## Errori & Eccezioni

- `FileNotFoundError` — percorsi non trovati.
- `ValueError` — schema invalido, dataset vuoti, bounds parametri.
- `RuntimeError` — errori training/FL, inconsistenze DP.
- `AssertionError` — controlli di allineamento shape.


## Note di Performance

- `trimmed_mean` o `coordinate_median` per robustezza con client volatili.
- Ridurre `batch_size` su nodi solo-CPU.
- Cache dataset estesi in `data/processed/`.


## Sicurezza & Privacy

- Nessun dato grezzo esce dai nodi; solo delta del modello in FL.
- Secure aggregation maschera aggiornamenti; DP limita leakage.
- Anchoring assicura integrità e non ripudio.


## Compatibilità all'Indietro

- Nuove chiavi config opzionali sono compatibili.
- Opzioni deprecate rimangono per almeno un ciclo minore.


## Changelog (API)

- **v2**: aggiunte API federate, helper privacy (Shamir + DP) e anchoring.
- **v1**: API iniziali per training centralizzato e scoring.


## Appendice A — API Legacy

> La sezione seguente mantiene il riferimento API precedente.

# 📚 Riferimento API - Sistema di Raccomandazione per Collocamento Mirato

**Documentazione Completa Codice per Sviluppatori e Ricercatori**

---

## 📋 Panoramica

Questo riferimento API fornisce documentazione comprensiva per tutte le classi, metodi e funzioni nel Sistema di Raccomandazione per Collocamento Mirato. La codebase è organizzata in componenti modulari con chiara separazione delle responsabilità.

### Struttura Moduli

```
utils/
├── scoring.py              # Algoritmi matching principali
├── parallel_training.py    # Pipeline training ML  
├── feature_engineering.py  # Preprocessing dati
└── visualization.py        # Generazione grafici

scripts/
├── 01_generate_dataset.py  # Entry point pipeline dati
├── 03_train_models.py      # Entry point pipeline training
└── 04_analyze_results.py   # Entry point pipeline analisi

streamlit_app.py            # Interfaccia web produzione
```

---

## 🎯 Classi Principali

### `EnhancedScoringSystem`

**Posizione**: `utils/scoring.py`

**Scopo**: Implementa l'algoritmo principale di matching candidato-azienda con supporto lingua italiana e elaborazione geografica.

#### Definizione Classe
```python
class EnhancedScoringSystem:
    def __init__(self)
```

**Attributi**:
- `geolocator`: Istanza geocodifica Nominatim
- `loc_cache`: Dizionario per caching risultati geocodifica
- `thresholds`: Soglie matching predefinite

#### Metodi

##### `geocode_with_cache(address: str) -> Tuple[float, float]`
Geocodifica indirizzi italiani con caching per ottimizzazione performance.

**Parametri**:
- `address` (str): Stringa indirizzo italiano (es. "Verona, Italy")

**Restituisce**:
- `Tuple[float, float]`: (latitudine, longitudine) o (NaN, NaN) se geocodifica fallisce

**Esempio**:
```python
scoring_system = EnhancedScoringSystem()
lat, lon = scoring_system.geocode_with_cache("Villafranca di Verona, Italy")
# Restituisce: (45.3506, 10.8444)
```

##### `haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float`
Calcola distanza geografica precisa usando formula Haversine.

**Parametri**:
- `lat1, lon1` (float): Coordinate prima località
- `lat2, lon2` (float): Coordinate seconda località

**Restituisce**:
- `float`: Distanza in chilometri, o NaN se coordinate invalide

**Esempio**:
```python
distance = scoring_system.haversine(45.4408, 10.9916, 45.3506, 10.8444)
# Restituisce: 15.2 (km tra Verona e Villafranca)
```

##### `compatibility_score(exclusions: str, company_text: str) -> float`
Analizza compatibilità semantica tra esclusioni candidato e attività azienda usando TF-IDF italiano.

**Parametri**:
- `exclusions` (str): Limitazioni lavorative candidato separate da virgola
- `company_text` (str): Descrizione attività/compatibilità azienda

**Restituisce**:
- `float`: Punteggio compatibilità [0.0-1.0], dove 1.0 = compatibilità perfetta

**Algoritmo**:
- Utilizza stop words italiane e tokenizzazione caratteri accentati
- Applica vettorizzazione TF-IDF con similarità coseno
- Scoring pesato: 70% similarità max + 30% similarità media
- Restituisce 1.0 - punteggio_similarità per interpretazione compatibilità

**Esempio**:
```python
exclusions = "Turni notturni, Lavori in quota"
company_text = "Lavoro d'ufficio con orario diurno flessibile"
score = scoring_system.compatibility_score(exclusions, company_text)
# Restituisce: 0.95 (alta compatibilità)
```

##### `generate_enhanced_training_data(df_candidates: pd.DataFrame, df_companies: pd.DataFrame) -> pd.DataFrame`
Genera dataset training sintetico usando regole matching probabilistiche.

**Parametri**:
- `df_candidates` (DataFrame): Dataset candidati esteso
- `df_companies` (DataFrame): Dataset aziende esteso

**Restituisce**:
- `DataFrame`: Dataset training con feature e outcome probabilistici

**Processo**:
1. Geocodifica tutti gli indirizzi se coordinate mancanti
2. Per ogni coppia candidato-azienda:
   - Calcola fattori compatibilità, distanza, attitudine
   - Applica formula scoring pesata
   - Genera outcome probabilistico con casualità controllata
3. Crea feature codificate one-hot per variabili categoriche

**Esempio**:
```python
df_train = scoring_system.generate_enhanced_training_data(df_candidates, df_companies)
# Restituisce: DataFrame con ~500K righe, 50+ feature
```

---

### `ParallelModelTrainer`

**Posizione**: `utils/parallel_training.py`

**Scopo**: Training ML ad alte prestazioni con elaborazione parallela e ottimizzazione iperparametri.

#### Definizione Classe
```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42)
```

**Attributi**:
- `random_state`: Seed riproducibilità
- `monitor`: Istanza monitoraggio risorse sistema
- `optimizer`: Istanza ottimizzazione iperparametri

#### Metodi

##### `parallel_hyperparameter_optimization(X: np.ndarray, y: np.ndarray) -> Dict`
Ottimizza iperparametri per multiple famiglie modelli in parallelo usando Optuna.

**Parametri**:
- `X` (ndarray): Feature training
- `y` (ndarray): Label training

**Restituisce**:
- `Dict`: Parametri migliori per ogni famiglia modelli

**Implementazione**:
- Usa ThreadPoolExecutor con 3 worker concorrenti
- Ottimizza RandomForest, XGBoost e LightGBM simultaneamente
- 50 trial per modello con algoritmo TPE
- Cross-validation 3-fold per valutazione robusta

**Esempio**:
```python
trainer = ParallelModelTrainer()
best_params = trainer.parallel_hyperparameter_optimization(X_train, y_train)
# Restituisce: {'random_forest': {...}, 'xgboost': {...}, 'lightgbm': {...}}
```

##### `create_optimized_models(best_params: Dict) -> List[Dict]`
Crea configurazioni modelli con iperparametri ottimizzati.

**Parametri**:
- `best_params` (Dict): Parametri ottimizzati da Optuna

**Restituisce**:
- `List[Dict]`: Configurazioni modelli pronte per training

**Modelli Creati**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized
- ExtraTrees
- GradientBoosting
- HistGradientBoosting
- MLP_Optimized

##### `parallel_model_training(model_configs: List, X_train, y_train, X_test, y_test) -> Dict`
Addestra modelli multipli concorrentemente con calibrazione probabilità.

**Parametri**:
- `model_configs` (List): Configurazioni modelli da create_optimized_models
- `X_train, y_train`: Dati training
- `X_test, y_test`: Dati testing

**Restituisce**:
- `Dict`: Risultati training con modelli, metriche e metadati

**Processo**:
1. Addestra fino a 6 modelli concorrentemente usando ThreadPoolExecutor
2. Applica CalibratedClassifierCV per calibrazione probabilità
3. Calcola metriche comprehensive (accuracy, precision, recall, F1, ROC-AUC)
4. Monitora risorse sistema durante training

**Esempio**:
```python
results = trainer.parallel_model_training(model_configs, X_train, y_train, X_test, y_test)
# Restituisce: {'LightGBM_Optimized': {'model': ..., 'metrics': {...}, 'status': 'success'}}
```

##### `create_ensemble_model(results: Dict, X_train, y_train) -> VotingClassifier`
Crea modello ensemble da modelli individuali addestrati con successo.

**Parametri**:
- `results` (Dict): Risultati da parallel_model_training
- `X_train, y_train`: Dati training per fitting ensemble

**Restituisce**:
- `VotingClassifier`: Modello ensemble calibrato con soft voting

##### `save_models(results: Dict, ensemble_model, save_dir: str = 'results')`
Salva tutti i modelli addestrati e metriche su disco.

**Parametri**:
- `results` (Dict): Risultati training
- `ensemble_model`: Istanza modello ensemble
- `save_dir` (str): Percorso directory output

**Output**:
- File modelli individuali: `{nome_modello}.joblib`
- Modello ensemble: `ensemble_model.joblib`
- Sommario metriche: `metrics_summary.csv`

---

### `JobMatchingDemo`

**Posizione**: `streamlit_app.py`

**Scopo**: Interfaccia Streamlit produzione con capacità matching real-time.

#### Definizione Classe
```python
class JobMatchingDemo:
    def __init__(self)
```

**Processo Inizializzazione**:
1. Configura geocodifica con focus italiano
2. Carica dati candidati e aziende (reali o demo)
3. Carica modelli ML addestrati se disponibili
4. Inizializza sistemi caching

#### Metodi

##### `load_data()`
Rileva automaticamente e carica dataset appropriato (reale vs demo).

**Logica**:
- Controlla file dati reali in `data/processed/`
- Ricade su generazione dati demo se dati reali non disponibili
- Gestisce parsing coordinate per indirizzi italiani
- Crea dati demo con distribuzione geografica italiana realistica

##### `find_matches(candidate_data: Dict, top_k: int = 5, distance_threshold: int = 30) -> List[Dict]`
Funzione matching real-time principale con filtri intelligenti.

**Parametri**:
- `candidate_data` (Dict): Informazioni candidato
- `top_k` (int): Numero top raccomandazioni da restituire
- `distance_threshold` (int): Distanza massima in chilometri

**Restituisce**:
- `List[Dict]`: Raccomandazioni aziende ordinate

**Processo**:
1. **Filtri Globali**: Validazione soglia attitudine
2. **Filtri Geografici**: Filtri aziende basati su distanza  
3. **Analisi Compatibilità**: Matching semantico TF-IDF
4. **Scoring Multi-fattore**: Combinazione pesata fattori
5. **Ranking**: Ordina per punteggio finale, restituisce top K

**Formula Scoring**:
```python
punteggio_finale = (
    0.35 × punteggio_compatibilità +
    0.25 × fattore_distanza + 
    0.20 × punteggio_attitudine +
    0.10 × retention_rate +
    0.05 × bonus_esperienza +
    0.05 × bonus_aziendali
)
```

**Esempio**:
```python
demo = JobMatchingDemo()
candidato = {
    'Area di Residenza': 'Verona, Italy',
    'Score Attitudine al Collocamento': 0.75,
    'Esclusioni': 'Turni notturni'
}
matches = demo.find_matches(candidato, top_k=5, distance_threshold=30)
# Restituisce: [{'Nome Azienda': 'Azienda_001', 'Score Finale': 89.5, ...}, ...]
```

---

## 🔧 Funzioni Utilità

### Feature Engineering (`utils/feature_engineering.py`)

##### `extend_candidates_dataset(df: pd.DataFrame) -> pd.DataFrame`
Migliora dataset candidati con feature ingegnerizzate.

**Miglioramenti**:
- **Simulazione disabilità**: Distribuzione realistica tipi disabilità italiani
- **Storia lavorativa**: Calcolo anni esperienza da primo impiego
- **Durata disoccupazione**: Calcoli basati su tempo
- **Mapping educazione**: Livelli educazione italiani con correlazioni disabilità

**Esempio**:
```python
from utils.feature_engineering import extend_candidates_dataset
df_esteso = extend_candidates_dataset(df_candidati)
# Aggiunge: Years_of_Experience, Durata Disoccupazione, Tipo di Disabilità migliorato
```

##### `extend_companies_dataset(df: pd.DataFrame) -> pd.DataFrame`
Migliora dataset aziende con feature business intelligence.

**Miglioramenti**:
- **Categorizzazione dimensione azienda**: Piccola/media/grande basata su numero dipendenti
- **Flag certificazioni**: Simulazione certificazione disability-friendly
- **Indicatori lavoro remoto**: Flag accordi lavoro moderni
- **Calcolo retention rate**: Metriche successo da dati storici

### Visualizzazione (`utils/visualization.py`)

##### `visualize_distribution(df: pd.DataFrame)`
Genera grafici distribuzione per analisi dataset.

**Grafici Creati**:
- Distribuzione punteggio attitudine
- Distribuzione punteggio compatibilità  
- Distribuzione distanza
- Bilanciamento outcome

##### `visualize_correlations(df: pd.DataFrame)`
Crea heatmap correlazione per analisi feature.

---

## 📊 Pipeline Preprocessing Dati

### `prepare_data_for_training(df_train: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict`

**Posizione**: `utils/parallel_training.py`

**Scopo**: Preprocessing dati comprensivo per training ML.

**Parametri**:
- `df_train` (DataFrame): Dataset training grezzo
- `test_size` (float): Proporzione set test
- `random_state` (int): Seed riproducibilità

**Restituisce**:
- `Dict`: Dati processati pronti per training ML

**Passi Pipeline**:
1. **Estrazione Target**: Separa variabile outcome
2. **Gestione Valori Mancanti**: Imputazione mediana per feature numeriche
3. **Split Train-Test**: Split stratificato preservando bilanciamento classi
4. **Scaling Robusto**: RobustScaler per resistenza outlier
5. **Selezione Feature**: SelectKBest con F-statistic (top 50 feature)
6. **Bilanciamento Classi**: Oversampling SMOTE per classe minoritaria

**Struttura Output**:
```python
{
    "X_train": np.ndarray,     # Feature training bilanciate
    "y_train": np.ndarray,     # Label training bilanciate  
    "X_test": np.ndarray,      # Feature test
    "y_test": np.ndarray,      # Label test
    "scaler": RobustScaler,    # Scaler fittato per inferenza
    "selector": SelectKBest    # Selector fittato per inferenza
}
```

---

## 🔍 Sistema Configurazione

### Caricamento Configurazione
```python
import yaml

def load_config(config_path: str = 'config.yaml') -> Dict:
    """Carica configurazione sistema da file YAML"""
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
```

### Struttura Configurazione Default
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
  user_agent: str             # Stringa user agent personalizzata
  cache_file: str             # Percorso file cache

italian_language:
  stop_words: List[str]       # Stop words italiane
  token_pattern: str          # Regex per token italiani
```

---

## 📈 Monitoraggio Performance

### `SystemResourceMonitor`

**Posizione**: `utils/parallel_training.py`

**Scopo**: Tracking real-time risorse sistema durante operazioni ML.

#### Metodi

##### `start()`
Inizia monitoraggio background uso CPU e memoria.

##### `stop()`
Ferma monitoraggio e finalizza statistiche.

##### `stats() -> Dict`
Restituisce statistiche utilizzo risorse medie.

**Esempio**:
```python
monitor = SystemResourceMonitor()
monitor.start()
# ... eseguire training ML ...
monitor.stop()
stats = monitor.stats()
# Restituisce: {'avg_cpu': 75.2, 'avg_mem': 68.5}
```

---

## 🔒 Gestione Errori e Validazione

### Pattern Validazione Input

```python
def validate_candidate_data(candidate: Dict) -> bool:
    """Valida struttura e valori dati candidato"""
    required_fields = [
        'Area di Residenza', 'Score Attitudine al Collocamento',
        'Years_of_Experience', 'Durata Disoccupazione'
    ]
    
    # Controlla campi richiesti
    if not all(field in candidate for field in required_fields):
        return False
    
    # Valida range
    if not 0.0 <= candidate['Score Attitudine al Collocamento'] <= 1.0:
        return False
    
    if candidate['Years_of_Experience'] < 0:
        return False
    
    return True
```

### Gestione Eccezioni

```python
try:
    result = scoring_system.compatibility_score(exclusions, company_text)
except Exception as e:
    logger.error(f"Scoring compatibilità fallito: {e}")
    result = 0.5  # Punteggio neutrale default
```

---

## 🚀 Esempi Utilizzo

### Pipeline Training Completa
```python
# 1. Caricare ed estendere dataset
from utils.feature_engineering import extend_candidates_dataset, extend_companies_dataset
from utils.scoring import EnhancedScoringSystem
from utils.parallel_training import ParallelModelTrainer, prepare_data_for_training

# Caricare dati grezzi
df_candidati = pd.read_csv('data/raw/Dataset_Candidati_Aggiornato.csv')
df_aziende = pd.read_csv('data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv')

# Estendere con feature
df_cand_ext = extend_candidates_dataset(df_candidati)
df_comp_ext = extend_companies_dataset(df_aziende)

# Generare dati training
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_comp_ext)

# Preparare per training ML
data = prepare_data_for_training(df_train)

# Addestrare modelli
trainer = ParallelModelTrainer()
best_params = trainer.parallel_hyperparameter_optimization(data['X_train'], data['y_train'])
model_configs = trainer.create_optimized_models(best_params)
results = trainer.parallel_model_training(model_configs, data['X_train'], data['y_train'], 
                                         data['X_test'], data['y_test'])

# Salvare modelli
ensemble = trainer.create_ensemble_model(results, data['X_train'], data['y_train'])
trainer.save_models(results, ensemble)
```

### Matching Real-time
```python
# Inizializzare sistema matching
demo = JobMatchingDemo()

# Definire candidato
candidato = {
    'Area di Residenza': 'Sommacampagna, Verona, Italy',
    'Score Attitudine al Collocamento': 0.80,
    'Years_of_Experience': 5,
    'Durata Disoccupazione': 12,
    'Titolo di Studio': 'Diploma',
    'Tipo di Disabilità': 'Motoria',
    'Esclusioni': 'Lavori in quota'
}

# Trovare match
matches = demo.find_matches(candidato, top_k=5, distance_threshold=30)

# Processare risultati
for i, match in enumerate(matches, 1):
    print(f"{i}. {match['Nome Azienda']}: {match['Score Finale']:.1f}% "
          f"({match['Distanza (km)']} km)")
```

---

## 📞 Supporto ed Estensione

### Aggiungere Nuove Feature

**Per aggiungere nuovo metodo scoring compatibilità**:
```python
class EnhancedScoringSystem:
    def nuovo_metodo_compatibilità(self, exclusions: str, company_text: str) -> float:
        # Implementare nuova logica
        return score
    
    def compatibility_score(self, exclusions: str, company_text: str) -> float:
        # Scegliere metodo basato su configurazione
        if self.config.get('usa_nuovo_metodo'):
            return self.nuovo_metodo_compatibilità(exclusions, company_text)
        else:
            return self.compatibility_score_originale(exclusions, company_text)
```

**Per aggiungere nuovo modello ML**:
```python
def create_optimized_models(self, best_params: Dict) -> List[Dict]:
    models = [...]  # modelli esistenti
    
    # Aggiungere nuovo modello
    models.append({
        'name': 'NuovoModello_Optimized',
        'class': NuovaClasseModello,
        'params': {**best_params.get('nuovo_modello', {}), 'random_state': self.random_state}
    })
    
    return models
```

### Integrazione Personalizzata

Per integrazioni specifiche organizzazione, estendere classi base:

```python
class CustomJobMatcher(JobMatchingDemo):
    def __init__(self, organization_config):
        super().__init__()
        self.org_config = organization_config
    
    def find_matches(self, candidate_data, **kwargs):
        # Applicare regole specifiche organizzazione
        base_matches = super().find_matches(candidate_data, **kwargs)
        return self.apply_org_filters(base_matches)
    
    def apply_org_filters(self, matches):
        # Logica filtri personalizzata
        return matches_filtrati
```

---

## 🎯 Integrazione Specifica CPI/SIL

### Estensioni per Legge 68/99

```python
class LawCompliantMatcher(JobMatchingDemo):
    def check_legge68_compliance(self, candidate_data: Dict, company_data: Dict) -> Tuple[bool, Dict]:
        """Verifica conformità Legge 68/99"""
        checks = {
            'invalidità_sufficiente': candidate_data.get('Percentuale_Invalidità', 0) >= 46,
            'azienda_obbligata': company_data['Numero Dipendenti'] >= 15,
            'posizioni_disponibili': company_data['Posizioni Aperte'] > 0,
            'quota_rispettata': self._check_quota_compliance(company_data)
        }
        return all(checks.values()), checks
    
    def _check_quota_compliance(self, company_data: Dict) -> bool:
        """Controlla rispetto quota obbligatoria"""
        dipendenti = company_data['Numero Dipendenti']
        if dipendenti < 15:
            return True  # Non obbligata
        elif 15 <= dipendenti <= 35:
            quota_richiesta = 1
        else:
            quota_richiesta = max(1, int(dipendenti * 0.07))  # 7%
        
        quota_attuale = company_data.get('Dipendenti_Categoria_Protetta', 0)
        return quota_attuale >= quota_richiesta
```

### Export Format CPI Standard

```python
def export_to_cpi_format(matches: List[Dict], candidate_id: str) -> pd.DataFrame:
    """Esporta risultati in formato standard CPI"""
    export_data = []
    for match in matches:
        export_data.append({
            'ID_Candidato': candidate_id,
            'Codice_Azienda': match.get('Codice_Azienda', ''),
            'Nome_Azienda': match['Nome Azienda'],
            'Percentuale_Compatibilità': match['Score Finale'],
            'Distanza_KM': match['Distanza (km)'],
            'Settore_Attività': match['Tipo di Attività'],
            'Posizioni_Disponibili': match['Posizioni Aperte'],
            'Data_Raccomandazione': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Algoritmo_Versione': '1.0'
        })
    
    return pd.DataFrame(export_data)
```

---

*Questo riferimento API fornisce documentazione comprensiva per integrare con ed estendere il Sistema di Raccomandazione per Collocamento Mirato. Per domande implementazione specifiche o necessità sviluppo personalizzato, contattare il team di sviluppo.*

---

**Versione Documento**: 1.0  
**Ultimo Aggiornamento**: Giugno 2025  
**Audience Target**: Sviluppatori, Ricercatori, Specialisti Integrazione
