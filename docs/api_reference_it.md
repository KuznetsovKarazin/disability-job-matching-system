# Riferimento API — Sistema di Matching Lavorativo per Disabili
_Ultimo aggiornamento: 2025-08-24 15:30_

<p align="center">
  <img src="https://img.shields.io/badge/status-attivo-success" />
  <img src="https://img.shields.io/badge/lingua-IT-blue" />
  <img src="https://img.shields.io/badge/federated-learning-purple" />
  <img src="https://img.shields.io/badge/privacy-DP%20%2B%20Shamir-teal" />
  <img src="https://img.shields.io/badge/blockchain-anchoring-orange" />
</p>

Questo documento definisce la **superficie API pubblica** del Sistema di Matching Lavorativo per Disabili.  
Copre schema di configurazione, funzioni e classi a livello di modulo, punti di ingresso CLI, contratti dati e comportamenti operativi.

**Ambito e Convenzioni**
- Python ≥ 3.8. I tipi statici sono descrittivi; l'applicazione runtime è best-effort.
- I percorsi dei file sono relativi alla root del repository salvo diversa indicazione.
- Tutti gli esempi assumono un ambiente virtuale attivato e un `config.yaml` popolato.
- L'interfaccia Streamlit utilizza queste API indirettamente; questo riferimento è destinato a sviluppatori e integratori.

## Indice
1. [Versioning e Stabilità](#versioning-e-stabilità)
2. [Schema di Configurazione](#schema-di-configurazione)
3. [Contratti Dati](#contratti-dati)
4. [Moduli](#moduli)
5. [CLI e Script](#cli-e-script)
6. [Integrazione Streamlit](#integrazione-streamlit)
7. [Errori ed Eccezioni](#errori-ed-eccezioni)
8. [Note sulle Performance](#note-sulle-performance)
9. [Note su Sicurezza e Privacy](#note-su-sicurezza-e-privacy)
10. [Compatibilità all'Indietro](#compatibilità-allindietro)

## Versioning e Stabilità

- **Intento semantico**: I cambiamenti che rompono le firme delle funzioni, la semantica dei parametri o i tipi di ritorno saranno annotati nel changelog.
- **API pubblica**: Solo gli elementi documentati qui sono considerati stabili. Gli helper non documentati possono cambiare senza preavviso.
- **Schema config**: Sono permesse aggiunte retrocompatibili (nuove chiavi opzionali); rimozioni o cambiamenti di tipo sono breaking.

## Schema di Configurazione

Chiavi e tipi di `config.yaml`. I valori mostrati sono default raccomandati.

```yaml
seed: 42                          # int — seed PRNG globale
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies:  data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir:  data/processed
  training_csv:   data/processed/Enhanced_Training_Dataset.csv
  results_dir:    results
ui:
  distance_max_km: 30             # int — raggio di ricerca predefinito nell'UI
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

**Note**
- I percorsi devono esistere prima dell'esecuzione; i writer creeranno directory intermedie quando possibile.
- `distance_max_km` è **30** di default.
- I parametri DP definiscono il budget privacy per esperimento; gli output di log dovrebbero includere ε, δ finali.

## Contratti Dati

### Input Grezzi
- **Candidati** — `Dataset_Candidati_Aggiornato.csv`
- **Aziende**  — `Dataset_Aziende_con_Stima_Assunzioni.csv`

Le colonne sono specifiche dell'istituzione; mantenere nomenclatura coerente tra regioni e versioni.

### Artifact Processati
- **Candidati Estesi** — `Dataset_Candidati_Aggiornato_Extended.csv`
- **Aziende Estese**  — `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv`
- **Tabella Training (canonica)** — `Enhanced_Training_Dataset.csv`

**Sommario Modelli Unificati**
- `results/merged_model_summary.csv` — metriche consolidate tra esecuzioni/modelli.

## Moduli
Classi e funzioni pubbliche esportate da ogni modulo. Le funzioni helper non elencate sono interne.

### utils.feature_engineering

**Scopo**: Preparazione dati, estensione e generazione tabella training canonica.

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

**Scopo**: Punteggio di matching e utilità.

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

**Scopo**: Training multi-threaded e orchestrazione modelli.

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

**Scopo**: Orchestrazione federata con aggregatori robusti.

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
        """Esegue i round FL; restituisce sommario metriche."""

    def evaluate_global(self) -> dict:
        """Valuta il modello aggregato su un set di holdout."""

    def save(self, out_dir: str) -> None:
        """Salva modello globale e metadati."""
```

### utils.enhanced_shamir_privacy

**Scopo**: Aggregazione sicura (Shamir) e helper Differential Privacy.

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
    """Applica maschere pairwise derivate da secret shares."""

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

**Scopo**: Commitment Merkle per risultati e modelli; utilità proof.

```python
class MerkleTree:
    def __init__(self, leaves: list[bytes]): ...
    def root(self) -> bytes
    def proof(self, index: int) -> list[tuple[bytes, str]]  # (sibling, 'L'|'R')

def build_manifest(paths: list[str]) -> dict
def commit_manifest(manifest: dict) -> tuple[bytes, "MerkleTree"]
def verify_inclusion(root: bytes, leaf: bytes, proof: list[tuple[bytes, str]]) -> bool
```

## CLI e Script

Ogni script espone una CLI basata su argparse. Esegui con `-h` per ispezionare tutte le opzioni.

```bash
# Estensione dati e visualizzazione
python scripts/01_generate_dataset.py --config config.yaml
python scripts/02_visualize_dataset.py --config config.yaml

# Training centralizzato e analisi
python scripts/03_train_models.py --config config.yaml --models LightGBM_Optimized MLP
python scripts/04_analyze_results.py --input results/merged_model_summary.csv

# Pipeline federate
python scripts/05_LightGBM_federated_training.py --rounds 10 --aggregator fedavg
python scripts/07_mlp_federated_training.py --rounds 10 --aggregator trimmed_mean
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6

# Anchoring e benchmark
python scripts/10_blockchain_anchoring_bench.py --scale 10000
```

**Codici di uscita**
- `0` — successo; artifact scritti.
- `1` — configurazione errata (path/schema/args).
- `2` — errore runtime (training, round FL, o fallimento plotting).

## Integrazione Streamlit

L'UI è un layer sottile sopra l'API:
- Carica configurazione e dataset.
- Chiama funzioni scoring per presentare match classificati.
- Opzionalmente visualizza risultati FL vs centralizzati e budget privacy.

**Endpoint health**: `/_stcore/health` (assicurati che `curl` sia in Docker per health check).

## Errori ed Eccezioni

Tipi di eccezione comuni esposti dall'API pubblica:
- `FileNotFoundError` — percorsi mancanti in `paths.*`.
- `ValueError` — schema invalido, dataset vuoti, o limiti parametri.
- `RuntimeError` — errori loop training/FL, inconsistenze accountant DP.
- `AssertionError` — check invarianti interni (es. allineamento shape).

## Note sulle Performance

- Usa `trimmed_mean` o `coordinate_median` per robustezza con client volatili.
- `batch_size` più piccolo migliora stabilità su nodi CPU-only.
- Cache dataset estesi sotto `data/processed/` per ridurre tempo preprocessing.
- **Realtà Performance**: L'apprendimento federato mostra una leggera degradazione delle performance (-0.06% F1 per LightGBM, -4% per MLP) rispetto al training centralizzato.

## Note su Sicurezza e Privacy

- Nessun dato grezzo lascia i nodi regionali sotto FL; solo i delta dei modelli vengono scambiati.
- L'aggregazione sicura maschera gli aggiornamenti per parametro; DP assicura perdita limitata.
- L'anchoring fornisce integrità e non-ripudio per artifact pubblicati.
- **Costo Privacy**: Impatto minimo sulle performance con parametri DP pratici (ε=1.0, δ=1e-06).

## Compatibilità all'Indietro

- Nuove chiavi config opzionali sono retrocompatibili.
- Le opzioni deprecate rimangono per almeno un ciclo minor con warning.

---

## Classi Principali (Riferimento Legacy)

### `EnhancedScoringSystem`

**Posizione**: `utils/scoring.py`

**Scopo**: Implementa l'algoritmo core di matching candidato-azienda con supporto linguaggio italiano e processamento geografico.

```python
class EnhancedScoringSystem:
    def __init__(self): ...
    
    def geocode_with_cache(self, address: str) -> tuple[float, float]:
        """Geocodifica indirizzi italiani con caching."""
    
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcola distanza geografica precisa usando formula Haversine."""
    
    def compatibility_score(self, exclusions: str, company_text: str) -> float:
        """Analizza compatibilità semantica usando TF-IDF italiano."""
    
    def generate_enhanced_training_data(self, df_candidates: "pd.DataFrame", df_companies: "pd.DataFrame") -> "pd.DataFrame":
        """Genera dataset training sintetico usando regole matching probabilistiche."""
```

**Formula Scoring**:
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

**Posizione**: `utils/parallel_training.py`

**Scopo**: Training ML ad alte performance con processamento parallelo e ottimizzazione iperparametri.

```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42): ...
    
    def parallel_hyperparameter_optimization(self, X: "np.ndarray", y: "np.ndarray") -> dict:
        """Ottimizza iperparametri per multiple famiglie di modelli usando Optuna."""
    
    def parallel_model_training(self, model_configs: list, X_train, y_train, X_test, y_test) -> dict:
        """Addestra modelli multipli concorrentemente con calibrazione probabilità."""
    
    def create_ensemble_model(self, results: dict, X_train, y_train) -> "VotingClassifier":
        """Crea modello ensemble da modelli individuali addestrati con successo."""
```

**Modelli Addestrati**:
- RandomForest_Optimized
- XGBoost_Optimized  
- LightGBM_Optimized (Migliore: F1=0.901)
- HistGradientBoosting
- GradientBoosting
- MLP_Optimized
- ExtraTrees

### `JobMatchingDemo`

**Posizione**: `streamlit_app.py`

**Scopo**: Interfaccia Streamlit di produzione con capacità matching real-time.

```python
class JobMatchingDemo:
    def __init__(self): ...
    
    def load_data(self):
        """Rileva automaticamente e carica dataset appropriato (reale vs demo)."""
    
    def find_matches(self, candidate_data: dict, top_k: int = 5, distance_threshold: int = 30) -> list[dict]:
        """Funzione core matching real-time con filtraggio intelligente."""
```

## Pipeline Preprocessing Dati

```python
def prepare_data_for_training(df_train: "pd.DataFrame", test_size: float = 0.2, random_state: int = 42) -> dict:
    """Preprocessing dati completo per training ML."""
```

**Passi Pipeline**:
1. **Estrazione Target**: Separa variabile outcome
2. **Gestione Valori Mancanti**: Imputazione mediana per feature numeriche
3. **Split Train-Test**: Split stratificato preservando bilanciamento classi
4. **Scaling Robusto**: RobustScaler per resistenza outlier
5. **Selezione Feature**: SelectKBest con F-statistic (top 50 feature)
6. **Bilanciamento Classi**: Oversampling SMOTE per classe minoritaria

## Risultati Performance

**Apprendimento Centralizzato (Baseline)**
- **Modello Migliore**: LightGBM_Optimized (F1=0.901, ROC-AUC=0.708)

**Apprendimento Federato LightGBM**
- **Centralizzato**: F1 ≈ 0.9012, ROC-AUC ≈ 0.716
- **Federato**: F1 ≈ 0.9007, ROC-AUC ≈ 0.687
- **Gap Performance**: -0.0005 F1-score

**Apprendimento Federato MLP**
- **Centralizzato**: F1 ≈ 0.828, Accuracy ≈ 0.735
- **Federato**: F1 ≈ 0.788, Accuracy ≈ 0.695
- **Privacy-Preserving**: Costo aggiuntivo minimo con ε=1.0, δ=1e-06

**Anchoring Blockchain**
- **Tempi build**: 100 record = 2.28s; 1k = 30.47s; 10k = 344.07s
- **Generazione proof**: 1.11ms - 20.65ms (scaling O(log n))
- **Verifica**: Media 24.49ms con 100% accuratezza

## Esempi d'Uso

### Pipeline Training Completa
```python
from utils.feature_engineering import extend_candidates_dataset, extend_companies_dataset
from utils.scoring import EnhancedScoringSystem
from utils.parallel_training import ParallelModelTrainer

# Carica ed estende dati
df_candidates = pd.read_csv('data/raw/Dataset_Candidati_Aggiornato.csv')
df_companies = pd.read_csv('data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv')

df_cand_ext = extend_candidates_dataset(df_candidates)
df_comp_ext = extend_companies_dataset(df_companies)

# Genera dati training
scoring_system = EnhancedScoringSystem()
df_train = scoring_system.generate_enhanced_training_data(df_cand_ext, df_comp_ext)

# Addestra modelli
trainer = ParallelModelTrainer()
results = trainer.parallel_model_training(configs, X_train, y_train, X_test, y_test)
```

### Matching Real-time
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

