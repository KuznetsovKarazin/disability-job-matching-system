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