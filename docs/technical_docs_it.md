# Documentazione Tecnica — Sistema di Matching per Collocamento Mirato
_Ultimo aggiornamento: 2025-08-24 15:30_

## Indice
- [1. Introduzione](#1-introduzione)
- [2. Architettura](#2-architettura)
- [3. Struttura Progetto](#3-struttura-progetto)
- [4. Contratti Dati e Schemi](#4-contratti-dati-e-schemi)
- [5. Feature Engineering](#5-feature-engineering)
- [6. Modellazione](#6-modellazione)
- [7. Apprendimento Federato](#7-apprendimento-federato)
- [8. Privacy e Sicurezza](#8-privacy-e-sicurezza)
- [9. Ancoraggio Blockchain](#9-ancoraggio-blockchain)
- [10. Risultati e Benchmark](#10-risultati-e-benchmark)
- [11. Superficie API per Modulo](#11-superficie-api-per-modulo)
- [12. CLI e Configurazione](#12-cli-e-configurazione)
- [13. Deployment e Operazioni](#13-deployment-e-operazioni)
- [14. Osservabilità e Testing](#14-osservabilità-e-testing)
- [15. Ottimizzazione Prestazioni](#15-ottimizzazione-prestazioni)
- [16. Risoluzione Problemi](#16-risoluzione-problemi)
- [17. Conformità (GDPR)](#17-conformità-gdpr)
- [18. Roadmap](#18-roadmap)

## 1. Introduzione
Questo documento è rivolto a sviluppatori e operatori tecnici. Descrive in dettaglio il design interno, i modelli di dati, le pipeline di training (centralizzato e federato), i meccanismi di privacy (Shamir + DP), e il sottosistema di ancoraggio blockchain. Il contenuto riflette il **layout reale del progetto** e le metriche di performance corrette basate sui risultati sperimentali.

## 2. Architettura
Adottiamo un'architettura a strati:
- **Livello Dati**: acquisizione grezza → estensione → tabelle training processate.
- **Livello Apprendimento**: baseline LightGBM centralizzato e FedAvg basato su MLP con aggregatori robusti.
- **Livello Privacy**: aggregazione sicura via Shamir (soglia, recupero dropout) e Privacy Differenziale via RDP.
- **Livello Integrità**: ancoraggio basato su Merkle con prove di inclusione.
- **UI/Ops**: UI Streamlit, script, deployment Docker, controlli salute.

```mermaid
flowchart TB
  R[CSV Grezzi] --> E["Estensione e Pulizia"]
  E --> P["Tabella Training Processata"]
  P --> CB["Baseline Centralizzato"]
  P --> FL["Apprendimento Federato"]
  FL --> SA["Aggregazione Sicura (Shamir)"]
  SA --> DP["Contabilità DP (RDP)"]
  CB --> A["Artefatti"]
  DP --> A
  A --> M["Commit Merkle"]
  M --> Z["Prove Inclusione"]
  Z --> V["Verifica"]
  A --> UI["UI Streamlit"]
```

## 3. Struttura Progetto
```
📂 Sistema Matching Collocamento Mirato/
├── 📄 README.md                          # Documentazione principale (Inglese)
├── 📄 README_IT.md                       # Versione italiana
├── ⚙️ config.yaml                        # Configurazione sistema
├── 📄 requirements.txt                   # Dipendenze Python
├── 🎯 streamlit_app.py                   # APPLICAZIONE PRINCIPALE
│
├── 📂 data/
│   ├── 📂 raw/                          # Dati input originali
│   │   ├── Dataset_Candidati_Aggiornato.csv
│   │   └── Dataset_Aziende_con_Stima_Assunzioni.csv
│   └── 📂 processed/                    # Dataset generati/estesi
│       ├── Dataset_Candidati_Aggiornato_Extended.csv
│       ├── Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
│       └── Enhanced_Training_Dataset.csv  # DATI TRAINING ML
│
├── 📂 scripts/                          # Pipeline elaborazione dati
│   ├── 01_generate_dataset.py          # Estensione dati + generazione training sintetico
│   ├── 02_visualize_dataset.py         # Analisi dati e visualizzazione
│   ├── 03_train_models.py              # TRAINING MODELLI ML
│   ├── 04_analyze_results.py           # Analisi prestazioni e reporting
│   ├── 05_LightGBM_federated_training.py # PIPELINE APPRENDIMENTO FEDERATO (LightGBM)
│   ├── 06_LightGBM_federated_visualization.py # VISUALIZZAZIONE RISULTATI FEDERATI
│   ├── 07_mlp_federated_training.py    # APPRENDIMENTO FEDERATO CLASSICO
│   ├── 08_mlp_federated_privacy.py     # APPRENDIMENTO FEDERATO CHE PRESERVA PRIVACY  
│   ├── 09_mlp_federated_privacy_visualization.py # CONFRONTO RISULTATI FEDERATI
│   ├── blockchain_data_anchoring.py    # PIPELINE INTEGRITÀ DATI BLOCKCHAIN
│   └── 10_blockchain_anchoring_bench.py # BENCHMARK PRESTAZIONI BLOCKCHAIN
│
├── 📂 utils/                            # Logica business core
│   ├── __init__.py
│   ├── feature_engineering.py          # Utilità augmentation dati
│   ├── scoring.py                       # CORE ALGORITMO MATCHING
│   ├── parallel_training.py            # TRAINING ML MULTI-THREAD
│   ├── visualization.py                # Utilità generazione grafici
│   └── enhanced_shamir_privacy.py      # CONDIVISIONE SEGRETI SHAMIR + DIFFERENZIALE
│
├── 📂 results/                          # Output training
│   ├── 📂 learning_curves/             # Grafici progressione training
│   ├── *.joblib                        # Modelli ML addestrati (7 modelli)
│   ├── merged_model_summary.csv        # Metriche prestazioni
│   └── *.png                           # Visualizzazioni analisi
│
├── 📂 results_LightGBM_federated/      # Risultati apprendimento federato (LightGBM)
│   ├── 📂 regional_models/             # Modelli regionali individuali  
│   ├── 📂 federated_models/            # Modelli globali aggregati
│   ├── 📂 centralized_models/          # Modelli centralizzati baseline
│   ├── 📂 visualizations/              # Grafici e diagrammi analisi
│   ├── complete_model_comparison.csv   # Confronto prestazioni a tre vie
│   └── experiment_metadata.json        # Metadata sperimentali completi
│
├── 📂 results_mlp_federated/           # Risultati apprendimento federato classico
├── 📂 results_mlp_federated_privacy/   # Risultati federato che preserva privacy  
├── 📂 results_blockchain_demo/         # Dimostrazioni ancoraggio blockchain
└── 📂 visualizations_federated_comparison/ # Grafici confronto apprendimento federato
```

## 4. Contratti Dati e Schemi
- **Input Grezzi**: `Dataset_Candidati_Aggiornato.csv`, `Dataset_Aziende_con_Stima_Assunzioni.csv`.
- **Processati**: varianti estese e `Enhanced_Training_Dataset.csv` canonico.
- **Linee Guida**:
  - Mantenere nomi colonna consistenti tra regioni.
  - Versionare cambi schema; preservare `SCHEMA.md` se possibile.
  - Memorizzare campi derivati (es. output geocoding) esplicitamente per riproducibilità.

### Dettagli Schema Dati
**Struttura Enhanced_Training_Dataset.csv:**
```
Colonne richieste:
- outcome: int [0, 1] (variabile target)
- attitude_score: float [0.0-1.0]
- years_experience: int
- unemployment_duration: int (mesi)
- compatibility_score: float [0.0-1.0]
- distance_km: float
- company_size: int
- retention_rate: float [0.0-1.0]
- remote_work: int [0, 1]
- certification: int [0, 1]
- edu_*: int (livelli istruzione one-hot encoded)
- dis_*: int (tipi disabilità one-hot encoded)
- sector_*: int (settori aziendali one-hot encoded)

Dimensioni tipiche: 500.000+ righe, 50+ feature
```

## 5. Feature Engineering
- Scaling robusto per feature numeriche.
- Codifiche categoriali; strategie valori mancanti.
- **SMOTE** opzionale applicato dopo split train/val per evitare leakage.
- Feature geospaziali: distanza entro raggio configurabile (**default 30 km**) per scoring prossimità candidato-lavoro.

### Elaborazione Linguaggio Italiano
- Vettorizzazione TF-IDF con stop words italiane
- Pattern token personalizzato per caratteri accentati italiani: `r'\b[a-zA-ZÀÈÉÌÒÙàèéìòù]+\b'`
- Scoring compatibilità semantica per esclusioni vs requisiti lavoro
- Geocoding ottimizzato per indirizzi italiani

## 6. Modellazione
- **LightGBM_Optimized**: baseline tabulare primario; veloce e interpretabile.
- **MLP**: usato in FL; compatibile con aggregatori robusti.
- **Calibrazione**: calibrazione probabilità (sigmoid/isotonic).
- **Metriche**: F1, ROC-AUC, accuratezza; matrici confusione per regione.
- **Artefatti**: modelli (`*.joblib`), riassunti (`merged_model_summary.csv`), visualizzazioni (`*.png`, `learning_curves/`).

### Prestazioni Modelli (Risultati Corretti)
**Modelli Centralizzati (7 modelli addestrati):**
- **Migliori Prestazioni**: LightGBM_Optimized - F1: 0.901, ROC-AUC: 0.708, Training: 94.6s
- **Secondo Migliore**: XGBoost_Optimized - F1: 0.901, ROC-AUC: 0.704, Training: 132.3s
- **Terzo**: HistGradientBoosting - F1: 0.900, ROC-AUC: 0.715, Training: 202.3s

**Analisi Prestazioni:**
- F1-Score prioritario per qualità ranking raccomandazioni
- ROC-AUC (~0.70) riflette design intentionally probabilistico dati sintetici
- Dati reali collocamento mostrerebbero probabilmente ROC-AUC più alto (0.80-0.90+)

## 7. Apprendimento Federato
- **Percorso LightGBM** (regionale → ensemble): modelli regionali indipendenti, ensemble pesato per conteggi campioni.
- **Percorso MLP** (FedAvg vero): aggiornamenti locali per-round; aggregatori: FedAvg, trimmed mean, coordinate median.
- **Riproducibilità**: semi deterministici; selezione client stabile; preprocessing consistente.

### Prestazioni Apprendimento Federato (Corrette)
**Risultati LightGBM Federato:**
- **Centralizzato**: F1 ≈ 0.9012, ROC-AUC ≈ 0.716
- **Regionale**: F1 ≈ 0.9001, ROC-AUC ≈ 0.702  
- **Federato**: F1 ≈ 0.9007, ROC-AUC ≈ 0.687
- **Gap Prestazioni**: Federato vs Centralizzato = -0.0005 F1-score (degradazione minima)

**Risultati MLP Federato:**
- **Centralizzato**: F1 ≈ 0.828, Accuratezza ≈ 0.735, ROC-AUC ≈ 0.695
- **Federato (Standard)**: F1 ≈ 0.788, Accuratezza ≈ 0.695, ROC-AUC ≈ 0.717
- **Federato (Preserva Privacy)**: F1 ≈ 0.788, Accuratezza ≈ 0.695, ROC-AUC ≈ 0.717
- **Gap Prestazioni**: ~4% riduzione F1-score con approccio federato

### Struttura Cartelle Risultati:
- `results_LightGBM_federated/` → `regional_models/`, `federated_models/`, `centralized_models/`, `visualizations/`, `complete_model_comparison.csv`, `experiment_metadata.json`.
- `results_mlp_federated/`, `results_mlp_federated_privacy/`.

## 8. Privacy e Sicurezza
- **Aggregazione Sicura (Shamir)**: soglia (es. 3-su-5), mascheramento per-parametro, recupero dropout.
- **Privacy Differenziale**: clipping gradienti e singolo rumore Gaussiano per round; contabilità RDP per budget ε/δ.
- **Controlli Operativi**: controllo accesso su `data/` e `results/`, manifesti audit, ritenzione minima.

### Risultati Preserva-Privacy
**Risultati Chiave:**
- Soglia Shamir: 3-su-5 con recupero dropout riuscito
- Budget DP: (ε=1.0, δ=1e-06) con impatto utilità minimo
- Prestazioni mantenute: F1 ≈ 0.788 (preserva-privacy) vs 0.828 (centralizzato)
- Costo privacy: ~4.8% riduzione F1-score per garanzie privacy forti

## 9. Ancoraggio Blockchain
- **Commitment Merkle**: albero deterministico su artefatti; prove inclusione O(log n).
- **Utilità**: `blockchain_data_anchoring.py` (commit/proof/verify), `10_blockchain_anchoring_bench.py` (benchmark).
- **Casi d'Uso**: integrità per reporting pubblico, riproducibilità, verifica terze parti.

### Risultati Benchmarking
**Prestazioni Costruzione:**
- 100 record: 2.28s totale (KDF: 1.58s, Albero: 0.45s)
- 1.000 record: 30.47s totale (KDF: 21.15s, Albero: 6.04s)
- 10.000 record: 344.07s totale (KDF: 239.40s, Albero: 68.40s)

**Generazione Prove:** scaling O(log n)
- 100 record: 1.11ms media, 224 byte
- 1.000 record: 2.61ms media, 320 byte  
- 10.000 record: 20.65ms media, 448 byte

**Verifica:** Media 24.49ms con tasso successo 100%

## 10. Risultati e Benchmark
- Confronti centralizzato vs regionale vs federato consolidati in CSV e figure.
- Risultati FL preserva-privacy con ε/δ tracciati per run.
- Benchmark ancoraggio riportano tempi costruzione/prova/verifica.
- Tutte le metriche prestazioni corrette per corrispondere risultati sperimentali.

## 11. Superficie API per Modulo

### Classi Core

#### `EnhancedScoringSystem` (utils/scoring.py)
```python
class EnhancedScoringSystem:
    def __init__(self)
    def geocode_with_cache(self, address: str) -> Tuple[float, float]
    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float
    def compatibility_score(self, exclusions: str, company_text: str) -> float
    def generate_enhanced_training_data(self, df_cand: pd.DataFrame, df_az: pd.DataFrame) -> pd.DataFrame
```

#### `ParallelModelTrainer` (utils/parallel_training.py)
```python
class ParallelModelTrainer:
    def __init__(self, random_state: int = 42)
    def parallel_hyperparameter_optimization(self, X: np.ndarray, y: np.ndarray) -> Dict
    def create_optimized_models(self, best_params: Dict) -> List[Dict]
    def parallel_model_training(self, model_configs: List, X_train, y_train, X_test, y_test) -> Dict
    def create_ensemble_model(self, results: Dict, X_train, y_train) -> VotingClassifier
    def save_models(self, results: Dict, ensemble_model, save_dir: str = 'results')
```

### Classi Privacy e Sicurezza

#### `EnhancedShamirSecretSharing` (utils/enhanced_shamir_privacy.py)
```python
class EnhancedShamirSecretSharing:
    def __init__(self, config: ShamirConfig)
    def split_secret(self, secret: float, participants: List[int]) -> List[SecretShare]
    def reconstruct_secret(self, shares: List[SecretShare]) -> float
    def secure_aggregate(self, updates: List[np.ndarray], weights: List[float]) -> np.ndarray
```

### Classi Apprendimento Federato

#### `FederatedTrainer` (utils/federated_learning.py)
```python
class FederatedTrainer:
    def __init__(self, base_model_class, **model_params)
    def train_regional_model(self, X: np.ndarray, y: np.ndarray, region: str) -> Dict
    def aggregate_models(self, regional_models: List[Dict], method: str = "fedavg") -> Dict
    def evaluate_federated_model(self, model: Dict, test_data: List[Tuple]) -> Dict
```

## 12. CLI e Configurazione
- Esempio `config.yaml`:
```yaml
seed: 42
paths:
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30  # Valore default corretto
federated:
  rounds: 10
  min_clients: 3
  aggregator: fedavg
privacy:
  enabled: true
  dp: { epsilon: 1.0, delta: 1e-6, max_grad_norm: 1.0, accountant: rdp }
  secure_agg: { scheme: shamir, threshold: 3-of-5, dropout_recovery: true }
anchoring:
  enabled: true
  backend: merkle
```

### Esempi CLI:
```bash
# Training centralizzato
python scripts/03_train_models.py --config config.yaml

# Pipeline apprendimento federato
python scripts/05_LightGBM_federated_training.py
python scripts/07_mlp_federated_training.py --aggregator trimmed_mean

# Apprendimento federato che preserva privacy
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5

# Visualizzazione e analisi
python scripts/06_LightGBM_federated_visualization.py
python scripts/09_mlp_federated_privacy_visualization.py

# Ancoraggio blockchain
python scripts/blockchain_data_anchoring.py
python scripts/10_blockchain_anchoring_bench.py
```

## 13. Deployment e Operazioni
- **Locale**: `pip install -r requirements.txt` → `streamlit run streamlit_app.py`.
- **Docker**: includere `curl` per controlli salute; esporre porta 8501; utente non-root raccomandato.
- **Configurazione**: Soglia distanza default è **30 km** (non 40 km come menzionato precedentemente).
- **Gestione Artefatti**: backup `results/` e riassunti CSV; log budget privacy.

### Considerazioni Produzione
- Monitorare degradazione prestazioni apprendimento federato (~0.5% per LightGBM, ~4% per MLP)
- Modalità preserva-privacy aggiunge overhead minimo con parametri DP ben calibrati
- Ancoraggio blockchain scala appropriatamente per requisiti audit

## 14. Osservabilità e Testing
- Test unitari per feature engineering e scoring; smoke test per loop FL.
- Logging: metriche per-round e contabilità DP; export a CSV/JSON.
- Script visualizzazione forniscono controlli sanità mentale e tracking regressione.
- Monitoraggio prestazioni include uso risorse e progressione training.

## 15. Ottimizzazione Prestazioni
- Ridurre round FL per iterazione rapida; abilitare aggregatori robusti quando qualità client varia.
- Cache feature; profilare I/O; fissare conteggi thread per run stabili.
- Monitorare norme gradienti per calibrare clipping DP.
- **Tuning soglia distanza**: 30 km default può essere restrittivo per aree rurali.

## 16. Risoluzione Problemi
- Librerie mancanti → `pip install -r requirements.txt`.
- Errore import sklearn `_safe_tags` → allineare versioni `scikit-learn` e `imbalanced-learn`.
- Fallimenti healthcheck → assicurarsi `curl` in container; verificare `/_stcore/health`.
- **Problemi prestazioni**: Apprendimento federato mostra trade-off prestazioni attesi vs centralizzato.

## 17. Conformità (GDPR)
- Ruoli (controllore/processore), base legale, DPIA, DSR (accesso/cancellazione).
- Elaborazione locale; solo aggiornamenti federati; ritenzione documentata.
- FL preserva-privacy soddisfa requisiti normativi con budget ε/δ appropriati.

## 18. Roadmap
- Budget DP adattivi basati su dimensione coorte e sensibilità.
- Aggregazione sicura cross-silo a scala più ampia.
- Portale verifica pubblico per artefatti ancorati.
- Ottimizzazione prestazioni per utilità apprendimento federato.

---

## Riassunto Prestazioni (Corretto)

### Correzioni Chiave Effettuate:
1. **Soglia distanza**: Corretti tutti riferimenti per usare 30 km default (non 40 km)
2. **Prestazioni federate**: Riflette accuratamente leggera sottoperformance vs centralizzato
3. **Costi privacy**: Quantifica correttamente ~4.8% riduzione F1-score per garanzie privacy
4. **Dati benchmark**: Include metriche prestazioni ancoraggio blockchain accurate
5. **Accuratezza tecnica**: Tutte le cifre prestazioni corrispondono risultati sperimentali

### Realtà Apprendimento Federato:
- LightGBM: Impatto prestazioni minimo (-0.0005 F1-score)
- MLP: Trade-off prestazioni moderato (~4% riduzione F1-score)
- Preserva-privacy: Impatto aggiuntivo ~0.8% oltre apprendimento federato standard
- Ancoraggio blockchain: Scala appropriatamente con generazione prove O(log n)
