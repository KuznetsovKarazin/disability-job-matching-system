# Guida Utente — Sistema di Matching per Collocamento Mirato
_Ultimo aggiornamento: 2025-08-24 15:45_

> Questa guida è scritta per **operatori** (CPI/SIL), **analisti**, e **sviluppatori**.
> Copre flussi di lavoro, operazioni UI, preparazione dataset, apprendimento federato, controlli privacy,
> ancoraggio blockchain, risoluzione problemi, e best practice.

**Documenti correlati**: `README.md`, `technical_documentation.md`, `deployment_guide.md`, `api_reference.md`.

## Indice
1. [Introduzione](#1-introduzione)
2. [Tour dell'Interfaccia](#2-tour-dellinterfaccia)
3. [Preparazione Dati e Contratti](#3-preparazione-dati-e-contratti)
4. [Esecuzione del Sistema](#4-esecuzione-del-sistema)
5. [Flussi di Lavoro Apprendimento Federato](#5-flussi-di-lavoro-apprendimento-federato)
6. [Privacy e Sicurezza (Vista Operatore)](#6-privacy-e-sicurezza-vista-operatore)
7. [Ancoraggio Blockchain (Vista Operatore)](#7-ancoraggio-blockchain-vista-operatore)
8. [Risultati, Report e Visualizzazioni](#8-risultati-report-e-visualizzazioni)
9. [Risoluzione Problemi](#9-risoluzione-problemi)
10. [FAQ](#10-faq)
11. [Best Practice](#11-best-practice)
12. [Supporto e Contatti](#12-supporto-e-contatti)
13. [Appendice A — Riferimento CLI](#13-appendice-a--riferimento-cli)
14. [Appendice B — Riferimento Configurazione](#14-appendice-b--riferimento-configurazione)
15. [Appendice C — Mappa Directory Risultati](#15-appendice-c--mappa-directory-risultati)

## 1. Introduzione

Il Sistema di Matching per Collocamento Mirato aiuta i servizi pubblici per l'impiego (CPI/SIL) a abbinare candidati con lavori inclusivi. Supporta sia training **centralizzato** che **federato**, e include capacità **privacy-preserving** e **ancoraggio integrità dati**.

**Concetti chiave**
- Training centralizzato con `data/processed/Enhanced_Training_Dataset.csv`.
- Apprendimento federato tra regioni senza condividere dati grezzi.
- Modalità privacy: aggregazione sicura (Shamir) e Privacy Differenziale (basata su RDP).
- Ancoraggio: commitment Merkle + prove per integrità a lungo termine.

**Ruoli**
- **Operatore**: esegue match, rivede raccomandazioni, genera report.
- **Analista**: valida dataset, monitora metriche, confronta modelli.
- **Sviluppatore**: mantiene pipeline, ottimizza modelli, configura deployment.

## 2. Tour dell'Interfaccia

Avvio:
```bash
streamlit run streamlit_app.py
```

**Sezioni Interfaccia Principale:**

**Dashboard Home**
- KPI (candidati, aziende, regioni), link rapidi.
- Indicatori stato e salute sistema.

**Ricerca Candidato**
- Inserimento informazioni candidato (manuale o da record esistenti)
- Configurazione parametri ricerca: raggio (**default 30 km**), soglia attitudine
- Visualizzazione raccomandazioni aziende classificate con punteggi compatibilità
- Esportazione risultati in CSV/PNG

**Dashboard Analytics**
- Metriche panoramica sistema e indicatori prestazioni
- Grafici distribuzione per disabilità e settori aziendali
- Tracciamento successo collocamenti e trend

**Gestione Dataset**
- Navigazione dati candidati e aziende
- Funzioni esportazione per analisi esterni
- Strumenti verifica qualità dati

**Informazioni Sistema (Info Sistema)**
- Stato tecnico e configurazione
- Metriche prestazioni modelli
- Stato privacy e sicurezza

### Pannello Configurazione Laterale

**Configurazione Sistema**:
- **Selezione Modello**: Scelta modello AI (se multipli disponibili)
- **Soglia Attitudine**: Prontezza minima collocamento (0.0-1.0)
- **Distanza Massima**: Raggio ricerca in chilometri (**5-50 km, default 30 km**)
- **Top Raccomandazioni**: Numero risultati da mostrare (3-10)

## 3. Preparazione Dati e Contratti

**Dati Input Grezzi (`data/raw/`)**
- `Dataset_Candidati_Aggiornato.csv` - Dati master candidati
- `Dataset_Aziende_con_Stima_Assunzioni.csv` - Dati aziende e ruoli

**Dati Processati (`data/processed/`)**
- `Dataset_Candidati_Aggiornato_Extended.csv` - Dati candidati migliorati
- `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv` - Dati aziende migliorati
- `Enhanced_Training_Dataset.csv` - Tabella training canonica

**Raccomandazioni Qualità Dati:**
- Mantenere schemi consistenti tra regioni.
- Documentare cambiamenti in `SCHEMA.md`; aggiornare `01_generate_dataset.py`.
- Validare geocoding; allineare con `ui.distance_max_km` (**30 km default**).
- Assicurare formato indirizzo italiano: "Città, Provincia, Italy"

## 4. Esecuzione del Sistema

### Training Centralizzato
```bash
# Addestra tutti i 7 modelli con ottimizzazione iperparametri
python scripts/03_train_models.py --config config.yaml

# Analizza risultati e genera report prestazioni
python scripts/04_analyze_results.py
```

### Preparazione Dati e Visualizzazione
```bash
# Genera dataset estesi e dati training
python scripts/01_generate_dataset.py

# Crea visualizzazioni analisi dati
python scripts/02_visualize_dataset.py
```

### LightGBM Federato (regionale → ensemble)
```bash
# Addestra modelli regionali e crea ensemble federato
python scripts/05_LightGBM_federated_training.py

# Genera visualizzazioni apprendimento federato
python scripts/06_LightGBM_federated_visualization.py
```

### MLP Federato (standard + privacy)
```bash
# Apprendimento federato standard con vari aggregatori
python scripts/07_mlp_federated_training.py --aggregator fedavg

# Apprendimento federato che preserva privacy con DP
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5

# Confronta risultati apprendimento federato
python scripts/09_mlp_federated_privacy_visualization.py
```

### Ancoraggio Blockchain
```bash
# Crea commitment Merkle e prove
python scripts/blockchain_data_anchoring.py

# Benchmark prestazioni ancoraggio
python scripts/10_blockchain_anchoring_bench.py
```

## 5. Flussi di Lavoro Apprendimento Federato

### 5.1 LightGBM (Ensemble Regionale)
**Approccio**: Addestra modelli per-regione; combina via ensemble pesato (peso ∝ conteggio campioni).

```bash
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py
```

**Prestazioni Attese**: 
- Centralizzato: F1 ≈ 0.9012
- Federato: F1 ≈ 0.9007 (-0.0005 degradazione)

### 5.2 MLP (FedAvg Vero)
**Opzioni Aggregazione:**

**FedAvg (default)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator fedavg --rounds 10 --batch_size 256
```

**Trimmed Mean (robusto a outlier)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator trimmed_mean --rounds 10 --batch_size 256
```

**Coordinate Median (più robusto)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator coordinate_median --rounds 10 --batch_size 256
```

**Prestazioni Attese**:
- Centralizzato: F1 ≈ 0.828
- Federato: F1 ≈ 0.788 (~4% riduzione)

### 5.3 FL Preserva-Privacy (Shamir + DP)
**Diversi Livelli Privacy:**

**Privacy Forte (ε=0.5)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 0.5 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Privacy Moderata (ε=1.0, raccomandato)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Privacy Rilassata (ε=2.0)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 2.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Prestazioni Attese**: F1 ≈ 0.788 (costo privacy aggiuntivo minimo)

## 6. Privacy e Sicurezza (Vista Operatore)

### Aggregazione Sicura
- **Condivisione Segreti Shamir**: Soglia 3-su-5 con mascheramento per-parametro
- **Recupero Dropout**: Gestione automatica disconnessioni client
- **Semi Deterministici**: Assicura risultati riproducibili tra esecuzioni

### Privacy Differenziale
- **Clipping Per-Round**: Limiti gradienti prima iniezione rumore
- **Singolo Rumore Gaussiano**: Applicato una volta per round aggregazione
- **Contabilità RDP**: Traccia budget privacy cumulativo (ε, δ)

### Checklist Sicurezza
- Dati grezzi non lasciano mai nodi regionali
- Proteggere directory `results_mlp_federated_privacy/`
- Ruotare credenziali accesso per `data/` e `results/`
- Monitorare consumo budget privacy

## 7. Ancoraggio Blockchain (Vista Operatore)

### Scopo
- **Prove Integrità**: Commitment Merkle per modelli e report
- **Verifica Lungo Termine**: Tracce audit tamper-evident
- **Conformità**: Soddisfare requisiti normativi per istituzioni pubbliche

### Esecuzione Ancoraggio
```bash
# Crea commitment e prove
python scripts/blockchain_data_anchoring.py

# Benchmark prestazioni
python scripts/10_blockchain_anchoring_bench.py
```

### Aspettative Prestazioni
- **100 record**: ~2.3s tempo costruzione, 1.1ms generazione prove
- **1.000 record**: ~30.5s tempo costruzione, 2.6ms generazione prove  
- **10.000 record**: ~344s tempo costruzione, 20.7ms generazione prove

**Output**: Manifesti, prove, log verifica in `results_blockchain_demo/`

## 8. Risultati, Report e Visualizzazioni

### Struttura Directory Risultati
- **`results/`**: Artefatti centralizzati (*.joblib), curve apprendimento, merged_model_summary.csv
- **`results_LightGBM_federated/`**: Confronto LightGBM regionale/federato/centralizzato
- **`results_mlp_federated/`**: Risultati apprendimento federato MLP standard
- **`results_mlp_federated_privacy/`**: Output MLP FL preserva-privacy
- **`visualizations_federated_comparison/`**: Grafici comparativi e analisi
- **`results_blockchain_demo/`**: Artefatti ancoraggio e dati verifica

### Interpretazione Prestazioni
- **Priorità F1-Score**: Metrica primaria per qualità raccomandazioni
- **Trade-off Prestazioni**:
  - LightGBM Federato: Impatto minimo (-0.0005 F1)
  - MLP Federato: Impatto moderato (~4% riduzione F1)
  - Modalità Privacy: Riduzione F1 aggiuntiva ~0.8%
- **Contesto ROC-AUC**: Valori ~0.70 riflettono design dati probabilistico intenzionale

## 9. Risoluzione Problemi

### Problemi Comuni

**Dipendenze Mancanti**
```bash
# Soluzione: Reinstallare requirements
pip install -r requirements.txt
```

**Errori Import Scikit-learn**
```bash
# Soluzione: Fissare versioni compatibili
pip install scikit-learn==1.6.1 imbalanced-learn==0.13.0
```

**Fallimenti Health Check Docker**
- Assicurarsi `curl` sia installato in container
- Verificare accessibilità endpoint `/_stcore/health`

**Confronti Risultati Vuoti**
- Confermare esistenza cartelle risultati e contenuto dati
- Controllare percorsi file CSV e permessi

**Errori Distanza Geografica**
- Usare formato: "Città, Provincia, Italy"
- Verificare spelling nomi città italiane
- Controllare cache geocoding: `data/processed/geocoding_cache.json`

### Problemi Prestazioni

**Tempi Risposta Lenti**
1. Pulire cache browser e ricaricare
2. Controllare stabilità connessione internet
3. Ridurre soglia distanza per elaborazione più veloce
4. Considerare batch size minori per apprendimento federato

**Nessun Match Candidato Trovato**
1. **Aumentare soglia distanza** da 30km a 40-50km per aree rurali
2. **Abbassare soglia attitudine** da 0.3 a 0.2-0.25
3. **Rivedere esclusioni** - assicurare non siano eccessivamente restrittive
4. **Verificare posizione** - deve essere indirizzo italiano valido

## 10. FAQ

**D: Posso usare il sistema senza condividere dati tra regioni?**
R: Sì - l'apprendimento federato MLP abilita collaborazione senza condivisione dati.

**D: Qual è il raggio ricerca predefinito?**
R: 30 km (configurabile via `config.yaml: ui.distance_max_km`).

**D: Quanto impatta la modalità privacy sulle prestazioni?**
R: ~4.8% riduzione F1-score per garanzie privacy forti (ε=1.0, δ=1e-6).

**D: Come posso verificare risultati mesi dopo?**
R: Usare prove ancoraggio blockchain memorizzate in `results_blockchain_demo/`.

**D: Sono richieste GPU?**
R: No - sistema è ottimizzato CPU e funziona bene su hardware standard.

**D: Perché alcuni punteggi sono più bassi in modalità federata vs centralizzata?**
R: È atteso per differenze distribuzione dati. LightGBM mostra impatto minimo, MLP mostra trade-off moderati.

## 11. Best Practice

### Operazioni Quotidiane

**Routine Mattutina**:
1. Controllare stato sistema nel tab Info Sistema
2. Rivedere cambiamenti analytics notturni
3. Verificare attualità informazioni candidati prioritari

**Elaborazione Candidati**:
1. Verificare sempre esclusioni con candidato prima ricerca
2. Usare dati candidato esistenti quando disponibili per consistenza
3. Documentare collocamenti riusciti per miglioramento sistema

**Valutazione Risultati**:
1. Concentrarsi su top 3 raccomandazioni per contatto iniziale
2. Considerare preferenze geografiche insieme a punteggi
3. Rivedere dettagli compatibilità oltre punteggi numerici

### Revisioni Settimanali

**Manutenzione Qualità Dati**:
- Aggiornare valutazioni candidati e disponibilità
- Verificare aperture posizioni aziende e requisiti
- Rimuovere o aggiornare inserzioni aziende inattive
- Rivedere accuratezza geocoding per nuovi indirizzi

**Monitoraggio Prestazioni**:
- Tracciare tassi successo collocamento per range punteggi
- Identificare aziende e settori più performanti
- Monitorare tempi risposta sistema e accuratezza
- Notare pattern in collocamenti riusciti vs non riusciti

### Ottimizzazione Configurazione

**Impostazioni Urbane vs Rurali**:
- **Aree urbane**: 20-25 km raggio per focus locale
- **Aree rurali**: 35-50 km raggio per opzioni adeguate
- **Regioni miste**: 30 km default solitamente ottimale

**Tuning Soglie**:
- **Periodi alto volume**: Aumentare soglia attitudine (0.4-0.5) per qualità
- **Periodi basso volume**: Diminuire soglia (0.2-0.3) per matching più ampio
- **Collocamenti specializzati**: Regolare basato su tipo disabilità e requisiti

## 12. Supporto e Contatti

### Supporto Tecnico
- **Contatto Primario**: michele.melch@gmail.com
- **Supporto Accademico**: oleksandr.kuznetsov@uniecampus.it
- **Tempo Risposta**: 24-48 ore per problemi non critici

### Partner Istituzionali
- **CPI Villafranca di Verona**: Guida operativa e validazione
- **SIL Veneto**: Coordinamento regionale e best practice
- **Università eCampus**: Sviluppo tecnico e ricerca

### Ottenere Aiuto
**Prima di Contattare Supporto**:
1. Annotare messaggi errore esatti e passi riproduzione
2. Controllare stato sistema nel tab Info Sistema
3. Provare soluzioni troubleshooting di base
4. Raccogliere dettagli configurazione sistema

**Includere nelle Richieste Supporto**:
- Screenshot dei problemi
- Messaggi errore (testo esatto)
- Passi per riprodurre il problema
- Dettagli configurazione sistema

## 13. Appendice A — Riferimento CLI

### Script Training Core

**01_generate_dataset.py** - Estensione dati e generazione training
```bash
python scripts/01_generate_dataset.py [--config config.yaml]
```

**02_visualize_dataset.py** - Analisi dati e visualizzazione
```bash
python scripts/02_visualize_dataset.py [--config config.yaml]
```

**03_train_models.py** - Training modelli centralizzati
```bash
python scripts/03_train_models.py [--config config.yaml]
```

**04_analyze_results.py** - Analisi prestazioni e reporting
```bash
python scripts/04_analyze_results.py [--config config.yaml]
```

### Script Apprendimento Federato

**05_LightGBM_federated_training.py** - Training federato LightGBM
```bash
python scripts/05_LightGBM_federated_training.py [--config config.yaml]
```

**06_LightGBM_federated_visualization.py** - Visualizzazione federata LightGBM
```bash
python scripts/06_LightGBM_federated_visualization.py [--config config.yaml]
```

**07_mlp_federated_training.py** - Training federato MLP
```bash
python scripts/07_mlp_federated_training.py [--config config.yaml] [--aggregator {fedavg,trimmed_mean,coordinate_median}] [--rounds ROUNDS] [--batch_size BATCH_SIZE]
```

**08_mlp_federated_privacy.py** - Training federato MLP preserva-privacy
```bash
python scripts/08_mlp_federated_privacy.py [--config config.yaml] [--dp.epsilon EPSILON] [--dp.delta DELTA] [--secure_agg.threshold THRESHOLD]
```

**09_mlp_federated_privacy_visualization.py** - Visualizzazione federata privacy
```bash
python scripts/09_mlp_federated_privacy_visualization.py [--config config.yaml]
```

### Script Blockchain

**blockchain_data_anchoring.py** - Crea commitment Merkle e prove
```bash
python scripts/blockchain_data_anchoring.py [--config config.yaml]
```

**10_blockchain_anchoring_bench.py** - Benchmark prestazioni ancoraggio
```bash
python scripts/10_blockchain_anchoring_bench.py [--config config.yaml]
```

## 14. Appendice B — Riferimento Configurazione

**Campi Configurazione Chiave:**
```yaml
seed: 42
paths:
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30  # Raggio ricerca default
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

## 15. Appendice C — Mappa Directory Risultati

**Struttura Directory:**
- `results/` — Modelli centralizzati e grafici prestazioni
- `results_LightGBM_federated/` — Artefatti apprendimento federato LightGBM  
- `results_mlp_federated/` — Output MLP federato standard
- `results_mlp_federated_privacy/` — Output MLP federato preserva-privacy
- `results_blockchain_demo/` — Artefatti ancoraggio blockchain
- `visualizations_federated_comparison/` — Grafici analisi comparativa

**File Chiave:**
- `merged_model_summary.csv` — Prestazioni modelli centralizzati consolidate
- `complete_model_comparison.csv` — Confronto federato vs centralizzato
- `experiment_metadata.json` — Configurazione sperimentale dettagliata e risultati

---

