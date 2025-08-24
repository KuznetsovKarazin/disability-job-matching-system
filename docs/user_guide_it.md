# Guida Utente — Sistema di Collocamento Mirato
_Ultimo aggiornamento: 2025-08-24 09:37_

> Guida per **operatori** (CPI/SIL), **analisti** e **sviluppatori**.
> Copre workflow, UI, preparazione dataset, apprendimento federato, privacy,
> anchoring, troubleshooting e best practice.

**Documenti correlati**: `README_IT.md`, `technical_documentation.md`, `deployment_guide.md`, `api_reference.md`.


## Indice
1. Introduzione
2. Tour dell'Interfaccia
3. Preparazione Dati & Contratti
4. Esecuzione del Sistema
5. Workflow di Apprendimento Federato
6. Privacy & Sicurezza (vista operatore)
7. Anchoring stile Blockchain (vista operatore)
8. Risultati, Report & Visualizzazioni
9. Troubleshooting
10. FAQ
11. Glossario
12. Supporto & Contatti
13. Appendice A — Riferimenti CLI
14. Appendice B — Riferimento `config.yaml`
15. Appendice C — Mappa delle Cartelle Risultati
16. Appendice D — Guida Precedente (testo integrale)

## 1. Introduzione

Il Sistema di Collocamento Mirato supporta CPI/SIL nell’abbinare candidati e posizioni inclusive.
Supporta training **centralizzato** e **federato**, include funzioni **privacy‑preserving** e **anchoring** per l’integrità.

**Concetti chiave**
- Training centralizzato con `data/processed/Enhanced_Training_Dataset.csv`.
- FL tra regioni senza condividere dati grezzi.
- Modalità privacy: secure aggregation (Shamir) e Privacy Differenziale (RDP).
- Anchoring: commit Merkle + prove.

**Ruoli**
- Operatore: avvia match, rivede raccomandazioni, genera report.
- Analista: valida dataset, monitora metriche, confronta modelli.
- Sviluppatore: mantiene pipeline, ottimizza modelli, configura deployment.

## 2. Tour dell'Interfaccia

Avvio:
```bash
streamlit run streamlit_app.py
```

**Home**
- KPI (candidati, aziende, regioni) e link rapidi.

**Matching**
- Input: raggio (default 30 km), filtri; Output: ranking (compatibilità, distanza, readiness).
- Export: CSV/PNG.

**Confronto Modelli**
- Centralizzato vs Regionale vs Federato; F1/ROC‑AUC; calibrazione; matrici.

**Privacy**
- Attiva/disattiva; (ε, δ) per round; stato secure aggregation.

**Integrità**
- Ultimo anchoring, prove e verifica; avvio di un nuovo anchoring.

## 3. Preparazione Dati & Contratti

**Raw (`data/raw/`)**
- Dataset_Candidati_Aggiornato.csv
- Dataset_Aziende_con_Stima_Assunzioni.csv

**Processed (`data/processed/`)**
- Dataset_Candidati_Aggiornato_Extended.csv
- Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
- Enhanced_Training_Dataset.csv  # tabella canonica

Raccomandazioni:
- Schemi coerenti tra regioni.
- Documentare cambi in `SCHEMA.md`; aggiornare `01_generate_dataset.py`.
- Validare geocodifica; allineare `ui.distance_max_km` (30 km).

## 4. Esecuzione del Sistema

**Centralizzato**
```bash
python scripts/03_train_models.py --config config.yaml
python scripts/04_analyze_results.py
```

**LightGBM Federato (regionale → ensemble)**
```bash
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py
```

**MLP Federato (standard + privacy)**
```bash
python scripts/07_mlp_federated_training.py --aggregator fedavg
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5
python scripts/09_mlp_federated_privacy_visualization.py
```

**Anchoring**
```bash
python scripts/blockchain_data_anchoring.py
python scripts/10_blockchain_anchoring_bench.py
```

## 5. Workflow di Apprendimento Federato

### 5.1 LightGBM (ensemble regionale)
- Modelli per regione; ensemble pesato (peso ∝ campioni).
```bash
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py
```

### 5.2 MLP (vero FedAvg)
**Aggregatore: fedavg**
```bash
python scripts/07_mlp_federated_training.py --aggregator fedavg --rounds 10 --batch_size 256
```
**Aggregatore: trimmed_mean**
```bash
python scripts/07_mlp_federated_training.py --aggregator trimmed_mean --rounds 10 --batch_size 256
```
**Aggregatore: coordinate_median**
```bash
python scripts/07_mlp_federated_training.py --aggregator coordinate_median --rounds 10 --batch_size 256
```

### 5.3 FL con Privacy (Shamir + DP)
**DP ε=0.5, δ=1e-6 (RDP)**
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 0.5 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```
**DP ε=1.0, δ=1e-6 (RDP)**
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```
**DP ε=2.0, δ=1e-6 (RDP)**
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 2.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

## 6. Privacy & Sicurezza (vista operatore)

**Secure Aggregation**
- Shamir (es. 3‑of‑5), mascheramento per‑parametro, recupero dropout.
- Semi deterministici per riproducibilità.

**Privacy Differenziale**
- Clipping per round e singola iniezione di rumore gaussiano.
- Accounting RDP di (ε, δ) cumulativi.

**Checklist**
- I dati grezzi non lasciano i nodi regionali.
- Proteggere `results_mlp_federated_privacy/`.
- Rotazione accessi a `data/` e `results/`.

## 7. Anchoring stile Blockchain (vista operatore)

**Obiettivo**
- Prove di integrità (commit Merkle + prove di inclusione) per modelli e report.

**Esecuzione**
```bash
python scripts/blockchain_data_anchoring.py
python scripts/10_blockchain_anchoring_bench.py
```

**Output**
- Manifest, prove, log in `results_blockchain_demo/`.

## 8. Risultati, Report & Visualizzazioni

- `results/`: artefatti centralizzati (`*.joblib`), `learning_curves/`, `*.png`, `merged_model_summary.csv`.
- `results_LightGBM_federated/`: artefatti LightGBM (regionale/federato/centralizzato) + `complete_model_comparison.csv`.
- `results_mlp_federated/`, `results_mlp_federated_privacy/`: output MLP FL e log privacy.
- `visualizations_federated_comparison/`: grafici comparativi.
- `results_blockchain_demo/`: artefatti di anchoring.

Interpretazione:
- Usare F1/ROC‑AUC; confrontare Centralizzato vs Regionale vs Federato.
- Con DP, trade‑off moderato su F1; regolare ε/round/aggregatore.

## 9. Troubleshooting

- Moduli mancanti → `pip install -r requirements.txt`
- Errore `_safe_tags` di scikit‑learn → versionare con `imbalanced-learn`
- Health check Docker → includere `curl`, verificare `/_stcore/health`
- Confronti vuoti → verificare cartelle risultati e path CSV

## 10. FAQ

**Uso senza condivisione dati?** Sì — MLP FL.
**Raggio predefinito?** 30 km (`config.yaml`).
**Impatto DP?** Lieve calo F1; regolare ε/round/aggregatore.
**Integrità nel tempo?** Verifica con prove in `results_blockchain_demo/`.
**GPU?** Non richieste; adatto a CPU.

## 11. Glossario

CPI/SIL, Apprendimento Federato, Secure Aggregation, Privacy Differenziale (RDP), Albero di Merkle.

## 12. Supporto & Contatti

CPI Villafranca di Verona • SIL Veneto • Università eCampus

## 13. Appendice A — Riferimenti CLI
### 01_generate_dataset.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/01_generate_dataset.py --config config.yaml
```
### 02_visualize_dataset.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/02_visualize_dataset.py --config config.yaml
```
### 03_train_models.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/03_train_models.py --config config.yaml
```
### 04_analyze_results.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/04_analyze_results.py --config config.yaml
```
### 05_LightGBM_federated_training.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/05_LightGBM_federated_training.py --config config.yaml
```
### 06_LightGBM_federated_visualization.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/06_LightGBM_federated_visualization.py --config config.yaml
```
### 07_mlp_federated_training.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/07_mlp_federated_training.py --config config.yaml
```
### 08_mlp_federated_privacy.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/08_mlp_federated_privacy.py --config config.yaml
```
### 09_mlp_federated_privacy_visualization.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/09_mlp_federated_privacy_visualization.py --config config.yaml
```
### blockchain_data_anchoring.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/blockchain_data_anchoring.py --config config.yaml
```
### 10_blockchain_anchoring_bench.py
Opzioni comuni:
```
--config config.yaml
--rounds 10
--batch_size 256
--aggregator [fedavg|trimmed_mean|coordinate_median]
```
Esempio:
```bash
python scripts/10_blockchain_anchoring_bench.py --config config.yaml
```

## 14. Appendice B — Riferimento `config.yaml`

Vedi README per l'esempio completo. Campi chiave: `paths.*`, `ui.distance_max_km`, `training.*`, `federated.*`, `privacy.*`, `anchoring.*`.

## 15. Appendice C — Mappa Cartelle Risultati

- `results/` — modelli e grafici centralizzati
- `results_LightGBM_federated/` — artefatti federati LightGBM
- `results_mlp_federated/`, `results_mlp_federated_privacy/` — output federati MLP
- `results_blockchain_demo/` — anchoring
- `visualizations_federated_comparison/` — grafici comparativi

## 16. Appendice D — Guida Precedente (testo integrale)

    # 🎯 Guida Utente - Sistema di Raccomandazione per Collocamento Mirato

**Manuale Operativo Completo per Operatori CPI e SIL**

---

## 📋 Indice

1. [Panoramica del Sistema](#panoramica-del-sistema)
2. [Primi Passi](#primi-passi)
3. [Tour dell'Interfaccia Principale](#tour-dellinterfaccia-principale)
4. [Ricerca Aziende per Candidati](#ricerca-aziende-per-candidati)
5. [Comprensione dei Risultati](#comprensione-dei-risultati)
6. [Dashboard Analytics](#dashboard-analytics)
7. [Gestione Dataset](#gestione-dataset)
8. [Impostazioni di Configurazione](#impostazioni-di-configurazione)
9. [Risoluzione Problemi](#risoluzione-problemi)
10. [Migliori Pratiche](#migliori-pratiche)

---

## 📊 Panoramica del Sistema

### Cos'è il Sistema di Raccomandazione per Collocamento Mirato?

Questo sistema è uno strumento avanzato basato su Intelligenza Artificiale progettato per aiutare i Centri per l'Impiego (CPI) e i Servizi di Integrazione Lavorativa (SIL) a trovare le migliori aziende compatibili per candidati con disabilità. Automatizza il complesso processo di valutazione delle esclusioni del candidato rispetto ai requisiti di compatibilità aziendale.

### Vantaggi Principali per gli Operatori

- **⏱️ Risparmio di Tempo**: Riduce il matching manuale da ore a secondi
- **🎯 Maggiore Accuratezza**: Scoring di compatibilità 90%+ vs valutazione soggettiva
- **📊 Decisioni Data-Driven**: Punteggi obiettivi basati su fattori multipli
- **🔍 Ricerca Comprensiva**: Valuta tutte le aziende nel raggio specificato
- **📈 Analytics**: Traccia pattern di collocamento e performance del sistema

### Chi Dovrebbe Usare Questa Guida?

- **Operatori Centro per l'Impiego** (personale CPI)
- **Coordinatori SIL** (personale Servizio Integrazione Lavorativa)
- **Case Manager** responsabili del collocamento candidati
- **Account Manager** che gestiscono rapporti aziendali

---

## 🚀 Primi Passi

### Requisiti di Sistema

- **Browser Web**: Chrome, Firefox, Safari, o Edge (versioni recenti)
- **Connessione Internet**: Richiesta per setup iniziale e geocodificazione
- **Risoluzione Schermo**: Minimo 1024x768 (1920x1080 raccomandato)
- **Nessuna Installazione Richiesta**: Il sistema funziona interamente nel browser

### Primo Accesso

1. **Aprire il Sistema**:
   ```
   Aprire il browser web e navigare all'URL del sistema
   (fornito dall'amministratore IT)
   ```

2. **Inizializzazione Sistema**:
   - Il sistema si caricherà automaticamente con dati demo al primo avvio
   - Il caricamento iniziale può richiedere 30-60 secondi
   - Vedrete la dashboard principale con candidati e aziende di esempio

3. **Verificare Stato Sistema**:
   - Controllare la scheda "Info Sistema" per confermare che tutti i componenti funzionino
   - Assicurarsi di vedere lo stato "✅ Interface: Operativo"

### Modalità Demo vs Produzione

**Modalità Demo** (Default):
- Utilizza dati sintetici realistici per training e dimostrazione
- Sicura per test e apprendimento del sistema
- Nessuna informazione reale di candidati o aziende

**Modalità Produzione** (Quando disponibili dati reali):
- Utilizza dati storici reali di collocamento
- Richiede setup dati appropriato dall'amministratore IT
- Fornisce raccomandazioni di matching reali

---

## 🖥️ Tour dell'Interfaccia Principale

### Layout Dashboard

L'interfaccia del sistema è organizzata in quattro schede principali:

#### 1. 🔍 Ricerca Candidato
- **Scopo**: Trovare aziende compatibili per singoli candidati
- **Uso Primario**: Operazioni quotidiane di collocamento
- **Utenti**: Case manager, coordinatori collocamento

#### 2. 📊 Analytics
- **Scopo**: Visualizzare statistiche di sistema e metriche di performance
- **Uso Primario**: Monitoraggio e reportistica
- **Utenti**: Supervisori, analisti dati

#### 3. 📋 Dataset
- **Scopo**: Esplorare ed esportare dati candidati/aziende
- **Uso Primario**: Verifica dati e gestione
- **Utenti**: Amministratori dati, controllo qualità

#### 4. ℹ️ Info Sistema
- **Scopo**: Stato sistema e informazioni tecniche
- **Uso Primario**: Troubleshooting e configurazione
- **Utenti**: Supporto IT, amministratori sistema

### Pannello Configurazione Laterale

Situato sul lato sinistro dell'interfaccia:

**🔧 Configurazione Sistema**:
- **Selezione Modello**: Scegli modello AI (se disponibili multipli)
- **Soglia Attitudine**: Propensione minima al lavoro (0.0-1.0)
- **Distanza Massima**: Raggio di ricerca in chilometri (5-50 km)
- **Top Raccomandazioni**: Numero di risultati da mostrare (3-10)

---

## 🔍 Ricerca Aziende per Candidati

### Passo 1: Accedere alla Ricerca Candidato

1. Cliccare sulla scheda **"🔍 Ricerca Candidato"**
2. Vedrete due colonne:
   - **Sinistra**: Input dati candidato
   - **Destra**: Raccomandazioni aziende (inizialmente vuota)

### Passo 2: Inserire Informazioni Candidato

Avete due opzioni per inserire i dati del candidato:

#### Opzione A: Usare Candidato Esistente

1. **Spuntare la casella** "Usa candidato esistente"
2. **Selezionare dal menu a tendina**: Scegliere tra candidati pre-caricati
3. **Rivedere dati auto-compilati**: Il sistema popola automaticamente tutti i campi
4. **Verificare informazioni**: Assicurarsi che i dati siano corretti e aggiornati

#### Opzione B: Inserimento Manuale

1. **Lasciare deselezionata** "Usa candidato esistente"
2. **Compilare dettagli candidato**:

   **Informazioni Base**:
   - **Area Residenza**: Selezionare area residenziale del candidato
   - **Titolo Studio**: Scegliere livello di istruzione
   - **Tipo Disabilità**: Selezionare tipo di disabilità

   **Punteggi Valutazione**:
   - **Attitudine**: Propensione al lavoro (scala 0.0-1.0)
     - 0.0-0.3: Bassa propensione
     - 0.4-0.6: Propensione moderata
     - 0.7-1.0: Alta propensione

   **Dati Esperienza**:
   - **Anni Esperienza**: Anni di esperienza lavorativa
   - **Mesi Disoccupazione**: Mesi di disoccupazione

   **Esclusioni**:
   - **Esclusioni**: Inserire limitazioni lavorative (separate da virgola)
   - Esempi: "Turni notturni, Lavori in quota"

### Passo 3: Configurare Parametri di Ricerca

**Regolare impostazioni nella barra laterale se necessario**:

- **Soglia Attitudine**: Abbassare per ricerca più ampia, alzare per qualità
- **Distanza Max**: Espandere per più opzioni, ridurre per focus locale
- **Top Raccomandazioni**: Più risultati per revisione comprensiva

### Passo 4: Eseguire Ricerca

1. **Cliccare** il pulsante "🔄 Trova Aziende Compatibili"
2. **Attendere elaborazione**: Solitamente richiede 2-5 secondi
3. **Rivedere risultati**: Il sistema mostra raccomandazioni ordinate

### Comprensione del Processo di Ricerca

Il sistema esegue automaticamente questi passi:

1. **Filtro Attitudine**: Esclude candidati sotto soglia minima
2. **Filtro Geografico**: Considera solo aziende entro limite distanza
3. **Analisi Compatibilità**: Usa AI per matching esclusioni vs attività aziendali
4. **Scoring Multi-fattore**: Combina compatibilità, distanza, attitudine e fattori aziendali
5. **Ranking**: Ordina risultati per punteggio finale di matching

---

## 📊 Comprensione dei Risultati

### Formato Visualizzazione Risultati

Ogni raccomandazione mostra:

**Header Azienda**:
- **Nome Azienda** e **Punteggio Complessivo** (percentuale)
- Indicatore visivo punteggio (più alto = match migliore)

**Riga Metriche Chiave 1**:
- **Settore**: Tipo di attività commerciale
- **Distanza**: Chilometri da residenza candidato
- **Dipendenti**: Dimensione azienda

**Riga Metriche Chiave 2**:
- **Compatibilità**: Punteggio match semantico (percentuale)
- **Remote**: Disponibilità lavoro remoto
- **Posizioni**: Posizioni aperte per candidati disabili

### Interpretazione Punteggi

**Range Punteggio Complessivo**:
- **85-100%**: Match eccellente - altamente raccomandato
- **70-84%**: Buon match - adatto per collocamento
- **55-69%**: Match discreto - può richiedere valutazione aggiuntiva
- **Sotto 55%**: Match scarso - non raccomandato

**Punteggio Compatibilità**:
- **90-100%**: Nessun conflitto trovato tra esclusioni e requisiti lavoro
- **70-89%**: Conflitti potenziali minori - colloquio raccomandato
- **50-69%**: Alcuni conflitti presenti - valutazione attenta necessaria
- **Sotto 50%**: Conflitti significativi - probabilmente incompatibile

### Analytics Visive

**Grafico Distribuzione Punteggi**:
- Grafico a barre che mostra punteggi relativi tra tutte le raccomandazioni
- Aiuta identificare vincitori chiari vs competizioni strette

**Grafico Scatter Distanza vs Compatibilità**:
- Mostra trade-off tra prossimità e adattamento lavoro
- Cerchi più grandi indicano punteggi complessivi più alti

### Azioni sui Risultati

**Nessun Risultato Trovato**:
Se non appaiono aziende:
1. **Aumentare soglia distanza** nella barra laterale
2. **Abbassare soglia attitudine** se appropriato
3. **Rivedere esclusioni** - potrebbero essere troppo restrittive
4. **Controllare località candidato** - assicurarsi che sia valida

---

## 📊 Dashboard Analytics

### Metriche Panoramica Sistema

**Indicatori Chiave di Performance**:
- **👥 Candidati Totali**: Candidati totali nel sistema
- **🏢 Aziende Totali**: Aziende totali disponibili
- **📈 Attitudine Media**: Propensione media al lavoro tra candidati
- **💼 Posizioni Aperte**: Posizioni aperte totali a livello sistema

### Grafici Distribuzione

**Distribuzione Tipi Disabilità**:
- Mostra ripartizione categorie disabilità candidati
- Aiuta identificare aree focus servizi
- Utile per pianificazione risorse

**Distribuzione Settori Aziende**:
- Visualizza varietà settori lavorativi disponibili
- Identifica opportunità collocamento per industria
- Guida sforzi sviluppo business

### Uso Analytics per Operazioni

**Monitoraggio Quotidiano**:
- Controllare posizioni aperte vs volume candidati
- Monitorare punteggi attitudine medi per trend
- Identificare settori con opportunità più alta

**Pianificazione Strategica**:
- Usare distribuzione disabilità per programmi specializzati
- Targetizzare outreach aziendale basato su gap settoriali
- Pianificare programmi training basati su pattern compatibilità

---

## 📋 Gestione Dataset

### Visualizzazione Dati Candidati

1. **Navigare alla** scheda "📋 Dataset"
2. **Selezionare** pulsante radio "Candidati"
3. **Rivedere tabella dati**:
   - Tutti i record candidati con informazioni complete
   - Colonne ordinabili per esplorazione dati
   - Funzionalità ricerca per record specifici

**Colonne Chiave Spiegate**:
- **ID_Candidato**: Identificatore unico
- **Score Attitudine al Collocamento**: Propensione al lavoro (0.0-1.0)
- **Years_of_Experience**: Esperienza professionale
- **Durata Disoccupazione**: Durata disoccupazione (mesi)
- **Esclusioni**: Limitazioni lavorative da valutazione medica

### Visualizzazione Dati Aziende

1. **Selezionare** pulsante radio "Aziende"
2. **Rivedere informazioni azienda**:
   - Dettagli business e informazioni contatto
   - Descrizioni compatibilità e requisiti
   - Informazioni geografiche e dimensioni

**Colonne Chiave Spiegate**:
- **Nome Azienda**: Identificatore azienda
- **Tipo di Attività**: Settore/attività business
- **Compatibilità**: Descrizione accomodamenti disabilità appropriati
- **Posizioni Aperte**: Posizioni disponibili per candidati disabili
- **Remote**: Disponibilità lavoro remoto (0=No, 1=Sì)
- **Certification**: Stato certificazione disability-friendly

### Funzioni Export Dati

**Export Dati Candidati**:
1. **Cliccare** pulsante "📥 Scarica CSV Candidati"
2. **Salvare file** nella posizione desiderata
3. **Usare per**: Analisi esterna, reportistica, backup

**Export Dati Aziende**:
1. **Cliccare** pulsante "📥 Scarica CSV Aziende"
2. **Il file include**: Tutte le informazioni azienda e disponibilità
3. **Usare per**: Outreach partner, pianificazione capacità

### Verifica Qualità Dati

**Controlli Regolari**:
- Verificare che esclusioni candidati siano attuali e accurate
- Confermare disponibilità posizioni aziende
- Aggiornare informazioni geografiche se aziende si trasferiscono
- Rivedere descrizioni compatibilità per accuratezza

---

## ⚙️ Impostazioni di Configurazione

### Regolazioni Soglie

**Soglia Attitudine**:
- **Default**: 0.3 (30%)
- **Più Bassa (0.1-0.2)**: Include candidati con propensione minore
- **Più Alta (0.4-0.6)**: Focus su candidati più pronti al lavoro
- **Impatto**: Influisce sulla dimensione del pool candidati

**Soglia Distanza** (Distanza Max):
- **Default**: 30 km
- **Aree urbane**: 20-25 km per focus locale
- **Aree rurali**: 40-50 km per opzioni adeguate
- **Impatto**: Bilancia fattibilità tragitto vs varietà opportunità

**Top Raccomandazioni**:
- **Default**: 5 risultati
- **Meno (3)**: Decisioni rapide
- **Più (7-10)**: Valutazione comprensiva
- **Impatto**: Profondità analisi vs semplicità

### Configurazione Avanzata

**Selezione Modello** (se disponibile):
- Scegliere tra diversi modelli AI
- Ogni modello può avere punti di forza diversi
- La selezione default è solitamente ottimale

**Quando Regolare Impostazioni**:

**Espandere Ricerca** quando:
- Pochi o nessun risultato per candidati qualificati
- Località rurali con opzioni locali limitate
- Requisiti disabilità specializzati

**Restringere Ricerca** quando:
- Troppi match di bassa qualità
- Necessità di focus su collocamenti più probabili
- Vincoli temporali richiedono decisioni rapide

---

## 🛠️ Risoluzione Problemi

### Problemi Comuni e Soluzioni

#### Problema: Nessun Risultato Trovato
**Sintomi**: Appare messaggio "Nessuna azienda trovata"
**Soluzioni**:
1. **Aumentare soglia distanza** a 40-50 km
2. **Abbassare soglia attitudine** a 0.2-0.3
3. **Rivedere esclusioni** - assicurarsi che non siano eccessivamente restrittive
4. **Controllare località** - verificare che area candidato sia località italiana valida

#### Problema: Tutti i Punteggi Molto Bassi
**Sintomi**: Tutte le raccomandazioni sotto 60%
**Soluzioni**:
1. **Rivedere accuratezza esclusioni** - potrebbero essere troppo ampie o inserite incorrettamente
2. **Controllare descrizioni compatibilità** - aziende potrebbero necessitare informazioni aggiornate
3. **Considerare soglie più basse** - impostazioni attuali potrebbero essere troppo severe

#### Problema: Sistema Carica Lentamente
**Sintomi**: Interfaccia impiega >30 secondi per rispondere
**Soluzioni**:
1. **Aggiornare pagina** browser
2. **Cancellare cache browser** e ricaricare
3. **Controllare velocità connessione** internet
4. **Provare browser diverso** se problemi persistono

#### Problema: Errori Geografici
**Sintomi**: "Calcolo distanza fallito" o distanze irrealistiche
**Soluzioni**:
1. **Verificare formato indirizzo** - usare formato "Città, Provincia, Italia"
2. **Controllare ortografia** nomi città italiane
3. **Usare città maggiori** invece di piccoli paesi se problemi persistono

### Ottenere Supporto Tecnico

**Prima di Contattare Supporto**:
1. **Annotare messaggio errore esatto** se appare
2. **Registrare passi** che hanno portato al problema
3. **Controllare stato sistema** nella scheda "Info Sistema"
4. **Provare soluzioni base** elencate sopra

**Informazioni Contatto**:
- **Supporto Tecnico**: michele.melch@gmail.com
- **Supporto Accademico**: oleksandr.kuznetsov@uniecampus.it
- **Includere nell'email**: Screenshot, messaggi errore, passi per riprodurre

---

## 🎯 Migliori Pratiche

### Operazioni Quotidiane

**Routine Mattutina**:
1. **Controllare stato sistema** nella scheda Info Sistema
2. **Rivedere analytics** per cambiamenti notturni
3. **Verificare candidati prioritari** abbiano informazioni attuali

**Elaborazione Candidati**:
1. **Sempre verificare esclusioni** con candidato prima della ricerca
2. **Usare dati candidato esistente** quando disponibili per consistenza
3. **Documentare collocamenti riusciti** per miglioramento sistema

**Valutazione Risultati**:
1. **Focus su top 3 raccomandazioni** per outreach iniziale
2. **Considerare preferenze geografiche** anche con punteggi alti
3. **Rivedere dettagli compatibilità** oltre al solo punteggio

### Revisioni Settimanali

**Qualità Dati**:
- Aggiornare informazioni candidato basate su nuove valutazioni
- Verificare disponibilità posizioni azienda e requisiti
- Rimuovere o aggiornare aziende inattive

**Analisi Performance**:
- Rivedere pattern collocamento riuscito vs non riuscito
- Identificare aziende con successo collocamento più alto
- Notare problemi sistematici con raccomandazioni

### Integrazione con Workflow Esistente

**Integrazione CPI**:
1. **Usare sistema per screening iniziale** candidati
2. **Combinare con valutazione manuale** per decisioni finali
3. **Documentare outcome collocamento** per miglioramento continuo

**Coordinamento SIL**:
1. **Condividere raccomandazioni** con case manager
2. **Coordinare follow-up** su match alto punteggio
3. **Tracciare successo collocamento lungo termine**

### Assicurazione Qualità

**Validazione Raccomandazioni**:
- **Cross-check esclusioni** contro requisiti azienda manualmente per top match
- **Verificare informazioni azienda** prima di fare contatto
- **Confermare preferenze candidato** allineate con raccomandazioni

**Miglioramento Continuo**:
- **Tracciare tassi successo collocamento** per range punteggi
- **Reportare problemi sistematici** al team tecnico
- **Suggerire miglioramenti** basati su esperienza campo

---

## 📞 Supporto e Risorse

### Riferimento Rapido

**Scorciatoie Chiave**:
- **Navigazione Schede**: Usare schede browser per candidati multipli
- **Impostazioni Barra Laterale**: Regolare soglie senza ricarica pagina
- **Funzioni Export**: Disponibili nella scheda Dataset per tutti i dati

**Soglie Importanti**:
- **Attitudine**: 0.3 default (regolare basato su pool candidati)
- **Distanza**: 30 km default (espandere per aree rurali)
- **Compatibilità**: 50% minimo per collocamento fattibile

### Risorse Training

**Training Nuovo Utente**:
1. **Iniziare con modalità demo** per comprendere interfaccia
2. **Praticare con candidati test** prima di operazioni reali
3. **Rivedere questa guida** sezione per sezione

**Feature Avanzate**:
- **Interpretazione analytics** per pianificazione strategica
- **Ottimizzazione configurazione** per scenari diversi
- **Tecniche integrazione** con workflow CPI/SIL esistenti

### Feedback e Miglioramento

**Come Fornire Feedback**:
- **Email suggerimenti** a michele.melch@gmail.com
- **Reportare bug** con passi dettagliati riproduzione
- **Condividere storie successo** per aiutare migliorare sistema

**Quale Feedback Aiuta**:
- Outcome collocamento mondo reale vs raccomandazioni sistema
- Suggerimenti usabilità interfaccia
- Feature aggiuntive che migliorerebbero operazioni
- Sfide integrazione con sistemi esistenti

---

## 🔍 Scenari d'Uso Comuni

### Scenario 1: Candidato con Disabilità Motoria

**Situazione**: Mario, 35 anni, disabilità motoria, non può fare lavori in quota
**Passi**:
1. Inserire "Lavori in quota" nelle esclusioni
2. Impostare distanza max 25 km (mobilità limitata)
3. Cercare aziende con certificazione disability-friendly
4. Prioritizzare risultati con lavoro remoto disponibile

**Risultato Atteso**: Aziende ufficio, call center, servizi amministrativi

### Scenario 2: Candidato con Disabilità Intellettiva

**Situazione**: Giulia, 28 anni, disabilità intellettiva lieve, no mansioni responsabilità
**Passi**:
1. Inserire "Mansioni di responsabilità" nelle esclusioni
2. Impostare soglia attitudine 0.4 (propensione media)
3. Focus su settori con supporto/tutoraggio
4. Considerare aziende con programmi inclusione

**Risultato Atteso**: Magazzini, assemblaggio, servizi pulizia con supervisione

### Scenario 3: Area Rurale con Poche Opzioni

**Situazione**: Candidato in piccolo comune, poche aziende locali
**Passi**:
1. Espandere distanza max a 45-50 km
2. Abbassare soglia compatibilità se necessario
3. Considerare lavoro remoto come priorità
4. Valutare trasporto pubblico per aziende lontane

**Risultato Atteso**: Mix aziende locali + remote work + pendolarismo

---

## 📋 Checklist Operativa Quotidiana

### Inizio Giornata
- [ ] Verificare stato sistema operativo
- [ ] Controllare nuovi candidati da elaborare
- [ ] Rivedere posizioni aziende aggiornate
- [ ] Confermare impostazioni soglie appropriate

### Elaborazione Candidato
- [ ] Verificare completezza dati candidato
- [ ] Confermare esclusioni con candidato
- [ ] Eseguire ricerca con parametri appropriati
- [ ] Valutare top 3-5 raccomandazioni
- [ ] Documentare azioni intraprese

### Fine Giornata
- [ ] Aggiornare stato collocamenti in corso
- [ ] Salvare/esportare dati se necessario
- [ ] Annotare problemi o suggerimenti
- [ ] Pianificare follow-up giorno seguente

---

*Questa Guida Utente è progettata per aiutare i professionisti del collocamento a massimizzare l'efficacia del Sistema di Raccomandazione per Collocamento Mirato. Per supporto aggiuntivo o domande specifiche sulla vostra implementazione, contattare il team di sviluppo.*

---

**Versione Documento**: 1.0  
**Ultimo Aggiornamento**: Giugno 2025  
**Prossima Revisione**: Dicembre 2025
