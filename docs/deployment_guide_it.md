# Guida al Deployment — Sistema di Collocamento Mirato
_Ultimo aggiornamento: 2025-08-23 23:52_

> **Ambito.** Questo documento descrive come distribuire il Sistema di Collocamento Mirato in ambienti di sviluppo, staging e produzione. Copre setup su singolo nodo, containerizzazione e topologie di apprendimento federato cross‑silo con opzioni di privacy e anchoring di integrità. È complementare al README e alla Documentazione Tecnica.

---

## Indice

1. [Destinatari & Assunzioni](#destinatari--assunzioni)  
2. [Requisiti di Sistema](#requisiti-di-sistema)  
3. [Architettura di Deployment](#architettura-di-deployment)  
4. [Struttura Progetto & Percorsi](#struttura-progetto--percorsi)  
5. [Ambienti](#ambienti)  
6. [Configurazione](#configurazione)  
7. [Deployment Locale (Bare‑Metal)](#deployment-locale-baremetal)  
8. [Deployment con Docker](#deployment-con-docker)  
9. [Reverse Proxy & TLS (opzionale)](#reverse-proxy--tls-opzionale)  
10. [Topologie di FL](#topologie-di-fl)  
11. [FL con Privacy (Shamir + DP)](#fl-con-privacy-shamir--dp)  
12. [Blockchain Anchoring](#blockchain-anchoring)  
13. [Schedulazione & Automazione](#schedulazione--automazione)  
14. [Monitoraggio, Log & Health Check](#monitoraggio-log--health-check)  
15. [Sicurezza & Operazioni GDPR](#sicurezza--operazioni-gdpr)  
16. [Performance & Capacity Planning](#performance--capacity-planning)  
17. [Backup, DR & Rollback](#backup-dr--rollback)  
18. [Troubleshooting](#troubleshooting)  
19. [Procedura di Upgrade](#procedura-di-upgrade)  
20. [Appendice A — File di Esempio](#appendice-a--file-di-esempio)  
21. [Appendice B — Kubernetes (Avanzato, Opzionale)](#appendice-b--kubernetes-avanzato-opzionale)  
22. [Appendice C — Guida Legacy (testo integrale)](#appendice-c--guida-legacy-testo-integrale)

---

## 1) Destinatari & Assunzioni

- **Destinatari**: DevOps/ML engineer e personale IT del settore pubblico che distribuisce per CPI/SIL.  
- **Assunzioni**:
  - Conoscenza base di Python, Docker e gestione servizi Linux.
  - Spazio di archiviazione disponibile per dataset e risultati.
  - Connettività di rete tra sedi (per FL) o uso di scambio file sicuro.

---

## 2) Requisiti di Sistema

**Sistemi Operativi**  
- Linux: Ubuntu 22.04/24.04 LTS (consigliato)  
- Windows 10/11 (sviluppo e PoC)  
- macOS 13+ (solo sviluppo)

**Hardware (baseline)**  
- CPU: 4 core; RAM: 16 GB; Disco: 50 GB  
- Per round FL o visualizzazioni pesanti: 8 core / 32 GB  
- GPU: **non richiesta**

**Rete**  
- Porta Streamlit: `8501`  
- Proxy inverso (80/443) opzionale  
- Per FL, vedere [Topologie](#topologie-di-fl)

---

## 3) Architettura di Deployment

Strati in produzione:

1. **Applicazione/UI** — Streamlit (`streamlit_app.py`).  
2. **Servizi di Training** — script `scripts/` schedulati (cron/Task Scheduler/systemd).  
3. **Coordinatore Federato** — host/processo che orchestra i round e raccoglie gli update.  
4. **Privacy** — secure aggregation (Shamir) e accounting DP.  
5. **Integrità** — commit Merkle e verifica delle prove.  
6. **Storage** — cartelle `data/` e `results*/`.  
7. **Osservabilità** — log, health endpoint, metriche CSV.

### Diagramma minimo (singolo host)

```
[ Operatori ] --> [ Streamlit (8501) ] --> [ Script: 01..10 ] --> [ results/* ]
                                   ^                                  
                                   +---- health: /_stcore/health
```

### FL cross‑silo (scambio file)

```
 Regione A         Regione B           Regione C            Coordinatore
+---------+       +---------+         +---------+          +-------------+
| ML loc. |       | ML loc. |         | ML loc. |          | aggregaz.   |
| script  |       | script  |         | script  |          | + anchoring |
+---------+       +---------+         +---------+          +-------------+
     \               |                    /                     ^
      \              |                   /                      |
       +----- scambio file sicuro / SFTP / share di rete -------+
```

---

## 4) Struttura Progetto & Percorsi

```
[vedi layout nel README principale: identico a quello del progetto]
```

---

## 5) Ambienti

- **Sviluppo**: macchina locale, campioni piccoli.  
- **Staging**: replica produzione (stesso OS e pacchetti), dataset anonimizzati.  
- **Produzione**: dipendenze fissate, `config.yaml` stabile, job pianificati, accessi minimi.

---

## 6) Configurazione

Creare e versionare `config.yaml`:

```yaml
seed: 42
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies: data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir: data/processed
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
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

---

## 7) Deployment Locale (Bare‑Metal)

**Prerequisiti**
```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Esecuzione UI**
```bash
streamlit run streamlit_app.py
# http://localhost:8501
```

**Preparazione dati**
```bash
python scripts/01_generate_dataset.py
python scripts/02_visualize_dataset.py
```

**Training centralizzato & analisi**
```bash
python scripts/03_train_models.py --config config.yaml
python scripts/04_analyze_results.py
```

---

## 8) Deployment con Docker

### 8.1 Build immagine
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
USER 1000
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build:
```bash
docker build -t djms:latest .
```

### 8.2 Run (singolo container)
```bash
docker run -d --name djms   -p 8501:8501   -v $PWD/data:/app/data   -v $PWD/results:/app/results   -v $PWD/results_LightGBM_federated:/app/results_LightGBM_federated   -v $PWD/results_mlp_federated:/app/results_mlp_federated   -v $PWD/results_mlp_federated_privacy:/app/results_mlp_federated_privacy   -v $PWD/results_blockchain_demo:/app/results_blockchain_demo   djms:latest
```

### 8.3 docker‑compose
```yaml
version: "3.9"
services:
  ui:
    image: djms:latest
    container_name: djms_ui
    ports: ["8501:8501"]
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 5s
      retries: 5
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./results_LightGBM_federated:/app/results_LightGBM_federated
      - ./results_mlp_federated:/app/results_mlp_federated
      - ./results_mlp_federated_privacy:/app/results_mlp_federated_privacy
      - ./results_blockchain_demo:/app/results_blockchain_demo
    environment:
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 9) Reverse Proxy & TLS (opzionale)

Esempio NGINX:
```nginx
server {{
  listen 80;
  server_name djms.esempio.it;
  location / {{
    proxy_pass http://127.0.0.1:8501/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }}
}}
```

---

## 10) Topologie di FL

### 10.1 Coordinamento su file (consigliato per i piloti)
- Ogni sede esegue `07_mlp_federated_training.py` sui propri dati.  
- Dopo le epoche locali, esporta gli update su share sicura.  
- Il coordinatore aggrega e pubblica il modello globale.  
- Ripetere per `federated.rounds`.

**Coordinatore**
```bash
python scripts/07_mlp_federated_training.py --mode aggregate --round 1
```

**Client**
```bash
python scripts/07_mlp_federated_training.py --mode client --round 1
```

### 10.2 Emulazione singolo host
Esecuzione sequenziale dei client; aggregazione in‑process.

---

## 11) FL con Privacy (Shamir + DP)

```bash
python scripts/08_mlp_federated_privacy.py   --dp.epsilon 1.0 --dp.delta 1e-6 --dp.max_grad_norm 1.0   --secure_agg.threshold 3-of-5 --round 1 --mode client
```

Coordinatore:
```bash
python scripts/08_mlp_federated_privacy.py --round 1 --mode aggregate
```

---

## 12) Blockchain Anchoring

- `blockchain_data_anchoring.py` per generare commit/prove;  
- `10_blockchain_anchoring_bench.py` per benchmark;  
- Artefatti in `results_blockchain_demo/`.

---

## 13) Schedulazione & Automazione

### 13.1 Cron
```cron
0 22 * * 1-5  /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/03_train_models.py --config config.yaml >> logs/train.log 2>&1'
30 22 * * 1-5 /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/04_analyze_results.py >> logs/analyze.log 2>&1'
0 23 * * 1-5  /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/blockchain_data_anchoring.py >> logs/anchor.log 2>&1'
```

### 13.2 Windows
```
cmd.exe /c "cd C:\djms && venv\Scripts\activate && python scripts\03_train_models.py --config config.yaml"
```

---

## 14) Monitoraggio, Log & Health Check

- Health: `/_stcore/health` (200).  
- Log: `logs/*.log` con rotazione settimanale.  
- Metriche: CSV in `results/` e `results_*/`.  
- Dashboard: `*_visualization.py`.

---

## 15) Sicurezza & GDPR

- Accessi minimi a `data/` e `results*/`.  
- Minimizzazione dati e retention.  
- In FL, condividere solo delta modello; soglia Shamir rispettata.

---

## 16) Performance & Capacity Planning

- Tarare batch e round.  
- LightGBM con Optuna; MLP batch 256/LR 1e-3.  
- SSD per risultati; compressione archivi.  
- Modalità privacy: validare in staging.

---

## 17) Backup, DR & Rollback

- Backup giornalieri; snapshot cifrati; immagini versionate.  
- Test periodici di restore su staging.

---

## 18) Troubleshooting

- Moduli mancanti → `pip install -r requirements.txt`.  
- Errori sklearn `_safe_tags` → allineare versioni.  
- Health check → installare `curl`.  
- CSV vuoti → completare i job di training.

---

## 19) Procedura di Upgrade

1. Finestra manutenzione e backup.  
2. Pull e revisione `requirements.txt`.  
3. Rebuild su staging; smoke test.  
4. Deploy in produzione; monitoraggio; rollback se necessario.

---

## 20) Appendice A — File di Esempio

### 20.1 `systemd` (UI)

```ini
[Unit]
Description=DJMS Streamlit UI
After=network.target

[Service]
Type=simple
User=djms
WorkingDirectory=/opt/djms
ExecStart=/opt/djms/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### 20.2 Logrotate

```
/opt/djms/logs/*.log {{
  weekly
  rotate 8
  compress
  missingok
  notifempty
  copytruncate
}}
```

### 20.3 NGINX
Vedi Sezione 9; completare con TLS secondo policy di ente.

---

## 21) Appendice B — Kubernetes (Avanzato, Opzionale)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: djms-ui
spec:
  replicas: 1
  selector: {{ matchLabels: {{ app: djms-ui }} }}
  template:
    metadata: {{ labels: {{ app: djms-ui }} }}
    spec:
      containers:
        - name: ui
          image: djms:latest
          ports: [{{containerPort: 8501}}]
          readinessProbe:
            httpGet: {{ path: /_stcore/health, port: 8501 }}
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: results
              mountPath: /app/results
      volumes:
        - name: data
          hostPath: {{ path: /opt/djms/data }}
        - name: results
          hostPath: {{ path: /opt/djms/results }}
---
apiVersion: v1
kind: Service
metadata:
  name: djms-ui-svc
spec:
  selector: {{ app: djms-ui }}
  ports:
    - port: 80
      targetPort: 8501
```

---

## 22) Appendice C — Guida Legacy (testo integrale)

# 🚀 Guida Deployment - Sistema di Raccomandazione per Collocamento Mirato

**Guida Completa Installazione e Configurazione per Ambienti di Produzione**

---

## 📋 Panoramica

Questa guida deployment fornisce istruzioni step-by-step per installare e configurare il Sistema di Raccomandazione per Collocamento Mirato in ambienti di produzione. La guida copre sia modalità demo (dati sintetici) che modalità produzione (dati reali collocamento).

### Modalità Deployment

- **🧪 Modalità Demo**: Utilizza dati sintetici per test e dimostrazione
- **🏭 Modalità Produzione**: Utilizza dati reali outcome collocamento per operazioni live
- **🔧 Modalità Sviluppo**: Ambiente sviluppo completo con tutti gli script

---

## 📋 Prerequisiti

### Requisiti Sistema

**Requisiti Minimi**:
- **Sistema Operativo**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8 o superiore (3.11 raccomandato)
- **RAM**: 8GB minimo (16GB raccomandati per training)
- **Storage**: 5GB spazio libero
- **CPU**: Processore multi-core (4+ core raccomandati)

**Requisiti Rete**:
- **Connessione Internet**: Richiesta per setup geocodifica iniziale
- **Firewall**: Porta 8501 (Streamlit default) o porta personalizzata
- **API Geografica**: Accesso al servizio geocodifica Nominatim

### Dipendenze Software

**Software Richiesto**:
```bash
# Python 3.8+ con pip
python --version  # Dovrebbe mostrare 3.8 o superiore
pip --version

# Git (per clonazione repository)
git --version

# Opzionale: strumenti ambiente virtuale
python -m venv --help
```

---

## 🔧 Metodi Installazione

### Metodo 1: Setup Demo Rapido (Raccomandato per Test)

**Passo 1: Clonare Repository**
```bash
# Clonare il progetto
git clone https://github.com/your-username/disability-job-matching.git
cd disability-job-matching

# Verificare struttura progetto
ls -la
# Dovrebbe mostrare: streamlit_app.py, config.yaml, requirements.txt, data/, scripts/, utils/
```

**Passo 2: Installare Dipendenze**
```bash
# Creare ambiente virtuale (raccomandato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installare requirements
pip install -r requirements.txt

# Verificare installazione
python -c "import streamlit, pandas, numpy, sklearn; print('Dipendenze OK')"
```

**Passo 3: Lanciare Demo**
```bash
# Avviare applicazione in modalità demo
streamlit run streamlit_app.py

# Il sistema automaticamente:
# 1. Carica con dati demo sintetici
# 2. Apre browser su http://localhost:8501
# 3. Mostra "Modalità Demo" nell'interfaccia
```

**Verifica**:
- Interfaccia carica con successo
- Scheda Analytics mostra dati esempio
- Si possono eseguire ricerche candidato
- Risultati appaiono entro 5 secondi

### Metodo 2: Setup Produzione (Dati Reali)

**Prerequisiti per Produzione**:
- Dati storici outcome collocamento
- File CSV formattati appropriatamente
- Approvazione conformità privacy dati

**Passo 1: Preparare Dati Produzione**

```bash
# Struttura dati produzione richiesta:
data/
├── raw/
│   ├── Dataset_Candidati_Aggiornato.csv      # Dati candidati reali
│   └── Dataset_Aziende_con_Stima_Assunzioni.csv  # Dati aziende reali
└── processed/
    └── Enhanced_Training_Dataset.csv         # Outcome collocamento reali
```

**Formato Enhanced_Training_Dataset.csv**:
```csv
outcome,attitude_score,years_experience,unemployment_duration,compatibility_score,distance_km,company_size,retention_rate,remote_work,certification,...
1,0.75,5,12,0.85,15.2,150,0.78,1,1,...
0,0.45,2,24,0.35,45.8,50,0.65,0,0,...
```

**Passo 2: Validazione Dati**
```bash
# Validare formato dati
python -c "
import pandas as pd
df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
print(f'Dati training: {df.shape[0]} righe, {df.shape[1]} colonne')
print(f'Distribuzione outcome: {df.outcome.value_counts()}')
print('Colonne richieste presenti:', all(col in df.columns for col in ['outcome', 'attitude_score', 'compatibility_score']))
"
```

**Passo 3: Addestrare Modelli Produzione**
```bash
# Addestrare modelli su dati reali
python scripts/03_train_models.py

# Output atteso:
# 📥 Loading training dataset...
# 🧹 Preparing data...
# 🎯 Optimizing hyperparameters...
# 🤖 Training [Nome Modello]...
# ✅ All models saved

# Verificare modelli creati
ls -la results/
# Dovrebbe mostrare: file *.joblib, metrics_summary.csv
```

**Passo 4: Lanciare Interfaccia Produzione**
```bash
# Avviare applicazione produzione
streamlit run streamlit_app.py

# Il sistema:
# 1. Rileva dati reali automaticamente
# 2. Carica modelli addestrati
# 3. Abilita feature produzione
```

### Metodo 3: Deployment Docker (Enterprise)

**Creare Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installare dipendenze sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiare requirements e installare dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiare applicazione
COPY . .

# Creare utente non-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Esporre porta
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Avviare applicazione
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Comandi Docker**:
```bash
# Build immagine
docker build -t disability-job-matcher .

# Eseguire container (modalità demo)
docker run -p 8501:8501 disability-job-matcher

# Eseguire con volume dati reali
docker run -p 8501:8501 -v /path/to/real/data:/app/data disability-job-matcher
```

---

## ⚙️ Configurazione

### Configurazione Ambiente

**Personalizzazione config.yaml**:
```yaml
# Esempio configurazione produzione
paths:
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"
  logs_dir: "logs"

matching_thresholds:
  attitude_min: 0.3           # Regolare basato su pool candidati
  distance_max_km: 30         # Regolare per deployment urbano/rurale
  match_probability_cutoff: 0.6

model_training:
  n_jobs: 4                   # Regolare basato su core CPU server
  optuna_trials: 50           # Ridurre per training veloce, aumentare per accuratezza

geocoding:
  delay: 0.5                  # Aumentare se si colpiscono rate limit
  cache_file: "data/processed/geocoding_cache.json"

streamlit:
  page_title: "Sistema Collocamento Mirato - [Nome Organizzazione]"
  default_top_k: 5
```

**Variabili Ambiente**:
```bash
# Configurazione ambiente opzionale
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### Configurazione Rete

**Impostazioni Firewall**:
```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload

# Verificare accessibilità porta
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

## 🔒 Configurazione Sicurezza

### Controllo Accesso

**Autenticazione Base (Streamlit)**:
```python
# Aggiungere a streamlit_app.py per protezione base
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "password_sicura":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorretta")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

**Permessi File**:
```bash
# Permessi file sicuri
chmod 600 config.yaml                    # File configurazione
chmod 600 data/processed/*.csv           # File dati  
chmod 755 scripts/*.py                   # Script eseguibili
chmod 644 requirements.txt               # File pubblici

# Directory sicure
chmod 750 data/                          # Directory dati
chmod 750 results/                       # Directory modelli
chmod 755 utils/                         # Directory codice
```

### Protezione Dati

**Gestione Dati Sensibili**:
```bash
# Creare directory dati sicura
sudo mkdir -p /opt/job-matcher/secure-data
sudo chown app-user:app-group /opt/job-matcher/secure-data
sudo chmod 750 /opt/job-matcher/secure-data

# Symlink all'applicazione
ln -s /opt/job-matcher/secure-data data/processed
```

**Strategia Backup**:
```bash
# Script backup automatizzato
#!/bin/bash
BACKUP_DIR="/opt/backups/job-matcher"
DATE=$(date +%Y%m%d_%H%M%S)

# Creare directory backup
mkdir -p $BACKUP_DIR/$DATE

# Backup file critici
cp -r data/processed $BACKUP_DIR/$DATE/
cp -r results/ $BACKUP_DIR/$DATE/
cp config.yaml $BACKUP_DIR/$DATE/

# Comprimere e crittografare
tar -czf $BACKUP_DIR/$DATE.tar.gz $BACKUP_DIR/$DATE
rm -rf $BACKUP_DIR/$DATE

echo "Backup completato: $BACKUP_DIR/$DATE.tar.gz"
```

---

## 📊 Monitoraggio e Logging

### Monitoraggio Applicazione

**Endpoint Health Check**:
```python
# Aggiungere a streamlit_app.py
import streamlit as st
import time

def health_check():
    """Controllo salute sistema"""
    checks = {
        'models_loaded': len(st.session_state.get('models', {})) > 0,
        'data_available': os.path.exists('data/processed/Enhanced_Training_Dataset.csv'),
        'config_valid': os.path.exists('config.yaml')
    }
    return all(checks.values()), checks

# Utilizzo nell'app
if st.sidebar.button("Health Check"):
    healthy, details = health_check()
    if healthy:
        st.success("✅ Sistema Sano")
    else:
        st.error("❌ Problemi Sistema Rilevati")
        st.json(details)
```

**Monitoraggio Performance**:
```python
# Aggiungere tracking performance
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        
    def log_request(self, operation, duration):
        self.request_count += 1
        logging.info(f"Operazione: {operation}, Durata: {duration:.2f}s, "
                    f"CPU: {psutil.cpu_percent()}%, "
                    f"Memoria: {psutil.virtual_memory().percent}%")

# Utilizzo
monitor = PerformanceMonitor()
start = time.time()
# ... eseguire operazione matching ...
monitor.log_request("candidate_matching", time.time() - start)
```

### Configurazione Logging

**Setup Logging**:
```python
# logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Creare directory logs
    os.makedirs('logs', exist_ok=True)
    
    # Configurare logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Utilizzo nell'applicazione principale
logger = setup_logging()
logger.info("Applicazione avviata")
```

**Rotazione Log**:
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

## 🔄 Procedure Manutenzione

### Attività Manutenzione Regolare

**Attività Quotidiane**:
```bash
#!/bin/bash
# daily_maintenance.sh

echo "$(date): Avvio manutenzione quotidiana"

# Controllare spazio disco
df -h | grep -E "/(dev|opt)" | awk '$5 > "80%" {print "WARNING: " $0}'

# Controllare salute applicazione
curl -f http://localhost:8501/_stcore/health || echo "WARNING: Health check app fallito"

# Validare integrità dati
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
    print(f'Dati OK: {df.shape[0]} record')
except Exception as e:
    print(f'Errore Dati: {e}')
"

# Cancellare vecchi file cache
find data/processed -name "*.cache" -mtime +7 -delete

echo "$(date): Manutenzione quotidiana completata"
```

**Attività Settimanali**:
```bash
#!/bin/bash
# weekly_maintenance.sh

# Aggiornare cache geocodifica se necessario
python -c "
import json
try:
    with open('data/processed/geocoding_cache.json', 'r') as f:
        cache = json.load(f)
    print(f'Cache geocodifica: {len(cache)} voci')
except:
    print('Nessuna cache geocodifica trovata')
"

# Controllare performance modelli
python scripts/04_analyze_results.py

# Backup modelli e configurazione
cp -r results/ /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
cp config.yaml /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
```

### Aggiornamenti Modelli

**Quando Riaddestare Modelli**:
- Nuovi dati outcome collocamento disponibili (raccomandato trimestrale)
- Cambiamenti significativi in demographics candidati/aziende
- Degradazione performance osservata
- Cambiamenti configurazione sistema

**Processo Riaddestramento**:
```bash
# 1. Backup modelli attuali
cp -r results/ results_backup_$(date +%Y%m%d)/

# 2. Aggiornare dati training
# Posizionare nuovo Enhanced_Training_Dataset.csv in data/processed/

# 3. Riaddestramento modelli
python scripts/03_train_models.py

# 4. Validare nuovi modelli
python scripts/04_analyze_results.py

# 5. Test con candidati esempio
streamlit run streamlit_app.py

# 6. Se soddisfatti, rimuovere backup
# rm -rf results_backup_*
```

---

## 🛠️ Risoluzione Problemi

### Problemi Comuni

**Problema 1: Applicazione Non Si Avvia**
```bash
# Sintomi: ModuleNotFoundError, errori import
# Soluzione: Verificare dipendenze
pip list | grep -E "(streamlit|pandas|numpy|sklearn)"
pip install -r requirements.txt --upgrade

# Controllare versione Python
python --version  # Dovrebbe essere 3.8+
```

**Problema 2: Nessun Modello Trovato**
```bash
# Sintomi: Warning "Nessun modello caricato"
# Soluzione: Addestrare modelli
ls -la results/  # Controllare se esistono file .joblib
python scripts/03_train_models.py  # Addestrare se mancanti
```

**Problema 3: Fallimenti Geocodifica**
```bash
# Sintomi: Errori calcolo distanza
# Soluzione: Controllare rete e cache
ping nominatim.openstreetmap.org
ls -la data/processed/geocoding_cache.json

# Reset cache se corrotto
rm data/processed/geocoding_cache.json
```

**Problema 4: Performance Scarse**
```bash
# Sintomi: Tempi risposta lenti
# Soluzioni:
# 1. Controllare risorse sistema
htop  # o top
free -h

# 2. Ridurre job paralleli
# Modificare config.yaml: model_training.n_jobs: 2

# 3. Cancellare cache browser
# 4. Riavviare applicazione
```

### Analisi Log Errori

**Pattern Errori Comuni**:
```bash
# Controllare errori recenti
tail -100 logs/app_$(date +%Y%m%d).log | grep ERROR

# Errori geocodifica
grep "Geocoding error" logs/*.log

# Errori predizione modelli  
grep "prediction error" logs/*.log

# Problemi memoria
grep -i "memory\|oom" logs/*.log
```

### Ottimizzazione Performance

**Ottimizzazione Memoria**:
```python
# Aggiungere a streamlit_app.py
import gc

@st.cache_data(ttl=3600)  # Cache per 1 ora
def load_cached_data():
    # Caricare operazioni dati pesanti
    pass

# Cancellare cache quando necessario
if st.button("Cancella Cache"):
    st.cache_data.clear()
    gc.collect()
    st.success("Cache cancellata")
```

**Ottimizzazione CPU**:
```yaml
# Regolazioni config.yaml per sistemi lenti
model_training:
  n_jobs: 2                 # Ridurre da 4
  optuna_trials: 25         # Ridurre da 50

matching_thresholds:
  distance_max_km: 25       # Ridurre spazio ricerca
```

---

## 📞 Supporto e Escalation

### Contatti Supporto

**Problemi Tecnici**:
- **Primario**: michele.melch@gmail.com
- **Accademico**: oleksandr.kuznetsov@uniecampus.it
- **Emergenza**: Includere "URGENTE" nell'oggetto email

**Informazioni da Includere**:
```bash
# Informazioni sistema
uname -a
python --version
pip list > pacchetti_installati.txt

# Log errori
tail -50 logs/app_$(date +%Y%m%d).log

# Configurazione
cat config.yaml

# Risorse sistema
free -h && df -h
```

### Procedure Escalation

**Livello 1: Problemi Applicazione**
- Controllare log e problemi comuni
- Riavviare applicazione
- Verificare configurazione

**Livello 2: Problemi Dati/Modelli**
- Validare integrità dati
- Riaddestramento modelli se necessario
- Controllare aggiornamenti dati

**Livello 3: Problemi Sistema/Infrastruttura**
- Contattare amministratore sistema
- Controllare connettività rete
- Rivedere log sicurezza

### Risorse Community

**Documentazione**:
- Repository GitHub: [Link al repository]
- Guida Utente: `docs/user_guide_italiano.md`
- Documentazione Tecnica: `docs/documentazione_tecnica_it.md`

**Aggiornamenti e Annunci**:
- GitHub Releases per aggiornamenti versione
- Notifiche email per aggiornamenti sicurezza critici

---

## 🎯 Configurazione Specifica CPI/SIL

### Integrazione con Sistemi Esistenti

**Esportazione Dati per Sistemi CPI**:
```python
# Script export per sistemi CPI esistenti
import pandas as pd
from datetime import datetime

def export_recommendations_for_cpi(recommendations, candidate_id):
    """Esporta raccomandazioni in formato CPI standard"""
    export_data = []
    for rec in recommendations:
        export_data.append({
            'ID_Candidato': candidate_id,
            'Nome_Azienda': rec['Nome Azienda'],
            'Score_Compatibilità': rec['Score Finale'],
            'Distanza_KM': rec['Distanza (km)'],
            'Settore': rec['Tipo di Attività'],
            'Data_Raccomandazione': datetime.now().strftime('%Y-%m-%d'),
            'Operatore': 'Sistema_AI'
        })
    
    df = pd.DataFrame(export_data)
    filename = f"raccomandazioni_{candidate_id}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(f"exports/{filename}", index=False)
    return filename
```

**Configurazione Multi-CPI**:
```yaml
# config_multi_cpi.yaml
organizations:
  cpi_villafranca:
    name: "CPI Villafranca di Verona"
    distance_max_km: 25        # Area urbana
    attitude_min: 0.4          # Soglia più alta
    contact: "dott.rotolani@cpi-villafranca.it"
  
  cpi_legnago:
    name: "CPI Legnago"  
    distance_max_km: 40        # Area rurale
    attitude_min: 0.3          # Soglia standard
    contact: "info@cpi-legnago.it"

  sil_veneto:
    name: "SIL Regione Veneto"
    distance_max_km: 50        # Copertura regionale
    attitude_min: 0.2          # Soglia più bassa per supporto
    contact: "sil@regione.veneto.it"
```

### Workflow Specifici Italiani

**Integrazione Legge 68/99**:
```python
# Controlli conformità Legge 68/99
def check_legge68_compliance(company_data, candidate_data):
    """Verifica conformità requisiti Legge 68/99"""
    checks = {
        'quota_obbligatoria': company_data['Numero Dipendenti'] >= 15,
        'posizioni_disponibili': company_data['Posizioni Aperte'] > 0,
        'certificazione_disabilità': candidate_data['Tipo di Disabilità'] in [
            'Motoria', 'Sensoriale', 'Intellettiva', 'Psichica'
        ],
        'percentuale_invalidità': candidate_data.get('Percentuale_Invalidità', 0) >= 46
    }
    
    return all(checks.values()), checks
```

---

*Questa guida deployment fornisce istruzioni complete per installare e mantenere il Sistema di Raccomandazione per Collocamento Mirato. Per domande deployment specifiche organizzazione o necessità configurazione personalizzata, contattare il team di sviluppo.*

---

**Versione Documento**: 1.0  
**Ultimo Aggiornamento**: Giugno 2025  
**Prossima Revisione**: Dicembre 2025
