# Guida Deployment — Sistema di Matching per Collocamento Mirato
_Ultimo aggiornamento: 2025-08-24 16:00_

> **Scopo.** Questo documento descrive come deployare il Sistema di Matching per Collocamento Mirato in ambienti di sviluppo, staging e produzione. Copre deployment single-node, setup containerizzati e topologie di training federato cross-silo con opzioni privacy-preserving e ancoraggio integrità. Tutte le aspettative prestazioni sono state aggiornate per corrispondere ai risultati sperimentali.

---

## Indice

1. [Audience e Assunzioni](#1-audience-e-assunzioni)  
2. [Requisiti Sistema](#2-requisiti-sistema)  
3. [Architettura per Deployment](#3-architettura-per-deployment)  
4. [Layout Progetto e Percorsi Runtime](#4-layout-progetto-e-percorsi-runtime)  
5. [Ambienti](#5-ambienti)  
6. [Configurazione](#6-configurazione)  
7. [Deployment Locale (Bare-Metal)](#7-deployment-locale-bare-metal)  
8. [Deployment Docker](#8-deployment-docker)  
9. [Reverse Proxy e TLS](#9-reverse-proxy-e-tls)  
10. [Topologie Apprendimento Federato](#10-topologie-apprendimento-federato)  
11. [FL Preserva-Privacy (Shamir + DP)](#11-fl-preserva-privacy-shamir--dp)  
12. [Ancoraggio Dati Blockchain](#12-ancoraggio-dati-blockchain)  
13. [Scheduling e Automazione](#13-scheduling-e-automazione)  
14. [Monitoraggio, Logging e Health Check](#14-monitoraggio-logging-e-health-check)  
15. [Sicurezza e Operazioni GDPR](#15-sicurezza-e-operazioni-gdpr)  
16. [Prestazioni e Capacity Planning](#16-prestazioni-e-capacity-planning)  
17. [Backup, DR e Rollback](#17-backup-dr-e-rollback)  
18. [Risoluzione Problemi](#18-risoluzione-problemi)  
19. [Procedura Upgrade](#19-procedura-upgrade)  
20. [Appendice A — File Esempio](#20-appendice-a--file-esempio)  

---

## 1) Audience e Assunzioni

- **Audience**: Ingegneri DevOps, ingegneri ML e staff IT settore pubblico che deployano per piloti CPI/SIL o produzione.  
- **Assunzioni**:
  - Familiarità base con ambienti Python, Docker e gestione servizi Linux.
  - Capacità provisioning storage per dataset e risultati.
  - Connettività rete disponibile tra siti (per FL) o capacità scambio file sicuro.

---

## 2) Requisiti Sistema

**Sistemi Operativi**  
- Linux: Ubuntu 22.04/24.04 LTS (raccomandato)  
- Windows 10/11 (supportato per sviluppo e PoC)  
- macOS 13+ (solo sviluppo)

**Hardware (baseline)**  
- **Minimo**: CPU 4 core, RAM 16 GB, Disco 50 GB  
- **Raccomandato**: CPU 8 core, RAM 32 GB, SSD 100 GB  
- **GPU**: **Non richiesta** (training CPU supportato e validato)

**Aspettative Prestazioni (corrette)**
- **LightGBM Federato**: Impatto prestazioni minimo (-0.0005 F1-score vs centralizzato)
- **MLP Federato**: Impatto moderato (~4% riduzione F1-score vs centralizzato)
- **Modalità Privacy**: Riduzione F1-score aggiuntiva ~0.8% oltre federato standard

**Networking**  
- Porta Streamlit default: `8501`  
- Reverse proxy opzionale (80/443) se esposto esternamente  
- Per FL, vedere [Topologie](#10-topologie-apprendimento-federato) per opzioni scambio dati

---

## 3) Architettura per Deployment

Livelli deployment produzione:

1. **Applicazione/UI** — App Streamlit (`streamlit_app.py`) per operatori  
2. **Servizi Training** — script sotto `scripts/` orchestrati da scheduler  
3. **Coordinatore Federato** — host/processo coordinatore per round FL  
4. **Livello Privacy** — aggregazione sicura (Shamir) e contabilità budget DP  
5. **Livello Integrità** — commitment Merkle e verifica prove  
6. **Storage** — cartelle `data/` e `results*/` (locali o network-attached)  
7. **Osservabilità** — log, endpoint salute e metriche

### Diagramma single-host minimo

```
[ Operatori ] --> [ Streamlit (8501) ] --> [ Script: 01..10 ] --> [ results/* ]
                                 ^                                  
                                 +---- salute: /_stcore/health
```

### FL cross-silo (scambio file)

```
 Regione A          Regione B           Regione C            Coordinatore
+----------+      +----------+        +----------+         +--------------+
| ML locale|      | ML locale|        | ML locale|         | aggregazione |
| script   |      | script   |        | script   |         | + ancoraggio |
+----------+      +----------+        +----------+         +--------------+
     \                |                   /                     ^
      \               |                  /                      |
       +------ scambio file sicuro / SFTP / drive condiviso ----+
```

---

## 4) Layout Progetto e Percorsi Runtime

Struttura progetto (corretta):

```
📂 Sistema Matching Collocamento Mirato/
├── README.md / README_IT.md
├── config.yaml                           # Configurazione core
├── requirements.txt
├── streamlit_app.py                      # Applicazione principale
├── data/
│   ├── raw/
│   │   ├── Dataset_Candidati_Aggiornato.csv
│   │   └── Dataset_Aziende_con_Stima_Assunzioni.csv
│   └── processed/
│       ├── Dataset_Candidati_Aggiornato_Extended.csv
│       ├── Dataset_Aziende_con_Stima_Assunzioni_Extended.csv
│       └── Enhanced_Training_Dataset.csv
├── scripts/
│   ├── 01_generate_dataset.py
│   ├── 02_visualize_dataset.py
│   ├── 03_train_models.py
│   ├── 04_analyze_results.py
│   ├── 05_LightGBM_federated_training.py
│   ├── 06_LightGBM_federated_visualization.py
│   ├── 07_mlp_federated_training.py
│   ├── 08_mlp_federated_privacy.py
│   ├── 09_mlp_federated_privacy_visualization.py
│   ├── blockchain_data_anchoring.py
│   └── 10_blockchain_anchoring_bench.py
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── scoring.py
│   ├── parallel_training.py
│   ├── visualization.py
│   └── enhanced_shamir_privacy.py
├── results/                              # Output modelli centralizzati
├── results_LightGBM_federated/          # Risultati LightGBM federati
├── results_mlp_federated/               # Risultati MLP federati standard
├── results_mlp_federated_privacy/       # Risultati MLP preserva-privacy
├── results_blockchain_demo/             # Risultati ancoraggio blockchain
└── visualizations_federated_comparison/ # Grafici analisi comparativa
```

---

## 5) Ambienti

- **Sviluppo**: Workstation locale, campioni piccoli, run iterativi  
- **Staging**: Replica produzione (stesso OS/Python), dataset anonimizzati  
- **Produzione**: Dipendenze bloccate, config stabile, job schedulati, accesso limitato

---

## 6) Configurazione

Creare e versionare `config.yaml` nella root repository (valori corretti):

```yaml
seed: 42
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies: data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir: data/processed
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30                    # Raggio ricerca default (CORRETTO)
training:
  model_set: ["LightGBM_Optimized", "MLP"]
  optuna_trials: 50
  calibration: "sigmoid"
federated:
  rounds: 10
  min_clients: 3
  aggregator: "fedavg"                   # fedavg | trimmed_mean | coordinate_median
  batch_size: 256
privacy:
  enabled: true
  dp:
    epsilon: 1.0                         # Valore raccomandato per bilancio utilità/privacy
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

> Tenere copie produzione di `config.yaml` fuori VCS se contengono percorsi privati.

---

## 7) Deployment Locale (Bare-Metal)

**Prerequisiti**
```bash
# Python 3.8+ con ambiente virtuale
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verificare installazione
python -c "import streamlit, pandas, numpy, sklearn; print('Dipendenze OK')"
```

**Avviare UI Streamlit**
```bash
streamlit run streamlit_app.py
# Accesso su: http://localhost:8501
```

**Preparazione Dati**
```bash
# Generare dataset estesi e dati training
python scripts/01_generate_dataset.py

# Creare visualizzazioni analisi dati
python scripts/02_visualize_dataset.py
```

**Training Centralizzato e Analisi**
```bash
# Addestrare tutti i 7 modelli con ottimizzazione iperparametri
python scripts/03_train_models.py --config config.yaml

# Analizzare risultati e generare report prestazioni
python scripts/04_analyze_results.py
```

**Flussi Apprendimento Federato**
```bash
# Training federato LightGBM (ensemble regionale)
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py

# Training federato MLP (FedAvg vero)
python scripts/07_mlp_federated_training.py --aggregator fedavg

# Training federato MLP preserva-privacy
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5
python scripts/09_mlp_federated_privacy_visualization.py
```

Gli artefatti appariranno sotto le directory `results/` e `results_*/`.

---

## 8) Deployment Docker

### 8.1 Dockerfile (sicurezza migliorata)
```dockerfile
FROM python:3.11-slim

# Installare dipendenze sistema incluso curl per health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiare requirements prima per caching migliore
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiare codice applicazione
COPY . .

# Creare utente non-root per sicurezza
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Esporre porta
EXPOSE 8501

# Health check con sintassi curl corretta
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Avviare applicazione con configurazione networking corretta
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
```

### 8.2 Comandi Build e Run
```bash
# Build immagine con tag appropriato
docker build -t djms:latest .

# Run con mount volumi appropriati e environment
docker run -d \
  --name djms \
  -p 8501:8501 \
  -v $PWD/data:/app/data:ro \
  -v $PWD/results:/app/results \
  -v $PWD/results_LightGBM_federated:/app/results_LightGBM_federated \
  -v $PWD/results_mlp_federated:/app/results_mlp_federated \
  -v $PWD/results_mlp_federated_privacy:/app/results_mlp_federated_privacy \
  -v $PWD/results_blockchain_demo:/app/results_blockchain_demo \
  -e STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
  djms:latest
```

### 8.3 Docker Compose (production-ready)
```yaml
version: "3.9"
services:
  ui:
    image: djms:latest
    container_name: djms_ui
    ports:
      - "8501:8501"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    volumes:
      - ./data:/app/data:ro
      - ./results:/app/results
      - ./results_LightGBM_federated:/app/results_LightGBM_federated
      - ./results_mlp_federated:/app/results_mlp_federated
      - ./results_mlp_federated_privacy:/app/results_mlp_federated_privacy
      - ./results_blockchain_demo:/app/results_blockchain_demo
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
      - STREAMLIT_SERVER_ENABLE_CORS=false
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 9) Reverse Proxy e TLS

### Configurazione NGINX (migliorata)
```nginx
upstream djms_backend {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name djms.example.org;
    
    # Header sicurezza
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=djms:10m rate=10r/m;
    limit_req zone=djms burst=5 nodelay;
    
    location / {
        proxy_pass http://djms_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout per operazioni modelli grandi
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    # Endpoint health check
    location /_stcore/health {
        proxy_pass http://djms_backend;
        access_log off;
    }
}
```

Aggiungere TLS via certbot o processo standard.

---

## 10) Topologie Apprendimento Federato

### 10.1 Coordinamento Basato su File (raccomandato per piloti CPI/SIL)

**Panoramica Workflow:**
1. Ogni regione esegue script training localmente su dati privati
2. Esportare delta/pesi modelli a cartella condivisa sicura (SFTP/SMB)
3. Coordinatore aggrega aggiornamenti e pubblica modello globale
4. Ripetere per round configurati

**Esecuzione Client Regionali:**
```bash
# Ogni sito CPI/SIL esegue localmente
python scripts/07_mlp_federated_training.py \
  --mode client \
  --region CPI_Verona \
  --round 1 \
  --export_path /secure/shared/round_001/
```

**Aggregazione Coordinatore:**
```bash
# Coordinatore centrale aggrega aggiornamenti
python scripts/07_mlp_federated_training.py \
  --mode aggregate \
  --round 1 \
  --import_path /secure/shared/round_001/ \
  --export_path /secure/shared/round_001/global/
```

### 10.2 Aspettative Prestazioni (corrette)

**LightGBM Federato (Ensemble Regionale):**
- Prestazioni attese: F1 ≈ 0.9007 vs centralizzato F1 ≈ 0.9012
- Impatto prestazioni: -0.0005 F1-score (degradazione minima)
- Tempo training: Simile a centralizzato per regione

**MLP Federato (FedAvg Vero):**
- Prestazioni attese: F1 ≈ 0.788 vs centralizzato F1 ≈ 0.828  
- Impatto prestazioni: ~4% riduzione F1-score
- Tempo training: Dipende da round e frequenza comunicazione

---

## 11) FL Preserva-Privacy (Shamir + DP)

### Configurazione e Prestazioni Attese

**Condivisione Segreti Shamir:**
- Schema soglia (3-su-5) con mascheramento per-parametro
- Recupero dropout per disconnessioni client
- Sicurezza crittografica senza degradazione prestazioni

**Privacy Differenziale:**
- Clipping per-round con iniezione singola rumore Gaussiano
- Contabilità RDP per tracking budget (ε, δ)
- Raccomandato: ε=1.0, δ=1e-6 per bilancio utilità/privacy

**Impatto Prestazioni (corretto):**
- MLP preserva-privacy: F1 ≈ 0.788 (simile a federato standard)
- Costo privacy aggiuntivo: ~0.8% riduzione F1-score vs federato standard
- Costo totale vs centralizzato: ~4.8% riduzione F1-score

**Comandi Esempio:**
```bash
# Privacy forte (ε=0.5)
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 0.5 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5

# Privacy raccomandato (ε=1.0) - buon bilancio utilità/privacy  
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 1.0 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5

# Privacy rilassata (ε=2.0)
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 2.0 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5
```

---

## 12) Ancoraggio Dati Blockchain

### Caratteristiche Prestazioni (da benchmark)

**Prestazioni Costruzione:**
- 100 record: 2.28s totale (KDF: 1.58s, Albero: 0.45s)
- 1.000 record: 30.47s totale (KDF: 21.15s, Albero: 6.04s)  
- 10.000 record: 344.07s totale (KDF: 239.40s, Albero: 68.40s)

**Generazione Prove (scaling O(log n)):**
- 100 record: 1.11ms media, 224 byte
- 1.000 record: 2.61ms media, 320 byte
- 10.000 record: 20.65ms media, 448 byte

**Prestazioni Verifica:**
- Media: 24.49ms con tasso successo 100%
- Nessun falso positivo o fallimento verifica nei benchmark

### Comandi Implementazione
```bash
# Creare commitment Merkle per tutti i risultati
python scripts/blockchain_data_anchoring.py \
  --inputs results results_LightGBM_federated results_mlp_federated results_mlp_federated_privacy \
  --output results_blockchain_demo

# Benchmark prestazioni ancoraggio sul tuo hardware
python scripts/10_blockchain_anchoring_bench.py
```

---

## 13) Scheduling e Automazione

### Linux (systemd + cron)
```bash
# Creare servizio per UI (opzionale - si può usare docker)
sudo systemctl enable djms-ui.service

# Calendario training via cron
crontab -e
```

```cron
# Training giornaliero alle 22:00 nei giorni feriali
0 22 * * 1-5 /opt/djms/scripts/run_training.sh >> /var/log/djms/training.log 2>&1

# Apprendimento federato settimanale domenica
0 20 * * 0 /opt/djms/scripts/run_federated.sh >> /var/log/djms/federated.log 2>&1

# Ancoraggio blockchain mensile
0 23 1 * * /opt/djms/scripts/run_anchoring.sh >> /var/log/djms/anchoring.log 2>&1
```

### Windows (Task Scheduler)
```powershell
# Creare task schedulato per training
schtasks /create /tn "DJMS Training" /tr "C:\djms\scripts\run_training.bat" /sc weekly /d TUE /st 22:00
```

### Scheduling Basato su Docker
```bash
# Usare cron host per invocare comandi docker exec
0 22 * * 1-5 docker exec djms_ui python scripts/03_train_models.py --config config.yaml >> /var/log/djms/training.log 2>&1
```

---

## 14) Monitoraggio, Logging e Health Check

### Monitoraggio Salute
```bash
# Controllare salute applicazione
curl -f http://localhost:8501/_stcore/health

# Monitorare risorse sistema
docker stats djms_ui

# Controllare log recenti
docker logs --tail 50 djms_ui
```

### Configurazione Logging
```python
# Setup logging migliorato
import logging
from datetime import datetime

def setup_production_logging():
    """Setup logging production-ready con rotazione"""
    import logging.handlers
    
    # Creare formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Setup handler
    handlers = [
        logging.StreamHandler(),  # Console
        logging.handlers.RotatingFileHandler(
            f'logs/djms_{datetime.now().strftime("%Y%m")}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
    
    # Configurare logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)
```

### Raccolta Metriche Prestazioni
```bash
#!/bin/bash
# collect_metrics.sh - Script raccolta metriche automatizzata

METRICS_DIR="/opt/djms/metrics"
DATE=$(date +%Y%m%d_%H%M%S)

# Metriche sistema
echo "timestamp,cpu_percent,memory_percent,disk_used_gb" > $METRICS_DIR/system_$DATE.csv
echo "$DATE,$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'),$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'),$(df -h | grep "/opt" | awk '{print $3}' | sed 's/G//')" >> $METRICS_DIR/system_$DATE.csv

# Metriche prestazioni modelli
if [ -f "/opt/djms/results/merged_model_summary.csv" ]; then
    cp /opt/djms/results/merged_model_summary.csv $METRICS_DIR/model_performance_$DATE.csv
fi

# Metriche apprendimento federato
if [ -f "/opt/djms/results_LightGBM_federated/complete_model_comparison.csv" ]; then
    cp /opt/djms/results_LightGBM_federated/complete_model_comparison.csv $METRICS_DIR/federated_performance_$DATE.csv
fi
```

---

## 15) Sicurezza e Operazioni GDPR

### Implementazione Controllo Accesso
```bash
# Permessi file sicuri
sudo chown -R djms:djms /opt/djms
chmod 750 /opt/djms/data/                 # Directory dati
chmod 750 /opt/djms/results*/             # Directory risultati
chmod 755 /opt/djms/scripts/              # Script
chmod 644 /opt/djms/config.yaml           # Configurazione
chmod 600 /opt/djms/.env                  # Segreti environment
```

### Checklist Conformità GDPR
- [ ] **Minimizzazione Dati**: Solo dati processati/aggregati in modalità federata
- [ ] **Limitazione Finalità**: Documentazione chiara finalità training ML  
- [ ] **Controlli Accesso**: Accesso role-based a directory dati e risultati
- [ ] **Politiche Ritenzione**: Pulizia automatizzata artefatti training intermedi
- [ ] **Procedure DSR**: Workflow accesso/cancellazione documentati
- [ ] **Valutazione Impatto Protezione Dati**: Completata per FL preserva-privacy
- [ ] **Accordi Processore**: Contratti con partecipanti regionali CPI/SIL

### Gestione Budget Privacy
```python
# Tracking budget privacy per produzione
class PrivacyBudgetManager:
    def __init__(self, max_epsilon=5.0, max_delta=1e-5):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        self.rounds = []
    
    def can_execute_round(self, epsilon, delta):
        """Controllare se round può essere eseguito entro budget"""
        return (self.current_epsilon + epsilon <= self.max_epsilon and
                self.current_delta + delta <= self.max_delta)
    
    def execute_round(self, epsilon, delta, round_info):
        """Registrare round eseguito"""
        if self.can_execute_round(epsilon, delta):
            self.current_epsilon += epsilon
            self.current_delta += delta
            self.rounds.append({
                'round': len(self.rounds) + 1,
                'epsilon': epsilon,
                'delta': delta,
                'total_epsilon': self.current_epsilon,
                'total_delta': self.current_delta,
                'timestamp': datetime.now(),
                **round_info
            })
            return True
        return False
```

---

## 16) Prestazioni e Capacity Planning

### Requisiti Risorse per Workload

**Training Centralizzato (7 modelli):**
- CPU: 4-8 core, 2-4 ore tempo training
- RAM: 16-32 GB picco durante ottimizzazione iperparametri
- Storage: 2-5 GB per modelli e risultati intermedi

**Training Federato LightGBM:**
- Per regione: CPU 2-4 core, RAM 8-16 GB
- Prestazioni attese: F1 ≈ 0.9007 (-0.0005 vs centralizzato)
- Tempo training: Simile a centralizzato per regione

**Training Federato MLP:**
- Per regione: CPU 2-4 core, RAM 8-16 GB  
- Prestazioni attese: F1 ≈ 0.788 (~4% vs centralizzato F1 ≈ 0.828)
- Tempo training: 30-60 minuti per round, 10-30 round tipici

**Modalità Preserva-Privacy:**
- Compute aggiuntivo: ~10-20% overhead per operazioni DP
- Prestazioni attese: F1 ≈ 0.788 (~0.8% costo aggiuntivo)
- Memoria: +2-4 GB per operazioni aggregazione sicura

### Linee Guida Capacity Planning
```yaml
# Regolazioni config.yaml per ambienti diversi

# Sviluppo/Testing
federated:
  rounds: 5
  batch_size: 128
model_training:
  optuna_trials: 20
  n_jobs: 2

# Produzione (risorse elevate)
federated:
  rounds: 15
  batch_size: 512
model_training:
  optuna_trials: 100
  n_jobs: 8

# Produzione (risorse limitate)
federated:
  rounds: 8
  batch_size: 256
model_training:
  optuna_trials: 30
  n_jobs: 4
```

---

## 17) Backup, DR e Rollback

### Strategia Backup
```bash
#!/bin/bash
# comprehensive_backup.sh

BACKUP_ROOT="/backup/djms"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$DATE"

mkdir -p $BACKUP_DIR

# Configurazioni critiche
cp /opt/djms/config.yaml $BACKUP_DIR/
cp /opt/djms/requirements.txt $BACKUP_DIR/
cp /opt/djms/.env $BACKUP_DIR/ 2>/dev/null || true

# Dati (se non troppo grandi)
cp -r /opt/djms/data/processed $BACKUP_DIR/

# Modelli e risultati
cp -r /opt/djms/results $BACKUP_DIR/
cp -r /opt/djms/results_* $BACKUP_DIR/

# Documentazione
cp -r /opt/djms/docs $BACKUP_DIR/

# Creare archivio
tar -czf $BACKUP_ROOT/djms_backup_$DATE.tar.gz -C $BACKUP_ROOT $DATE
rm -rf $BACKUP_DIR

# Mantenere ultimi 7 giorni localmente, 4 settimane offsite
find $BACKUP_ROOT -name "djms_backup_*.tar.gz" -mtime +7 -delete
```

### Test Disaster Recovery
```bash
#!/bin/bash
# dr_test.sh - Test procedura restore

RESTORE_DIR="/tmp/djms_restore_test"
BACKUP_FILE="$1"

# Estrarre backup
mkdir -p $RESTORE_DIR
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# Test configurazione
cd $RESTORE_DIR/*/
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test caricamento modelli
python -c "
import joblib
import glob
models = glob.glob('results/*.joblib')
if models:
    model = joblib.load(models[0])
    print(f'Modello caricato con successo: {models[0]}')
else:
    print('Nessun modello trovato')
"

echo "Test DR completato per $BACKUP_FILE"
```

---

## 18) Risoluzione Problemi

### Problemi Comuni e Soluzioni

**Applicazione Non Si Avvia**
```bash
# Controllare dipendenze
pip list | grep -E "(streamlit|pandas|scikit-learn)"
pip install -r requirements.txt --upgrade

# Controllare versione Python
python --version  # Dovrebbe essere 3.8+

# Controllare disponibilità porta
netstat -tlnp | grep 8501
```

**Problemi Docker**
```bash
# Fallimenti health check
docker exec djms_ui curl -f http://localhost:8501/_stcore/health
docker logs djms_ui --tail 50

# Problemi risorse container
docker stats djms_ui
docker exec djms_ui df -h
docker exec djms_ui free -m
```

**Problemi Prestazioni**
```bash
# Sottoperformance apprendimento federato
# Atteso: LightGBM -0.0005 F1, MLP ~4% riduzione F1
# Se significativamente peggio, controllare split dati e bilancio regionale

# Monitorare log training
tail -f logs/training.log

# Controllare risorse sistema durante training
htop
iotop

# Verificare esistenza e validità file modelli
ls -la results/*.joblib
python -c "import joblib; print('Modello valido:', joblib.load('results/LightGBM_Optimized.joblib') is not None)"
```

**Problemi Geocoding**
```bash
# Connettività rete
ping nominatim.openstreetmap.org

# Problemi cache
ls -la data/processed/geocoding_cache.json
# Se corrotto, rimuovere e rigenerare
rm data/processed/geocoding_cache.json
python scripts/01_generate_dataset.py
```

**Problemi Apprendimento Federato**
```bash
# Controllare split dati regionali
python -c "
import pandas as pd
import glob
files = glob.glob('data/federated/CPI_*.csv')
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: {len(df)} campioni')
"

# Verificare risultati aggregazione
ls -la results_mlp_federated/
ls -la results_mlp_federated_privacy/
```

---

## 19) Procedura Upgrade

### Checklist Pre-Upgrade
1. **Annunciare finestra manutenzione** agli operatori
2. **Backup sistema corrente** (config, dati, modelli, risultati)
3. **Test upgrade su ambiente staging** prima
4. **Rivedere changelog** e breaking change
5. **Preparare piano rollback**

### Passi Upgrade
```bash
#!/bin/bash
# upgrade_procedure.sh

set -e  # Uscire su qualsiasi errore

BACKUP_DIR="/opt/djms_backup_$(date +%Y%m%d_%H%M%S)"
DJMS_ROOT="/opt/djms"

echo "Avvio procedura upgrade DJMS..."

# 1. Creare backup
echo "Creazione backup sistema..."
cp -r $DJMS_ROOT $BACKUP_DIR

# 2. Fermare servizi
echo "Fermata servizi..."
docker-compose down || systemctl stop djms-ui || true

# 3. Aggiornare codice
echo "Aggiornamento codebase..."
cd $DJMS_ROOT
git fetch origin
git checkout main
git pull origin main

# 4. Aggiornare dipendenze
echo "Aggiornamento dipendenze Python..."
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Rebuild immagine Docker (se si usano container)
echo "Rebuild immagine Docker..."
docker build -t djms:latest .

# 6. Eseguire migrazioni/aggiornamenti (se presenti)
echo "Esecuzione aggiornamenti sistema..."
python scripts/01_generate_dataset.py --verify-only || true

# 7. Avviare servizi
echo "Avvio servizi..."
docker-compose up -d || systemctl start djms-ui

# 8. Health check
echo "Esecuzione health check..."
sleep 30
curl -f http://localhost:8501/_stcore/health

echo "Upgrade completato con successo!"
echo "Backup memorizzato in: $BACKUP_DIR"
```

### Validazione Post-Upgrade
```bash
# Verificare funzionalità critiche
python scripts/03_train_models.py --dry-run
streamlit run streamlit_app.py --help

# Controllare compatibilità modelli
python -c "
import joblib
import glob
models = glob.glob('results/*.joblib')
for model_file in models:
    try:
        model = joblib.load(model_file)
        print(f'✓ {model_file} - OK')
    except Exception as e:
        print(f'✗ {model_file} - ERRORE: {e}')
"

# Verificare componenti apprendimento federato
python scripts/07_mlp_federated_training.py --validate-setup
```

---

## 20) Appendice A — File Esempio

### File Servizio systemd
```ini
# /etc/systemd/system/djms-ui.service
[Unit]
Description=Servizio UI Streamlit DJMS
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=djms
Group=djms
WorkingDirectory=/opt/djms
Environment=PATH=/opt/djms/venv/bin
ExecStart=/opt/djms/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 --browser.gatherUsageStats=false
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=djms-ui

# Impostazioni sicurezza
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/djms/results /opt/djms/results_* /opt/djms/logs

[Install]
WantedBy=multi-user.target
```

### Configurazione Logrotate
```
# /etc/logrotate.d/djms
/opt/djms/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 djms djms
    postrotate
        # Riavviare servizio per riaprire file log
        systemctl reload djms-ui || true
    endscript
}

/var/log/djms/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 644 djms djms
}
```

### Template Variabili Ambiente
```bash
# Template file .env
DJMS_ENV=production
DJMS_SECRET_KEY=your-secret-key-here
DJMS_DEBUG=false

# Impostazioni database (se implementato)
DATABASE_URL=sqlite:///opt/djms/data/djms.db

# Chiavi API servizi esterni (se necessarie)
NOMINATIM_USER_AGENT=djms-production-v1
GEOCODING_API_KEY=your-api-key-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/djms/application.log

# Sicurezza
MAX_UPLOAD_SIZE=100MB
ALLOWED_HOSTS=djms.example.org,localhost
```

### Wrapper Script Training
```bash
#!/bin/bash
# /opt/djms/scripts/run_training.sh

set -euo pipefail

DJMS_ROOT="/opt/djms"
LOG_DIR="/var/log/djms"
DATE=$(date +%Y%m%d_%H%M%S)

cd $DJMS_ROOT
source venv/bin/activate

echo "[$DATE] Avvio pipeline training..." >> $LOG_DIR/training.log

# Controlli pre-training
if [ ! -f "data/processed/Enhanced_Training_Dataset.csv" ]; then
    echo "[$DATE] Dati training non trovati, generazione..." >> $LOG_DIR/training.log
    python scripts/01_generate_dataset.py >> $LOG_DIR/training.log 2>&1
fi

# Eseguire training centralizzato
python scripts/03_train_models.py --config config.yaml >> $LOG_DIR/training.log 2>&1

# Eseguire analisi
python scripts/04_analyze_results.py >> $LOG_DIR/training.log 2>&1

# Eseguire apprendimento federato (settimanale)
if [ "$(date +%u)" -eq 7 ]; then  # Domenica
    echo "[$DATE] Esecuzione apprendimento federato settimanale..." >> $LOG_DIR/training.log
    python scripts/05_LightGBM_federated_training.py >> $LOG_DIR/training.log 2>&1
    python scripts/07_mlp_federated_training.py >> $LOG_DIR/training.log 2>&1
fi

echo "[$DATE] Pipeline training completata con successo" >> $LOG_DIR/training.log
```

---

