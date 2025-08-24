# Deployment Guide — Disability Job Matching System
_Last updated: 2025-08-23 23:52_

> **Scope.** This document describes how to deploy the Disability Job Matching System in development, staging, and production environments. It covers single-node deployments, containerized setups, and cross‑silo federated training topologies with privacy-preserving options and integrity anchoring. It complements the project’s README and Technical Documentation.

---

## Table of Contents

1. [Audience & Assumptions](#audience--assumptions)  
2. [System Requirements](#system-requirements)  
3. [Architecture for Deployment](#architecture-for-deployment)  
4. [Project Layout & Runtime Paths](#project-layout--runtime-paths)  
5. [Environments](#environments)  
6. [Configuration](#configuration)  
7. [Local (Bare‑Metal) Deployment](#local-baremetal-deployment)  
8. [Docker Deployment](#docker-deployment)  
9. [Reverse Proxy & TLS (optional)](#reverse-proxy--tls-optional)  
10. [Federated Learning Topologies](#federated-learning-topologies)  
11. [Privacy‑Preserving FL (Shamir + DP)](#privacypreserving-fl-shamir--dp)  
12. [Blockchain Data Anchoring](#blockchain-data-anchoring)  
13. [Scheduling & Automation](#scheduling--automation)  
14. [Monitoring, Logging & Health Checks](#monitoring-logging--health-checks)  
15. [Security & GDPR Operations](#security--gdpr-operations)  
16. [Performance & Capacity Planning](#performance--capacity-planning)  
17. [Backup, DR & Rollback](#backup-dr--rollback)  
18. [Troubleshooting](#troubleshooting)  
19. [Upgrade Procedure](#upgrade-procedure)  
20. [Appendix A — Example Files](#appendix-a--example-files)  
21. [Appendix B — Kubernetes (Advanced, Optional)](#appendix-b--kubernetes-advanced-optional)  
22. [Appendix C — Legacy Deployment Guide (verbatim)](#appendix-c--legacy-deployment-guide-verbatim)

---

## 1) Audience & Assumptions

- **Audience**: DevOps engineers, ML engineers, and public‑sector IT staff deploying for CPI/SIL pilots or production.  
- **Assumptions**:
  - You have basic familiarity with Python environments, Docker, and Linux service management.
  - You can provision storage for datasets and results.
  - Network connectivity is available between sites (for FL) or you will use a secure file‑based exchange.

---

## 2) System Requirements

**Operating Systems**  
- Linux: Ubuntu 22.04/24.04 LTS (recommended)  
- Windows 10/11 (supported for development and PoC)  
- macOS 13+ (development only)

**Hardware (baseline)**  
- CPU: 4 cores; RAM: 16 GB; Disk: 50 GB  
- For federated aggregation rounds or large visualizations: 8 cores / 32 GB recommended  
- GPU: **not required** (CPU training supported and validated)

**Networking**  
- Default Streamlit port: `8501`  
- Optional reverse proxy (80/443) if exposing externally  
- For FL, see [Topologies](#federated-learning-topologies) for data exchange options

---

## 3) Architecture for Deployment

Layers in production:

1. **Application/UI** — Streamlit app (`streamlit_app.py`) for operators (demo/ops tool).  
2. **Training Services** — scripts under `scripts/` orchestrated by scheduler (cron/Task Scheduler/systemd).  
3. **Federated Coordinator** — a **coordinator host** (or process) that sequences FL rounds and collects updates.  
4. **Privacy Layer** — secure aggregation (Shamir) and DP budget accounting.  
5. **Integrity Layer** — Merkle commitments and proof verification for artifacts.  
6. **Storage** — `data/` and `results*/` folders (local or network‑attached).  
7. **Observability** — logs, health endpoints, and metrics CSV.

### Minimal single‑host diagram

```
[ Operators ] --> [ Streamlit (8501) ] --> [ Scripts: 01..10 ] --> [ results/* ]
                                 ^                                  
                                 +---- health: /_stcore/health
```

### Cross‑silo FL (file‑based exchange)

```
 Region A           Region B            Region C             Coordinator
+----------+      +----------+        +----------+         +--------------+
| local ML |      | local ML |        | local ML |         | aggregation  |
| scripts  |      | scripts  |        | scripts  |         | + anchoring  |
+----------+      +----------+        +----------+         +--------------+
     \                |                   /                     ^
      \               |                  /                      |
       +------ secure file exchange / SFTP / shared drive ------+
```

---

## 4) Project Layout & Runtime Paths

Project skeleton (as shipped):

```
📁 Disability Job Matching System/
├── README.md / README_IT.md
├── config.yaml
├── requirements.txt
├── streamlit_app.py
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
├── results/
│   ├── learning_curves/
│   ├── *.joblib
│   ├── merged_model_summary.csv
│   └── *.png
├── results_LightGBM_federated/
│   ├── regional_models/
│   ├── federated_models/
│   ├── centralized_models/
│   ├── visualizations/
│   ├── complete_model_comparison.csv
│   └── experiment_metadata.json
├── results_mlp_federated/
├── results_mlp_federated_privacy/
├── results_blockchain_demo/
└── visualizations_federated_comparison/
```

---

## 5) Environments

- **Development**: local workstation, small samples; iterative runs.  
- **Staging**: replica of production setup (same OS and Python packages), anonymized datasets.  
- **Production**: locked dependencies (`requirements.txt`), stable `config.yaml`, scheduled jobs, restricted user access.

---

## 6) Configuration

Create and version `config.yaml` in repository root. Key flags:

```yaml
seed: 42
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies: data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir: data/processed
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30      # default search radius
training:
  model_set: ["LightGBM_Optimized", "MLP"]
  optuna_trials: 50
  calibration: "sigmoid"
federated:
  rounds: 10
  min_clients: 3
  aggregator: "fedavg"     # fedavg | trimmed_mean | coordinate_median
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

> Keep production copies of `config.yaml` outside VCS if they embed private paths. Use environment overrides when necessary.

---

## 7) Local (Bare‑Metal) Deployment

**Prerequisites**
```bash
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run Streamlit UI**
```bash
streamlit run streamlit_app.py
# http://localhost:8501
```

**Data Preparation**
```bash
python scripts/01_generate_dataset.py
python scripts/02_visualize_dataset.py
```

**Centralized Training & Analysis**
```bash
python scripts/03_train_models.py --config config.yaml
python scripts/04_analyze_results.py
```

Artifacts will appear under `results/` and `results_*/` families.

---

## 8) Docker Deployment

### 8.1 Build the Image
Use this Dockerfile (ensure `curl` is present for health checks):

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

### 8.2 Run (single container)
```bash
docker run -d --name djms   -p 8501:8501   -v $PWD/data:/app/data   -v $PWD/results:/app/results   -v $PWD/results_LightGBM_federated:/app/results_LightGBM_federated   -v $PWD/results_mlp_federated:/app/results_mlp_federated   -v $PWD/results_mlp_federated_privacy:/app/results_mlp_federated_privacy   -v $PWD/results_blockchain_demo:/app/results_blockchain_demo   djms:latest
```

### 8.3 docker‑compose (UI + scheduled jobs)
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

> Schedule training via host cron/Task Scheduler invoking `docker exec` (see Section 13).

---

## 9) Reverse Proxy & TLS (optional)

Example NGINX site:
```nginx
server {{
  listen 80;
  server_name djms.example.org;
  location / {{
    proxy_pass http://127.0.0.1:8501/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }}
}}
```

Add TLS via your standard process (e.g., certbot). Set `--server.enableCORS=false` only if you control the proxy and trust headers.

---

## 10) Federated Learning Topologies

Two reference modes compatible with current scripts:

### 10.1 File‑Based Coordination (recommended for pilots)
- Each region runs `07_mlp_federated_training.py` locally on private data.
- After local epochs, each site **exports** model deltas (or weights) to a secure shared folder (SFTP/SMB).
- The coordinator aggregates exported updates and publishes a new global model back to the same share.
- Repeat for `federated.rounds` from `config.yaml`.

**Coordinator sample (pseudo‑workflow)**
```bash
# Round 1: notify clients (out of band)
# Wait for files under: results_mlp_federated/round_001/client_*/update.bin
python scripts/07_mlp_federated_training.py --mode aggregate --round 1
# Publish: results_mlp_federated/round_001/global/model.bin
```

**Clients**
```bash
python scripts/07_mlp_federated_training.py --mode client --round 1
# Produces: results_mlp_federated/round_001/client_<REGION>/update.bin
```

> If you adopt privacy mode, switch to `08_mlp_federated_privacy.py` with DP and Shamir options (see Section 11).

### 10.2 Single‑Host Emulation
For development, run sequential clients on one machine with different regional CSV subsets (`utils/federated_data_splitter.py` behavior can be emulated by pointing to regional files). Aggregation happens in‑process.

---

## 11) Privacy‑Preserving FL (Shamir + DP)

- **Shamir Secret Sharing** (`utils/enhanced_shamir_privacy.py`): split each client’s update into shares; send subsets to peers/coordinator. Aggregator reconstructs when threshold is met (e.g., `3-of-5`).  
- **Differential Privacy**: per‑round clipping and a **single** injection of calibrated Gaussian noise; RDP accounting accumulates privacy budget (ε, δ).

**Client command (example)**
```bash
python scripts/08_mlp_federated_privacy.py   --dp.epsilon 1.0 --dp.delta 1e-6 --dp.max_grad_norm 1.0   --secure_agg.threshold 3-of-5 --round 1 --mode client
```

**Coordinator (aggregation)**
```bash
python scripts/08_mlp_federated_privacy.py --round 1 --mode aggregate
```

> Store privacy proofs/ledgers under `results_mlp_federated_privacy/round_x/` alongside model artifacts.

---

## 12) Blockchain Data Anchoring

- Use `scripts/blockchain_data_anchoring.py` to produce Merkle commitments for artifacts (models, metrics, manifests).  
- Run `10_blockchain_anchoring_bench.py` to benchmark proof build/verify times on your hardware.  
- Store anchors and proofs under `results_blockchain_demo/`.

**Daily anchoring (example)**
```bash
python scripts/blockchain_data_anchoring.py   --inputs results results_LightGBM_federated results_mlp_federated results_mlp_federated_privacy   --out results_blockchain_demo
```

---

## 13) Scheduling & Automation

### 13.1 Linux (cron)
```cron
# m h dom mon dow  command
0 22 * * 1-5  /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/03_train_models.py --config config.yaml >> logs/train.log 2>&1'
30 22 * * 1-5 /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/04_analyze_results.py >> logs/analyze.log 2>&1'
0 23 * * 1-5  /usr/bin/bash -lc 'cd /opt/djms && source venv/bin/activate && python scripts/blockchain_data_anchoring.py >> logs/anchor.log 2>&1'
```

### 13.2 Windows (Task Scheduler)
Create tasks that execute:
```
cmd.exe /c "cd C:\djms && venv\Scripts\activate && python scripts\03_train_models.py --config config.yaml"
```

### 13.3 Docker Cron (host‑side)
Invoke commands using `docker exec djms_ui ...` at scheduled times.

---

## 14) Monitoring, Logging & Health Checks

- **Health**: Streamlit `GET /_stcore/health` (HTTP 200 when ready).  
- **Logs**: redirect script outputs to `logs/*.log`; rotate weekly.  
- **Metrics**: CSV snapshots in `results/merged_model_summary.csv` and `results_*/complete_model_comparison.csv`.  
- **Dashboards**: use `*_visualization.py` scripts to render charts into `results*/visualizations/`.

---

## 15) Security & GDPR Operations

- **Access Control**: restrict `data/` and `results*/` to least privilege.  
- **Data Minimization**: use `processed/` views; avoid distributing raw files.  
- **DSR Workflows**: document access/erasure paths in your org’s SOP.  
- **FL Controls**: only share model deltas; never raw data; keep cohort size ≥ threshold for Shamir.  
- **Retention**: purge intermediate artifacts on a schedule; anchor only manifests and hashes when possible.

---

## 16) Performance & Capacity Planning

- **CPU/RAM**: adjust `federated.batch_size` and number of rounds to meet SLAs.  
- **LightGBM**: prefer `num_leaves`/`max_depth` tuned by `Optuna`; monitor overfitting in small regions.  
- **MLP**: start with batch 256 and learning rate 1e-3; reduce batch on small nodes.  
- **I/O**: place `results*/` on SSD; compress archives for cold storage.  
- **Privacy Mode**: expect modest F1 delta with ε≈1.0; measure on staging before enabling in prod.

---

## 17) Backup, DR & Rollback

- **Daily backups**: `config.yaml`, `data/processed/`, `results*/`, and `docs/`.  
- **Off‑site**: keep encrypted snapshots off‑site weekly.  
- **Rollback**: tag container images; retain last 3 images and last 3 `requirements.txt` revisions.  
- **Verification**: periodically restore to a staging host and validate Streamlit + training scripts.

---

## 18) Troubleshooting

- `ModuleNotFoundError` → ensure `pip install -r requirements.txt`.  
- `sklearn.utils._tags._safe_tags` import error → align `scikit-learn` and `imbalanced-learn` versions per `requirements.txt`.  
- Health check fails → ensure `curl` installed in Docker image; port mapping `-p 8501:8501`.  
- Empty `complete_model_comparison.csv` → verify inputs and that training ran to completion.  
- Slow proofs in anchoring → benchmark and adjust batch sizes; anchor manifests instead of full artifacts.

---

## 19) Upgrade Procedure

1. Announce change window; back up `config.yaml`, `results*/`, and `docs/`.  
2. Pull changes; review `requirements.txt` diffs.  
3. Rebuild container (if used) and run smoke tests on staging.  
4. Deploy to production; monitor health and logs; roll back if KPIs regress.

---

## 20) Appendix A — Example Files

### 20.1 `systemd` unit (UI)

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

### 20.3 NGINX (with TLS passthrough stub)
See Section 9 for base config; terminate TLS per your policy.

---

## 21) Appendix B — Kubernetes (Advanced, Optional)

Minimal manifest (single pod + service):

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

> Use Ingress + TLS as per your cluster standards.

---

## 22) Appendix C — Legacy Deployment Guide (verbatim)

# 🚀 Deployment Guide - Disability Job Matching System

**Complete Installation and Setup Guide for Production Environments**

---

## 📋 Overview

This deployment guide provides step-by-step instructions for installing and configuring the Disability Job Matching System in production environments. The guide covers both demo mode (synthetic data) and production mode (real employment data) deployments.

### Deployment Modes

- **🧪 Demo Mode**: Uses synthetic data for testing and demonstration
- **🏭 Production Mode**: Uses real employment outcome data for live operations
- **🔧 Development Mode**: Full development environment with all scripts

---

## 📋 Prerequisites

### System Requirements

**Minimum Requirements**:
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 8GB minimum (16GB recommended for training)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

**Network Requirements**:
- **Internet Connection**: Required for initial geocoding setup
- **Firewall**: Port 8501 (Streamlit default) or custom port
- **Geographic API**: Access to Nominatim geocoding service

### Software Dependencies

**Required Software**:
```bash
# Python 3.8+ with pip
python --version  # Should show 3.8 or higher
pip --version

# Git (for repository cloning)
git --version

# Optional: Virtual environment tools
python -m venv --help
```

---

## 🔧 Installation Methods

### Method 1: Quick Demo Setup (Recommended for Testing)

**Step 1: Clone Repository**
```bash
# Clone the project
git clone https://github.com/your-username/disability-job-matching.git
cd disability-job-matching

# Verify project structure
ls -la
# Should see: streamlit_app.py, config.yaml, requirements.txt, data/, scripts/, utils/
```

**Step 2: Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, numpy, sklearn; print('Dependencies OK')"
```

**Step 3: Launch Demo**
```bash
# Start the application in demo mode
streamlit run streamlit_app.py

# System will automatically:
# 1. Load with synthetic demo data
# 2. Open browser at http://localhost:8501
# 3. Show "Demo Mode" in interface
```

**Verification**:
- Interface loads successfully
- Analytics tab shows sample data
- Can perform candidate searches
- Results appear within 5 seconds

### Method 2: Production Setup (Real Data)

**Prerequisites for Production**:
- Historical employment outcome data
- Properly formatted CSV files
- Data privacy compliance approval

**Step 1: Prepare Production Data**

```bash
# Production data structure required:
data/
├── raw/
│   ├── Dataset_Candidati_Aggiornato.csv      # Real candidate data
│   └── Dataset_Aziende_con_Stima_Assunzioni.csv  # Real company data
└── processed/
    └── Enhanced_Training_Dataset.csv         # Real employment outcomes
```

**Enhanced_Training_Dataset.csv Format**:
```csv
outcome,attitude_score,years_experience,unemployment_duration,compatibility_score,distance_km,company_size,retention_rate,remote_work,certification,...
1,0.75,5,12,0.85,15.2,150,0.78,1,1,...
0,0.45,2,24,0.35,45.8,50,0.65,0,0,...
```

**Step 2: Data Validation**
```bash
# Validate data format
python -c "
import pandas as pd
df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
print(f'Training data: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Outcome distribution: {df.outcome.value_counts()}')
print('Required columns present:', all(col in df.columns for col in ['outcome', 'attitude_score', 'compatibility_score']))
"
```

**Step 3: Train Production Models**
```bash
# Train models on real data
python scripts/03_train_models.py

# Expected output:
# 📥 Loading training dataset...
# 🧹 Preparing data...
# 🎯 Optimizing hyperparameters...
# 🤖 Training [Model Name]...
# ✅ All models saved

# Verify models created
ls -la results/
# Should see: *.joblib files, metrics_summary.csv
```

**Step 4: Launch Production Interface**
```bash
# Start production application
streamlit run streamlit_app.py

# System will:
# 1. Detect real data automatically
# 2. Load trained models
# 3. Enable production features
```

### Method 3: Docker Deployment (Enterprise)

**Create Dockerfile**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Start application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Docker Commands**:
```bash
# Build image
docker build -t disability-job-matcher .

# Run container (demo mode)
docker run -p 8501:8501 disability-job-matcher

# Run with real data volume
docker run -p 8501:8501 -v /path/to/real/data:/app/data disability-job-matcher
```

---

## ⚙️ Configuration

### Environment Configuration

**config.yaml Customization**:
```yaml
# Production configuration example
paths:
  training_dataset: "data/processed/Enhanced_Training_Dataset.csv"
  model_output_dir: "results"
  logs_dir: "logs"

matching_thresholds:
  attitude_min: 0.3           # Adjust based on candidate pool
  distance_max_km: 30         # Adjust for urban/rural deployment
  match_probability_cutoff: 0.6

model_training:
  n_jobs: 4                   # Adjust based on server CPU cores
  optuna_trials: 50           # Reduce for faster training, increase for accuracy

geocoding:
  delay: 0.5                  # Increase if hitting rate limits
  cache_file: "data/processed/geocoding_cache.json"

streamlit:
  page_title: "Sistema Collocamento Mirato - [Organization Name]"
  default_top_k: 5
```

**Environment Variables**:
```bash
# Optional environment configuration
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_ENABLE_CORS=false
```

### Network Configuration

**Firewall Settings**:
```bash
# Ubuntu/Debian
sudo ufw allow 8501/tcp

# CentOS/RHEL
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload

# Verify port accessibility
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

## 🔒 Security Configuration

### Access Control

**Basic Authentication (Streamlit)**:
```python
# Add to streamlit_app.py for basic protection
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

**File Permissions**:
```bash
# Secure file permissions
chmod 600 config.yaml                    # Config files
chmod 600 data/processed/*.csv           # Data files  
chmod 755 scripts/*.py                   # Executable scripts
chmod 644 requirements.txt               # Public files

# Secure directories
chmod 750 data/                          # Data directory
chmod 750 results/                       # Models directory
chmod 755 utils/                         # Code directory
```

### Data Protection

**Sensitive Data Handling**:
```bash
# Create secure data directory
sudo mkdir -p /opt/job-matcher/secure-data
sudo chown app-user:app-group /opt/job-matcher/secure-data
sudo chmod 750 /opt/job-matcher/secure-data

# Symlink to application
ln -s /opt/job-matcher/secure-data data/processed
```

**Backup Strategy**:
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups/job-matcher"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup critical files
cp -r data/processed $BACKUP_DIR/$DATE/
cp -r results/ $BACKUP_DIR/$DATE/
cp config.yaml $BACKUP_DIR/$DATE/

# Compress and encrypt
tar -czf $BACKUP_DIR/$DATE.tar.gz $BACKUP_DIR/$DATE
rm -rf $BACKUP_DIR/$DATE

echo "Backup completed: $BACKUP_DIR/$DATE.tar.gz"
```

---

## 📊 Monitoring and Logging

### Application Monitoring

**Health Check Endpoint**:
```python
# Add to streamlit_app.py
import streamlit as st
import time

def health_check():
    """System health check"""
    checks = {
        'models_loaded': len(st.session_state.get('models', {})) > 0,
        'data_available': os.path.exists('data/processed/Enhanced_Training_Dataset.csv'),
        'config_valid': os.path.exists('config.yaml')
    }
    return all(checks.values()), checks

# Usage in app
if st.sidebar.button("Health Check"):
    healthy, details = health_check()
    if healthy:
        st.success("✅ System Healthy")
    else:
        st.error("❌ System Issues Detected")
        st.json(details)
```

**Performance Monitoring**:
```python
# Add performance tracking
import time
import psutil
import logging

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        
    def log_request(self, operation, duration):
        self.request_count += 1
        logging.info(f"Operation: {operation}, Duration: {duration:.2f}s, "
                    f"CPU: {psutil.cpu_percent()}%, "
                    f"Memory: {psutil.virtual_memory().percent}%")

# Usage
monitor = PerformanceMonitor()
start = time.time()
# ... perform matching operation ...
monitor.log_request("candidate_matching", time.time() - start)
```

### Logging Configuration

**Logging Setup**:
```python
# logging_config.py
import logging
import os
from datetime import datetime

def setup_logging():
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Usage in main application
logger = setup_logging()
logger.info("Application started")
```

**Log Rotation**:
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

## 🔄 Maintenance Procedures

### Regular Maintenance Tasks

**Daily Tasks**:
```bash
#!/bin/bash
# daily_maintenance.sh

echo "$(date): Starting daily maintenance"

# Check disk space
df -h | grep -E "/(dev|opt)" | awk '$5 > "80%" {print "WARNING: " $0}'

# Check application health
curl -f http://localhost:8501/_stcore/health || echo "WARNING: App health check failed"

# Validate data integrity
python -c "
import pandas as pd
try:
    df = pd.read_csv('data/processed/Enhanced_Training_Dataset.csv')
    print(f'Data OK: {df.shape[0]} records')
except Exception as e:
    print(f'Data ERROR: {e}')
"

# Clear old cache files
find data/processed -name "*.cache" -mtime +7 -delete

echo "$(date): Daily maintenance completed"
```

**Weekly Tasks**:
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update geocoding cache if needed
python -c "
import json
try:
    with open('data/processed/geocoding_cache.json', 'r') as f:
        cache = json.load(f)
    print(f'Geocoding cache: {len(cache)} entries')
except:
    print('No geocoding cache found')
"

# Check model performance
python scripts/04_analyze_results.py

# Backup models and config
cp -r results/ /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
cp config.yaml /opt/backups/job-matcher/weekly_$(date +%Y%m%d)/
```

### Model Updates

**When to Retrain Models**:
- New employment outcome data available (quarterly recommended)
- Significant changes in candidate/company demographics
- Performance degradation observed
- System configuration changes

**Retraining Process**:
```bash
# 1. Backup current models
cp -r results/ results_backup_$(date +%Y%m%d)/

# 2. Update training data
# Place new Enhanced_Training_Dataset.csv in data/processed/

# 3. Retrain models
python scripts/03_train_models.py

# 4. Validate new models
python scripts/04_analyze_results.py

# 5. Test with sample candidates
streamlit run streamlit_app.py

# 6. If satisfied, remove backup
# rm -rf results_backup_*
```

---

## 🛠️ Troubleshooting

### Common Issues

**Issue 1: Application Won't Start**
```bash
# Symptoms: ModuleNotFoundError, Import errors
# Solution: Verify dependencies
pip list | grep -E "(streamlit|pandas|numpy|sklearn)"
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+
```

**Issue 2: No Models Found**
```bash
# Symptoms: "No models loaded" warning
# Solution: Train models
ls -la results/  # Check if .joblib files exist
python scripts/03_train_models.py  # Train if missing
```

**Issue 3: Geocoding Failures**
```bash
# Symptoms: Distance calculation errors
# Solution: Check network and cache
ping nominatim.openstreetmap.org
ls -la data/processed/geocoding_cache.json

# Reset cache if corrupted
rm data/processed/geocoding_cache.json
```

**Issue 4: Poor Performance**
```bash
# Symptoms: Slow response times
# Solutions:
# 1. Check system resources
htop  # or top
free -h

# 2. Reduce parallel jobs
# Edit config.yaml: model_training.n_jobs: 2

# 3. Clear browser cache
# 4. Restart application
```

### Error Logs Analysis

**Common Error Patterns**:
```bash
# Check recent errors
tail -100 logs/app_$(date +%Y%m%d).log | grep ERROR

# Geocoding errors
grep "Geocoding error" logs/*.log

# Model prediction errors  
grep "prediction error" logs/*.log

# Memory issues
grep -i "memory\|oom" logs/*.log
```

### Performance Optimization

**Memory Optimization**:
```python
# Add to streamlit_app.py
import gc

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_cached_data():
    # Load heavy data operations
    pass

# Clear cache when needed
if st.button("Clear Cache"):
    st.cache_data.clear()
    gc.collect()
    st.success("Cache cleared")
```

**CPU Optimization**:
```yaml
# config.yaml adjustments for slower systems
model_training:
  n_jobs: 2                 # Reduce from 4
  optuna_trials: 25         # Reduce from 50

matching_thresholds:
  distance_max_km: 25       # Reduce search space
```

---

## 📞 Support and Escalation

### Support Contacts

**Technical Issues**:
- **Primary**: michele.melch@gmail.com
- **Academic**: oleksandr.kuznetsov@uniecampus.it
- **Emergency**: Include "URGENT" in subject line

**Information to Include**:
```bash
# System information
uname -a
python --version
pip list > installed_packages.txt

# Error logs
tail -50 logs/app_$(date +%Y%m%d).log

# Configuration
cat config.yaml

# System resources
free -h && df -h
```

### Escalation Procedures

**Level 1: Application Issues**
- Check logs and common issues
- Restart application
- Verify configuration

**Level 2: Data/Model Issues**
- Validate data integrity
- Retrain models if needed
- Check for data updates

**Level 3: System/Infrastructure Issues**
- Contact system administrator
- Check network connectivity
- Review security logs

### Community Resources

**Documentation**:
- GitHub Repository: [Link to repository]
- User Guide: `docs/user_guide_english.md`
- Technical Docs: `docs/technical_documentation.md`

**Updates and Announcements**:
- GitHub Releases for version updates
- Email notifications for critical security updates

---

*This deployment guide provides comprehensive instructions for installing and maintaining the Disability Job Matching System. For organization-specific deployment questions or custom configuration needs, contact the development team.*

---

**Document Version**: 1.0  
**Last Updated**: June 2025  
**Next Review**: December 2025
