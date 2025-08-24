# Deployment Guide — Disability Job Matching System
_Last updated: 2025-08-24 16:00_

> **Scope.** This document describes how to deploy the Disability Job Matching System in development, staging, and production environments. It covers single-node deployments, containerized setups, and cross-silo federated training topologies with privacy-preserving options and integrity anchoring. All performance expectations have been updated to match experimental results.

---

## Table of Contents

1. [Audience & Assumptions](#1-audience--assumptions)  
2. [System Requirements](#2-system-requirements)  
3. [Architecture for Deployment](#3-architecture-for-deployment)  
4. [Project Layout & Runtime Paths](#4-project-layout--runtime-paths)  
5. [Environments](#5-environments)  
6. [Configuration](#6-configuration)  
7. [Local (Bare-Metal) Deployment](#7-local-bare-metal-deployment)  
8. [Docker Deployment](#8-docker-deployment)  
9. [Reverse Proxy & TLS](#9-reverse-proxy--tls)  
10. [Federated Learning Topologies](#10-federated-learning-topologies)  
11. [Privacy-Preserving FL (Shamir + DP)](#11-privacy-preserving-fl-shamir--dp)  
12. [Blockchain Data Anchoring](#12-blockchain-data-anchoring)  
13. [Scheduling & Automation](#13-scheduling--automation)  
14. [Monitoring, Logging & Health Checks](#14-monitoring-logging--health-checks)  
15. [Security & GDPR Operations](#15-security--gdpr-operations)  
16. [Performance & Capacity Planning](#16-performance--capacity-planning)  
17. [Backup, DR & Rollback](#17-backup-dr--rollback)  
18. [Troubleshooting](#18-troubleshooting)  
19. [Upgrade Procedure](#19-upgrade-procedure)  
20. [Appendix A — Example Files](#20-appendix-a--example-files)  

---

## 1) Audience & Assumptions

- **Audience**: DevOps engineers, ML engineers, and public-sector IT staff deploying for CPI/SIL pilots or production.  
- **Assumptions**:
  - Basic familiarity with Python environments, Docker, and Linux service management.
  - Ability to provision storage for datasets and results.
  - Network connectivity available between sites (for FL) or secure file-based exchange capability.

---

## 2) System Requirements

**Operating Systems**  
- Linux: Ubuntu 22.04/24.04 LTS (recommended)  
- Windows 10/11 (supported for development and PoC)  
- macOS 13+ (development only)

**Hardware (baseline)**  
- **Minimum**: CPU 4 cores, RAM 16 GB, Disk 50 GB  
- **Recommended**: CPU 8 cores, RAM 32 GB, SSD 100 GB  
- **GPU**: **Not required** (CPU training supported and validated)

**Performance Expectations (corrected)**
- **LightGBM Federated**: Minimal performance impact (-0.0005 F1-score vs centralized)
- **MLP Federated**: Moderate impact (~4% F1-score reduction vs centralized)
- **Privacy Mode**: Additional ~0.8% F1-score reduction beyond standard federated

**Networking**  
- Default Streamlit port: `8501`  
- Optional reverse proxy (80/443) if exposing externally  
- For FL, see [Topologies](#10-federated-learning-topologies) for data exchange options

---

## 3) Architecture for Deployment

Production deployment layers:

1. **Application/UI** — Streamlit app (`streamlit_app.py`) for operators  
2. **Training Services** — scripts under `scripts/` orchestrated by scheduler  
3. **Federated Coordinator** — coordinator host/process for FL rounds  
4. **Privacy Layer** — secure aggregation (Shamir) and DP budget accounting  
5. **Integrity Layer** — Merkle commitments and proof verification  
6. **Storage** — `data/` and `results*/` folders (local or network-attached)  
7. **Observability** — logs, health endpoints, and metrics

### Minimal single-host diagram

```
[ Operators ] --> [ Streamlit (8501) ] --> [ Scripts: 01..10 ] --> [ results/* ]
                                 ^                                  
                                 +---- health: /_stcore/health
```

### Cross-silo FL (file-based exchange)

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

Project structure (corrected):

```
📂 Disability Job Matching System/
├── README.md / README_IT.md
├── config.yaml                           # Core configuration
├── requirements.txt
├── streamlit_app.py                      # Main application
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
├── results/                              # Centralized model outputs
├── results_LightGBM_federated/          # LightGBM federated results
├── results_mlp_federated/               # Standard MLP federated results
├── results_mlp_federated_privacy/       # Privacy-preserving MLP results
├── results_blockchain_demo/             # Blockchain anchoring results
└── visualizations_federated_comparison/ # Comparative analysis charts
```

---

## 5) Environments

- **Development**: Local workstation, small samples, iterative runs  
- **Staging**: Production replica (same OS/Python), anonymized datasets  
- **Production**: Locked dependencies, stable config, scheduled jobs, restricted access

---

## 6) Configuration

Create and version `config.yaml` in repository root (corrected values):

```yaml
seed: 42
paths:
  raw_candidates: data/raw/Dataset_Candidati_Aggiornato.csv
  raw_companies: data/raw/Dataset_Aziende_con_Stima_Assunzioni.csv
  processed_dir: data/processed
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30                    # Default search radius (CORRECTED)
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
    epsilon: 1.0                         # Recommended value for utility/privacy balance
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

> Keep production copies of `config.yaml` outside VCS if they embed private paths.

---

## 7) Local (Bare-Metal) Deployment

**Prerequisites**
```bash
# Python 3.8+ with virtual environment
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, pandas, numpy, sklearn; print('Dependencies OK')"
```

**Launch Streamlit UI**
```bash
streamlit run streamlit_app.py
# Access at: http://localhost:8501
```

**Data Preparation**
```bash
# Generate extended datasets and training data
python scripts/01_generate_dataset.py

# Create data analysis visualizations
python scripts/02_visualize_dataset.py
```

**Centralized Training & Analysis**
```bash
# Train all 7 models with hyperparameter optimization
python scripts/03_train_models.py --config config.yaml

# Analyze results and generate performance reports
python scripts/04_analyze_results.py
```

**Federated Learning Workflows**
```bash
# LightGBM federated training (regional ensemble)
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py

# MLP federated training (true FedAvg)
python scripts/07_mlp_federated_training.py --aggregator fedavg

# Privacy-preserving MLP federated training
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5
python scripts/09_mlp_federated_privacy_visualization.py
```

Artifacts will appear under `results/` and `results_*/` directories.

---

## 8) Docker Deployment

### 8.1 Dockerfile (improved security)
```dockerfile
FROM python:3.11-slim

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check with proper curl syntax
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start application with proper networking configuration
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
```

### 8.2 Build and Run Commands
```bash
# Build image with appropriate tag
docker build -t djms:latest .

# Run with proper volume mounts and environment
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

## 9) Reverse Proxy & TLS

### NGINX Configuration (improved)
```nginx
upstream djms_backend {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name djms.example.org;
    
    # Security headers
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
        
        # Timeouts for large model operations
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /_stcore/health {
        proxy_pass http://djms_backend;
        access_log off;
    }
}
```

Add TLS via certbot or your standard process.

---

## 10) Federated Learning Topologies

### 10.1 File-Based Coordination (recommended for CPI/SIL pilots)

**Workflow Overview:**
1. Each region runs training scripts locally on private data
2. Export model deltas/weights to secure shared folder (SFTP/SMB)
3. Coordinator aggregates updates and publishes global model
4. Repeat for configured rounds

**Regional Client Execution:**
```bash
# Each CPI/SIL site runs locally
python scripts/07_mlp_federated_training.py \
  --mode client \
  --region CPI_Verona \
  --round 1 \
  --export_path /secure/shared/round_001/
```

**Coordinator Aggregation:**
```bash
# Central coordinator aggregates updates
python scripts/07_mlp_federated_training.py \
  --mode aggregate \
  --round 1 \
  --import_path /secure/shared/round_001/ \
  --export_path /secure/shared/round_001/global/
```

### 10.2 Performance Expectations (corrected)

**LightGBM Federated (Regional Ensemble):**
- Expected performance: F1 ≈ 0.9007 vs centralized F1 ≈ 0.9012
- Performance impact: -0.0005 F1-score (minimal degradation)
- Training time: Similar to centralized per region

**MLP Federated (True FedAvg):**
- Expected performance: F1 ≈ 0.788 vs centralized F1 ≈ 0.828  
- Performance impact: ~4% F1-score reduction
- Training time: Depends on rounds and communication frequency

---

## 11) Privacy-Preserving FL (Shamir + DP)

### Configuration and Expected Performance

**Shamir Secret Sharing:**
- Threshold scheme (3-of-5) with per-parameter masking
- Dropout recovery for client disconnections
- Cryptographic security without performance degradation

**Differential Privacy:**
- Per-round clipping with single Gaussian noise injection
- RDP accounting for (ε, δ) budget tracking
- Recommended: ε=1.0, δ=1e-6 for utility/privacy balance

**Performance Impact (corrected):**
- Privacy-preserving MLP: F1 ≈ 0.788 (similar to standard federated)
- Additional privacy cost: ~0.8% F1-score reduction vs standard federated
- Total cost vs centralized: ~4.8% F1-score reduction

**Example Commands:**
```bash
# Strong privacy (ε=0.5)
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 0.5 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5

# Recommended privacy (ε=1.0) - good utility/privacy balance  
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 1.0 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5

# Relaxed privacy (ε=2.0)
python scripts/08_mlp_federated_privacy.py \
  --dp.epsilon 2.0 --dp.delta 1e-6 \
  --secure_agg.threshold 3-of-5
```

---

## 12) Blockchain Data Anchoring

### Performance Characteristics (from benchmarks)

**Build Performance:**
- 100 records: 2.28s total (KDF: 1.58s, Tree: 0.45s)
- 1,000 records: 30.47s total (KDF: 21.15s, Tree: 6.04s)  
- 10,000 records: 344.07s total (KDF: 239.40s, Tree: 68.40s)

**Proof Generation (O(log n) scaling):**
- 100 records: 1.11ms avg, 224 bytes
- 1,000 records: 2.61ms avg, 320 bytes
- 10,000 records: 20.65ms avg, 448 bytes

**Verification Performance:**
- Average: 24.49ms with 100% success rate
- No false positives or verification failures in benchmarks

### Implementation Commands
```bash
# Create Merkle commitments for all results
python scripts/blockchain_data_anchoring.py \
  --inputs results results_LightGBM_federated results_mlp_federated results_mlp_federated_privacy \
  --output results_blockchain_demo

# Benchmark anchoring performance on your hardware
python scripts/10_blockchain_anchoring_bench.py
```

---

## 13) Scheduling & Automation

### Linux (systemd + cron)
```bash
# Create service for UI (optional - can use docker)
sudo systemctl enable djms-ui.service

# Training schedule via cron
crontab -e
```

```cron
# Daily training at 22:00 on weekdays
0 22 * * 1-5 /opt/djms/scripts/run_training.sh >> /var/log/djms/training.log 2>&1

# Weekly federated learning on Sundays
0 20 * * 0 /opt/djms/scripts/run_federated.sh >> /var/log/djms/federated.log 2>&1

# Monthly blockchain anchoring
0 23 1 * * /opt/djms/scripts/run_anchoring.sh >> /var/log/djms/anchoring.log 2>&1
```

### Windows (Task Scheduler)
```powershell
# Create scheduled task for training
schtasks /create /tn "DJMS Training" /tr "C:\djms\scripts\run_training.bat" /sc weekly /d TUE /st 22:00
```

### Docker-based Scheduling
```bash
# Use host cron to invoke docker exec commands
0 22 * * 1-5 docker exec djms_ui python scripts/03_train_models.py --config config.yaml >> /var/log/djms/training.log 2>&1
```

---

## 14) Monitoring, Logging & Health Checks

### Health Monitoring
```bash
# Check application health
curl -f http://localhost:8501/_stcore/health

# Monitor system resources
docker stats djms_ui

# Check recent logs
docker logs --tail 50 djms_ui
```

### Logging Configuration
```python
# Enhanced logging setup
import logging
from datetime import datetime

def setup_production_logging():
    """Setup production-ready logging with rotation"""
    import logging.handlers
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Setup handlers
    handlers = [
        logging.StreamHandler(),  # Console
        logging.handlers.RotatingFileHandler(
            f'logs/djms_{datetime.now().strftime("%Y%m")}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)
```

### Performance Metrics Collection
```bash
# Automated metrics collection script
#!/bin/bash
# collect_metrics.sh

METRICS_DIR="/opt/djms/metrics"
DATE=$(date +%Y%m%d_%H%M%S)

# System metrics
echo "timestamp,cpu_percent,memory_percent,disk_used_gb" > $METRICS_DIR/system_$DATE.csv
echo "$DATE,$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'),$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}'),$(df -h | grep "/opt" | awk '{print $3}' | sed 's/G//')" >> $METRICS_DIR/system_$DATE.csv

# Model performance metrics
if [ -f "/opt/djms/results/merged_model_summary.csv" ]; then
    cp /opt/djms/results/merged_model_summary.csv $METRICS_DIR/model_performance_$DATE.csv
fi

# Federated learning metrics
if [ -f "/opt/djms/results_LightGBM_federated/complete_model_comparison.csv" ]; then
    cp /opt/djms/results_LightGBM_federated/complete_model_comparison.csv $METRICS_DIR/federated_performance_$DATE.csv
fi
```

---

## 15) Security & GDPR Operations

### Access Control Implementation
```bash
# Secure file permissions
sudo chown -R djms:djms /opt/djms
chmod 750 /opt/djms/data/                 # Data directory
chmod 750 /opt/djms/results*/             # Results directories
chmod 755 /opt/djms/scripts/              # Scripts
chmod 644 /opt/djms/config.yaml           # Configuration
chmod 600 /opt/djms/.env                  # Environment secrets
```

### GDPR Compliance Checklist
- [ ] **Data Minimization**: Only processed/aggregated data in federated mode
- [ ] **Purpose Limitation**: Clear documentation of ML training purposes  
- [ ] **Access Controls**: Role-based access to data and results directories
- [ ] **Retention Policies**: Automated cleanup of intermediate training artifacts
- [ ] **DSR Procedures**: Documented access/erasure workflows
- [ ] **Data Protection Impact Assessment**: Completed for privacy-preserving FL
- [ ] **Processor Agreements**: Contracts with regional CPI/SIL participants

### Privacy Budget Management
```python
# Privacy budget tracking for production
class PrivacyBudgetManager:
    def __init__(self, max_epsilon=5.0, max_delta=1e-5):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.current_epsilon = 0.0
        self.current_delta = 0.0
        self.rounds = []
    
    def can_execute_round(self, epsilon, delta):
        """Check if round can be executed within budget"""
        return (self.current_epsilon + epsilon <= self.max_epsilon and
                self.current_delta + delta <= self.max_delta)
    
    def execute_round(self, epsilon, delta, round_info):
        """Record executed round"""
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

## 16) Performance & Capacity Planning

### Resource Requirements by Workload

**Centralized Training (7 models):**
- CPU: 4-8 cores, 2-4 hours training time
- RAM: 16-32 GB peak during hyperparameter optimization
- Storage: 2-5 GB for models and intermediate results

**LightGBM Federated Training:**
- Per region: CPU 2-4 cores, RAM 8-16 GB
- Expected performance: F1 ≈ 0.9007 (-0.0005 vs centralized)
- Training time: Similar to centralized per region

**MLP Federated Training:**
- Per region: CPU 2-4 cores, RAM 8-16 GB  
- Expected performance: F1 ≈ 0.788 (~4% vs centralized F1 ≈ 0.828)
- Training time: 30-60 minutes per round, 10-30 rounds typical

**Privacy-Preserving Mode:**
- Additional compute: ~10-20% overhead for DP operations
- Expected performance: F1 ≈ 0.788 (~0.8% additional cost)
- Memory: +2-4 GB for secure aggregation operations

### Capacity Planning Guidelines
```yaml
# config.yaml adjustments for different environments

# Development/Testing
federated:
  rounds: 5
  batch_size: 128
model_training:
  optuna_trials: 20
  n_jobs: 2

# Production (high-resource)
federated:
  rounds: 15
  batch_size: 512
model_training:
  optuna_trials: 100
  n_jobs: 8

# Production (resource-constrained)
federated:
  rounds: 8
  batch_size: 256
model_training:
  optuna_trials: 30
  n_jobs: 4
```

---

## 17) Backup, DR & Rollback

### Backup Strategy
```bash
#!/bin/bash
# comprehensive_backup.sh

BACKUP_ROOT="/backup/djms"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$DATE"

mkdir -p $BACKUP_DIR

# Critical configurations
cp /opt/djms/config.yaml $BACKUP_DIR/
cp /opt/djms/requirements.txt $BACKUP_DIR/
cp /opt/djms/.env $BACKUP_DIR/ 2>/dev/null || true

# Data (if not too large)
cp -r /opt/djms/data/processed $BACKUP_DIR/

# Models and results
cp -r /opt/djms/results $BACKUP_DIR/
cp -r /opt/djms/results_* $BACKUP_DIR/

# Documentation
cp -r /opt/djms/docs $BACKUP_DIR/

# Create archive
tar -czf $BACKUP_ROOT/djms_backup_$DATE.tar.gz -C $BACKUP_ROOT $DATE
rm -rf $BACKUP_DIR

# Retain last 7 days locally, 4 weeks offsite
find $BACKUP_ROOT -name "djms_backup_*.tar.gz" -mtime +7 -delete
```

### Disaster Recovery Testing
```bash
#!/bin/bash
# dr_test.sh - Test restore procedure

RESTORE_DIR="/tmp/djms_restore_test"
BACKUP_FILE="$1"

# Extract backup
mkdir -p $RESTORE_DIR
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# Test configuration
cd $RESTORE_DIR/*/
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test model loading
python -c "
import joblib
import glob
models = glob.glob('results/*.joblib')
if models:
    model = joblib.load(models[0])
    print(f'Successfully loaded model: {models[0]}')
else:
    print('No models found')
"

echo "DR test completed for $BACKUP_FILE"
```

---

## 18) Troubleshooting

### Common Issues and Solutions

**Application Won't Start**
```bash
# Check dependencies
pip list | grep -E "(streamlit|pandas|scikit-learn)"
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Should be 3.8+

# Check port availability
netstat -tlnp | grep 8501
```

**Docker Issues**
```bash
# Health check failures
docker exec djms_ui curl -f http://localhost:8501/_stcore/health
docker logs djms_ui --tail 50

# Container resource issues
docker stats djms_ui
docker exec djms_ui df -h
docker exec djms_ui free -m
```

**Performance Issues**
```bash
# Federated learning underperformance
# Expected: LightGBM -0.0005 F1, MLP ~4% F1 reduction
# If significantly worse, check data splits and regional balance

# Monitor training logs
tail -f logs/training.log

# Check system resources during training
htop
iotop

# Verify model files exist and are valid
ls -la results/*.joblib
python -c "import joblib; print('Model valid:', joblib.load('results/LightGBM_Optimized.joblib') is not None)"
```

**Geocoding Issues**
```bash
# Network connectivity
ping nominatim.openstreetmap.org

# Cache issues
ls -la data/processed/geocoding_cache.json
# If corrupted, remove and regenerate
rm data/processed/geocoding_cache.json
python scripts/01_generate_dataset.py
```

**Federated Learning Issues**
```bash
# Check regional data splits
python -c "
import pandas as pd
import glob
files = glob.glob('data/federated/CPI_*.csv')
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: {len(df)} samples')
"

# Verify aggregation results
ls -la results_mlp_federated/
ls -la results_mlp_federated_privacy/
```

---

## 19) Upgrade Procedure

### Pre-Upgrade Checklist
1. **Announce maintenance window** to operators
2. **Backup current system** (config, data, models, results)
3. **Test upgrade on staging** environment first
4. **Review changelog** and breaking changes
5. **Prepare rollback plan**

### Upgrade Steps
```bash
#!/bin/bash
# upgrade_procedure.sh

set -e  # Exit on any error

BACKUP_DIR="/opt/djms_backup_$(date +%Y%m%d_%H%M%S)"
DJMS_ROOT="/opt/djms"

echo "Starting DJMS upgrade procedure..."

# 1. Create backup
echo "Creating system backup..."
cp -r $DJMS_ROOT $BACKUP_DIR

# 2. Stop services
echo "Stopping services..."
docker-compose down || systemctl stop djms-ui || true

# 3. Update code
echo "Updating codebase..."
cd $DJMS_ROOT
git fetch origin
git checkout main
git pull origin main

# 4. Update dependencies
echo "Updating Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Rebuild Docker image (if using containers)
echo "Rebuilding Docker image..."
docker build -t djms:latest .

# 6. Run migrations/updates (if any)
echo "Running system updates..."
python scripts/01_generate_dataset.py --verify-only || true

# 7. Start services
echo "Starting services..."
docker-compose up -d || systemctl start djms-ui

# 8. Health check
echo "Performing health check..."
sleep 30
curl -f http://localhost:8501/_stcore/health

echo "Upgrade completed successfully!"
echo "Backup stored at: $BACKUP_DIR"
```

### Post-Upgrade Validation
```bash
# Verify critical functionality
python scripts/03_train_models.py --dry-run
streamlit run streamlit_app.py --help

# Check model compatibility
python -c "
import joblib
import glob
models = glob.glob('results/*.joblib')
for model_file in models:
    try:
        model = joblib.load(model_file)
        print(f'✓ {model_file} - OK')
    except Exception as e:
        print(f'✗ {model_file} - ERROR: {e}')
"

# Verify federated learning components
python scripts/07_mlp_federated_training.py --validate-setup
```

---

## 20) Appendix A — Example Files

### systemd Service File
```ini
# /etc/systemd/system/djms-ui.service
[Unit]
Description=DJMS Streamlit UI Service
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

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/djms/results /opt/djms/results_* /opt/djms/logs

[Install]
WantedBy=multi-user.target
```

### Logrotate Configuration
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
        # Restart service to reopen log files
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

### Environment Variables Template
```bash
# .env file template
DJMS_ENV=production
DJMS_SECRET_KEY=your-secret-key-here
DJMS_DEBUG=false

# Database settings (if implemented)
DATABASE_URL=sqlite:///opt/djms/data/djms.db

# External service API keys (if needed)
NOMINATIM_USER_AGENT=djms-production-v1
GEOCODING_API_KEY=your-api-key-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/djms/application.log

# Security
MAX_UPLOAD_SIZE=100MB
ALLOWED_HOSTS=djms.example.org,localhost
```

### Training Script Wrapper
```bash
#!/bin/bash
# /opt/djms/scripts/run_training.sh

set -euo pipefail

DJMS_ROOT="/opt/djms"
LOG_DIR="/var/log/djms"
DATE=$(date +%Y%m%d_%H%M%S)

cd $DJMS_ROOT
source venv/bin/activate

echo "[$DATE] Starting training pipeline..." >> $LOG_DIR/training.log

# Pre-training checks
if [ ! -f "data/processed/Enhanced_Training_Dataset.csv" ]; then
    echo "[$DATE] Training data not found, generating..." >> $LOG_DIR/training.log
    python scripts/01_generate_dataset.py >> $LOG_DIR/training.log 2>&1
fi

# Run centralized training
python scripts/03_train_models.py --config config.yaml >> $LOG_DIR/training.log 2>&1

# Run analysis
python scripts/04_analyze_results.py >> $LOG_DIR/training.log 2>&1

# Run federated learning (weekly)
if [ "$(date +%u)" -eq 7 ]; then  # Sunday
    echo "[$DATE] Running weekly federated learning..." >> $LOG_DIR/training.log
    python scripts/05_LightGBM_federated_training.py >> $LOG_DIR/training.log 2>&1
    python scripts/07_mlp_federated_training.py >> $LOG_DIR/training.log 2>&1
fi

echo "[$DATE] Training pipeline completed successfully" >> $LOG_DIR/training.log
```

---

