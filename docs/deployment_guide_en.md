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