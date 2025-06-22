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