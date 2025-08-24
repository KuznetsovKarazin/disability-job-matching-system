# User Guide — Disability Job Matching System
_Last updated: 2025-08-24 15:45_

> This guide is written for **operators** (CPI/SIL), **analysts**, and **developers**.
> It covers workflows, UI operations, dataset preparation, federated learning, privacy controls,
> blockchain anchoring, troubleshooting, and best practices.

**Related documents**: `README.md`, `technical_documentation.md`, `deployment_guide.md`, `api_reference.md`.

## Table of Contents
1. [Introduction](#1-introduction)
2. [UI Tour](#2-ui-tour)
3. [Data Preparation & Contracts](#3-data-preparation--contracts)
4. [Running the System](#4-running-the-system)
5. [Federated Learning Workflows](#5-federated-learning-workflows)
6. [Privacy & Security (Operator View)](#6-privacy--security-operator-view)
7. [Blockchain Anchoring (Operator View)](#7-blockchain-anchoring-operator-view)
8. [Results, Reports & Visualizations](#8-results-reports--visualizations)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQ](#10-faq)
11. [Best Practices](#11-best-practices)
12. [Support & Contacts](#12-support--contacts)
13. [Appendix A — CLI Reference](#13-appendix-a--cli-reference)
14. [Appendix B — Configuration Reference](#14-appendix-b--configuration-reference)
15. [Appendix C — Results Directory Map](#15-appendix-c--results-directory-map)

## 1. Introduction

The Disability Job Matching System helps public employment services (CPI/SIL) match candidates with inclusive jobs. It supports both **centralized** and **federated** training, and includes **privacy-preserving** capabilities and **data integrity anchoring**.

**Key concepts**
- Centralized training with `data/processed/Enhanced_Training_Dataset.csv`.
- Federated learning across regions without sharing raw data.
- Privacy modes: secure aggregation (Shamir) and Differential Privacy (RDP-based).
- Anchoring: Merkle commitments + proofs for long-term integrity.

**Roles**
- **Operator**: runs matches, reviews recommendations, generates reports.
- **Analyst**: validates datasets, monitors metrics, compares models.
- **Developer**: maintains pipelines, optimizes models, configures deployments.

## 2. UI Tour

Launch:
```bash
streamlit run streamlit_app.py
```

**Main Interface Sections:**

**Home Dashboard**
- KPIs (candidates, companies, regions), quick links.
- System status and health indicators.

**Candidate Search (Ricerca Candidato)**
- Input candidate information (manual or from existing records)
- Configure search parameters: radius (**default 30 km**), attitude threshold
- View ranked company recommendations with compatibility scores
- Export results to CSV/PNG

**Analytics Dashboard**
- System overview metrics and performance indicators
- Distribution charts for disabilities and company sectors
- Placement success tracking and trends

**Dataset Management**
- Browse candidate and company data
- Export functions for external analysis
- Data quality verification tools

**System Information (Info Sistema)**
- Technical status and configuration
- Model performance metrics
- Privacy and security status

### Sidebar Configuration Panel

**System Configuration (Configurazione Sistema)**:
- **Model Selection**: Choose AI model (if multiple available)
- **Attitude Threshold**: Minimum employment readiness (0.0-1.0)
- **Maximum Distance**: Search radius in kilometers (**5-50 km, default 30 km**)
- **Top Recommendations**: Number of results to show (3-10)

## 3. Data Preparation & Contracts

**Raw Input Data (`data/raw/`)**
- `Dataset_Candidati_Aggiornato.csv` - Candidate master data
- `Dataset_Aziende_con_Stima_Assunzioni.csv` - Company and role data

**Processed Data (`data/processed/`)**
- `Dataset_Candidati_Aggiornato_Extended.csv` - Enhanced candidate data
- `Dataset_Aziende_con_Stima_Assunzioni_Extended.csv` - Enhanced company data
- `Enhanced_Training_Dataset.csv` - Canonical training table

**Data Quality Recommendations:**
- Keep schemas consistent across regions.
- Document changes in `SCHEMA.md`; update `01_generate_dataset.py`.
- Validate geocoding; align with `ui.distance_max_km` (**30 km default**).
- Ensure Italian address format: "City, Province, Italy"

## 4. Running the System

### Centralized Training
```bash
# Train all 7 models with hyperparameter optimization
python scripts/03_train_models.py --config config.yaml

# Analyze results and generate performance reports
python scripts/04_analyze_results.py
```

### Data Preparation and Visualization
```bash
# Generate extended datasets and training data
python scripts/01_generate_dataset.py

# Create data analysis visualizations
python scripts/02_visualize_dataset.py
```

### LightGBM Federated (regional → ensemble)
```bash
# Train regional models and create federated ensemble
python scripts/05_LightGBM_federated_training.py

# Generate federated learning visualizations
python scripts/06_LightGBM_federated_visualization.py
```

### MLP Federated (standard + privacy)
```bash
# Standard federated learning with various aggregators
python scripts/07_mlp_federated_training.py --aggregator fedavg

# Privacy-preserving federated learning with DP
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --secure_agg.threshold 3-of-5

# Compare federated learning results
python scripts/09_mlp_federated_privacy_visualization.py
```

### Blockchain Anchoring
```bash
# Create Merkle commitments and proofs
python scripts/blockchain_data_anchoring.py

# Benchmark anchoring performance
python scripts/10_blockchain_anchoring_bench.py
```

## 5. Federated Learning Workflows

### 5.1 LightGBM (Regional Ensembling)
**Approach**: Train per-region models; combine via weighted ensemble (weight ∝ sample count).

```bash
python scripts/05_LightGBM_federated_training.py
python scripts/06_LightGBM_federated_visualization.py
```

**Expected Performance**: 
- Centralized: F1 ≈ 0.9012
- Federated: F1 ≈ 0.9007 (-0.0005 degradation)

### 5.2 MLP (True FedAvg)
**Aggregation Options:**

**FedAvg (default)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator fedavg --rounds 10 --batch_size 256
```

**Trimmed Mean (robust to outliers)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator trimmed_mean --rounds 10 --batch_size 256
```

**Coordinate Median (most robust)**:
```bash
python scripts/07_mlp_federated_training.py --aggregator coordinate_median --rounds 10 --batch_size 256
```

**Expected Performance**:
- Centralized: F1 ≈ 0.828
- Federated: F1 ≈ 0.788 (~4% reduction)

### 5.3 Privacy-preserving FL (Shamir + DP)
**Different Privacy Levels:**

**Strong Privacy (ε=0.5)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 0.5 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Moderate Privacy (ε=1.0, recommended)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 1.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Relaxed Privacy (ε=2.0)**:
```bash
python scripts/08_mlp_federated_privacy.py --dp.epsilon 2.0 --dp.delta 1e-6 --secure_agg.threshold 3-of-5
```

**Expected Performance**: F1 ≈ 0.788 (minimal additional privacy cost)

## 6. Privacy & Security (Operator View)

### Secure Aggregation
- **Shamir Secret Sharing**: 3-of-5 threshold with per-parameter masking
- **Dropout Recovery**: Automatic handling of client disconnections
- **Deterministic Seeds**: Ensures reproducible results across runs

### Differential Privacy
- **Per-round Clipping**: Gradient bounds before noise injection
- **Single Gaussian Noise**: Applied once per aggregation round
- **RDP Accounting**: Tracks cumulative privacy budget (ε, δ)

### Security Checklist
- Raw data never leaves regional nodes
- Protect `results_mlp_federated_privacy/` directory
- Rotate access credentials for `data/` and `results/`
- Monitor privacy budget consumption

## 7. Blockchain Anchoring (Operator View)

### Purpose
- **Integrity Proofs**: Merkle commitments for models and reports
- **Long-term Verification**: Tamper-evident audit trails
- **Compliance**: Meet regulatory requirements for public institutions

### Running Anchoring
```bash
# Create commitments and proofs
python scripts/blockchain_data_anchoring.py

# Performance benchmarking
python scripts/10_blockchain_anchoring_bench.py
```

### Performance Expectations
- **100 records**: ~2.3s build time, 1.1ms proof generation
- **1,000 records**: ~30.5s build time, 2.6ms proof generation  
- **10,000 records**: ~344s build time, 20.7ms proof generation

**Outputs**: Manifests, proofs, verification logs in `results_blockchain_demo/`

## 8. Results, Reports & Visualizations

### Results Directory Structure
- **`results/`**: Centralized artifacts (*.joblib), learning curves, merged_model_summary.csv
- **`results_LightGBM_federated/`**: Regional/federated/centralized LightGBM comparison
- **`results_mlp_federated/`**: Standard MLP federated learning results
- **`results_mlp_federated_privacy/`**: Privacy-preserving MLP FL outputs
- **`visualizations_federated_comparison/`**: Comparative charts and analysis
- **`results_blockchain_demo/`**: Anchoring artifacts and verification data

### Performance Interpretation
- **F1-Score Priority**: Primary metric for recommendation quality
- **Performance Trade-offs**:
  - LightGBM Federated: Minimal impact (-0.0005 F1)
  - MLP Federated: Moderate impact (~4% F1 reduction)
  - Privacy Mode: Additional ~0.8% F1 reduction
- **ROC-AUC Context**: Values ~0.70 reflect intentional probabilistic data design

## 9. Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
# Solution: Reinstall requirements
pip install -r requirements.txt
```

**Scikit-learn Import Errors**
```bash
# Solution: Pin compatible versions
pip install scikit-learn==1.6.1 imbalanced-learn==0.13.0
```

**Docker Health Check Failures**
- Ensure `curl` is installed in container
- Verify `/_stcore/health` endpoint accessibility

**Empty Result Comparisons**
- Confirm result folders exist and contain data
- Check CSV file paths and permissions

**Geographic Distance Errors**
- Use format: "City, Province, Italy"
- Verify Italian city name spelling
- Check geocoding cache: `data/processed/geocoding_cache.json`

### Performance Issues

**Slow Response Times**
1. Clear browser cache and reload
2. Check internet connection stability
3. Reduce distance threshold for faster processing
4. Consider lower batch sizes for federated learning

**No Candidate Matches Found**
1. **Increase distance threshold** from 30km to 40-50km for rural areas
2. **Lower attitude threshold** from 0.3 to 0.2-0.25
3. **Review exclusions** - ensure not overly restrictive
4. **Verify location** - must be valid Italian address

## 10. FAQ

**Q: Can I use the system without sharing data between regions?**
A: Yes - MLP federated learning enables collaboration without data sharing.

**Q: What is the default search radius?**
A: 30 km (configurable via `config.yaml: ui.distance_max_km`).

**Q: How much does privacy mode impact performance?**
A: ~4.8% F1-score reduction for strong privacy guarantees (ε=1.0, δ=1e-6).

**Q: How can I verify results months later?**
A: Use blockchain anchoring proofs stored in `results_blockchain_demo/`.

**Q: Are GPUs required?**
A: No - system is CPU-optimized and works well on standard hardware.

**Q: Why are some scores lower in federated vs centralized mode?**
A: This is expected due to data distribution differences. LightGBM shows minimal impact, MLP shows moderate trade-offs.

## 11. Best Practices

### Daily Operations

**Morning Routine**:
1. Check system status in Info Sistema tab
2. Review overnight analytics changes
3. Verify priority candidate information currency

**Candidate Processing**:
1. Always verify exclusions with candidate before searching
2. Use existing candidate data when available for consistency
3. Document successful placements for system improvement

**Result Evaluation**:
1. Focus on top 3 recommendations for initial outreach
2. Consider geographic preferences alongside scores
3. Review compatibility details beyond numerical scores

### Weekly Reviews

**Data Quality Maintenance**:
- Update candidate assessments and availability
- Verify company position openings and requirements
- Remove or update inactive company listings
- Review geocoding accuracy for new addresses

**Performance Monitoring**:
- Track placement success rates by score ranges
- Identify highest-performing companies and sectors
- Monitor system response times and accuracy
- Note patterns in successful vs unsuccessful placements

### Configuration Optimization

**Urban vs Rural Settings**:
- **Urban areas**: 20-25 km radius for local focus
- **Rural areas**: 35-50 km radius for adequate options
- **Mixed regions**: 30 km default usually optimal

**Threshold Tuning**:
- **High-volume periods**: Increase attitude threshold (0.4-0.5) for quality
- **Low-volume periods**: Decrease threshold (0.2-0.3) for broader matching
- **Specialized placements**: Adjust based on disability type and requirements

## 12. Support & Contacts

### Technical Support
- **Primary Contact**: michele.melch@gmail.com
- **Academic Support**: oleksandr.kuznetsov@uniecampus.it
- **Response Time**: 24-48 hours for non-critical issues

### Institutional Partners
- **CPI Villafranca di Verona**: Operational guidance and validation
- **SIL Veneto**: Regional coordination and best practices
- **Università eCampus**: Technical development and research

### Getting Help
**Before Contacting Support**:
1. Note exact error messages and reproduction steps
2. Check system status in Info Sistema tab
3. Try basic troubleshooting solutions
4. Gather system configuration details

**Include in Support Requests**:
- Screenshots of issues
- Error messages (exact text)
- Steps to reproduce the problem
- System configuration details

## 13. Appendix A — CLI Reference

### Core Training Scripts

**01_generate_dataset.py** - Data extension and training generation
```bash
python scripts/01_generate_dataset.py [--config config.yaml]
```

**02_visualize_dataset.py** - Data analysis and visualization
```bash
python scripts/02_visualize_dataset.py [--config config.yaml]
```

**03_train_models.py** - Centralized model training
```bash
python scripts/03_train_models.py [--config config.yaml]
```

**04_analyze_results.py** - Performance analysis and reporting
```bash
python scripts/04_analyze_results.py [--config config.yaml]
```

### Federated Learning Scripts

**05_LightGBM_federated_training.py** - LightGBM federated training
```bash
python scripts/05_LightGBM_federated_training.py [--config config.yaml]
```

**06_LightGBM_federated_visualization.py** - LightGBM federated visualization
```bash
python scripts/06_LightGBM_federated_visualization.py [--config config.yaml]
```

**07_mlp_federated_training.py** - MLP federated training
```bash
python scripts/07_mlp_federated_training.py [--config config.yaml] [--aggregator {fedavg,trimmed_mean,coordinate_median}] [--rounds ROUNDS] [--batch_size BATCH_SIZE]
```

**08_mlp_federated_privacy.py** - Privacy-preserving MLP federated training
```bash
python scripts/08_mlp_federated_privacy.py [--config config.yaml] [--dp.epsilon EPSILON] [--dp.delta DELTA] [--secure_agg.threshold THRESHOLD]
```

**09_mlp_federated_privacy_visualization.py** - Privacy federated visualization
```bash
python scripts/09_mlp_federated_privacy_visualization.py [--config config.yaml]
```

### Blockchain Scripts

**blockchain_data_anchoring.py** - Create Merkle commitments and proofs
```bash
python scripts/blockchain_data_anchoring.py [--config config.yaml]
```

**10_blockchain_anchoring_bench.py** - Benchmark anchoring performance
```bash
python scripts/10_blockchain_anchoring_bench.py [--config config.yaml]
```

## 14. Appendix B — Configuration Reference

**Key Configuration Fields:**
```yaml
seed: 42
paths:
  training_csv: data/processed/Enhanced_Training_Dataset.csv
  results_dir: results
ui:
  distance_max_km: 30  # Default search radius
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

## 15. Appendix C — Results Directory Map

**Directory Structure:**
- `results/` — Centralized models & performance plots
- `results_LightGBM_federated/` — LightGBM federated learning artifacts  
- `results_mlp_federated/` — Standard MLP federated outputs
- `results_mlp_federated_privacy/` — Privacy-preserving MLP federated outputs
- `results_blockchain_demo/` — Blockchain anchoring artifacts
- `visualizations_federated_comparison/` — Comparative analysis charts

**Key Files:**
- `merged_model_summary.csv` — Consolidated centralized model performance
- `complete_model_comparison.csv` — Federated vs centralized comparison
- `experiment_metadata.json` — Detailed experimental configuration and results

---

