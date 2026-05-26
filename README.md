
# Federated Learning Fraud Detection Attack Simulation Toolkit

## Overview

This is a comprehensive federated learning (FL) attack simulation toolkit designed to study security vulnerabilities in FL systems using LightGBM models for fraud detection with the IEEE-CIS dataset.

## Features

- **6 Attack Types Implemented**:
  - Data Poisoning: Label Flip, Backdoor
  - Model Poisoning: Byzantine, Scaling
  - Identity Attacks: Free-Ride, Sybil

- **Robust Detection & Defense**:
  - Multiple detection mechanisms (update norms, cosine similarity, label distribution, etc.)
  - Attack classification
  - Rule-based and anomaly detection

- **Aggregation Methods**:
  - Rotation Aggregation
  - FedAvg
  - Krum
  - Trimmed Mean

## Getting Started

### Prerequisites

1. **Python 3.8+**
2. **IEEE-CIS Fraud Detection Dataset** (obtained separately)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Siddheshh21/Federated-Learning--Fraud-Detection--v1.git
   cd Federated-Learning--Fraud-Detection--v1
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain and set up the dataset**:
   - Download the IEEE-CIS Fraud Detection dataset
   - Partition it horizontally across 5 simulated clients in the `data/` directory
   - Directory structure for `data/`:
     ```
     data/
     ├── Client_1/
     │   ├── local_train.csv
     │   ├── client_test.csv
     │   ├── server_share.csv
     │   └── validation.csv
     ├── Client_2/
     │   └── (same structure as Client_1)
     ├── Client_3/
     │   └── (same structure as Client_1)
     ├── Client_4/
     │   └── (same structure as Client_1)
     ├── Client_5/
     │   └── (same structure as Client_1)
     └── test_data.csv
     ```

### Running the Project

1. **Clean FL Training**:
   ```bash
   python main.py
   ```

2. **Attack Testing**:
   ```bash
   python run_test_attack.py
   ```

## Project Structure

```
Federated-Learning--Fraud-Detection--v1/
├── src/
│   ├── attacks_comprehensive.py      # All attack implementations
│   ├── detection.py                   # Attack detection & defense
│   ├── enhanced_federated_loop.py    # Enhanced FL with attacks
│   ├── original_fl_rotation.py       # Core FL training loop
│   ├── evaluation.py                  # Evaluation metrics
│   ├── json_output_handler.py         # JSON output generation
│   ├── interactive_attack_tester.py  # Interactive attack tester
│   ├── config.py                      # Configuration management
│   ├── logger.py                      # Logging utilities
│   └── config/
│       └── experiment.yaml            # Experiment config
├── main.py                            # Entry point for clean FL
├── check_data.py                      # Data checking utility
├── requirements.txt                   # Dependencies
├── .gitignore                        # Exclude large files
└── README.md                         # This file
```

## Dependencies

- numpy &gt;= 1.21.0
- pandas &gt;= 1.3.0
- lightgbm &gt;= 3.3.0
- scikit-learn &gt;= 0.24.0
- matplotlib &gt;= 3.4.0
- seaborn &gt;= 0.11.0
- pyyaml &gt;= 6.0

## Notes

- **Data/Artifacts Not Included**: Large dataset files, trained models, and training artifacts are not stored in the repository.
- **Clean Baseline**: For attack testing, you'll need a clean FL baseline; run `python main.py` first to generate one.

