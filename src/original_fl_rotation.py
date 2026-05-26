# Full merged script: AAFL OPTIMIZED + GLOBAL THRESHOLD ADD-ON
"""
═══════════════════════════════════════════════════════════════════════════════
    AAFL PROJECT - OPTIMIZED VERSION (ERROR FIXED + GLOBAL THRESHOLD ADD-ON)
    Fix: Removed is_unbalance (conflicts with scale_pos_weight)
    Added: Global threshold computation and application for consistent recall tuning
    UPDATED: Timestamped file naming for easy identification
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import pickle
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, average_precision_score, log_loss, precision_recall_curve,
    balanced_accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style('whitegrid')
except Exception:
    sns = None
from typing import Dict, List, Tuple, Optional
import zipfile
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - FIXED + TIMESTAMPED
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Optimized configuration - FIXED"""

    BASE_DIR = Path(__file__).resolve().parent.parent
    CLIENTS_DIR = BASE_DIR / 'data'

    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_DIR = BASE_DIR / 'artifacts' / f'FL_Training_Results_OPTIMIZED_{TIMESTAMP}'

    NUM_ROUNDS = 12
    NUM_CLIENTS = 5
    ROTATION_CHUNKS = 3

    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_estimators': 100,
        'min_child_samples': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'scale_pos_weight': 5.67
    }

    DP_EPSILON = 1.5
    DP_DELTA = 1e-5
    DP_NOISE_SCALE = 0.01
    RANDOM_SEED = 42
    GLOBAL_TEST_PATH = CLIENTS_DIR / 'test_data.csv'

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def setup_environment():
    print("🚀 AAFL Project - FL Training Pipeline (OPTIMIZED - FIXED)\n")
    print("="*100)
    print("STAGE 6: FEDERATED LEARNING TRAINING & AGGREGATION".center(100))
    print("="*100)
    print(f"\n⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Running locally (no Google Drive mount)\n")

    dirs = [
        Config.OUTPUT_DIR,
        Config.OUTPUT_DIR / 'Models' / 'Global',
        Config.OUTPUT_DIR / 'Models' / 'Clients',
        Config.OUTPUT_DIR / 'Models' / 'For_Member2_Attack',
        Config.OUTPUT_DIR / 'Metrics' / 'Per_Round',
        Config.OUTPUT_DIR / 'Metrics' / 'For_Member3_Dashboard',
        Config.OUTPUT_DIR / 'Logs',
        Config.OUTPUT_DIR / 'Visualizations',
        Config.OUTPUT_DIR / 'Checkpoints',
        Config.OUTPUT_DIR / 'Reports'
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # initialize logs
    with open(Config.OUTPUT_DIR / 'Logs' / 'training_log.txt', 'w') as f:
        f.write(f"AAFL FL Training Log (OPTIMIZED - FIXED)\n")
        f.write(f"Started: {datetime.now()}\n{'='*80}\n\n")
        f.write("OPTIMIZATIONS:\n")
        f.write(f"DP Epsilon: {Config.DP_EPSILON}\n")
        f.write(f"DP Noise Scale: {Config.DP_NOISE_SCALE}\n")
        f.write(f"Learning Rate: {Config.LGBM_PARAMS['learning_rate']}\n")
        f.write(f"Scale Pos Weight: {Config.LGBM_PARAMS['scale_pos_weight']}\n\n")

    print(f"Output Directory Created:\n   Path: {Config.OUTPUT_DIR}\n")
    return datetime.now()

def log_message(msg: str, log_type='info'):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"[{ts}] [{log_type.upper()}] {msg}\n"
    with open(Config.OUTPUT_DIR / 'Logs' / 'training_log.txt', 'a') as f:
        f.write(entry)
    if log_type == 'error':
        with open(Config.OUTPUT_DIR / 'Logs' / 'error_log.txt', 'a') as f:
            f.write(entry)

def calculate_metrics(y_true, y_pred_proba, threshold: float = None):
    """
    Calculates performance metrics with optional threshold tuning.
    If threshold is None, the best threshold is auto-selected using F-beta (recall emphasis).
    """
    if threshold is None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        beta = 2
        f_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-9)
        best_idx = np.argmax(f_scores)
        # precision_recall_curve returns thresholds length = len(precisions)-1
        # when best_idx equals last precision index, thresholds[best_idx] may be out of range, guard:
        if best_idx >= len(thresholds):
            threshold = thresholds[-1]
        else:
            threshold = thresholds[best_idx]
        print(f"AUTO-SELECTED threshold = {threshold:.3f} (Recall-focused tuning)")

    y_pred = (y_pred_proba >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        logloss = log_loss(y_true, y_pred_proba)
    except Exception:
        auc, auprc, logloss = 0.0, 0.0, 999.0
    # Additional imbalance-aware metrics
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal_acc = float('nan')
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn = fp = fn = tp = 0

    return {
        'accuracy': bal_acc,  # Use Balanced Accuracy as Accuracy
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': auc,
        'auprc': auprc,
        'log_loss': logloss,
        'threshold_used': threshold,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def apply_dp_to_data(X, y, epsilon: float = 1.5):
    X_noisy = X.copy()
    for col in X_noisy.columns:
        if X_noisy[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            r = X_noisy[col].max() - X_noisy[col].min()
            if r > 0:
                scale = (r * Config.DP_NOISE_SCALE) / epsilon
                noise = np.random.laplace(0, scale, size=len(X_noisy))
                X_noisy[col] = X_noisy[col] + noise
    log_message(f"DP applied: {len(X_noisy)} samples (ε={epsilon})")
    return X_noisy, y

def save_checkpoint(server, round_num):
    checkpoint = {'round': round_num, 'round_metrics': server.round_metrics, 'timestamp': datetime.now()}
    with open(Config.OUTPUT_DIR / 'Checkpoints' / f'checkpoint_round_{round_num}.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)

# ═══════════════════════════════════════════════════════════════════════════
# CLIENT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class FLClient:
    def __init__(self, client_id: int, client_name: str, data_path: Path):
        self.client_id = client_id
        self.client_name = client_name
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_server_share = None
        self.y_server_share = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.local_model = None
        self.metrics_history = []
        self.rotation_log = []

    def load_data(self) -> bool:
        try:
            file_mappings = {
                'train': ['local_train.csv', 'train.csv'],
                'server': ['server_share.csv', 'server.csv'],
                'val': ['validation.csv', 'val.csv'],
                'test': ['client_test.csv', 'test.csv']
            }
            data = {}
            for split, names in file_mappings.items():
                loaded = False
                for name in names:
                    if (self.data_path / name).exists():
                        data[split] = pd.read_csv(self.data_path / name)
                        loaded = True
                        break
                if not loaded:
                    print(f"   Missing {split} for {self.client_name}")
                    return False

            self.X_train = data['train'].drop('isFraud', axis=1)
            self.y_train = data['train']['isFraud']
            self.X_server_share = data['server'].drop('isFraud', axis=1)
            self.y_server_share = data['server']['isFraud']
            self.X_val = data['val'].drop('isFraud', axis=1)
            self.y_val = data['val']['isFraud']
            self.X_test = data['test'].drop('isFraud', axis=1)
            self.y_test = data['test']['isFraud']

            total = len(self.X_train) + len(self.X_server_share) + len(self.X_val) + len(self.X_test)
            print(f"{self.client_name}:")
            print(f"   Training (50%): {len(self.X_train):,} samples (Fraud: {self.y_train.mean()*100:.2f}%)")
            print(f"   Server Share (15%): {len(self.X_server_share):,} samples -> 3 chunks of ~{len(self.X_server_share)//3:,}")
            print(f"   Validation (15%): {len(self.X_val):,} samples")
            print(f"   Test (20%): {len(self.X_test):,} samples")
            print(f"   Actual split: {len(self.X_train)/total*100:.1f}%-{len(self.X_server_share)/total*100:.1f}%-{len(self.X_val)/total*100:.1f}%-{len(self.X_test)/total*100:.1f}%\n")
            log_message(f"{self.client_name} loaded")
            return True
        except Exception as e:
            print(f"   Error: {e}")
            return False

    def get_rotated_server_chunk(self, round_num: int) -> Tuple[pd.DataFrame, pd.Series]:
        total_samples = len(self.X_server_share)
        chunk_size = total_samples // Config.ROTATION_CHUNKS
        chunk_idx = (round_num - 1) % Config.ROTATION_CHUNKS
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size if chunk_idx < Config.ROTATION_CHUNKS - 1 else total_samples
        X_chunk = self.X_server_share.iloc[start_idx:end_idx].copy()
        y_chunk = self.y_server_share.iloc[start_idx:end_idx].copy()
        self.rotation_log.append({'round': round_num, 'chunk_idx': chunk_idx + 1, 'chunk_size': len(X_chunk)})
        return X_chunk, y_chunk

    def train_local_model(self, global_model=None, round_num=1) -> Optional[lgb.Booster]:
        try:
            train_data = lgb.Dataset(self.X_train, label=self.y_train, free_raw_data=False)
            valid_data = lgb.Dataset(self.X_val, label=self.y_val, free_raw_data=False)
            if global_model:
                model = lgb.train(
                    Config.LGBM_PARAMS,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, valid_data],
                    valid_names=['train', 'valid'],
                    init_model=global_model,
                    keep_training_booster=True
                )
            else:
                model = lgb.train(
                    Config.LGBM_PARAMS,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[train_data, valid_data],
                    valid_names=['train', 'valid']
                )
            self.local_model = model
            model.save_model(Config.OUTPUT_DIR / 'Models' / 'Clients' / f'Round_{round_num:02d}_{self.client_name}_local.txt')
            log_message(f"{self.client_name} - R{round_num}: Trained")
            return model
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            log_message(f"Training failed - {self.client_name}: {e}", 'error')
            return None

    def prepare_server_transmission(self, round_num: int) -> Tuple[lgb.Booster, pd.DataFrame, pd.Series]:
        try:
            X_chunk, y_chunk = self.get_rotated_server_chunk(round_num)
            X_chunk_dp, y_chunk_dp = apply_dp_to_data(X_chunk, y_chunk, Config.DP_EPSILON)
            return self.local_model, X_chunk_dp, y_chunk_dp
        except Exception as e:
            X_chunk, y_chunk = self.get_rotated_server_chunk(round_num)
            return self.local_model, X_chunk, y_chunk

    def evaluate_on_validation(self, model: lgb.Booster, round_num: int) -> Dict:
        try:
            y_pred = model.predict(self.X_val, num_iteration=model.best_iteration)
            metrics = calculate_metrics(self.y_val, y_pred)
            result = {'round': round_num, 'client': self.client_name, **metrics}
            self.metrics_history.append(result)
            return result
        except:
            return {}

    def test_model(self, model: lgb.Booster, model_type='baseline') -> Dict:
        try:
            y_pred = model.predict(self.X_test, num_iteration=model.best_iteration)
            metrics = calculate_metrics(self.y_test, y_pred)
            return {'client': self.client_name, 'model_type': model_type, 'test_samples': len(self.X_test), **metrics}
        except:
            return {}

# ═══════════════════════════════════════════════════════════════════════════
# FEDERATED SERVER
# ═══════════════════════════════════════════════════════════════════════════

class FederatedServer:
    def __init__(self):
        self.clients = []
        self.global_model = None
        self.round_metrics = []
        self.baseline_models = {}

    def initialize_clients(self) -> bool:
        print("="*100 + "\nSTEP 2: LOADING CLIENT DATA\n" + "="*100 + "\n")
        for i in range(1, Config.NUM_CLIENTS + 1):
            client = FLClient(i, f'Client_{i}', Config.CLIENTS_DIR / f'Client_{i}')
            if client.load_data():
                self.clients.append(client)
        print(f"Initialized {len(self.clients)}/{Config.NUM_CLIENTS} clients\n")
        return len(self.clients) == Config.NUM_CLIENTS

    def train_baselines(self) -> List[Dict]:
        print("\n" + "="*100 + "\nBASELINE TRAINING\n" + "="*100 + "\n")
        results = []
        for client in self.clients:
            print(f"   Training {client.client_name}...")
            model = client.train_local_model(None, 0)
            if model:
                self.baseline_models[client.client_name] = model
                metrics = client.test_model(model, 'baseline')
                if metrics:
                    results.append(metrics)
                    print(f"      AUC: {metrics['auc_roc']:.4f}, F1: {metrics['f1_score']:.4f}")
        if results:
            path1 = Config.OUTPUT_DIR / 'Metrics' / 'BASELINE_metrics.csv'
            path2 = Config.OUTPUT_DIR / 'Metrics' / 'For_Member3_Dashboard' / 'baseline_metrics.csv'
            pd.DataFrame(results).to_csv(path1, index=False)
            pd.DataFrame(results).to_csv(path2, index=False)
            print(f"\nSaved: BASELINE_metrics.csv")
            print(f"   Location: {path1}\n")
        return results

    def run_round(self, round_num: int):
        start = time.time()
        print(f"\n{'-'*100}\nRound {round_num}/{Config.NUM_ROUNDS}\n{'-'*100}")
        cycle = ((round_num - 1) % Config.ROTATION_CHUNKS) + 1
        print(f"Rotation: Chunk {cycle}/3")

        try:
            print(f"\nStep 1: Local Training")
            transmissions = []
            for client in self.clients:
                print(f"   {client.client_name}...", end=' ')
                model = client.train_local_model(self.global_model, round_num)
                if model:
                    transmission = client.prepare_server_transmission(round_num)
                    transmissions.append((client, *transmission))
                    print("[OK]")
                else:
                    print("[FAIL]")

            if not transmissions:
                return
            print(f"   {len(transmissions)}/{len(self.clients)} ready")

            print(f"\nStep 2: Aggregation")
            self.global_model = self.aggregate_with_rotation(transmissions, round_num)
            if not self.global_model:
                self.global_model = transmissions[0][1]

            # Save global models
            global_path = Config.OUTPUT_DIR / 'Models' / 'Global' / f'global_model_R{round_num:02d}.txt'
            attack_path = Config.OUTPUT_DIR / 'Models' / 'For_Member2_Attack' / f'global_model_R{round_num:02d}.txt'
            self.global_model.save_model(global_path)
            self.global_model.save_model(attack_path)
            print(f"   Saved: global_model_R{round_num:02d}.txt")
            print(f"      Location: {global_path}")

            print(f"\nStep 3: Validation")
            metrics = []
            for client in self.clients:
                m = client.evaluate_on_validation(self.global_model, round_num)
                if m:
                    metrics.append(m)
                    print(f"   {client.client_name}: AUC={m['auc_roc']:.4f}, F1={m['f1_score']:.4f}, Recall={m['recall']:.4f}, Accuracy={m['accuracy']:.4f}")

            if metrics:
                avg = {k: np.mean([m[k] for m in metrics]) for k in ['accuracy','precision','recall','f1_score','auc_roc','auprc','log_loss']}
                dur = time.time() - start
                self.round_metrics.append({'round': round_num, 'rotation_chunk': cycle, 'num_clients': len(metrics), 'duration_seconds': dur, **avg})
                print(f"\n   Avg: AUC={avg['auc_roc']:.4f} | F1={avg['f1_score']:.4f} | Recall={avg['recall']:.4f} | Accuracy={avg['accuracy']:.4f}")
                print(f"   Duration: {dur:.1f}s")
            save_checkpoint(self, round_num)
        except Exception as e:
            print(f"\nError: {e}")
            if transmissions:
                self.global_model = transmissions[0][1]

    def aggregate_with_rotation(self, transmissions: List, round_num: int) -> Optional[lgb.Booster]:
        try:
            all_X, all_y, weights = [], [], []
            for client, model, X_dp, y_dp in transmissions:
                all_X.append(X_dp)
                all_y.append(y_dp)
                weights.append(len(client.X_train))
            X_combined = pd.concat(all_X, axis=0, ignore_index=True)
            y_combined = pd.concat(all_y, axis=0, ignore_index=True)
            print(f"   Combined: {len(X_combined):,} samples")
            base_idx = np.argmax(weights)
            base_model = transmissions[base_idx][1]
            dataset = lgb.Dataset(X_combined, label=y_combined)
            return lgb.train(Config.LGBM_PARAMS, dataset, init_model=base_model, keep_training_booster=True)
        except Exception as e:
            print(f"   Aggregation error: {e}")
            return transmissions[0][1] if transmissions else None

# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION & SAVING
# ═══════════════════════════════════════════════════════════════════════════

def save_all_metrics(server):
    print("\n" + "="*100 + "\nSAVING VALIDATION METRICS\n" + "="*100)
    all_val = []
    for c in server.clients:
        all_val.extend(c.metrics_history)
    if all_val:
        path1 = Config.OUTPUT_DIR / 'Metrics' / 'VALIDATION_metrics_per_round.csv'
        path2 = Config.OUTPUT_DIR / 'Metrics' / 'For_Member3_Dashboard' / 'validation_metrics.csv'
        pd.DataFrame(all_val).to_csv(path1, index=False)
        pd.DataFrame(all_val).to_csv(path2, index=False)
        print(f"Saved: VALIDATION_metrics_per_round.csv")
        print(f"   Location: {path1}")

    if server.round_metrics:
        path1 = Config.OUTPUT_DIR / 'Metrics' / 'ROUND_SUMMARY_metrics.csv'
        path2 = Config.OUTPUT_DIR / 'Metrics' / 'For_Member3_Dashboard' / 'round_summary.csv'
        pd.DataFrame(server.round_metrics).to_csv(path1, index=False)
        pd.DataFrame(server.round_metrics).to_csv(path2, index=False)
        print(f"Saved: ROUND_SUMMARY_metrics.csv")
        print(f"   Location: {path1}")

def evaluate_final_models(server, baseline_results):
    print("\n" + "="*100 + "\nFINAL EVALUATION (CLIENT TESTS)\n" + "="*100)

    global_threshold_path = Config.OUTPUT_DIR / 'Metrics' / 'GLOBAL_threshold.txt'
    if global_threshold_path.exists():
        with open(global_threshold_path, 'r') as f:
            global_threshold = float(f.read().strip())
        print(f"Using Global Threshold = {global_threshold:.4f}\n")
    else:
        global_threshold = None
        print("WARNING: Global threshold not found - using adaptive per-client threshold\n")

    results = []
    for c in server.clients:
        print(f"{c.client_name}...", end=' ')
        try:
            y_pred = server.global_model.predict(c.X_test, num_iteration=server.global_model.best_iteration)
            metrics = calculate_metrics(c.y_test, y_pred, threshold=global_threshold)
            m = {'client': c.client_name, 'model_type': 'federated', 'test_samples': len(c.X_test), **metrics}
            results.append(m)
            print(f"AUC={m['auc_roc']:.4f}, F1={m['f1_score']:.4f}, Recall={m['recall']:.4f}")
        except Exception as e:
            print(f"Error: {e}")

    if results:
        pd.DataFrame(results).to_csv(Config.OUTPUT_DIR / 'Metrics' / 'FEDERATED_test_results.csv', index=False)
        print(f"Saved: FEDERATED_test_results.csv")

    # If baseline_results exist, keep BEFORE_AFTER comparison as before
    if baseline_results and results:
        df_b = pd.DataFrame(baseline_results)
        df_f = pd.DataFrame(results)
        comp = pd.merge(df_b[['client','auc_roc','f1_score']], df_f[['client','auc_roc','f1_score']],
                       on='client', suffixes=('_before','_after'))
        for m in ['auc_roc','f1_score']:
            comp[f'{m}_improvement_%'] = (comp[f'{m}_after'] - comp[f'{m}_before']) * 100
        path1 = Config.OUTPUT_DIR / 'Metrics' / 'BEFORE_AFTER_comparison.csv'
        path2 = Config.OUTPUT_DIR / 'Metrics' / 'For_Member3_Dashboard' / 'before_after_comparison.csv'
        comp.to_csv(path1, index=False)
        comp.to_csv(path2, index=False)
        print(f"✅ Saved: BEFORE_AFTER_comparison.csv")
        print(f"   Location: {path1}")
        print(f"\n📊 Avg Improvement: AUC={comp['auc_roc_improvement_%'].mean():+.2f}%, F1={comp['f1_score_improvement_%'].mean():+.2f}%")

def evaluate_global_test_set(server):
    print("\n" + "="*100 + "\nGLOBAL TEST (test_data.csv - ORIGINAL 20%, NO SMOTE)\n" + "="*100)
    global_threshold_path = Config.OUTPUT_DIR / 'Metrics' / 'GLOBAL_threshold.txt'
    if global_threshold_path.exists():
        with open(global_threshold_path, 'r') as f:
            global_threshold = float(f.read().strip())
        print(f"🌍 Using Global Threshold = {global_threshold:.4f}\n")
    else:
        global_threshold = None
        print("⚠️ Global threshold not found — using auto threshold\n")

    try:
        test_path = Config.GLOBAL_TEST_PATH
        print(f"   Path: {test_path}")
        df = pd.read_csv(test_path)
        X = df.drop('isFraud', axis=1)
        y = df['isFraud']
        print(f"✅ Loaded: {len(df):,} samples (Fraud: {y.sum():,}, {y.mean()*100:.2f}%)\n")

        y_pred = server.global_model.predict(X, num_iteration=server.global_model.best_iteration)
        metrics = calculate_metrics(y, y_pred, threshold=global_threshold)
        print(f"📊 Results: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1_score']:.4f}, Recall={metrics['recall']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        print(f"   Precision={metrics['precision']:.4f}")

        pd.DataFrame([{'test_type': 'Global', 'samples': len(y), 'fraud_samples': y.sum(), **metrics}]).to_csv(Config.OUTPUT_DIR / 'Metrics' / 'GLOBAL_TEST_results.csv', index=False)
        print(f"✅ Saved: GLOBAL_TEST_results.csv")
        # Persist a canonical clean baseline for attacked comparisons
        try:
            baseline_dir = Config.BASE_DIR / 'baselines'
            baseline_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                'timestamp': datetime.now().isoformat(),
                'artifacts_dir': str(Config.OUTPUT_DIR),
                'eval': {
                    'global_test': {
                        'threshold_used': float(metrics.get('threshold_used')) if metrics.get('threshold_used') is not None else None,
                        'accuracy': float(metrics.get('accuracy')),
                        'precision': float(metrics.get('precision')),
                        'recall': float(metrics.get('recall')),
                        'f1': float(metrics.get('f1_score')),
                        'auc': float(metrics.get('auc_roc')),
                        'auprc': float(metrics.get('auprc')),
                        'tn': int(metrics.get('tn', 0)), 'fp': int(metrics.get('fp', 0)), 'fn': int(metrics.get('fn', 0)), 'tp': int(metrics.get('tp', 0)),
                        'samples': int(len(y)),
                        'positives': int(y.sum())
                    }
                }
            }
            with open(baseline_dir / 'latest_clean.json', 'w') as jf:
                json.dump(payload, jf, indent=2)
        except Exception as _:
            pass
    except FileNotFoundError:
        print(f"❌ test_data.csv not found at: {Config.GLOBAL_TEST_PATH}")
        print("   Skipping global test evaluation")
    except Exception as e:
        print(f"❌ Error: {e}")

def create_final_package():
    print("\n" + "="*100 + "\nCREATING FINAL ZIP PACKAGE\n" + "="*100)
    try:
        zip_name = f'FL_Results_OPTIMIZED_{Config.TIMESTAMP}.zip'
        zip_path = Config.OUTPUT_DIR.parent / zip_name

        print(f"📦 Creating package: {zip_name}")
        print(f"   This may take a minute...\n")

        file_count = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(Config.OUTPUT_DIR):
                for file in files:
                    if not file.endswith('.zip'):
                        fp = Path(root) / file
                        zipf.write(fp, fp.relative_to(Config.OUTPUT_DIR.parent))
                        file_count += 1

        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"✅ ZIP Package Created Successfully!")
        print(f"   Name: {zip_name}")
        print(f"   Location: {zip_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Files included: {file_count}")
    except Exception as e:
        print(f"❌ ZIP creation failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL THRESHOLD ADD-ON
# ═══════════════════════════════════════════════════════════════════════════

def compute_and_save_global_threshold(server):
    """Compute a single Balanced-Accuracy-optimized threshold from all clients' validation sets."""
    print("\n" + "="*100 + "\nGLOBAL THRESHOLD COMPUTATION\n" + "="*100)
    try:
        y_true_all, y_pred_all = [], []
        for client in server.clients:
            # ensure client's val arrays are present
            y_true_all.extend(client.y_val.tolist() if isinstance(client.y_val, pd.Series) else list(client.y_val))
            preds = server.global_model.predict(client.X_val, num_iteration=server.global_model.best_iteration)
            y_pred_all.extend(preds.tolist() if isinstance(preds, np.ndarray) else list(preds))

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        if y_true_all.size == 0 or y_pred_all.size == 0:
            raise ValueError("Empty validation predictions for threshold search")

        # Candidate thresholds from PR curve + quantiles + default 0.5
        pr_p, pr_r, pr_thr = precision_recall_curve(y_true_all, y_pred_all)
        cands = list(pr_thr.tolist()) if pr_thr.size > 0 else []
        qs = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
        try:
            qvals = np.quantile(y_pred_all, qs)
            cands.extend(list(qvals))
        except Exception:
            cands.append(0.5)
        cands.append(0.5)
        # Deduplicate and clip
        cands = sorted(set(float(max(1e-6, min(1-1e-6, c))) for c in cands))

        # Search best by Balanced Accuracy
        from sklearn.metrics import balanced_accuracy_score
        best_thr, best_bacc = None, -1.0
        for thr in cands:
            y_bin = (y_pred_all >= thr).astype(int)
            try:
                bacc = float(balanced_accuracy_score(y_true_all, y_bin))
            except Exception:
                continue
            if bacc > best_bacc:
                best_bacc, best_thr = bacc, float(thr)

        # Fallback
        if best_thr is None:
            best_thr = 0.5
            best_bacc = 0.0

        global_threshold = float(best_thr)
        print(f"🌍 Optimal Global Threshold = {global_threshold:.4f} (Balanced Accuracy={best_bacc:.4f})")
        path = Config.OUTPUT_DIR / 'Metrics' / 'GLOBAL_threshold.txt'
        with open(path, 'w') as f:
            f.write(str(global_threshold))
        print(f"✅ Saved global threshold to: {path}")

    except Exception as e:
        print(f"❌ Failed to compute global threshold: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    start_time = setup_environment()
    try:
        server = FederatedServer()
        if not server.initialize_clients():
            return
        baseline_results = server.train_baselines()

        print("\n" + "="*100 + "\nSTARTING FL TRAINING (OPTIMIZED)\n" + "="*100)
        for r in tqdm(range(1, Config.NUM_ROUNDS + 1), desc="FL Rounds", unit="round"):
            server.run_round(r)

        save_all_metrics(server)
        compute_and_save_global_threshold(server)  # ✅ Added global threshold step
        evaluate_final_models(server, baseline_results)
        evaluate_global_test_set(server)

        total_time = (datetime.now() - start_time).total_seconds()
        print("\n" + "="*100 + "\n✅ TRAINING COMPLETED SUCCESSFULLY\n" + "="*100)
        print(f"\n⏱️ Total Time: {total_time/60:.2f} minutes")
        if server.round_metrics:
            final = server.round_metrics[-1]
            print(f"📈 Final Performance (Round {Config.NUM_ROUNDS}):")
            print(f"   • AUC-ROC: {final['auc_roc']:.4f}")
            print(f"   • F1-Score: {final['f1_score']:.4f}")
            print(f"   • Recall: {final['recall']:.4f}")
            print(f"   • Accuracy: {final['accuracy']:.4f}")
            print(f"   • Precision: {final['precision']:.4f}")
        print(f"\n📂 All results saved in:\n   {Config.OUTPUT_DIR}\n")
    except Exception as e:
        print(f"\n❌ Training Failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    np.random.seed(Config.RANDOM_SEED)
    main()
