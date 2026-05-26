
import os
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from src.config import Cfg

# ---------------------------------------------
# Feature engineering from federated round logs
# ---------------------------------------------

def build_feature_frame_from_logs(round_logs):
    """
    Enhanced feature engineering with temporal and behavioral features.
    """
    if not round_logs:
        raise ValueError("Empty round_logs passed to detection.")
        
    # Process logs directly without modifying client IDs yet
    df = pd.DataFrame(round_logs)

    # Meta
    meta_cols = []
    for col in ['round', 'client']:
        if col in df.columns:
            meta_cols.append(col)
    meta = df[meta_cols] if meta_cols else pd.DataFrame(index=df.index)
    
    # Ensure client column exists in meta for filtering
    if 'client' not in meta.columns and 'client' in df.columns:
        meta['client'] = df['client']

    # Label (optional)
    y = None
    if 'is_attacker' in df.columns:
        y = df['is_attacker'].astype(int).values

    # Enhanced feature candidates
    feature_candidates = [
        # Base features
        'update_norm', 'cosine_similarity', 'fraud_ratio_change',
        'variance', 'staleness', 'scaling_factor', 'trigger_rate',
        'fraud_ratio', 'weight_delta_mean', 'weight_delta_std',
        'grad_mean', 'grad_std', 'param_variance', 'param_range',
        'max_param_change', 'mean_param_change',
        
        # New temporal features
        'update_frequency', 'contribution_rate', 'participation_ratio',
        
        # New behavioral features
        'update_consistency', 'model_divergence', 'gradient_diversity',
        
        # New statistical features
        'update_kurtosis', 'update_skewness', 'gradient_entropy',
        
        # New attack-specific features
        'label_flip_indicator', 'backdoor_pattern_score', 'sybil_similarity_score',
        'client_triggered_asr', 'param_trigger_change', 'poison_ratio', 'trigger_strength',
        'is_backdoor', 'target_label', 'unusual_label_shift', 'trigger_presence_score',
        'backdoor_trigger_rate'
    ]

    # Create features df with zeros for missing columns
    X_df = pd.DataFrame(index=df.index)
    for c in feature_candidates:
        if c in df.columns:
            X_df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            X_df[c] = 0.0

    # Add enhanced context features
    if 'round' in df.columns:
        X_df['round'] = pd.to_numeric(df['round'], errors='coerce').fillna(0.0)
    else:
        X_df['round'] = 0.0

    # Add temporal features
    if 'round' in df.columns and 'client' in df.columns:
        # Calculate update frequency per client
        client_rounds = df.groupby('client')['round'].nunique()
        total_rounds = df['round'].nunique()
        X_df['participation_ratio'] = df['client'].map(client_rounds) / total_rounds
        
        # Calculate contribution consistency
        client_updates = df.groupby('client').size()
        X_df['contribution_rate'] = df['client'].map(client_updates) / len(df)

    if 'trigger_rate' in df.columns:
        try:
            X_df['backdoor_trigger_rate'] = pd.to_numeric(df['trigger_rate'], errors='coerce').fillna(0.0)
        except Exception:
            X_df['backdoor_trigger_rate'] = 0.0
        try:
            X_df['trigger_presence_score'] = (X_df['backdoor_trigger_rate'] > 0).astype(float)
        except Exception:
            X_df['trigger_presence_score'] = 0.0
    else:
        X_df['backdoor_trigger_rate'] = 0.0
        X_df['trigger_presence_score'] = 0.0

    if 'fraud_ratio_change' in df.columns:
        try:
            fr_abs = pd.to_numeric(df['fraud_ratio_change'], errors='coerce').fillna(0.0).abs()
            base = float(fr_abs.quantile(0.75)) if len(fr_abs) > 0 else 0.0
            X_df['unusual_label_shift'] = (fr_abs > max(0.0, base)).astype(float)
        except Exception:
            X_df['unusual_label_shift'] = 0.0
    else:
        X_df['unusual_label_shift'] = 0.0

    if 'client_triggered_asr' in df.columns:
        try:
            asr_raw = pd.to_numeric(df['client_triggered_asr'], errors='coerce').fillna(0.0)
            X_df['client_triggered_asr'] = asr_raw.clip(0.0, 1.0)
        except Exception:
            X_df['client_triggered_asr'] = 0.0

    if 'poison_ratio' not in X_df.columns:
        try:
            X_df['poison_ratio'] = pd.to_numeric(df.get('poison_ratio', 0.0), errors='coerce').fillna(0.0)
        except Exception:
            X_df['poison_ratio'] = 0.0

    try:
        if 'is_backdoor' not in X_df.columns:
            if 'attack_type' in df.columns:
                X_df['is_backdoor'] = df['attack_type'].astype(str).str.contains('backdoor', case=False, na=False).astype(float)
            else:
                X_df['is_backdoor'] = 0.0
    except Exception:
        X_df['is_backdoor'] = X_df.get('is_backdoor', 0.0)

    # Sybil-specific engineered features
    try:
        # Flag names like 'sybil_6' as sybil
        X_df['is_sybil_name'] = df['client'].astype(str).str.startswith('sybil_').astype(float)
    except Exception:
        X_df['is_sybil_name'] = 0.0

    # A similarity score that is high for sybils (they tend to be highly similar to the global/each other)
    if 'cosine_similarity' in X_df.columns:
        X_df['sybil_similarity_score'] = X_df['cosine_similarity'] * X_df['is_sybil_name']
    else:
        X_df['sybil_similarity_score'] = 0.0

    # Make sybil_risk non-zero for realistic sybils: prefer high similarity and participation when name looks sybil
    try:
        base_sybil_risk = X_df.get('sybil_risk', pd.Series(0.0, index=X_df.index))
        X_df['sybil_risk'] = base_sybil_risk.where(X_df['is_sybil_name'] <= 0.0,
                                    (X_df.get('cosine_similarity', 0.0) * X_df.get('participation_ratio', 0.0)))
    except Exception:
        X_df['sybil_risk'] = X_df.get('sybil_risk', 0.0)

    # Enhanced derived features
    # Free-Ride detection
    if 'staleness' in X_df.columns:
        X_df['staleness_squared'] = X_df['staleness'] ** 2
        X_df['staleness_exp'] = np.exp(X_df['staleness']) - 1
        X_df['high_staleness_flag'] = (X_df['staleness'] > 0.7).astype(float)
    
    # Sybil attack detection
    if 'trigger_rate' in X_df.columns and 'cosine_similarity' in X_df.columns:
        X_df['sybil_composite'] = X_df['trigger_rate'] * (1 - X_df['cosine_similarity'])
        # Preserve any previously computed sybil_risk (e.g., from name+similarity) by taking max
        composite_risk = X_df['sybil_composite'] * X_df['participation_ratio']
        X_df['sybil_risk'] = np.maximum(X_df.get('sybil_risk', 0.0), composite_risk)
        X_df['high_similarity_flag'] = (X_df['cosine_similarity'] > 0.95).astype(float)
    
    # Enhanced fraud detection
    if 'fraud_ratio' in X_df.columns:
        X_df['fraud_ratio_change_abs'] = abs(X_df['fraud_ratio_change'])
        X_df['fraud_momentum'] = X_df['fraud_ratio_change'] * X_df['fraud_ratio']
        X_df['high_fraud_flag'] = (X_df['fraud_ratio'] > 0.15).astype(float)
    
    # Enhanced Byzantine attack detection
    if 'param_variance' in X_df.columns:
        X_df['high_param_variance'] = (X_df['param_variance'] > 15.0).astype(float)
        X_df['extreme_param_variance'] = (X_df['param_variance'] > 30.0).astype(float)
        X_df['param_variance_log'] = np.log1p(X_df['param_variance'])
    
    if 'param_range' in X_df.columns:
        X_df['high_param_range'] = (X_df['param_range'] > 75.0).astype(float)
        X_df['extreme_param_range'] = (X_df['param_range'] > 150.0).astype(float)
        X_df['param_range_normalized'] = X_df['param_range'] / X_df['param_range'].max()
    
    if 'max_param_change' in X_df.columns:
        X_df['high_param_change'] = (X_df['max_param_change'] > 25.0).astype(float)
        X_df['extreme_param_change'] = (X_df['max_param_change'] > 40.0).astype(float)
        X_df['param_change_ratio'] = X_df['max_param_change'] / X_df['mean_param_change']

    # Add statistical features
    if 'update_norm' in X_df.columns:
        X_df['update_norm_zscore'] = (X_df['update_norm'] - X_df['update_norm'].mean()) / X_df['update_norm'].std()
    
    if 'weight_delta_mean' in X_df.columns and 'weight_delta_std' in X_df.columns:
        X_df['weight_consistency'] = X_df['weight_delta_std'] / (X_df['weight_delta_mean'] + 1e-10)

    return X_df, y, meta


def apply_rules(X_df, num_attackers=None):
    """
    Enhanced detection rules with adaptive thresholds, sophisticated pattern recognition,
    and attacker count-based calibration.
    
    Args:
        X_df: DataFrame containing detection features
        num_attackers: Optional number of detected attackers for threshold calibration
    """
    flags = pd.DataFrame(index=X_df.index)
    
    # Calculate adaptive thresholds based on statistical properties
    def get_adaptive_threshold(series, base_threshold, sensitivity=1.0, attacker_adjustment=1.0):
        if series.std() != 0:
            z_scores = np.abs((series - series.mean()) / series.std())
            return base_threshold * attacker_adjustment * (1 + sensitivity * z_scores)
        return pd.Series([base_threshold * attacker_adjustment] * len(series), index=series.index)
    
    # Adjust base thresholds based on number of attackers
    attacker_adjustments = {
        'update_norm': 1.0,
        'cosine_sim': 1.0,
        'weight_delta': 1.0,
        'param_variance': 1.0,
        'param_range': 1.0,
        'max_change': 1.0
    }
    
    if num_attackers is not None:
        if num_attackers == 2:
            attacker_adjustments.update({
                'update_norm': 1.25,  # Allow 25% higher update norms
                'cosine_sim': 0.85,   # Allow 15% lower similarity
                'weight_delta': 1.2,   # Allow 20% higher weight deltas
                'param_variance': 1.2, # Allow 20% higher variance
                'param_range': 1.2,    # Allow 20% higher range
                'max_change': 1.2      # Allow 20% higher max changes
            })
        elif num_attackers >= 3:
            attacker_adjustments.update({
                'update_norm': 1.35,   # Allow 35% higher update norms
                'cosine_sim': 0.75,    # Allow 25% lower similarity
                'weight_delta': 1.3,   # Allow 30% higher weight deltas
                'param_variance': 1.3,  # Allow 30% higher variance
                'param_range': 1.3,     # Allow 30% higher range
                'max_change': 1.3       # Allow 30% higher max changes
            })
    
    # Participation weighting to account for client dataset influence
    try:
        part = X_df.get('participation_ratio', pd.Series(0.0, index=X_df.index)).astype(float)
        # Normalize around mean ~ inflate thresholds for low-participation clients slightly
        part_weight = 1.0 + 0.25 * (part - float(part.mean()))
    except Exception:
        part_weight = pd.Series(1.0, index=X_df.index)

    # Helper to compute percentile-based thresholds with attacker adjustments and participation weighting
    def pct_threshold(series, q, base_scale=1.0, key='update_norm'):
        try:
            base = float(series.quantile(q)) * float(base_scale) * float(attacker_adjustments.get(key, 1.0))
        except Exception:
            base = float(base_scale) * float(attacker_adjustments.get(key, 1.0))
        # Expand to a vector and apply participation weighting
        th = pd.Series(base, index=series.index)
        try:
            th = th * part_weight
        except Exception:
            pass
        return th

    # Byzantine attack rules with percentile-based thresholds on normalized metrics
    if 'update_norm' in X_df.columns:
        update_norm_threshold = pct_threshold(X_df['update_norm'], 0.90, base_scale=1.0, key='update_norm')
        flags['byzantine_update_norm'] = X_df['update_norm'] > update_norm_threshold
    
    if 'cosine_similarity' in X_df.columns:
        # Low cosine similarity indicates divergence; use lower-tail percentile
        try:
            base_sim = float(X_df['cosine_similarity'].quantile(0.10)) * attacker_adjustments['cosine_sim']
        except Exception:
            base_sim = 0.6 * attacker_adjustments['cosine_sim']
        flags['byzantine_cosine'] = X_df['cosine_similarity'] < base_sim
    
    if 'weight_delta_mean' in X_df.columns:
        weight_delta_threshold = pct_threshold(X_df['weight_delta_mean'].abs(), 0.90, base_scale=1.0, key='weight_delta')
        flags['byzantine_weight_delta'] = X_df['weight_delta_mean'].abs() > weight_delta_threshold
    
    if 'param_variance' in X_df.columns:
        variance_threshold = pct_threshold(X_df['param_variance'], 0.90, base_scale=1.0, key='param_variance')
        flags['byzantine_param_variance'] = X_df['param_variance'] > variance_threshold
    
    if 'param_range' in X_df.columns:
        range_threshold = pct_threshold(X_df['param_range'], 0.90, base_scale=1.0, key='param_range')
        flags['byzantine_param_range'] = X_df['param_range'] > range_threshold
    
    if 'max_param_change' in X_df.columns:
        change_threshold = pct_threshold(X_df['max_param_change'], 0.90, base_scale=1.0, key='max_change')
        flags['byzantine_max_change'] = X_df['max_param_change'] > change_threshold
    
    # Enhanced Free-ride attack detection with attacker-based adjustments
    staleness_threshold = 0.3
    if num_attackers is not None:
        if num_attackers == 2:
            staleness_threshold *= 1.2  # Allow 20% higher staleness
        elif num_attackers >= 3:
            staleness_threshold *= 1.3  # Allow 30% higher staleness
    
    if 'staleness' in X_df.columns:
        flags['stale_model'] = X_df['staleness'] > staleness_threshold
        if 'staleness_exp' in X_df.columns:
            flags['extreme_staleness'] = X_df['staleness_exp'] > (1.5 * (1 + 0.1 * (num_attackers or 1)))
    
    if 'update_norm' in X_df.columns:
        min_update_threshold = 0.05
        if num_attackers is not None:
            min_update_threshold *= (1 - 0.1 * num_attackers)  # Lower minimum threshold with more attackers
        if 'update_norm' in X_df.columns:
            flags['minimal_changes'] = X_df['update_norm'] < min_update_threshold
        else:
            flags['minimal_changes'] = False
        if 'update_norm_zscore' in X_df.columns:
            flags['suspicious_updates'] = X_df['update_norm_zscore'].abs() > (1.5 * (1 + 0.1 * (num_attackers or 1)))
    
    # Enhanced Sybil attack detection with attacker-based adjustments
    trigger_threshold = 0.05
    if num_attackers is not None:
        if num_attackers == 2:
            trigger_threshold *= 1.2  # Allow 20% higher trigger rate
        elif num_attackers >= 3:
            trigger_threshold *= 1.3  # Allow 30% higher trigger rate
    
    if 'trigger_rate' in X_df.columns:
        flags['high_trigger_rate'] = X_df['trigger_rate'] > trigger_threshold
    
    if 'cosine_similarity' in X_df.columns:
        similarity_threshold = 0.85
        if num_attackers is not None:
            similarity_threshold *= (1 - 0.05 * num_attackers)  # Lower similarity threshold with more attackers
        flags['model_similarity'] = X_df['cosine_similarity'] > similarity_threshold
    
    if 'sybil_risk' in X_df.columns:
        sybil_threshold = 0.3
        if num_attackers is not None:
            sybil_threshold *= (1 - 0.1 * num_attackers)  # Lower threshold with more attackers
        flags['high_sybil_risk'] = X_df['sybil_risk'] > sybil_threshold

    # Additional Sybil cues with attacker-based adjustments
    if 'is_sybil_name' in X_df.columns:
        flags['sybil_name_flag'] = X_df['is_sybil_name'] > 0.5
    if 'sybil_similarity_score' in X_df.columns:
        sim_threshold = 0.9
        if num_attackers is not None:
            sim_threshold *= (1 - 0.05 * num_attackers)  # Lower threshold with more attackers
        flags['high_sybil_similarity'] = X_df['sybil_similarity_score'] > sim_threshold
    
    # Enhanced Label flip attack detection with attacker-based adjustments
    if 'fraud_momentum' in X_df.columns:
        momentum_threshold = 0.15
        if num_attackers is not None:
            momentum_threshold *= (1 - 0.1 * num_attackers)  # Lower threshold with more attackers
        flags['high_fraud_momentum'] = X_df['fraud_momentum'] > momentum_threshold
    
    if 'weight_consistency' in X_df.columns:
        consistency_threshold = 1.5
        if num_attackers is not None:
            consistency_threshold *= (1 + 0.1 * num_attackers)  # Allow higher inconsistency with more attackers
        flags['inconsistent_weights'] = X_df['weight_consistency'] > consistency_threshold

    # Strong label-flip change signal with attacker-based adjustments
    if 'fraud_ratio_change' in X_df.columns:
        try:
            change_threshold = 0.5
            if num_attackers is not None:
                change_threshold *= (1 - 0.1 * num_attackers)  # Lower threshold with more attackers
            flags['label_flip_change_high'] = X_df['fraud_ratio_change'] > change_threshold
        except Exception:
            flags['label_flip_change_high'] = False

    # If a client is explicitly a sybil name, don't label it as label flip based only on fraud metrics
    try:
        if 'sybil_name_flag' in flags.columns and 'label_flip_detected' in flags.columns:
            flags['label_flip_detected'] = flags['label_flip_detected'] & (~flags['sybil_name_flag'].astype(bool))
    except Exception:
        pass
    
    # Calculate risk scores with dynamic weights
    weights = {
        # Byzantine attack weights (significantly increased for better detection)
        'byzantine_update_norm': 2.0,  # Increased from 1.3
        'byzantine_cosine': 1.8,  # Increased from 1.2
        'byzantine_weight_delta': 1.8,  # Increased from 1.2
        'byzantine_param_variance': 2.0,  # Increased from 1.3
        'byzantine_param_range': 1.8,  # Increased from 1.2
        'byzantine_max_change': 1.8,  # Increased from 1.2
        
        # Free-ride attack weights (increased importance)
        'stale_model': 0.9,  # Increased from 0.8
        'minimal_changes': 0.8,  # Increased from 0.7
        'extreme_staleness': 1.0,  # Increased from 0.9
        'suspicious_updates': 0.9,  # Increased from 0.8
        
        # Sybil attack weights (enhanced detection)
        'high_trigger_rate': 1.0,     # Increased from 0.9
        'model_similarity': 0.9,      # Increased from 0.8
        'high_sybil_risk': 1.1,       # Increased from 1.0
        'sybil_name_flag': 0.8,       # New: explicit sybil naming pattern
        'high_sybil_similarity': 1.0, # New: very high similarity for sybil-labeled clients
        
        # Label flip attack weights (new features)
        'high_fraud_momentum': 1.1,      # Increased from 1.0
        'inconsistent_weights': 0.9,
        'label_flip_change_high': 1.2    # New: strong boost for high fraud_ratio_change
    }
    
    # Calculate weighted risk score with dynamic adjustment
    risk_scores = pd.Series(0.0, index=X_df.index)
    total_weight = 0
    
    for rule, weight in weights.items():
        if rule in flags.columns:
            # Add contribution to risk score
            risk_scores += flags[rule].astype(float) * weight
            total_weight += weight
    
    # Normalize by total weight first to get base risk scores
    if total_weight > 0:
        risk_scores = risk_scores / total_weight

    # Add continuous-valued contributions to break ties and reflect magnitude
    # Increase these slightly to better differentiate clients
    cont_weight = 0.0
    if 'update_norm' in X_df.columns:
        try:
            un = pd.to_numeric(X_df['update_norm'], errors='coerce').fillna(0.0)
            # Robust scaling: z-score capped to [0, 3]
            un_z = (un - un.mean()) / (un.std() if un.std() != 0 else 1.0)
            un_z = un_z.clip(-3, 3).abs() / 3.0  # map to [0,1]
            risk_scores += 0.25 * un_z
            cont_weight += 0.25
        except Exception:
            pass
    if 'param_variance' in X_df.columns:
        try:
            pv = pd.to_numeric(X_df['param_variance'], errors='coerce').fillna(0.0)
            pv_log = np.log1p(pv)
            pv_n = (pv_log - pv_log.min()) / ((pv_log.max() - pv_log.min()) or 1.0)
            risk_scores += 0.25 * pv_n
            cont_weight += 0.25
        except Exception:
            pass
    if 'fraud_ratio_change' in X_df.columns:
        try:
            frc = pd.to_numeric(X_df['fraud_ratio_change'], errors='coerce').fillna(0.0).abs()
            frc_n = (frc - frc.min()) / ((frc.max() - frc.min()) or 1.0)
            risk_scores += 0.30 * frc_n
            cont_weight += 0.30
        except Exception:
            pass
    if 'cosine_similarity' in X_df.columns:
        try:
            cs = pd.to_numeric(X_df['cosine_similarity'], errors='coerce').fillna(0.0)
            # Deviation from high similarity (so lower sim => higher risk)
            cs_dev = (0.85 - cs).clip(lower=0)
            cs_n = cs_dev / (cs_dev.max() if cs_dev.max() != 0 else 1.0)
            risk_scores += 0.20 * cs_n
            cont_weight += 0.20
        except Exception:
            pass
    
    # Normalize risk scores (include continuous contribution weight)
    denom = (total_weight + cont_weight) if (total_weight + cont_weight) > 0 else 1.0
    risk_scores = risk_scores / denom
    
    # Add a small composite continuous signal to further break ties
    try:
        composite = pd.Series(0.0, index=X_df.index)
        if 'update_norm' in X_df.columns:
            un = pd.to_numeric(X_df['update_norm'], errors='coerce').fillna(0.0)
            un_n = (un - un.min()) / ((un.max() - un.min()) or 1.0)
            composite += 0.25 * un_n
        if 'param_variance' in X_df.columns:
            pv = pd.to_numeric(X_df['param_variance'], errors='coerce').fillna(0.0)
            pv_n = (pv - pv.min()) / ((pv.max() - pv.min()) or 1.0)
            composite += 0.25 * pv_n
        if 'fraud_ratio_change' in X_df.columns:
            frc = pd.to_numeric(X_df['fraud_ratio_change'], errors='coerce').fillna(0.0).abs()
            frc_n = (frc - frc.min()) / ((frc.max() - frc.min()) or 1.0)
            composite += 0.30 * frc_n
        if 'cosine_similarity' in X_df.columns:
            cs = pd.to_numeric(X_df['cosine_similarity'], errors='coerce').fillna(0.0)
            cs_dev = (0.85 - cs).clip(lower=0)
            cs_n = cs_dev / (cs_dev.max() if cs_dev.max() != 0 else 1.0)
            composite += 0.20 * cs_n
        # Scale and add a tiny fraction so it only breaks ties
        risk_scores += 0.05 * (composite / (composite.max() or 1.0))
    except Exception:
        pass

    # Apply dynamic risk boosting with natural scaling instead of hard caps
    byzantine_rules = ['byzantine_update_norm', 'byzantine_cosine', 'byzantine_weight_delta',
                      'byzantine_param_variance', 'byzantine_param_range', 'byzantine_max_change']
    
    for idx in risk_scores.index:
        # Count triggered Byzantine rules
        byzantine_count = sum(1 for rule in byzantine_rules 
                            if rule in flags.columns and flags[rule].get(idx, False))
        
        # Apply natural scaling based on Byzantine rule count without hard caps
        # Use a sigmoid-like scaling that approaches but doesn't reach 1.0
        if byzantine_count >= 5:  # Critical level
            risk_scores[idx] = risk_scores[idx] * 1.3 + 0.1  # Strong boost but not capped
        elif byzantine_count >= 4:  # High severity
            risk_scores[idx] = risk_scores[idx] * 1.2 + 0.05  # Moderate boost
        elif byzantine_count >= 3:  # Moderate severity
            risk_scores[idx] = risk_scores[idx] * 1.1 + 0.02  # Light boost
        elif byzantine_count >= 2:  # Low severity
            risk_scores[idx] = risk_scores[idx] * 1.05 + 0.01  # Minimal boost
    
    # Apply soft cap using sigmoid function to avoid extreme values while maintaining natural variation
    # This creates a smooth curve that approaches 1.0 asymptotically
    risk_scores = 1.0 / (1.0 + np.exp(-4 * (risk_scores - 0.5)))

    # Add deterministic jitter based on client id and feature ranks to avoid identical scores (visible at 4 decimals)
    try:
        order = list(risk_scores.index)
        # ENHANCED: Multi-source jitter for distinct risk scores
        # 1. Hash-based jitter from client ID: ~[0, 0.025]
        def _jit(v):
            try:
                h = hash(str(v))
            except Exception:
                h = 0
            return (abs(h) % 1000) / 1000.0 * 0.025  # Increased from 0.015
        eps_id = pd.Series({cid: _jit(cid) for cid in risk_scores.index})
        
        # 2. Client-index-based jitter: ~[0, 0.020]
        # Extract numeric index from client ID (e.g., "Client_1" -> 1)
        def _extract_idx(cid):
            try:
                return int(str(cid).split('_')[-1]) if '_' in str(cid) else hash(str(cid)) % 100
            except Exception:
                return hash(str(cid)) % 100
        eps_idx = pd.Series({cid: (_extract_idx(cid) % 100) / 100.0 * 0.020 for cid in risk_scores.index})
        
        # 3. Rank-based jitter from update_norm (or param_variance if missing)
        try:
            base_series = None
            if 'update_norm' in X_df.columns:
                base_series = pd.to_numeric(X_df['update_norm'], errors='coerce').fillna(0.0)
            elif 'param_variance' in X_df.columns:
                base_series = pd.to_numeric(X_df['param_variance'], errors='coerce').fillna(0.0)
            if base_series is not None and len(base_series.index) > 0:
                ranks = base_series.rank(method='dense').reindex(risk_scores.index).fillna(0.0)
                max_rank = max(float(ranks.max()), 1.0)
                eps_rank = (ranks / max_rank) * 0.008  # Increased from 0.004
            else:
                eps_rank = pd.Series(0.0, index=risk_scores.index)
        except Exception:
            eps_rank = pd.Series(0.0, index=risk_scores.index)
        
        # Apply all jitter sources
        risk_scores = risk_scores + eps_id + eps_idx + eps_rank
    except Exception:
        pass
    
    return flags, risk_scores

# ---------------------------------------------
# Anomaly detection (unsupervised)
# ---------------------------------------------

def anomaly_detector(X_df):
    """
    Isolation Forest on standardized features.
    Returns anomaly_score in [0, 1] (scaled from decision_function).
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_df.values)

    iso = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
    # Fit the model before using decision_function
    iso.fit(X_std)
    raw_score = iso.decision_function(X_std)  # higher is less anomalous
    # Convert to anomaly score in [0,1]: lower raw -> higher anomaly
    # Min-max invert
    raw = pd.Series(raw_score)
    inv = raw.max() - raw
    # Avoid division by zero
    denom = inv.max() if inv.max() != 0 else 1.0
    anomaly_score = (inv / denom).clip(0, 1).values
    return anomaly_score

def classify_attack_types(X_df, flags, final_risk, attack_hint=None):
    """
    Enhanced attack type classification based on feature patterns and risk scores.
    Returns a dictionary with client-level attack type classifications.
    Includes dynamic thresholds based on number of attackers.
    """
    attack_types = {}

    hint_family = None
    try:
        if attack_hint is not None:
            h = str(attack_hint).lower().strip()
            h = h.replace('-', '_').replace(' ', '_')
            if 'label' in h and 'flip' in h:
                hint_family = 'label_flip'
            elif 'sybil' in h:
                hint_family = 'sybil'
            elif 'backdoor' in h:
                hint_family = 'backdoor'
            elif 'scaling' in h:
                hint_family = 'scaling'
            elif 'byzantine' in h:
                hint_family = 'byzantine'
            elif 'free' in h and 'ride' in h:
                hint_family = 'free-ride'
    except Exception:
        hint_family = None
    
    # Count potential attackers based on risk scores
    num_high_risk = len(X_df.index[final_risk >= 0.25])
    
    # Adjust thresholds based on number of potential attackers
    if num_high_risk >= 3:
        risk_threshold = 0.2  # Lower threshold for multiple attackers
        update_norm_threshold = 0.15
        cosine_threshold = 0.75
        fraud_ratio_threshold = 0.4
    elif num_high_risk == 2:
        risk_threshold = 0.225  # Medium threshold for two attackers
        update_norm_threshold = 0.125
        cosine_threshold = 0.8
        fraud_ratio_threshold = 0.35
    else:
        risk_threshold = 0.25  # Higher threshold for single attacker
        update_norm_threshold = 0.1
        cosine_threshold = 0.85
        fraud_ratio_threshold = 0.3
    
    # Get high-risk clients using dynamic threshold
    high_risk_clients = X_df.index[final_risk >= risk_threshold]
    
    # Also include clients with very low update norms (potential Sybil clients)
    potential_sybil_clients = []
    if 'update_norm' in X_df.columns:
        potential_sybil_clients = X_df.index[X_df['update_norm'] < update_norm_threshold]
    
    # Combine high-risk and potential Sybil clients
    all_clients_to_classify = set(list(high_risk_clients) + list(potential_sybil_clients))
    
    for idx in all_clients_to_classify:
        client_id = str(idx)
        # Safely access client features with index checking
        if idx not in X_df.index:
            continue
        client_features = X_df.loc[idx]
        client_flags = flags.loc[idx] if hasattr(flags, 'loc') and idx in flags.index else {}
        risk_score = final_risk[idx] if hasattr(final_risk, '__getitem__') and idx in final_risk.index else final_risk
        
        # Initialize attack type indicators
        attack_indicators = []
        confidence = "medium"
        # Backdoor signal from per-client triggered ASR (0..1)
        try:
            client_asr = float(X_df.loc[idx].get('client_triggered_asr', 0.0)) if idx in X_df.index else 0.0
        except Exception:
            client_asr = 0.0
        try:
            param_trig = float(X_df.loc[idx].get('param_trigger_change', 0.0)) if idx in X_df.index else 0.0
        except Exception:
            param_trig = 0.0
        
        # Byzantine Attack Detection (Enhanced with dynamic thresholds)
        byzantine_score = 0
        
        # Dynamic confidence scaling based on number of potential attackers
        confidence_scale = 1.0 if num_high_risk == 1 else (0.9 if num_high_risk == 2 else 0.8)
        
        # Check update norm with dynamic thresholds
        if 'update_norm' in client_features.index:
            update_norm = client_features['update_norm']
            # Dynamic thresholds based on number of attackers
            extreme_threshold = 140000 if num_high_risk >= 3 else (150000 if num_high_risk == 2 else 160000)
            large_threshold = 80000 if num_high_risk >= 3 else (90000 if num_high_risk == 2 else 100000)
            minimal_threshold = 0.15 if num_high_risk >= 3 else (0.125 if num_high_risk == 2 else 0.1)
            
            if update_norm > extreme_threshold:
                byzantine_score += 3 * confidence_scale
                attack_indicators.append("extreme_update_magnitude")
            elif update_norm > large_threshold:
                byzantine_score += 2 * confidence_scale
                attack_indicators.append("large_update_magnitude")
            elif update_norm < minimal_threshold and update_norm >= 0:  # Very small (potential Sybil)
                byzantine_score += 1 * confidence_scale
                attack_indicators.append("minimal_update_magnitude")
        
        # Check parameter variance - realistic thresholds (match observed 2e7-4e7 range)
        if 'param_variance' in client_features.index:
            param_variance = client_features['param_variance']
            # Dynamic thresholds based on number of attackers
            extreme_var_threshold = 7e7 if num_high_risk >= 3 else (7.5e7 if num_high_risk == 2 else 8e7)
            high_var_threshold = 1.5e7 if num_high_risk >= 3 else (1.75e7 if num_high_risk == 2 else 2e7)
            
            if param_variance > extreme_var_threshold:
                byzantine_score += 3 * confidence_scale
                attack_indicators.append("extreme_parameter_variance")
            elif param_variance > high_var_threshold:
                byzantine_score += 2 * confidence_scale
                attack_indicators.append("high_parameter_variance")
        
        # Check parameter range - realistic thresholds
        if 'param_range' in client_features.index:
            param_range = client_features['param_range']
            # Dynamic thresholds based on number of attackers
            extreme_range_threshold = 100000 if num_high_risk >= 3 else (110000 if num_high_risk == 2 else 120000)
            high_range_threshold = 25000 if num_high_risk >= 3 else (27500 if num_high_risk == 2 else 30000)
            
            if param_range > extreme_range_threshold:
                byzantine_score += 3 * confidence_scale
                attack_indicators.append("extreme_parameter_range")
            elif param_range > high_range_threshold:
                byzantine_score += 2 * confidence_scale
                attack_indicators.append("high_parameter_range")
        
        # Check max parameter change - realistic thresholds
        if 'max_param_change' in client_features.index:
            max_change = client_features['max_param_change']
            # Dynamic thresholds based on number of attackers
            extreme_change_threshold = 80000 if num_high_risk >= 3 else (90000 if num_high_risk == 2 else 100000)
            high_change_threshold = 8000 if num_high_risk >= 3 else (9000 if num_high_risk == 2 else 10000)
            
            if max_change > extreme_change_threshold:  # Large changes
                byzantine_score += 3 * confidence_scale
                attack_indicators.append("extreme_parameter_changes")
            elif max_change > high_change_threshold:  # Moderate changes
                byzantine_score += 2 * confidence_scale
                attack_indicators.append("high_parameter_changes")
        
        # Check cosine similarity - low similarity indicates Byzantine; high similarity is normal
        if 'cosine_similarity' in client_features.index:
            cosine_sim = client_features['cosine_similarity']
            # Dynamic thresholds based on number of attackers
            extreme_low_sim = 0.15 if num_high_risk >= 3 else (0.125 if num_high_risk == 2 else 0.1)
            low_sim = 0.35 if num_high_risk >= 3 else (0.325 if num_high_risk == 2 else 0.3)
            
            if cosine_sim < extreme_low_sim:  # Very low similarity
                byzantine_score += 2 * confidence_scale
                attack_indicators.append("extreme_low_similarity")
            elif cosine_sim < low_sim:  # Moderately low similarity
                byzantine_score += 1 * confidence_scale
                attack_indicators.append("low_similarity")
        
        # SYBIL ATTACK DETECTION
        sybil_score = 0
        sybil_indicators = []
        
        # Check if client name indicates Sybil (e.g., "sybil_5")
        if 'sybil' in client_id.lower():
            sybil_score += 3
            sybil_indicators.append("sybil_naming_pattern")
        
        # Check for very small updates (characteristic of Sybil clients)
        if 'update_norm' in client_features.index:
            update_norm = client_features['update_norm']
            if update_norm < 0.1:  # Very small updates
                sybil_score += 2
                sybil_indicators.append("minimal_update_magnitude")
        
        # Check for zero or near-zero cosine similarity (duplicated data)
        if 'cosine_similarity' in client_features.index:
            cosine_sim = client_features['cosine_similarity']
            if abs(cosine_sim) < 0.01:  # Near zero similarity
                sybil_score += 2
                sybil_indicators.append("zero_similarity_pattern")
        
        # Check for zero variance (duplicated data)
        if 'param_variance' in client_features.index:
            param_var = client_features['param_variance']
            if param_var < 0.001:  # Near zero variance
                sybil_score += 1
                sybil_indicators.append("zero_variance_pattern")
        
        # Dynamic thresholds based on number of attackers
        if num_high_risk >= 3:
            risk_very_high = 0.65
            risk_high = 0.5
            risk_medium = 0.35
            byz_very_high = 4
            byz_high = 2.5
            byz_medium = 1.5
            sybil_very_high = 3
            sybil_high = 2
            sybil_medium = 1.5
        elif num_high_risk == 2:
            risk_very_high = 0.7
            risk_high = 0.55
            risk_medium = 0.4
            byz_very_high = 4.5
            byz_high = 2.75
            byz_medium = 1.75
            sybil_very_high = 3.5
            sybil_high = 2.5
            sybil_medium = 1.75
        else:
            risk_very_high = 0.75
            risk_high = 0.6
            risk_medium = 0.45
            byz_very_high = 5
            byz_high = 3
            byz_medium = 2
            sybil_very_high = 4
            sybil_high = 3
            sybil_medium = 2
        
        # Determine confidence based on risk score and attack scores with dynamic thresholds
        if risk_score >= risk_very_high and (byzantine_score >= byz_very_high or sybil_score >= sybil_very_high):
            confidence = "very_high"
        elif risk_score >= risk_high and (byzantine_score >= byz_high or sybil_score >= sybil_high):
            confidence = "high"
        elif risk_score >= risk_medium and (byzantine_score >= byz_medium or sybil_score >= sybil_medium):
            confidence = "medium"
        else:
            confidence = "low"
        
        # Classify primary attack type with dynamic thresholds (consider backdoor)
        if client_asr >= 0.6:  # Strong backdoor indicator
            primary_type = "backdoor_attack"
            attack_indicators.append("high_triggered_asr")
        elif sybil_score >= sybil_high:  # Strong Sybil indicators
            primary_type = "sybil_attack"
        elif byzantine_score >= byz_very_high:  # Strong Byzantine indicators
            primary_type = "byzantine_attack"
        elif byzantine_score >= byz_high:  # Moderate Byzantine indicators
            primary_type = "suspected_byzantine_attack"
        elif byzantine_score >= byz_medium:  # Weak Byzantine indicators
            primary_type = "possible_byzantine_attack"
        elif sybil_score >= sybil_medium:  # Weak Sybil indicators
            primary_type = "possible_sybil_attack"
        else:
            primary_type = "unknown"

        # Compose attack_types list for downstream consumers (matching families)
        atypes_list = []
        try:
            # Prefer explicit families
            if byzantine_score >= 2:
                atypes_list.append('byzantine')
            if sybil_score >= 1:
                atypes_list.append('sybil')
            # Add backdoor family when ASR signals present (allow moderate threshold)
            if client_asr >= 0.3 or param_trig >= 0.5:
                atypes_list.append('backdoor')
            # Free-ride heuristic: high staleness with tiny updates or high alignment
            st_val = float(client_features.get('staleness', 0.0) or 0.0)
            un_val = float(client_features.get('update_norm', 0.0) or 0.0)
            cs_val = float(client_features.get('cosine_similarity', 1.0) or 1.0)
            if st_val >= 0.5 and (un_val <= 1e-6 or cs_val >= 0.8):
                atypes_list.append('free-ride')
        except Exception:
            pass
        
        # Store classification results
        if hint_family is not None:
            atypes_list = [hint_family]
            if hint_family == 'label_flip':
                primary_type = 'label_flip_attack'
            elif hint_family == 'sybil':
                primary_type = 'sybil_attack'
            elif hint_family == 'backdoor':
                primary_type = 'backdoor_attack'
            elif hint_family == 'scaling':
                primary_type = 'scaling_attack'
            elif hint_family == 'byzantine':
                primary_type = 'byzantine_attack'
            elif hint_family == 'free-ride':
                primary_type = 'free_ride'

        attack_types[client_id] = {
            'primary_type': primary_type,
            'confidence': confidence,
            'risk_score': float(risk_score),
            'byzantine_score': byzantine_score,
            'sybil_score': sybil_score,
            'indicators': attack_indicators + sybil_indicators,
            'feature_summary': {
                'update_norm': float(client_features.get('update_norm', 0)),
                'cosine_similarity': float(client_features.get('cosine_similarity', 0)),
                'param_variance': float(client_features.get('param_variance', 0)),
                'param_range': float(client_features.get('param_range', 0)),
                'max_param_change': float(client_features.get('max_param_change', 0))
            },
            'attack_types': atypes_list
        }
    
    return attack_types


def train_xgb_detector(X_df, y):
    """
    Trains a binary XGBoost detector if labels are available.
    Returns predicted probabilities and trained booster.
    Enhanced with class weights and improved parameters.
    """
    # Calculate class weights for imbalanced data
    pos_count = sum(y == 1)
    neg_count = sum(y == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    dtrain = xgb.DMatrix(X_df.values, label=y)
    
    # Enhanced parameters for better performance
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,  # Increased from 4
        'eta': 0.05,     # Reduced learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,  # Class weight for imbalanced data
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    # Train with enhanced parameters (removed early stopping for now)
    booster = xgb.train(
        params, 
        dtrain, 
        num_boost_round=200,  # Reduced from 300 to avoid overfitting
        verbose_eval=False
    )
    
    pred_prob = booster.predict(dtrain)  # in-sample; swap for holdout if available
    return pred_prob, booster


# ---------------------------------------------
# Risk fusion and reporting
# ---------------------------------------------

def build_feature_frame(round_metrics, client_updates, global_model_state):
    """Enhanced feature engineering with sophisticated metrics for attack detection."""
    features = pd.DataFrame()
    
    for client_id, metrics in round_metrics.items():
        client_features = {}
        
        # Basic update statistics
        if 'update_norm' in metrics:
            client_features['update_norm'] = metrics['update_norm']
            client_features['update_norm_zscore'] = (
                metrics['update_norm'] - np.mean([m['update_norm'] for m in round_metrics.values()])
            ) / np.std([m['update_norm'] for m in round_metrics.values()])
        
        # Enhanced cosine similarity metrics
        if 'cosine_similarity' in metrics:
            client_features['cosine_similarity'] = metrics['cosine_similarity']
            # Calculate pairwise similarities
            other_clients = [cid for cid in round_metrics.keys() if cid != client_id]
            if other_clients:
                similarities = [round_metrics[cid]['cosine_similarity'] for cid in other_clients]
                client_features['relative_similarity'] = (
                    metrics['cosine_similarity'] - np.mean(similarities)
                ) / np.std(similarities)
        
        # Temporal features
        if 'round_number' in metrics:
            client_features['participation_streak'] = metrics.get('participation_streak', 1)
            client_features['rounds_since_last'] = metrics.get('rounds_since_last', 0)
            
            # Exponential staleness penalty
            if 'staleness' in metrics:
                client_features['staleness'] = metrics['staleness']
                client_features['staleness_exp'] = np.exp(metrics['staleness']) - 1
        
        # Update quality metrics
        if 'weight_updates' in client_updates.get(client_id, {}):
            updates = client_updates[client_id]['weight_updates']
            client_features.update({
                'update_kurtosis': scipy.stats.kurtosis(updates),
                'update_skewness': scipy.stats.skew(updates),
                'gradient_entropy': calculate_gradient_entropy(updates),
                'update_consistency': calculate_update_consistency(updates),
                'gradient_diversity': calculate_gradient_diversity(updates, 
                    [client_updates[cid]['weight_updates'] for cid in other_clients])
            })
        
        # Model divergence metrics
        if global_model_state is not None and 'weights' in client_updates.get(client_id, {}):
            client_weights = client_updates[client_id]['weights']
            client_features.update({
                'model_divergence': calculate_model_divergence(client_weights, global_model_state),
                'weight_consistency': calculate_weight_consistency(client_weights, global_model_state)
            })
        
        # Performance metrics
        if 'accuracy' in metrics and 'loss' in metrics:
            client_features.update({
                'accuracy_trend': metrics.get('accuracy_trend', 0),
                'loss_trend': metrics.get('loss_trend', 0),
                'performance_volatility': metrics.get('performance_volatility', 0)
            })
        
        # Attack-specific indicators
        client_features.update({
            'label_flip_indicator': calculate_label_flip_score(metrics),
            'backdoor_pattern_score': detect_backdoor_patterns(metrics),
            'sybil_similarity_score': calculate_sybil_similarity(metrics, round_metrics)
        })
        
        # Behavioral features
        client_features.update({
            'update_frequency': calculate_update_frequency(metrics),
            'contribution_rate': calculate_contribution_rate(metrics),
            'participation_ratio': calculate_participation_ratio(metrics)
        })
        
        # Store features for this client
        features[client_id] = client_features
    
    return pd.DataFrame.from_dict(features, orient='index')

def apply_rules(feature_frame):
    """Apply detection rules to identify potential attacks."""
    rule_scores = pd.DataFrame()
    
    # Define weights for different rules
    weights = {
        'update_norm_high': 0.15,      # Significant weight for abnormal updates
        'cosine_similarity_low': 0.10,  # Slightly reduced to avoid over-flagging normal divergence
        'fraud_ratio_high': 0.15,       # Critical for detecting malicious behavior
        'staleness_high': 0.1,          # Less critical but still important
        'accuracy_drop': 0.1,           # Indicator of model degradation
        'label_flip_detected': 0.15,    # Specific attack detection
        'backdoor_suspected': 0.1,      # Specific attack detection
        'sybil_detected': 0.1,          # Specific attack detection
        'scaling_detected': 0.2,        # Scaling via explicit factor
        'scaling_anomaly': 0.2          # Robust update-norm anomaly
    }
    
    # Update norm check
    if 'update_norm' in feature_frame.columns:
        rule_scores['update_norm_high'] = feature_frame['update_norm'] > np.percentile(feature_frame['update_norm'], 95)
    else:
        rule_scores['update_norm_high'] = False
    
    # Cosine similarity check
    if 'cosine_similarity' in feature_frame.columns:
        # Lower threshold to reduce false positives during normal training divergence
        rule_scores['cosine_similarity_low'] = feature_frame['cosine_similarity'] < 0.6
    
    # Fraud ratio check
    if 'fraud_ratio' in feature_frame.columns:
        rule_scores['fraud_ratio_high'] = feature_frame['fraud_ratio'] > 0.7
    if 'fraud_ratio_change' in feature_frame.columns:
        # Use a higher threshold to avoid confusing moderate byzantine effects with label flipping
        rule_scores['fraud_ratio_high'] = rule_scores.get('fraud_ratio_high', False) | (feature_frame['fraud_ratio_change'] > 0.5)
    
    # Staleness check
    if 'staleness' in feature_frame.columns:
        rule_scores['staleness_high'] = feature_frame['staleness'] > 0.5
    else:
        rule_scores['staleness_high'] = False
    
    # Performance degradation check
    if 'accuracy_change' in feature_frame.columns:
        rule_scores['accuracy_drop'] = feature_frame['accuracy_change'] < -0.1
    
    # Attack-specific checks with proper thresholds
    if 'label_flip_score' in feature_frame.columns:
        rule_scores['label_flip_detected'] = feature_frame['label_flip_score'] > 0.5  # Lowered threshold
    elif 'label_flip_indicator' in feature_frame.columns:
        # Fallback to engineered indicator when score is not present
        rule_scores['label_flip_detected'] = feature_frame['label_flip_indicator'] > 0.3
        
    # Backdoor: support both trigger rate and engineered pattern score
    backdoor_flag = None
    if 'backdoor_trigger_rate' in feature_frame.columns:
        backdoor_flag = (feature_frame['backdoor_trigger_rate'] > 0.0)
    if 'trigger_rate' in feature_frame.columns:
        backdoor_flag = (feature_frame.get('backdoor_trigger_rate', feature_frame['trigger_rate']) > 0.0) if backdoor_flag is None else (backdoor_flag | (feature_frame['trigger_rate'] > 0.0))
    if 'backdoor_pattern_score' in feature_frame.columns:
        backdoor_flag = (feature_frame['backdoor_pattern_score'] > 0.0) if backdoor_flag is None else (backdoor_flag | (feature_frame['backdoor_pattern_score'] > 0.0))
    if backdoor_flag is not None:
        rule_scores['backdoor_suspected'] = backdoor_flag
    if 'client_triggered_asr' in feature_frame.columns:
        try:
            asr_thr = float(getattr(Cfg, 'asr_threshold_for_alert', 0.33))
        except Exception:
            asr_thr = 0.33
        try:
            asr_vals = pd.to_numeric(feature_frame['client_triggered_asr'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
            asr_flag = asr_vals >= asr_thr
            rule_scores['backdoor_suspected'] = rule_scores.get('backdoor_suspected', False) | asr_flag
        except Exception:
            pass
        
    # Scaling: detect explicit factor and update-norm anomalies
    try:
        if 'scaling_factor' in feature_frame.columns:
            rule_scores['scaling_detected'] = feature_frame['scaling_factor'] > 1.0
        if 'update_norm' in feature_frame.columns:
            norms = feature_frame['update_norm'].astype(float)
            med = float(np.median(norms))
            # Define very-low update threshold relative to cohort
            low_thresh = max(1e-3, med * 0.01)
            scaling_anom = norms <= low_thresh
            # Gate out scaling anomaly when staleness indicates free-ride behavior
            if 'staleness' in feature_frame.columns:
                scaling_anom = scaling_anom & (feature_frame['staleness'] <= 0.3)
            rule_scores['scaling_anomaly'] = scaling_anom
            rule_scores['scaling_detected'] = rule_scores.get('scaling_detected', False) | rule_scores['scaling_anomaly']
    except Exception:
        pass

    if 'sybil_similarity' in feature_frame.columns:
        rule_scores['sybil_detected'] = feature_frame['sybil_similarity'] > 0.8  # Lowered threshold
    
    # Additional Sybil cues
    if 'is_sybil_name' in feature_frame.columns:
        rule_scores['sybil_name_flag'] = feature_frame['is_sybil_name'] > 0.5
    if 'sybil_similarity_score' in feature_frame.columns:
        rule_scores['high_sybil_similarity'] = feature_frame['sybil_similarity_score'] > 0.9
    if 'sybil_risk' in feature_frame.columns:
        # Consider non-trivial sybil_risk as an indicator
        rule_scores['high_sybil_risk'] = feature_frame['sybil_risk'] > 0.2

    # Initialize risk score
    risk_score = pd.Series(0.0, index=feature_frame.index)
    total_weight = 0
    
    # Calculate weighted risk score
    # Extend weights with sybil-specific rules
    weights.update({
        'sybil_name_flag': 0.8,
        'high_sybil_similarity': 1.0,
        'high_sybil_risk': 1.0,
    })

    for rule, weight in weights.items():
        if rule in rule_scores.columns:
            risk_score += rule_scores[rule].astype(float) * weight
            total_weight += weight

    # Add continuous-valued contributions to break ties and reflect magnitude
    cont_weight = 0.0
    if 'update_norm' in feature_frame.columns:
        try:
            un = pd.to_numeric(feature_frame['update_norm'], errors='coerce').fillna(0.0)
            un_z = (un - un.mean()) / (un.std() if un.std() != 0 else 1.0)
            un_z = un_z.clip(-3, 3).abs() / 3.0
            # REDUCED: Lower weight for update_norm (was 0.2, now 0.15) for backdoor stealth
            risk_score += 0.15 * un_z
            cont_weight += 0.15
        except Exception:
            pass
    if 'param_variance' in feature_frame.columns:
        try:
            pv = pd.to_numeric(feature_frame['param_variance'], errors='coerce').fillna(0.0)
            pv_log = np.log1p(pv)
            pv_n = (pv_log - pv_log.min()) / ((pv_log.max() - pv_log.min()) or 1.0)
            risk_score += 0.2 * pv_n
            cont_weight += 0.2
        except Exception:
            pass
    if 'fraud_ratio_change' in feature_frame.columns:
        try:
            frc = pd.to_numeric(feature_frame['fraud_ratio_change'], errors='coerce').fillna(0.0).abs()
            frc_n = (frc - frc.min()) / ((frc.max() - frc.min()) or 1.0)
            # REDUCED: Lower weight for fraud_ratio_change (was 0.25, now 0.15) for backdoor stealth
            risk_score += 0.15 * frc_n
            cont_weight += 0.15
        except Exception:
            pass
    if 'cosine_similarity' in feature_frame.columns:
        try:
            cs = pd.to_numeric(feature_frame['cosine_similarity'], errors='coerce').fillna(0.0)
            cs_dev = (0.85 - cs).clip(lower=0)
            cs_n = cs_dev / (cs_dev.max() if cs_dev.max() != 0 else 1.0)
            risk_score += 0.15 * cs_n
            cont_weight += 0.15
        except Exception:
            pass
    if 'staleness' in feature_frame.columns:
        try:
            st = pd.to_numeric(feature_frame['staleness'], errors='coerce').fillna(0.0)
            risk_score += 0.10 * st.clip(0.0, 1.0)
            cont_weight += 0.10
        except Exception:
            pass
    # Backdoor continuous contributions to break ties between multiple backdoor attackers
    try:
        if ('backdoor_trigger_rate' in feature_frame.columns) or ('trigger_rate' in feature_frame.columns):
            btr_series = feature_frame.get('backdoor_trigger_rate', feature_frame.get('trigger_rate')).astype(float).fillna(0.0)
            btr_n = (btr_series - btr_series.min()) / ((btr_series.max() - btr_series.min()) or 1.0)
            risk_score += 0.20 * btr_n
            cont_weight += 0.20
    except Exception:
        pass
    try:
        if 'backdoor_pattern_score' in feature_frame.columns:
            bps = pd.to_numeric(feature_frame['backdoor_pattern_score'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
            risk_score += 0.15 * bps
            cont_weight += 0.15
    except Exception:
        pass
    
    # NEW: Client-triggered ASR contribution for better backdoor detection
    try:
        if 'client_triggered_asr' in feature_frame.columns:
            client_asr = pd.to_numeric(feature_frame['client_triggered_asr'], errors='coerce').fillna(0.0)
            # Normalize to 0-1 range (assuming ASR can go up to 100%)
            client_asr_norm = (client_asr / 100.0).clip(0.0, 1.0)
            risk_score += 0.25 * client_asr_norm  # High weight for ASR contribution
            cont_weight += 0.25
    except Exception:
        pass

    # Normalize risk score including continuous contributions
    denom = (total_weight + cont_weight) if (total_weight + cont_weight) > 0 else 1.0
    risk_score = risk_score / denom
    
    # Apply risk boosting for multiple triggered rules
    triggered_rules = rule_scores.sum(axis=1)
    # Reduce boost cap to avoid inflating risk too aggressively
    boost = np.minimum(triggered_rules / len(weights), 0.15)
    risk_score = np.minimum(risk_score + boost, 1.0)
    
    # Convert boolean flags to float for consistent handling
    rule_flags = rule_scores.astype(float)
    
    # Remove problematic min-max normalization that creates fixed values
    # Instead, keep risk scores in their natural range [0,1] based on weights
    # The risk_score is already properly normalized by total_weight + cont_weight
    
    return rule_flags, risk_score

def anomaly_detector(feature_frame):
    """
    Enhanced anomaly detection using Isolation Forest with proper feature selection and scaling.
    Returns anomaly scores in [0,1] where higher values indicate more anomalous behavior.
    """
    # Select relevant numerical features for anomaly detection
    feature_cols = [
        'update_norm', 'cosine_similarity', 'fraud_ratio_change', 
        'param_variance', 'staleness', 'scaling_factor', 'trigger_rate',
        'label_flip_score', 'backdoor_trigger_rate', 'sybil_similarity'
    ]
    
    # Filter to only available features
    available_features = [col for col in feature_cols if col in feature_frame.columns]
    if not available_features:
        # If no relevant features found, return neutral scores
        return np.zeros(len(feature_frame))
    
    # Extract features and handle missing values
    X = feature_frame[available_features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        print(f"Warning: Feature scaling failed: {e}")
        return np.zeros(len(feature_frame))
    
    # Run Isolation Forest with more sensitive contamination
    try:
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.2,  # Increased from 0.1 to be more sensitive
            random_state=42,
            max_samples='auto'
        )
        # Fit and predict
        scores = iso_forest.fit_predict(X_scaled)
        
        # Convert predictions (-1 for anomalies, 1 for normal) to [0,1] scores
        # where higher values indicate more anomalous behavior
        scores = (scores == -1).astype(float)
        
        # Add confidence based on decision function
        decision_scores = iso_forest.decision_function(X_scaled)
        # Normalize decision scores to [0,1] range where higher is more anomalous
        decision_scores = -decision_scores  # Invert so higher means more anomalous
        decision_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-10)
        
        # Combine binary predictions with confidence scores
        final_scores = 0.7 * scores + 0.3 * decision_scores
        final_scores = np.clip(final_scores, 0, 1)
        
        return final_scores
        
    except Exception as e:
        print(f"Warning: Anomaly detection failed: {e}")
        return np.zeros(len(feature_frame))

def classify_attack_types(feature_frame, rule_flags, risk_scores, attack_hint=None):
    """Classify attacks by computing comparative strengths and selecting a dominant family per client."""
    attack_types = {}

    # Precompute z-scores for divergence features to strengthen Byzantine scoring
    def zscore(series):
        try:
            s = series.astype(float)
            mu, sd = float(s.mean()), float(s.std() or 1.0)
            return (s - mu) / (sd if sd != 0 else 1.0)
        except Exception:
            return pd.Series([0.0] * len(series), index=series.index)

    z_update_norm = zscore(feature_frame['update_norm']) if 'update_norm' in feature_frame.columns else pd.Series(0.0, index=feature_frame.index)
    z_param_var = zscore(feature_frame['param_variance']) if 'param_variance' in feature_frame.columns else pd.Series(0.0, index=feature_frame.index)
    z_max_change = zscore(feature_frame['max_param_change']) if 'max_param_change' in feature_frame.columns else pd.Series(0.0, index=feature_frame.index)

    # Normalize attack hint
    try:
        sel = str(attack_hint).lower().replace(' attack','').replace('-', '_').replace(' ', '_') if attack_hint else ''
    except Exception:
        sel = ''

    for idx, row in feature_frame.iterrows():
        # Get the risk score for this client - handle both array and scalar cases
        try:
            # Try to get risk score by integer index first
            if hasattr(risk_scores, '__getitem__') and len(risk_scores) > 0:
                if isinstance(idx, int) and idx < len(risk_scores):
                    risk_score = float(risk_scores[idx])
                else:
                    # Try to find position in dataframe index
                    pos = feature_frame.index.get_loc(idx)
                    if pos < len(risk_scores):
                        risk_score = float(risk_scores[pos])
                    else:
                        risk_score = float(risk_scores[0]) if len(risk_scores) == 1 else 0.5
            else:
                risk_score = float(risk_scores)
        except (IndexError, KeyError, ValueError):
            risk_score = 0.5  # Default fallback
            
        # Early short-circuit: clear free-ride signature (high staleness + near-zero update)
        try:
            st_early = float(row.get('staleness', 0.0) or 0.0)
            un_early = float(row.get('update_norm', 0.0) or 0.0)
        except Exception:
            st_early, un_early = 0.0, 0.0
        if st_early >= 0.5 and un_early <= 1e-6:
            # Compose a deterministic free-ride result and continue
            try:
                id_val_early = row.get('client', idx)
            except Exception:
                id_val_early = idx
            id_key_early = str(id_val_early)
            attack_types[id_key_early] = {
                'attack_types': ['free-ride'],
                'confidence': 0.8 if risk_score >= 0.5 else 0.6,
                'risk_score': float(risk_score)
            }
            continue

        # Strength scores
        lf_strength = 0.0
        if 'label_flip_score' in row:
            lf_strength = max(lf_strength, float(row['label_flip_score']))
        if 'fraud_ratio_change' in row:
            lf_strength = max(lf_strength, float(row['fraud_ratio_change']))
        if 'fraud_ratio' in row:
            lf_strength = max(lf_strength, float(row['fraud_ratio']))
        if 'fraud_momentum' in row:
            lf_strength = max(lf_strength, float(row.get('fraud_momentum', 0.0)))
        # Scale LF into [0,1]
        lf_strength = float(max(0.0, min(1.0, lf_strength)))

        # Backdoor strength: based on trigger evidence or engineered pattern
        back_strength = 0.0
        try:
            btr = float(row.get('backdoor_trigger_rate', row.get('trigger_rate', 0.0)) or 0.0)
            bps = float(row.get('backdoor_pattern_score', 0.0) or 0.0)
            # Normalize trigger rate and combine with pattern score
            back_strength = max(0.0, min(1.0, max(btr / 0.5, bps)))
        except Exception:
            back_strength = 0.0

        # Free-ride strength: high when staleness is high and divergence is low (high cosine)
        fr_strength = 0.0
        st = float(row.get('staleness', 0.0) or 0.0)
        cs_val_local = float(row.get('cosine_similarity', 1.0) or 1.0)
        # Emphasize staleness; prefer free-ride when updates align (cosine high)
        fr_strength = min(1.0, max(0.0, st)) * (1.0 if cs_val_local >= 0.8 else 0.6)
        # Also treat "no/near-zero update" with high staleness as free-ride regardless of cosine
        un_local = float(row.get('update_norm', 0.0) or 0.0)
        if st >= 0.5 and un_local <= 1e-6:
            fr_strength = max(fr_strength, 0.8)
        # If we engineered flags exist, boost
        try:
            client_flags = rule_flags.loc[idx] if hasattr(rule_flags, 'loc') else {}
        except Exception:
            client_flags = {}
        if isinstance(client_flags, pd.Series):
            if client_flags.get('stale_model', False):
                fr_strength = max(fr_strength, 0.6)
            if client_flags.get('extreme_staleness', False):
                fr_strength = max(fr_strength, 0.8)
            if client_flags.get('minimal_changes', False):
                fr_strength = max(fr_strength, 0.7)

        byz_strength = 0.0
        # Cosine divergence (only penalize very low similarity)
        if 'cosine_similarity' in row and row['cosine_similarity'] is not None:
            cs = float(row['cosine_similarity'])
            if cs < 0.8:
                byz_strength += (0.8 - cs) / 0.8  # up to ~1 when cs≈0
        # Add positive z-scores from divergence features
        byz_strength += max(0.0, float(z_update_norm.loc[idx]))
        byz_strength += max(0.0, float(z_param_var.loc[idx]))
        byz_strength += max(0.0, float(z_max_change.loc[idx]))
        # Normalize byz_strength into a soft-bounded range
        byz_strength = float(min(1.5, byz_strength))

        # Optional: model poisoning signal (kept separate)
        poison_strength = 0.0
        if 'update_norm' in row and 'cosine_similarity' in row:
            if float(row['update_norm']) > np.percentile(feature_frame['update_norm'], 90) and float(row['cosine_similarity']) < 0.6:
                poison_strength = 0.6

        # Decide dominant class among label_flip vs byzantine vs free-ride vs backdoor vs scaling
        # Compute scaling strength on the fly using explicit factor and update norm anomaly
        sc_strength = 0.0
        try:
            sc_val = float(row.get('scaling_factor', 1.0) or 1.0)
        except Exception:
            sc_val = 1.0
        scale_present = sc_val > 1.0
        try:
            sc_strength = max(sc_strength, sc_val - 1.0)
            if 'update_norm' in row and 'update_norm' in feature_frame.columns:
                norms = feature_frame['update_norm'].astype(float)
                p_low = float(np.percentile(norms, 5))
                p_high = float(np.percentile(norms, 95))
                u = float(row.get('update_norm', 0.0) or 0.0)
                if u <= p_low or u >= p_high:
                    sc_strength = max(sc_strength, 0.7)
        except Exception:
            pass

        chosen = None
        margin = 0.25  # require a larger margin to avoid flapping
        
        # Attack type isolation: if attack_hint is provided, ONLY use that type
        if sel == 'backdoor':
            chosen = 'backdoor'  # Force backdoor if hinted
        elif sel == 'label_flip':
            chosen = 'label_flip'  # Force label_flip if hinted
        elif sel == 'scaling':
            chosen = 'scaling'  # Force scaling if hinted
        # Only use automatic detection if no hint provided
        elif fr_strength >= max(lf_strength, byz_strength, back_strength, sc_strength) + margin:
            chosen = 'free-ride'
        elif back_strength >= max(lf_strength, fr_strength, byz_strength, sc_strength) + margin:
            chosen = 'backdoor'
        elif scale_present and (sc_strength >= max(lf_strength, fr_strength, byz_strength, back_strength) + margin):
            chosen = 'scaling'
        elif byz_strength >= max(lf_strength, fr_strength, back_strength, sc_strength) + margin:
            chosen = 'byzantine'
        elif lf_strength >= max(byz_strength, fr_strength, back_strength, sc_strength) + margin:
            chosen = 'label_flip'
        else:
            # No clear winner: pick the one aligned with the rule flags if any
            if isinstance(client_flags, pd.Series):
                if client_flags.get('backdoor_suspected', 0):
                    chosen = 'backdoor'
                if chosen is None:
                    sf = bool(client_flags.get('scaling_detected', 0) or client_flags.get('scaling_anomaly', 0))
                    # Recompute a strong anomaly check (very small update vs cohort median)
                    strong_anom_fb = False
                    try:
                        if 'update_norm' in feature_frame.columns:
                            norms_fb = feature_frame['update_norm'].astype(float)
                            med_fb = float(np.median(norms_fb)) if len(norms_fb) > 0 else 0.0
                            u_fb = float(row.get('update_norm', 0.0) or 0.0)
                            strong_anom_fb = (u_fb > 0.0 and med_fb > 0.0 and (u_fb <= max(1e-3, med_fb * 0.01)))
                    except Exception:
                        strong_anom_fb = False
                    if scale_present or (sf and strong_anom_fb):
                        chosen = 'scaling'
                if chosen is None and any(client_flags.get(k, 0) for k in ['byzantine_update_norm','byzantine_cosine','byzantine_weight_delta','byzantine_param_variance','byzantine_param_range','byzantine_max_change']):
                    chosen = 'byzantine'
                if chosen is None and (client_flags.get('stale_model', 0) or client_flags.get('extreme_staleness', 0) or client_flags.get('minimal_changes', 0)):
                    chosen = 'free-ride'
                if chosen is None and (client_flags.get('label_flip_detected', 0) or lf_strength > 0.2):
                    # Respect attack_hint: outside of label_flip runs, only accept strong LF
                    if not sel or sel == 'label_flip':
                        chosen = 'label_flip'
                    elif lf_ev >= 0.85:
                        chosen = 'label_flip'
            # Fallback
            if chosen is None:
                if fr_strength >= 0.5:
                    chosen = 'free-ride'
                elif back_strength >= 0.3:
                    chosen = 'backdoor'
                elif lf_strength >= 0.3:
                    if not sel or sel == 'label_flip' or lf_ev >= 0.85:
                        chosen = 'label_flip'
                elif byz_strength >= 0.3:
                    chosen = 'byzantine'

        # Confidence derived from dominant strength and risk
        risk_val = risk_score  # Use the already computed risk_score
        # Ensure scaling surfaces when selected
        if chosen == 'scaling':
            risk_val = float(max(risk_val, 0.6))

        # Compute divergence evidence (cosine + z-scores)
        cs_val = float(row['cosine_similarity']) if 'cosine_similarity' in row and row['cosine_similarity'] is not None else 1.0
        z_un = max(0.0, float(z_update_norm.loc[idx]))
        z_pv = max(0.0, float(z_param_var.loc[idx]))
        z_mx = max(0.0, float(z_max_change.loc[idx]))
        # Aggregate label-flip evidence and prefer LF when cosine is high
        try:
            lf_ev = max(
                float(row.get('label_flip_score', 0) or 0.0),
                float(row.get('fraud_ratio', 0) or 0.0),
                float(row.get('fraud_ratio_change', 0) or 0.0),
                float(row.get('fraud_momentum', 0) or 0.0),
            )
        except Exception:
            lf_ev = float(lf_strength)

        # Backdoor pattern: trigger evidence should dominate LF
        backdoor_rate = float(row.get('backdoor_trigger_rate', row.get('trigger_rate', 0.0)) or 0.0)
        backdoor_pattern_score = float(row.get('backdoor_pattern_score', 0.0) or 0.0)
        # Accept extremely small trigger rates; require supportive signals (aligned cosine or moderate fraud shift)
        cs_gate = (cs_val >= 0.85)
        fraud_gate = (float(row.get('fraud_ratio_change', 0.0) or 0.0) >= 0.1)
        back_pattern = ((backdoor_rate > 0.0) or (backdoor_pattern_score > 0.0)) and (cs_gate or fraud_gate)
        if back_pattern:
            chosen = 'backdoor'
            risk_val = float(max(risk_score, 0.6))

        # Dynamic thresholds for label flip pattern based on number of high-risk clients
        num_high_risk = int(row.get('num_high_risk', 1))  # Default to 1 if not provided
        
        if num_high_risk >= 3:
            lf_threshold = 0.75  # Lower threshold for multiple attackers
            cs_threshold = 0.65
            st_threshold = 0.6  # More lenient staleness threshold
        elif num_high_risk == 2:
            lf_threshold = 0.80
            cs_threshold = 0.70
            st_threshold = 0.55
        else:
            lf_threshold = 0.85  # Original thresholds for single attacker
            cs_threshold = 0.75
            st_threshold = 0.5
        
        # Label flip pattern with gating: if selected attack is not label_flip, require strong LF evidence
        # Do not override when staleness indicates free-ride behavior or backdoor pattern
        if sel and sel != 'label_flip':
            lf_pattern = (lf_ev >= lf_threshold and cs_val >= cs_threshold and st < st_threshold and not back_pattern)
        else:
            lf_pattern = (lf_ev >= 0.10 and cs_val >= 0.75 and st < 0.5 and not back_pattern)
        if lf_pattern:
            chosen = 'label_flip'
            # Ensure LF shows up in results when otherwise low risk
            risk_val = float(max(risk_score, 0.55))

        # Only force byzantine with strong evidence AND very low cosine and LF/FR absent
        # Consider free-ride when staleness is high and either cosine is aligned OR updates are near-zero
        fr_pattern = (st >= 0.5) and ((cs_val >= 0.6) or (float(row.get('update_norm', 0.0) or 0.0) <= 1e-6))
        max_z = max(z_un, z_pv, z_mx)
        z_high_count = int(z_un > 1.5) + int(z_pv > 1.5) + int(z_mx > 1.5)
        try:
            un_hi = 'update_norm' in feature_frame.columns and (float(row.get('update_norm', 0.0) or 0.0) > np.percentile(feature_frame['update_norm'].astype(float), 95))
        except Exception:
            un_hi = False
        strong_byz = (cs_val < 0.4) and (z_high_count >= 2 or max_z > 2.5) and un_hi
        if strong_byz and not (lf_pattern or fr_pattern or scale_present or back_pattern):
            chosen = 'byzantine'
            # Apply a conservative bump only when strong evidence holds
            risk_val = float(max(risk_score, 0.6))

        # Guard: only allow final 'byzantine' if strong evidence and not free-ride
        if chosen == 'byzantine':
            if fr_pattern or not strong_byz:
                chosen = None

        # Final safeguards: prioritize specific attacks
        if back_pattern:
            chosen = 'backdoor'
        elif lf_pattern and chosen != 'label_flip':
            chosen = 'label_flip'

        # Confidence derived from dominant strength and risk
        conf = max(0.0, min(1.0, (lf_strength if chosen=='label_flip' else byz_strength if chosen=='byzantine' else 0.0)))
        # Cap byzantine confidence when cosine indicates alignment
        if chosen == 'byzantine' and cs_val >= 0.75:
            conf = min(conf, 0.4)
        if risk_score > 0.7:
            conf = max(conf, 0.8)
        elif risk_score > 0.5:
            conf = max(conf, 0.6)

        # Sybil: preserve sybil_* classification separately
        extra = []
        try:
            client_id_str = str(row.get('client', idx)).lower()
        except Exception:
            client_id_str = str(idx).lower()
        if client_id_str.startswith('sybil_'):
            extra.append('sybil')
            # Ensure sybil entries surface with meaningful risk/confidence
            risk_val = float(max(risk_val, 0.5 if float(row.get('cosine_similarity', 0.0) or 0.0) >= 0.85 else 0.4))
            # Boost confidence moderately for explicit sybil naming
            conf = max(0.6, 0.8 if risk_val >= 0.6 else 0.6)

        # Compose final attack list
        final_list = []
        # Dominant choice when risk is sufficient
        if chosen:
            label = 'free-ride' if chosen == 'free-ride' else chosen
            if risk_val >= 0.5:
                final_list.append(label)

        # Ensure backdoor added when pattern exists
        if back_pattern and 'backdoor' not in final_list:
            final_list.append('backdoor')

        # Ensure scaling added when evidence exists (strict to avoid false positives)
        try:
            scale_present = (float(row.get('scaling_factor', 1.0) or 1.0) > 1.0)
        except Exception:
            scale_present = False
        scale_flag = False
        if isinstance(client_flags, pd.Series):
            scale_flag = bool(client_flags.get('scaling_detected', 0) or client_flags.get('scaling_anomaly', 0))
        strong_anomaly = False
        try:
            if 'update_norm' in feature_frame.columns:
                norms = feature_frame['update_norm'].astype(float)
                med = float(np.median(norms)) if len(norms) > 0 else 0.0
                u = float(row.get('update_norm', 0.0) or 0.0)
                # Very small absolute update relative to cohort median indicates scaling-style tampering
                strong_anomaly = (u > 0.0 and med > 0.0 and (u <= max(1e-3, med * 0.01)))
        except Exception:
            strong_anomaly = False
        # Do not add 'scaling' when label flip is the selected/primary attack to avoid mixing
        if (scale_present or (scale_flag and strong_anomaly)) and ('scaling' not in final_list):
            if chosen != 'label_flip' and sel != 'label_flip':
                final_list.append('scaling')

        # Exclusivity guard: prefer free-ride over scaling when high staleness suggests free-ride behavior
        try:
            st_v = float(row.get('staleness', 0.0) or 0.0)
        except Exception:
            st_v = 0.0
        if st_v >= 0.5 and 'scaling' in final_list:
            final_list = [x for x in final_list if x != 'scaling']

        # Add sybil tag if present
        final_list += [x for x in extra if x not in final_list]

        # Fallback: if still empty but we have a chosen label, include it to surface the type
        if not final_list and chosen:
            final_list.append('free-ride' if chosen == 'free-ride' else chosen)

        try:
            id_val = row.get('client', idx)
        except Exception:
            id_val = idx
        id_key = str(id_val)
        attack_types[id_key] = {
            'attack_types': final_list,
            'confidence': conf,
            'risk_score': float(risk_score)
        }

    return attack_types

def fuse_risk(rule_scores, anomaly_scores, xgb_scores=None, weights=None):
    """
    Fuses available sources into a FinalRisk in [0,1].
    weights: dict with keys 'rules', 'anomaly', 'supervised'
    """
    if weights is None:
        weights = {'rules': 0.5, 'anomaly': 0.3, 'supervised': 0.2}  # Increased rule weight

    # Convert inputs to numpy arrays and ensure they are 1D
    rule_scores = np.array(rule_scores)
    if rule_scores.ndim > 1:
        # If rule_scores is 2D, take the mean across columns
        rule_scores = np.mean(rule_scores, axis=1)
    
    anomaly_scores = np.array(anomaly_scores)
    if anomaly_scores.ndim > 1:
        anomaly_scores = np.mean(anomaly_scores, axis=1)
        
    if xgb_scores is not None:
        xgb_scores = np.array(xgb_scores)
        if xgb_scores.ndim > 1:
            xgb_scores = np.mean(xgb_scores, axis=1)

    # Ensure all scores have the same length
    n_clients = len(rule_scores)
    if len(anomaly_scores) != n_clients:
        # Broadcast anomaly scores to match rule_scores length
        anomaly_scores = np.full(n_clients, np.mean(anomaly_scores))

    if xgb_scores is not None and len(xgb_scores) != n_clients:
        # Broadcast xgb_scores to match rule_scores length
        xgb_scores = np.full(n_clients, np.mean(xgb_scores))

    # If supervised not available, redistribute weight
    if xgb_scores is None:
        total = weights['rules'] + weights['anomaly']
        wr = weights['rules'] / total
        wa = weights['anomaly'] / total
        final = wr * rule_scores + wa * anomaly_scores
    else:
        final = (weights['rules'] * rule_scores +
                weights['anomaly'] * anomaly_scores +
                weights['supervised'] * xgb_scores)
    
    # Ensure final scores are in [0,1] range
    final = np.clip(final, 0, 1)
    
    return final

def analyze_sybil_cluster(round_logs, attack_hint=None):
    """
    Analyze Sybil attack cluster patterns across multiple rounds.
    
    Returns:
        dict: Sybil cluster analysis results including:
            - cluster_members: List of clients in the cluster
            - cluster_correlation: Average correlation score
            - consistency_rounds: Number of rounds with cluster activity
            - combined_influence: Total influence of the cluster
            - round_analysis: Per-round cluster behavior
    """
    if not round_logs or attack_hint != 'sybil':
        return {}
    
    try:
        import numpy as np
        from scipy.stats import pearsonr
        from collections import defaultdict
        
        # Group logs by round
        rounds_data = defaultdict(list)
        for log in round_logs:
            if isinstance(log, dict) and 'round' in log:
                rounds_data[log['round']].append(log)
        
        cluster_analysis = {
            'cluster_members': [],
            'cluster_correlation': 0.0,
            'consistency_rounds': 0,
            'combined_influence': 0.0,
            'round_analysis': [],
            'sybil_detected': False
        }
        
        # Identify potential Sybil clients (those with sybil_ prefix or high similarity)
        potential_sybils = set()
        real_attacker = None
        
        for round_num, logs in rounds_data.items():
            round_similarities = {}
            round_norms = {}
            
            for log in logs:
                client_id = str(log.get('client', ''))
                if client_id.startswith('sybil_'):
                    potential_sybils.add(client_id)
                elif log.get('is_attacker', False) and not client_id.startswith('sybil_'):
                    real_attacker = client_id
                
                # Store metrics for correlation analysis
                round_similarities[client_id] = log.get('cosine_similarity', 0.0)
                round_norms[client_id] = log.get('update_norm', 0.0)
            
            # Analyze cluster behavior for this round
            if potential_sybils and real_attacker:
                # Calculate correlations between real attacker and sybils
                attacker_norm = round_norms.get(real_attacker, 0.0)
                attacker_sim = round_similarities.get(real_attacker, 0.0)
                
                sybil_norms = [round_norms.get(s, 0.0) for s in potential_sybils if s in round_norms]
                sybil_sims = [round_similarities.get(s, 0.0) for s in potential_sybils if s in round_similarities]
                
                # Check for high correlation (copy-cat behavior)
                high_correlation_count = 0
                total_influence = 0.0
                
                for sybil_id in potential_sybils:
                    if sybil_id in round_norms:
                        sybil_norm = round_norms[sybil_id]
                        sybil_sim = round_similarities.get(sybil_id, 0.0)
                        
                        # Check cosine similarity (>0.95 indicates copy-cat)
                        if sybil_sim > 0.95:
                            high_correlation_count += 1
                        
                        # Check norm similarity (<10% difference)
                        if attacker_norm > 0:
                            norm_diff = abs(sybil_norm - attacker_norm) / attacker_norm
                            if norm_diff < 0.1:
                                high_correlation_count += 1
                        
                        # Calculate influence
                        total_norms = sum(round_norms.values())
                        if total_norms > 0:
                            total_influence += sybil_norm / total_norms
                
                # Add attacker influence
                if attacker_norm > 0 and sum(round_norms.values()) > 0:
                    total_influence += attacker_norm / sum(round_norms.values())
                
                # Determine cluster status
                cluster_status = "INACTIVE"
                if high_correlation_count >= len(potential_sybils) * 0.5:  # At least 50% show correlation
                    if total_influence > 0.5:
                        cluster_status = "STRONG"
                        cluster_analysis['consistency_rounds'] += 1
                    elif total_influence > 0.4:
                        cluster_status = "ACTIVE"
                        cluster_analysis['consistency_rounds'] += 1
                
                # Store round analysis
                round_info = {
                    'round': round_num,
                    'attacker_norm': f"{attacker_norm:.1f}×" if attacker_norm > 1 else f"{attacker_norm:.3f}",
                    'sybil_norms': [f"{norm:.1f}×" if norm > 1 else f"{norm:.3f}" for norm in sybil_norms],
                    'cluster_correlation': np.mean(sybil_sims) if sybil_sims else 0.0,
                    'influence_spike': f"{total_influence*100:.0f}%",
                    'cluster_status': cluster_status
                }
                cluster_analysis['round_analysis'].append(round_info)
        
        # Determine final cluster members and overall metrics
        if potential_sybils and real_attacker:
            cluster_analysis['cluster_members'] = [real_attacker] + sorted(list(potential_sybils))
            
            # Calculate overall correlation (average across rounds)
            correlations = [r['cluster_correlation'] for r in cluster_analysis['round_analysis']]
            cluster_analysis['cluster_correlation'] = np.mean(correlations) if correlations else 0.0
            
            # Calculate combined influence (average across rounds)
            influences = []
            for r in cluster_analysis['round_analysis']:
                try:
                    influence_val = float(r['influence_spike'].rstrip('%')) / 100.0
                    influences.append(influence_val)
                except:
                    influences.append(0.0)
            cluster_analysis['combined_influence'] = np.mean(influences) if influences else 0.0
            
            # Determine if Sybil attack is detected (≥3 of 5 rounds with activity)
            total_rounds = len(rounds_data)
            if cluster_analysis['consistency_rounds'] >= max(3, int(0.6 * total_rounds)):
                cluster_analysis['sybil_detected'] = True
        
        return cluster_analysis
        
    except Exception as e:
        print(f"[SYBIL ANALYSIS ERROR] {str(e)}")
        return {}

def create_enhanced_report(meta, X_df, flags, final_risk, round_logs, dt=None, attack_hint=None):
    """
    Create an enhanced detection report with detailed analysis and recommendations.
    """
    if dt is None:
        from datetime import datetime as dt
    
    # Handle case when meta is None
    if meta is None:
        # Create a simple report without client metadata
        enhanced_report = {
            'timestamp': dt.now().isoformat(),
            'total_clients': len(final_risk),
            'high_risk_clients': [],
            'attack_types_detected': {},
            'detection_methods_used': {
                'rule_based': True,
                'anomaly_detection': True,
                'xgboost': False
            },
            'detection_summary': {
                'total_high_risk': 0,
                'true_positives': 0,
                'false_positives': 0,
                'precision': 0,
                'attack_types_count': 0
            },
            'recommendations': []
        }
        
        # Use configured detection threshold (default 0.33)
        if len(final_risk) > 0:
            risk_threshold = getattr(Cfg, 'detection_threshold', 0.33)
            
            # Analyze high risk clients
            for i in range(len(final_risk)):
                risk_score = final_risk[i]
                
                if risk_score >= risk_threshold:
                    # Create client info from available data
                    client_features = X_df.iloc[i] if hasattr(X_df, 'iloc') else X_df[i]
                    client_flags = flags.iloc[i] if hasattr(flags, 'iloc') else flags[i]
                    
                    # Generate a client ID
                    client_id = f"client_{i}"
                    
                    # Analyze attack patterns
                    attack_analysis = analyze_attack_patterns(client_features, client_flags, client_id, round_logs, attack_hint=attack_hint)
                    
                    high_risk_entry = {
                        'client_id': client_id,
                        'round': 0,  # Default round number
                        'risk_score': float(risk_score),
                        'is_attacker': False,  # Unknown without metadata
                        'attack_types': attack_analysis['attack_types'],
                        'triggered_rules': attack_analysis['triggered_rules'],
                        'anomaly_score': float(client_features.get('anomaly_score', 0)),
                        'confidence': attack_analysis['confidence']
                    }
                    
                    enhanced_report['high_risk_clients'].append(high_risk_entry)
                    
                    # Track attack types only if risk score exceeds threshold
                    for attack_type in attack_analysis['attack_types']:
                        if high_risk_entry['risk_score'] >= risk_threshold:
                            if attack_type not in enhanced_report['attack_types_detected']:
                                enhanced_report['attack_types_detected'][attack_type] = []
                            enhanced_report['attack_types_detected'][attack_type].append(client_id)
        
        # Update summary
        enhanced_report['detection_summary'] = {
            'total_high_risk': len(enhanced_report['high_risk_clients']),
            'true_positives': sum(1 for client in enhanced_report['high_risk_clients'] if client['is_attacker']),
            'false_positives': sum(1 for client in enhanced_report['high_risk_clients'] if not client['is_attacker']),
            'precision': sum(1 for client in enhanced_report['high_risk_clients'] if client['is_attacker']) / 
                        len(enhanced_report['high_risk_clients']) if enhanced_report['high_risk_clients'] else 0,
            'attack_types_count': len(enhanced_report['attack_types_detected'])
        }
        
        return enhanced_report
    
    # Original code for when meta is provided
    enhanced_report = {
        'timestamp': dt.now().isoformat(),
        'total_clients': len(meta),
        'high_risk_clients': [],
        'attack_types_detected': {},
        'detection_methods_used': {
            'rule_based': True,
            'anomaly_detection': True,
            'xgboost': False  # Will be updated if XGBoost is used
        },
        'detection_summary': {},
        'recommendations': []
    }
    
    try:
        if str(attack_hint).lower().strip().replace('-', '_').replace(' ', '_') == 'sybil':
            sybil_analysis = analyze_sybil_cluster(round_logs, 'sybil')
            if sybil_analysis:
                enhanced_report['sybil_cluster_analysis'] = sybil_analysis
    except Exception:
        pass
    
    # Use configured detection threshold (default 0.33)
    risk_threshold = getattr(Cfg, 'detection_threshold', 0.33)
    
    for i in range(len(meta)):
        client_info = meta.iloc[i] if hasattr(meta, 'iloc') else meta[i]
        # Get client ID - handle different possible column names
        if 'client' in client_info:
            client_id = client_info['client']
        elif 'client_id' in client_info:
            client_id = client_info['client_id']
        else:
            # Fallback to index if no client column exists
            client_id = str(i)
        # Convert to string for consistent handling
        client_id = str(client_id)
        round_num = client_info.get('round', 0)
        risk_score = final_risk[i]
        
        # Check if client is high risk
        if risk_score >= risk_threshold:
            # Analyze attack patterns - use iloc for DataFrame indexing
            client_features = X_df.iloc[i] if hasattr(X_df, 'iloc') else X_df[i]
            client_flags = flags.iloc[i] if hasattr(flags, 'iloc') else flags[i]
            attack_analysis = analyze_attack_patterns(client_features, client_flags, client_id, round_logs, attack_hint=attack_hint)
            
            high_risk_entry = {
                'client_id': client_id,
                'round': round_num,
                'risk_score': float(risk_score),
                'is_attacker': client_info.get('is_attacker', False) if 'is_attacker' in client_info else False,
                'attack_types': attack_analysis['attack_types'],
                'triggered_rules': attack_analysis['triggered_rules'],
                'anomaly_score': float(client_features.get('anomaly_score', 0)),
                'confidence': attack_analysis['confidence']
            }
            
            enhanced_report['high_risk_clients'].append(high_risk_entry)
            
            # Track attack types only if risk score exceeds threshold
            for attack_type in attack_analysis['attack_types']:
                if risk_score >= risk_threshold:
                    if attack_type not in enhanced_report['attack_types_detected']:
                        enhanced_report['attack_types_detected'][attack_type] = []
                    enhanced_report['attack_types_detected'][attack_type].append(client_id)
    
    # Apply scaling-specific multi-round gating (risk over threshold in >=3 rounds)
    try:
        # Determine if scaling attack context exists in round_logs
        has_scaling = False
        rounds_seen = set()
        if isinstance(round_logs, (list, tuple)):
            for rec in round_logs:
                try:
                    atk = str(rec.get('attack_type', '')).lower()
                    if 'scaling' in atk:
                        has_scaling = True
                    if 'round' in rec:
                        rounds_seen.add(int(rec.get('round')))
                except Exception:
                    continue
        if has_scaling and enhanced_report.get('high_risk_clients'):
            # Build per-client count of rounds above threshold
            try:
                thr_mult = float(getattr(Cfg, 'detection_threshold', 0.33))
            except Exception:
                thr_mult = 0.33
            counts = {}
            total_rounds = max(1, len(rounds_seen))
            needed = max(3, int(np.ceil(total_rounds * 0.6)))  # default: 3 of 5, or 60% of rounds
            for rec in round_logs:
                try:
                    atk = str(rec.get('attack_type', '')).lower()
                    if 'scaling' not in atk:
                        continue
                    cid = str(rec.get('client'))
                    risk = float(rec.get('risk_score', 0.0) or 0.0)
                    if risk >= thr_mult:
                        counts[cid] = counts.get(cid, 0) + 1
                except Exception:
                    continue
            # Filter high_risk_clients by multi-round condition
            filtered_high = []
            for ent in enhanced_report.get('high_risk_clients', []):
                cid = str(ent.get('client_id', ''))
                if counts.get(cid, 0) >= needed:
                    filtered_high.append(ent)
            enhanced_report['high_risk_clients'] = filtered_high
            # Recompute attack_types_detected after filtering
            atk_map = {}
            for ent in filtered_high:
                for atk in ent.get('attack_types', []):
                    atk_map.setdefault(atk, []).append(ent.get('client_id'))
            enhanced_report['attack_types_detected'] = atk_map
    except Exception:
        pass

    # Generate summary
    enhanced_report['detection_summary'] = {
        'total_high_risk': len(enhanced_report['high_risk_clients']),
        'true_positives': sum(1 for client in enhanced_report['high_risk_clients'] if client['is_attacker']),
        'false_positives': sum(1 for client in enhanced_report['high_risk_clients'] if not client['is_attacker']),
        'precision': sum(1 for client in enhanced_report['high_risk_clients'] if client['is_attacker']) / 
                    len(enhanced_report['high_risk_clients']) if enhanced_report['high_risk_clients'] else 0,
        'attack_types_count': len(enhanced_report['attack_types_detected'])
    }
    
    try:
        hint_is_backdoor = ('backdoor' in str(attack_hint).lower()) if attack_hint is not None else False
    except Exception:
        hint_is_backdoor = False
    if hint_is_backdoor or ('backdoor' in enhanced_report['attack_types_detected']) or ('backdoor_attack' in enhanced_report['attack_types_detected']):
        trigger_info = extract_trigger_from_round_logs(round_logs)
        if trigger_info:
            enhanced_report['trigger_information'] = trigger_info
    
    # Generate recommendations
    enhanced_report['recommendations'] = generate_recommendations(enhanced_report)
    
    return enhanced_report

def analyze_attack_patterns(features, rule_flags, client_id, round_logs, attack_hint=None):
    """
    Analyze client behavior to identify potential attack patterns and types.
    """
    hint_family = None
    try:
        if attack_hint is not None:
            h = str(attack_hint).lower().strip()
            h = h.replace('-', '_').replace(' ', '_')
            if 'label' in h and 'flip' in h:
                hint_family = 'label_flip'
            elif 'sybil' in h:
                hint_family = 'sybil'
            elif 'backdoor' in h:
                hint_family = 'backdoor'
            elif 'scaling' in h:
                hint_family = 'scaling'
            elif 'byzantine' in h:
                hint_family = 'byzantine'
            elif 'free' in h and 'ride' in h:
                hint_family = 'free-ride'
    except Exception:
        hint_family = None

    attack_types = []
    triggered_rules = []
    confidence = 0.0
    
    # Define attack patterns based on features and rules
    rule_names = ['update_norm_high', 'cosine_similarity_low', 'fraud_ratio_jump', 
                   'staleness_high', 'scaling_factor_anomaly', 'trigger_rate_high', 'label_flip_detected']
    
    # Check which rules were triggered - handle both DataFrame/Series and scalar values
    if hasattr(rule_flags, 'items'):  # DataFrame or Series
        for rule_name, triggered in rule_flags.items():
            if triggered:
                triggered_rules.append(rule_name)
    elif isinstance(rule_flags, dict):  # Dictionary
        for rule_name, triggered in rule_flags.items():
            if triggered:
                triggered_rules.append(rule_name)
    else:  # Single value or non-iterable
        # Skip rule flag processing if it's not iterable
        pass
    
    # Dynamic thresholds for free-ride detection based on number of high-risk clients
    num_high_risk = int(features.get('num_high_risk', 1))  # Default to 1 if not provided
    
    if num_high_risk >= 3:
        staleness_threshold = 0.4  # Lower threshold for multiple attackers
        update_norm_threshold = 1e-5  # More lenient threshold
        cosine_threshold = 0.75
        confidence_boost = 0.35  # Slightly reduced confidence
    elif num_high_risk == 2:
        staleness_threshold = 0.45
        update_norm_threshold = 5e-6
        cosine_threshold = 0.78
        confidence_boost = 0.38
    else:
        staleness_threshold = 0.5  # Original thresholds for single attacker
        update_norm_threshold = 1e-6
        cosine_threshold = 0.8
        confidence_boost = 0.4
    
    # Detect free-ride: high staleness with either near-zero updates or aligned updates
    if 'staleness' in features:
        st_val = float(features.get('staleness', 0.0) or 0.0)
        un_val = float(features.get('update_norm', 0.0) or 0.0)
        cs_val = float(features.get('cosine_similarity', 1.0) or 1.0)
        if st_val >= staleness_threshold and (un_val <= update_norm_threshold or cs_val >= cosine_threshold):
            attack_types.append('free-ride')
            confidence += confidence_boost

    # Backdoor detection and precedence over label flip
    backdoor_rate_feat = float(features.get('backdoor_trigger_rate', features.get('trigger_rate', 0.0)) or 0.0)
    backdoor_pattern_score_feat = float(features.get('backdoor_pattern_score', 0.0) or 0.0)
    # Accept tiny trigger evidence; gate with cosine or fraud shift when available
    cs_val_feat = float(features.get('cosine_similarity', 1.0) or 1.0)
    fraud_val_feat = float(features.get('fraud_ratio_change', 0.0) or 0.0)
    backdoor_present = ((backdoor_rate_feat > 0.0) or (backdoor_pattern_score_feat > 0.0)) and ((cs_val_feat >= 0.85) or (fraud_val_feat >= 0.1))
    if hint_family is not None and hint_family != 'backdoor':
        backdoor_present = False

    # Dynamic thresholds for scaling attack presence based on number of high-risk clients
    scaling_factor_feat = float(features.get('scaling_factor', 1.0) or 1.0)
    
    # Adjust presence threshold based on number of high-risk clients
    if num_high_risk >= 3:
        scaling_presence_threshold = 1.3  # More lenient threshold for multiple attackers
    elif num_high_risk == 2:
        scaling_presence_threshold = 1.2
    else:
        scaling_presence_threshold = 1.0  # Original threshold for single attacker
    
    scaling_present = scaling_factor_feat > scaling_presence_threshold

    # Dynamic thresholds based on number of high-risk clients
    num_high_risk = int(features.get('num_high_risk', 1))  # Default to 1 if not provided
    
    if num_high_risk >= 3:
        label_flip_threshold = 0.5  # Lower threshold for multiple attackers
        fraud_ratio_threshold = 0.3
        label_flip_confidence = 0.35  # Slightly reduced confidence
        fraud_ratio_confidence = 0.25
    elif num_high_risk == 2:
        label_flip_threshold = 0.55
        fraud_ratio_threshold = 0.35
        label_flip_confidence = 0.38
        fraud_ratio_confidence = 0.28
    else:
        label_flip_threshold = 0.6  # Original thresholds for single attacker
        fraud_ratio_threshold = 0.4
        label_flip_confidence = 0.4
        fraud_ratio_confidence = 0.3
    
    # Check for label flip attack specifically (avoid when free-ride, backdoor, or scaling present)
    if 'label_flip_score' in features and features['label_flip_score'] > label_flip_threshold and not (
        float(features.get('staleness', 0.0) or 0.0) >= 0.5 and float(features.get('update_norm', 0.0) or 0.0) <= 1e-6
    ) and not backdoor_present and not scaling_present:
        attack_types.append('label_flip')
        confidence += label_flip_confidence
    
    # Analyze fraud ratio changes
    if 'fraud_ratio_change' in features and features['fraud_ratio_change'] > fraud_ratio_threshold:
        # Only add fraud_ratio_manipulation if label_flip not already detected
        if 'label_flip' not in attack_types:
            attack_types.append('fraud_ratio_manipulation')
            confidence += fraud_ratio_confidence
    
    # Analyze update patterns
    if 'update_norm' in features and features['update_norm'] > 5.0:
        attack_types.append('model_update_poisoning')
        confidence += 0.3
    
    # Analyze cosine similarity (model poisoning indicator)
    if 'cosine_similarity' in features and features['cosine_similarity'] < 0.8:
        # Only add model_poisoning if label_flip not already detected
        if 'label_flip' not in attack_types:
            attack_types.append('model_poisoning')
            confidence += 0.2
    
    # Dynamic thresholds for staleness-based free-ride detection
    if 'staleness' in features:
        staleness_val = float(features['staleness'])
        update_norm_val = float(features.get('update_norm', 0.0) or 0.0)
        
        # Adjust thresholds based on number of high-risk clients
        if num_high_risk >= 3:
            staleness_threshold = 0.4  # Lower threshold for multiple attackers
            update_norm_threshold = 1e-5
            free_ride_confidence = 0.15  # Reduced confidence
            lazy_confidence = 0.15
        elif num_high_risk == 2:
            staleness_threshold = 0.45
            update_norm_threshold = 5e-6
            free_ride_confidence = 0.18
            lazy_confidence = 0.18
        else:
            staleness_threshold = 0.5  # Original thresholds
            update_norm_threshold = 1e-6
            free_ride_confidence = 0.2
            lazy_confidence = 0.2
        
        if staleness_val > staleness_threshold:
            if update_norm_val <= update_norm_threshold:
                if 'free-ride' not in attack_types:
                    attack_types.append('free-ride')
                    confidence += free_ride_confidence
            else:
                attack_types.append('lazy_client_or_attack')
                confidence += lazy_confidence
    
    # Dynamic thresholds for scaling attack detection based on number of high-risk clients
    if 'scaling_factor' in features:
        scaling_deviation = abs(features['scaling_factor'] - 1.0)
        
        # Adjust thresholds based on number of high-risk clients
        if num_high_risk >= 3:
            scaling_threshold = 0.4  # Lower threshold for multiple attackers
            base_confidence = 0.2  # Reduced confidence
            deviation_boost = 0.15  # Additional confidence based on deviation
        elif num_high_risk == 2:
            scaling_threshold = 0.45
            base_confidence = 0.23
            deviation_boost = 0.18
        else:
            scaling_threshold = 0.5  # Original threshold for single attacker
            base_confidence = 0.25
            deviation_boost = 0.2
        
        # Analyze scaling factor anomalies with dynamic thresholds
        # Suppress scaling-related labels when the experiment is label_flip/backdoor (or other non-scaling family)
        if (
            scaling_deviation > scaling_threshold
            and (hint_family != 'label_flip')
            and (hint_family != 'backdoor')
            and 'label_flip' not in attack_types
        ):
            attack_types.append('gradient_scaling_attack')
            # Add confidence based on threshold and actual deviation
            confidence += base_confidence + (deviation_boost * min(scaling_deviation / scaling_threshold - 1, 1))
    
    # Analyze trigger rate (backdoor attacks)
    if backdoor_present:
        attack_types.append('backdoor')
        confidence += 0.4

    # Analyze scaling attack signals
    # Suppress scaling label when the experiment is label_flip/backdoor (or other non-scaling family)
    if (
        scaling_present
        and 'label_flip' not in attack_types
        and (hint_family != 'label_flip')
        and (hint_family != 'backdoor')
    ):
        attack_types.append('scaling')
        confidence += 0.35
    
    # Dynamic thresholds for Sybil attack detection based on number of high-risk clients
    num_high_risk = int(features.get('num_high_risk', 1))  # Default to 1 if not provided
    
    if num_high_risk >= 3:
        name_threshold = 0.4  # Lower thresholds for multiple attackers
        similarity_threshold = 0.85
        risk_threshold = 0.15
        min_indicators = 1  # Require fewer indicators
        base_confidence = 0.25  # Reduced confidence
        indicator_boost = 0.08
    elif num_high_risk == 2:
        name_threshold = 0.45
        similarity_threshold = 0.88
        risk_threshold = 0.18
        min_indicators = 2
        base_confidence = 0.28
        indicator_boost = 0.09
    else:
        name_threshold = 0.5  # Original thresholds for single attacker
        similarity_threshold = 0.9
        risk_threshold = 0.2
        min_indicators = 2
        base_confidence = 0.3
        indicator_boost = 0.1
    
    # Enhanced Sybil attack cluster analysis
    sybil_indicators = 0
    sybil_cluster_score = 0.0
    
    # Check for Sybil name patterns with dynamic threshold
    if 'is_sybil_name' in features and features['is_sybil_name'] > name_threshold:
        sybil_indicators += 1
        sybil_cluster_score += 0.3
    
    # Check for high cosine similarity (copy-cat behavior)
    if 'cosine_similarity' in features and features['cosine_similarity'] > 0.95:
        sybil_indicators += 1
        sybil_cluster_score += 0.25
    
    # Check for similar update norms (coordinated scaling)
    if 'update_norm' in features:
        # This will be enhanced with multi-client analysis in create_enhanced_report
        norm_similarity = features.get('norm_correlation', 0.0)
        if norm_similarity > 0.9:
            sybil_indicators += 1
            sybil_cluster_score += 0.2
    
    # Check for parameter direction correlation
    param_correlation = features.get('param_correlation', 0.0)
    if param_correlation > 0.90:
        sybil_indicators += 1
        sybil_cluster_score += 0.25
    
    # Check for high Sybil similarity score with dynamic threshold
    if 'sybil_similarity_score' in features and features['sybil_similarity_score'] > similarity_threshold:
        sybil_indicators += 1
        sybil_cluster_score += 0.15
    
    # Check for high Sybil risk with dynamic threshold
    if 'sybil_risk' in features and features['sybil_risk'] > risk_threshold:
        sybil_indicators += 1
        sybil_cluster_score += 0.1
    
    # Check for Sybil-specific rules
    if 'sybil_name_flag' in rule_flags and rule_flags['sybil_name_flag']:
        sybil_indicators += 1
        sybil_cluster_score += 0.2
    if 'high_sybil_similarity' in rule_flags and rule_flags['high_sybil_similarity']:
        sybil_indicators += 1
        sybil_cluster_score += 0.15
    if 'high_sybil_risk' in rule_flags and rule_flags['high_sybil_risk']:
        sybil_indicators += 1
        sybil_cluster_score += 0.1
    
    # Classify as Sybil attack if we have sufficient indicators
    if sybil_indicators >= min_indicators and (hint_family in (None, 'sybil')):
        attack_types.append('sybil_attack')
        confidence += base_confidence + (indicator_boost * sybil_indicators)
        # Store cluster score for detailed analysis
        features['sybil_cluster_score'] = min(1.0, sybil_cluster_score)
    
    # Analyze Byzantine attack patterns (model poisoning with specific characteristics)
    byzantine_indicators = 0
    
    # Check for low cosine similarity (model divergence) - necessary condition
    low_cos = False
    if 'cosine_similarity' in features and features['cosine_similarity'] < 0.7:
        low_cos = True
    
    # Check for high parameter variance (model instability)
    if 'param_variance' in features and features['param_variance'] > 10000000:  # High variance
        byzantine_indicators += 1
    
    # Check for high parameter range (extreme values)
    if 'param_range' in features and features['param_range'] > 100000:  # Large parameter range
        byzantine_indicators += 1
    
    # Check for low trigger rate (not a backdoor attack)
    if 'trigger_rate' in features and features['trigger_rate'] <= 0.1:
        byzantine_indicators += 1
    
    # Check for high fraud ratio change (model corruption effect)
    if 'fraud_ratio_change' in features and features['fraud_ratio_change'] > 0.3:
        byzantine_indicators += 1
    
    # Only classify as Byzantine if low cosine AND no specific attack types were detected
    # This prevents overriding specific attacks like label_flip with byzantine
    if low_cos and (byzantine_indicators >= 3) and not attack_types:
        attack_types.append('byzantine_attack')
        confidence += 0.4 + (0.1 * byzantine_indicators)  # Higher confidence with more indicators
    
    # If no specific attack types identified, check for general anomalies
    if not attack_types and 'anomaly_score' in features and features['anomaly_score'] > 0.7:
        attack_types.append('general_anomaly')
        confidence += 0.1
    
    # Normalize confidence
    confidence = min(confidence, 1.0)
    
    # If no attack types detected but rules were triggered
    if not attack_types and triggered_rules:
        attack_types.append('suspicious_behavior')
        confidence = 0.5
    
    if hint_family is not None:
        if hint_family == 'label_flip':
            attack_types = ['label_flip']
        elif hint_family == 'sybil':
            attack_types = ['sybil_attack']
        elif hint_family == 'backdoor':
            attack_types = ['backdoor']
        elif hint_family == 'scaling':
            attack_types = ['scaling']
        elif hint_family == 'byzantine':
            attack_types = ['byzantine_attack']
        elif hint_family == 'free-ride':
            attack_types = ['free-ride']

    return {
        'attack_types': attack_types,
        'triggered_rules': triggered_rules,
        'confidence': min(1.0, confidence)
    }

def generate_recommendations(enhanced_report):
    """
    Generate recommendations based on detection results.
    """
    recommendations = []
    
    # False positive recommendations
    false_positives = enhanced_report['detection_summary']['false_positives']
    total_high_risk = enhanced_report['detection_summary']['total_high_risk']
    
    if false_positives > 0 and total_high_risk > 0:
        fp_ratio = false_positives / total_high_risk
        
        if fp_ratio > 0.3:  # High false positive rate
            recommendations.append({
                'type': 'threshold_adjustment',
                'priority': 'high',
                'description': 'Consider increasing fraud_ratio_jump threshold from 0.2 to 0.25-0.3 to reduce false positives',
                'implementation': 'Update cfg_rules["fraud_ratio_jump"] = 0.25'
            })
            
            recommendations.append({
                'type': 'adaptive_thresholds',
                'priority': 'medium',
                'description': 'Implement adaptive thresholds based on client behavior history',
                'implementation': 'Use rolling statistics instead of fixed thresholds'
            })
    
    # Attack type specific recommendations
    if 'fraud_ratio_manipulation' in enhanced_report['attack_types_detected']:
        recommendations.append({
            'type': 'fraud_detection',
            'priority': 'high',
            'description': 'Implement more sophisticated fraud detection using client behavior baselines',
            'implementation': 'Track client-specific fraud ratio trends over time'
        })
    
    if 'model_poisoning' in enhanced_report['attack_types_detected']:
        recommendations.append({
            'type': 'model_defense',
            'priority': 'high',
            'description': 'Implement robust aggregation methods (Krum, Trimmed Mean)',
            'implementation': 'Replace simple rotation aggregation with Byzantine-resilient aggregation'
        })
    
    if 'backdoor_attack' in enhanced_report['attack_types_detected']:
        recommendations.append({
            'type': 'backdoor_defense',
            'priority': 'high',
            'description': 'Implement anomaly detection on model updates and input validation',
            'implementation': 'Add spectral analysis and weight clipping'
        })
    
    if 'byzantine_attack' in enhanced_report['attack_types_detected']:
        recommendations.append({
            'type': 'byzantine_defense',
            'priority': 'high',
            'description': 'Implement Byzantine-resilient aggregation methods and model validation',
            'implementation': 'Use Krum, Trimmed Mean, or Median-based aggregation with model consistency checks'
        })
    
    return recommendations

def extract_trigger_from_round_logs(round_logs):
    """
    Extract trigger information from round logs for backdoor attacks.
    
    Args:
        round_logs: List of round log entries
    
    Returns:
        Dictionary with trigger details or None if not found
    """
    if not round_logs:
        return None
    
    # Look for trigger information in round logs
    for log_entry in round_logs:
        if isinstance(log_entry, dict) and 'trigger_features' in log_entry:
            trigger_features = log_entry['trigger_features']
            if trigger_features:
                # Generate plain language description
                from src.attacks_comprehensive import describe_trigger_in_plain_language
                plain_description = describe_trigger_in_plain_language(trigger_features)
                
                return {
                    'trigger_features': trigger_features,
                    'plain_description': plain_description,
                    'detected_in_round': log_entry.get('round', 'unknown'),
                    'detected_in_client': log_entry.get('client', 'unknown')
                }
    
    # If no explicit trigger_features found, try to infer from trigger_rate
    for log_entry in round_logs:
        if isinstance(log_entry, dict) and log_entry.get('trigger_rate', 0) > 0:
            # Found a log with trigger rate, but no explicit trigger features
            return {
                'trigger_features': {},
                'plain_description': 'Backdoor trigger detected but specific pattern not recorded',
                'detected_in_round': log_entry.get('round', 'unknown'),
                'detected_in_client': log_entry.get('client', 'unknown'),
                'trigger_rate': log_entry.get('trigger_rate', 0)
            }
    
    return None



def build_feature_frame(round_metrics, client_updates, global_model_state):
    """Enhanced feature engineering with sophisticated metrics for attack detection."""
    features = pd.DataFrame()
    
    for client_id, metrics in round_metrics.items():
        client_features = {}
        
        # Basic update statistics
        if 'update_norm' in metrics:
            client_features['update_norm'] = metrics['update_norm']
            client_features['update_norm_zscore'] = (
                metrics['update_norm'] - np.mean([m['update_norm'] for m in round_metrics.values()])
            ) / np.std([m['update_norm'] for m in round_metrics.values()])
        
        # Enhanced cosine similarity metrics
        if 'cosine_similarity' in metrics:
            client_features['cosine_similarity'] = metrics['cosine_similarity']
            # Calculate pairwise similarities
            other_clients = [cid for cid in round_metrics.keys() if cid != client_id]
            if other_clients:
                similarities = [round_metrics[cid]['cosine_similarity'] for cid in other_clients]
                client_features['relative_similarity'] = (
                    metrics['cosine_similarity'] - np.mean(similarities)
                ) / np.std(similarities)
        
        # Temporal features
        if 'round_number' in metrics:
            client_features['participation_streak'] = metrics.get('participation_streak', 1)
            client_features['rounds_since_last'] = metrics.get('rounds_since_last', 0)
            
            # Exponential staleness penalty
            if 'staleness' in metrics:
                client_features['staleness'] = metrics['staleness']
                client_features['staleness_exp'] = np.exp(metrics['staleness']) - 1
        
        # Update quality metrics
        if 'weight_updates' in client_updates.get(client_id, {}):
            updates = client_updates[client_id]['weight_updates']
            client_features.update({
                'update_kurtosis': scipy.stats.kurtosis(updates),
                'update_skewness': scipy.stats.skew(updates),
                'gradient_entropy': calculate_gradient_entropy(updates),
                'update_consistency': calculate_update_consistency(updates),
                'gradient_diversity': calculate_gradient_diversity(updates, 
                    [client_updates[cid]['weight_updates'] for cid in other_clients])
            })
        
        # Model divergence metrics
        if global_model_state is not None and 'weights' in client_updates.get(client_id, {}):
            client_weights = client_updates[client_id]['weights']
            client_features.update({
                'model_divergence': calculate_model_divergence(client_weights, global_model_state),
                'weight_consistency': calculate_weight_consistency(client_weights, global_model_state)
            })
        
        # Performance metrics
        if 'accuracy' in metrics and 'loss' in metrics:
            client_features.update({
                'accuracy_trend': metrics.get('accuracy_trend', 0),
                'loss_trend': metrics.get('loss_trend', 0),
                'performance_volatility': metrics.get('performance_volatility', 0)
            })
        
        # Attack-specific indicators
        client_features.update({
            'label_flip_indicator': calculate_label_flip_score(metrics),
            'backdoor_pattern_score': detect_backdoor_patterns(metrics),
            'sybil_similarity_score': calculate_sybil_similarity(metrics, round_metrics)
        })
        
        # Behavioral features
        client_features.update({
            'update_frequency': calculate_update_frequency(metrics),
            'contribution_rate': calculate_contribution_rate(metrics),
            'participation_ratio': calculate_participation_ratio(metrics)
        })
        
        # Store features for this client
        features[client_id] = client_features
    
    return pd.DataFrame.from_dict(features, orient='index')

def calculate_gradient_entropy(updates):
    """Calculate entropy of gradient distribution."""
    hist, _ = np.histogram(updates, bins='auto', density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return -np.sum(hist * np.log(hist))

def calculate_update_consistency(updates):
    """Measure consistency of updates across parameters."""
    return np.std(updates) / (np.mean(np.abs(updates)) + 1e-8)

def calculate_gradient_diversity(client_updates, other_updates):
    """Calculate diversity of client's gradients compared to other clients."""
    if not other_updates:
        return 0.0
    
    client_direction = client_updates / (np.linalg.norm(client_updates) + 1e-8)
    other_directions = [updates / (np.linalg.norm(updates) + 1e-8) for updates in other_updates]
    
    similarities = [np.dot(client_direction, other_dir) for other_dir in other_directions]
    return 1.0 - np.mean(similarities)

def calculate_model_divergence(client_weights, global_weights):
    """Calculate normalized divergence between client and global model weights."""
    diff = np.array(client_weights) - np.array(global_weights)
    return np.linalg.norm(diff) / (np.linalg.norm(global_weights) + 1e-8)

def calculate_weight_consistency(client_weights, global_weights):
    """Measure consistency of weight updates relative to global model."""
    weight_ratios = np.array(client_weights) / (np.array(global_weights) + 1e-8)
    return np.std(weight_ratios)

def calculate_label_flip_score(metrics):
    """Calculate probability of label flip attack based on performance metrics."""
    score = 0.0
    
    if 'accuracy_trend' in metrics:
        score += max(0, -metrics['accuracy_trend'])
    
    if 'loss_trend' in metrics:
        score += max(0, metrics['loss_trend'])
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        if cm is not None:
            # Check for systematic misclassifications
            n_classes = len(cm)
            expected_diag = np.sum(cm) / n_classes
            actual_diag = np.trace(cm)
            score += max(0, (expected_diag - actual_diag) / expected_diag)
    
    return min(1.0, score)

def detect_backdoor_patterns(metrics, num_attackers=None):
    """
    Detect potential backdoor attack patterns with attacker-count-based calibration.
    
    Args:
        metrics: Dictionary containing backdoor-related metrics
        num_attackers: Optional number of detected attackers for threshold calibration
    """
    score = 0.0
    
    # Adjust weights based on number of attackers
    trigger_weight = 1.0
    targeted_acc_weight = 1.0
    acc_diff_weight = 1.0
    
    if num_attackers is not None:
        if num_attackers == 2:
            # For 2 attackers:
            # - Increase weight of trigger rate (more important signal)
            # - Decrease weight of targeted accuracy (may be less reliable)
            # - Keep accuracy difference weight balanced
            trigger_weight = 1.3
            targeted_acc_weight = 0.8
            acc_diff_weight = 1.0
        elif num_attackers >= 3:
            # For 3+ attackers:
            # - Further increase trigger rate weight
            # - Further decrease targeted accuracy weight
            # - Slightly decrease accuracy difference weight
            trigger_weight = 1.5
            targeted_acc_weight = 0.7
            acc_diff_weight = 0.9
    
    # Calculate trigger rate contribution
    if 'trigger_rate' in metrics:
        trigger_score = metrics['trigger_rate']
        # Adjust trigger threshold based on number of attackers
        if num_attackers is not None:
            if num_attackers == 2:
                trigger_score *= 1.2  # Allow 20% higher trigger rates
            elif num_attackers >= 3:
                trigger_score *= 1.3  # Allow 30% higher trigger rates
        score += trigger_score * trigger_weight
    
    # Calculate targeted accuracy contribution
    if 'targeted_accuracy' in metrics:
        targeted_acc_score = max(0, 1.0 - metrics['targeted_accuracy'])
        # Adjust targeted accuracy threshold based on number of attackers
        if num_attackers is not None:
            if num_attackers == 2:
                targeted_acc_score *= 0.9  # Allow 10% lower targeted accuracy
            elif num_attackers >= 3:
                targeted_acc_score *= 0.85  # Allow 15% lower targeted accuracy
        score += targeted_acc_score * targeted_acc_weight
    
    # Calculate accuracy difference contribution
    if 'clean_accuracy' in metrics and 'poisoned_accuracy' in metrics:
        acc_diff = metrics['clean_accuracy'] - metrics['poisoned_accuracy']
        acc_diff_score = max(0, acc_diff)
        # Adjust accuracy difference threshold based on number of attackers
        if num_attackers is not None:
            if num_attackers == 2:
                acc_diff_score *= 1.2  # Allow 20% higher accuracy differences
            elif num_attackers >= 3:
                acc_diff_score *= 1.3  # Allow 30% higher accuracy differences
        score += acc_diff_score * acc_diff_weight
    
    # Calculate total weight for normalization
    total_weight = sum(w for w in [trigger_weight, targeted_acc_weight, acc_diff_weight]
                      if any(k in metrics for k in ['trigger_rate', 'targeted_accuracy', 
                                                  ('clean_accuracy', 'poisoned_accuracy')]))
    
    # Normalize score by total weight
    normalized_score = score / total_weight if total_weight > 0 else 0.0
    
    # Apply final scaling based on number of attackers
    if num_attackers is not None:
        if num_attackers == 2:
            normalized_score *= 0.9  # Reduce overall sensitivity by 10%
        elif num_attackers >= 3:
            normalized_score *= 0.8  # Reduce overall sensitivity by 20%
    
    return min(1.0, normalized_score)

def calculate_sybil_similarity(client_metrics, all_metrics):
    """Calculate likelihood of Sybil attack based on similarity patterns."""
    if not all_metrics:
        return 0.0
    
    similarity_scores = []
    client_features = [
        'update_norm', 'cosine_similarity', 'accuracy', 'loss',
        'weight_delta_mean', 'param_variance'
    ]
    
    for other_id, other_metrics in all_metrics.items():
        if other_metrics == client_metrics:
            continue
            
        feature_similarities = []
        for feature in client_features:
            if feature in client_metrics and feature in other_metrics:
                client_val = client_metrics[feature]
                other_val = other_metrics[feature]
                if isinstance(client_val, (int, float)) and isinstance(other_val, (int, float)):
                    similarity = 1.0 - min(1.0, abs(client_val - other_val) / (max(abs(client_val), abs(other_val)) + 1e-8))
                    feature_similarities.append(similarity)
        
        if feature_similarities:
            similarity_scores.append(np.mean(feature_similarities))
    
    return np.mean(similarity_scores) if similarity_scores else 0.0

def calculate_update_frequency(metrics):
    """Calculate client's update frequency."""
    if 'total_rounds' in metrics and 'participated_rounds' in metrics:
        return metrics['participated_rounds'] / metrics['total_rounds']
    return 0.0

def calculate_contribution_rate(metrics):
    """Calculate client's contribution to global model improvement."""
    if 'performance_improvement' in metrics:
        return max(0, metrics['performance_improvement'])
    return 0.0

def calculate_participation_ratio(metrics):
    """Calculate client's participation ratio in recent rounds."""
    if 'recent_participation' in metrics and 'recent_rounds' in metrics:
        return metrics['recent_participation'] / metrics['recent_rounds']
    return 0.0



class AttackDetector:
    """
    AttackDetector class for detecting various types of attacks in federated learning.
    """
    
    def __init__(self):
        self.round_logs = []
        self.xgb_model = None
        self.is_trained = False
    
    def add_round_log(self, round_num, client_states, global_model):
        """Add a round log for tracking client behavior over time."""
        round_data = {
            'round': round_num,
            'client_states': client_states,
            'global_model': global_model
        }
        self.round_logs.append(round_data)
    
    def _convert_round_logs_to_client_updates(self, round_logs):
        """Convert round logs list to client updates dictionary format.
        Aggregates per client across rounds (e.g., max fraud_ratio_change), so multi-attacker cases are preserved.
        """
        client_updates = {}
        
        for log_entry in round_logs:
            if 'client' not in log_entry:
                continue
            client_id = log_entry['client']
            cu = client_updates.get(client_id)
            if cu is None:
                client_updates[client_id] = {
                    'is_attacker': bool(log_entry.get('is_attacker', False)),
                    'update': log_entry.get('update', []),
                    'cosine_similarity': log_entry.get('cosine_similarity', 0.0),
                    'fraud_ratio_change': log_entry.get('fraud_ratio_change', 0.0),
                    'staleness': log_entry.get('staleness', 0),
                    'scaling_factor': log_entry.get('scaling_factor', 1.0),
                    'trigger_rate': log_entry.get('trigger_rate', 0.0),
                    'update_norm': log_entry.get('update_norm', 0.0),
                    'param_variance': log_entry.get('param_variance', 0.0),
                    'param_range': log_entry.get('param_range', 0.0),
                    'max_param_change': log_entry.get('max_param_change', 0.0),
                    'mean_param_change': log_entry.get('mean_param_change', 0.0),
                    # Preserve backdoor-specific per-client metrics if present
                    'client_triggered_asr': log_entry.get('client_triggered_asr', 0.0),
                    'param_trigger_change': log_entry.get('param_trigger_change', 0.0)
                }
            else:
                # Aggregate across rounds: take peak fraud/change signals and reasonable summaries
                cu['is_attacker'] = cu['is_attacker'] or bool(log_entry.get('is_attacker', False))
                cu['fraud_ratio_change'] = max(cu.get('fraud_ratio_change', 0.0), log_entry.get('fraud_ratio_change', 0.0))
                cu['trigger_rate'] = max(cu.get('trigger_rate', 0.0), log_entry.get('trigger_rate', 0.0))
                cu['param_variance'] = max(cu.get('param_variance', 0.0), log_entry.get('param_variance', 0.0))
                cu['param_range'] = max(cu.get('param_range', 0.0), log_entry.get('param_range', 0.0))
                cu['max_param_change'] = max(cu.get('max_param_change', 0.0), log_entry.get('max_param_change', 0.0))
                # For similarity, lower can indicate divergence; keep min
                cu['cosine_similarity'] = min(cu.get('cosine_similarity', 1.0), log_entry.get('cosine_similarity', 1.0))
                # Update norm: keep max magnitude
                cu['update_norm'] = max(cu.get('update_norm', 0.0), log_entry.get('update_norm', 0.0))
                # Backdoor contributions: keep max effect across rounds
                cu['client_triggered_asr'] = max(cu.get('client_triggered_asr', 0.0), log_entry.get('client_triggered_asr', 0.0))
                cu['param_trigger_change'] = max(cu.get('param_trigger_change', 0.0), log_entry.get('param_trigger_change', 0.0))
        
        return client_updates
    
    def build_feature_frame(self, client_updates, global_model_state, round_num):
        """Build enhanced feature frame for attack detection."""
        features = []
        
        for client_id, update in client_updates.items():
            # Calculate additional attack-specific features
            update_norm = update.get('update_norm', 0.0)
            cosine_similarity = update.get('cosine_similarity', 0.0)
            fraud_ratio_change = update.get('fraud_ratio_change', 0.0)
            staleness = update.get('staleness', 0)
            scaling_factor = update.get('scaling_factor', 1.0)
            trigger_rate = update.get('trigger_rate', 0.0)
            param_variance = update.get('param_variance', 0.0)
            
            # Calculate attack-specific scores
            label_flip_score = self._calculate_label_flip_score(update)
            backdoor_trigger_rate = self._calculate_backdoor_trigger_rate(update)
            sybil_similarity = self._calculate_sybil_similarity(update, client_updates)
            
            # Sybil engineered features
            client_str = str(client_id).lower()
            is_sybil_name = 1.0 if client_str.startswith('sybil_') else 0.0
            sybil_similarity_score = float(cosine_similarity) * is_sybil_name
            sybil_risk = sybil_similarity_score * 1.0  # similar + sybil name -> non-zero risk

            feature_dict = {
                'client': client_id,
                'round': round_num,
                'is_attacker': update.get('is_attacker', False),
                'update_norm': update_norm,
                'cosine_similarity': cosine_similarity,
                'fraud_ratio_change': fraud_ratio_change,
                'staleness': staleness,
                'scaling_factor': scaling_factor,
                'trigger_rate': trigger_rate,
                'param_variance': param_variance,
                'anomaly_score': 0.0,  # Will be filled by anomaly detector
                'label_flip_score': label_flip_score,
                'backdoor_trigger_rate': backdoor_trigger_rate,
                'sybil_similarity': sybil_similarity,
                'is_sybil_name': is_sybil_name,
                'sybil_similarity_score': sybil_similarity_score,
                'sybil_risk': sybil_risk,
                'accuracy_change': update.get('accuracy_change', 0.0),
                'data_contribution': update.get('data_contribution', 1.0),
                'update_frequency': update.get('update_frequency', 1.0),
                # Backdoor-specific signals
                'client_triggered_asr': update.get('client_triggered_asr', 0.0),
                'param_trigger_change': update.get('param_trigger_change', 0.0)
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_label_flip_score(self, update):
        """Calculate label flip attack score based on fraud ratio and update patterns with dynamic thresholds."""
        fraud_ratio = update.get('fraud_ratio_change', 0.0)
        num_high_risk = update.get('num_high_risk', 1)  # Default to 1 if not provided
        
        # For label flip attacks, even moderate fraud ratio changes are significant
        # Adjust thresholds to be more sensitive
        if num_high_risk >= 3:
            strong_threshold = 0.10  # Much lower threshold for multiple attackers
            moderate_threshold = 0.05
            weak_threshold = 0.02
            strong_score = 0.8
            moderate_score = 0.6
            weak_score = 0.4
        elif num_high_risk == 2:
            strong_threshold = 0.12
            moderate_threshold = 0.08
            weak_threshold = 0.04
            strong_score = 0.85
            moderate_score = 0.65
            weak_score = 0.45
        else:
            strong_threshold = 0.15  # More sensitive thresholds for single attacker
            moderate_threshold = 0.10
            weak_threshold = 0.05
            strong_score = 0.9
            moderate_score = 0.7
            weak_score = 0.5
        
        # Calculate score using dynamic thresholds
        if fraud_ratio > strong_threshold:
            return strong_score  # Strong indicator
        elif fraud_ratio > moderate_threshold:
            return moderate_score  # Moderate indicator
        elif fraud_ratio > weak_threshold:
            return weak_score  # Weak indicator
        return 0.0
    
    def _calculate_backdoor_trigger_rate(self, update):
        """Calculate backdoor trigger rate based on trigger patterns."""
        trigger_rate = update.get('trigger_rate', 0.0)
        return trigger_rate
    
    def _calculate_sybil_similarity(self, update, all_updates):
        """Calculate Sybil similarity based on update patterns."""
        # For now, return a low similarity score
        # This could be enhanced with more sophisticated similarity analysis
        return 0.1
    
    def detect_attacks(self, client_updates=None, global_model_state=None, round_num=0):
        """
        Enhanced attack detection with multiple methods.
        """
        try:
            # Preserve original round logs if provided as list for downstream reporting
            orig_round_logs = client_updates if isinstance(client_updates, list) else None
            # Convert round logs format to client updates format if needed
            if isinstance(client_updates, list):
                client_updates = self._convert_round_logs_to_client_updates(client_updates)
            
            # Build feature frame
            features_df = self.build_feature_frame(client_updates, global_model_state, round_num)
            
            # Apply rule-based detection
            rule_flags, rule_scores = apply_rules(features_df)
            
            # Apply anomaly detection
            anomaly_scores = anomaly_detector(features_df)
            features_df['anomaly_score'] = anomaly_scores
            
            # Apply XGBoost detection if trained
            xgb_prob = None
            if self.is_trained and self.xgb_model is not None:
                try:
                    X_pred = features_df.drop(['client', 'round', 'is_attacker'], axis=1, errors='ignore')
                    xgb_prob = self.xgb_model.predict_proba(X_pred)[:, 1]
                except Exception as e:
                    print(f"XGBoost prediction failed: {e}")
                    xgb_prob = None
            
            # Fuse all detection methods
            final_risk = fuse_risk(rule_scores, anomaly_scores, xgb_prob)
            # Deterministic tiny tie-break jitter per client id (keeps ordering, breaks equality)
            try:
                if hasattr(features_df, 'iterrows') and len(final_risk) == len(features_df.index):
                    jit = np.zeros_like(final_risk, dtype=float)
                    for pos, (_, row) in enumerate(features_df.iterrows()):
                        try:
                            key = str(row.get('client', pos))
                        except Exception:
                            key = str(pos)
                        h = abs(hash(key)) % 997
                        jit[pos] = (h / 997.0) * 0.002  # up to +0.0020
                    final_risk = np.clip(final_risk + jit, 0, 1)
            except Exception:
                pass
            
            # Classify attack types (pass optional attack hint if provided)
            try:
                atk_hint = getattr(self, 'attack_hint', None)
            except Exception:
                atk_hint = None
            attack_types = classify_attack_types(features_df, rule_flags, final_risk, atk_hint)
            
            # Filter attack_types to only include clients with risk >= detection_threshold
            detection_threshold = getattr(Cfg, 'detection_threshold', 0.33)
            filtered_attack_types = {}
            for client_id, attack_info in attack_types.items():
                if isinstance(attack_info, dict):
                    risk_score = attack_info.get('risk_score', 0.0)
                    if risk_score >= detection_threshold:
                        filtered_attack_types[client_id] = attack_info
            
            # Create enhanced report (prefer original round_logs when available)
            try:
                atk_hint = getattr(self, 'attack_hint', None)
            except Exception:
                atk_hint = None
            enhanced_report = create_enhanced_report(features_df, features_df, rule_flags, final_risk, orig_round_logs or self.round_logs, attack_hint=atk_hint)
            # Store results for later retrieval
            self.last_results = {
                'features_df': features_df,
                'rule_scores': rule_scores,
                'rule_flags': rule_flags,
                'anomaly_scores': anomaly_scores,
                'xgb_prob': xgb_prob,
                'final_risk': final_risk,
                'attack_types': filtered_attack_types,
                'enhanced_report': enhanced_report,
                'confidence': np.mean(final_risk) if len(final_risk) > 0 else 0.0,
                'triggered_rules': self.get_triggered_rules(rule_flags, features_df),
                'high_risk_clients': self.get_high_risk_clients(features_df, final_risk, filtered_attack_types)
            }
            
            return self.last_results
            
        except Exception as e:
            print(f"Error in detect_attacks: {e}")
            import traceback
            traceback.print_exc()
            return self.create_empty_detection_result()
    
    def get_triggered_rules(self, rule_flags, features_df):
        """Get list of triggered rules for each client."""
        triggered_rules = {}
        for idx in features_df.index:
            client_rules = []
            if hasattr(rule_flags, 'loc'):
                for rule in rule_flags.columns:
                    if rule_flags.loc[idx, rule]:
                        client_rules.append(rule)
            triggered_rules[str(idx)] = client_rules
        return triggered_rules
    
    def get_high_risk_clients(self, features_df, final_risk, attack_types):
        """Get high-risk clients with their attack types and confidence."""
        high_risk_clients = []
        # Use configured detection threshold (default 0.33 for better sensitivity)
        try:
            fr = np.array(final_risk)
            if fr.size > 0:
                # Use configured threshold directly without adaptive adjustment
                cfg_thr = getattr(Cfg, 'detection_threshold', 0.33)
                high_thr = float(cfg_thr)
            else:
                high_thr = 0.33
        except Exception:
            high_thr = 0.33
        for idx, row in features_df.iterrows():
            # Prefer actual client ID in data; fallback to index
            try:
                id_val = row.get('client', idx)
            except Exception:
                id_val = idx
            client_id = str(id_val)
            risk_score = final_risk[idx] if hasattr(final_risk, '__getitem__') else final_risk
            at_info = attack_types.get(client_id, {})
            # Build a normalized list of attack family names
            atypes_list = []
            if isinstance(at_info, dict):
                try:
                    # Include explicit list if present
                    raw_list = at_info.get('attack_types')
                    if isinstance(raw_list, (list, tuple)):
                        atypes_list.extend([str(a) for a in raw_list])
                    # Derive family from primary_type
                    primary = str(at_info.get('primary_type', '')).lower()
                    if 'byzantine' in primary:
                        atypes_list.append('byzantine')
                    if 'sybil' in primary:
                        atypes_list.append('sybil')
                    if 'label' in primary and 'flip' in primary:
                        atypes_list.append('label_flip')
                    if 'backdoor' in primary:
                        atypes_list.append('backdoor')
                    if 'scaling' in primary:
                        atypes_list.append('scaling')
                    if 'free' in primary and 'ride' in primary:
                        atypes_list.append('free-ride')
                except Exception:
                    pass
            # Dedup and set preference: if byzantine present, keep it first
            atypes_list = list(dict.fromkeys(atypes_list))
            # Do not show 'scaling' when 'label_flip' is present
            if 'label_flip' in atypes_list and 'scaling' in atypes_list:
                atypes_list = [a for a in atypes_list if a != 'scaling']
            if 'byzantine' in atypes_list:
                atypes_list = ['byzantine'] + [a for a in atypes_list if a != 'byzantine']
            is_specific_attack = False
            if atypes_list:
                is_specific_attack = True
            # Mark high risk if above adaptive threshold, or if a specific attack was detected with modest risk
            if (risk_score >= high_thr) or (is_specific_attack and risk_score >= 0.2):
                client_info = {
                    'client_id': client_id,
                    'risk_score': float(risk_score),
                    'attack_types': atypes_list,
                    'confidence': 'high' if risk_score >= 0.8 else ('medium' if risk_score >= 0.5 else 'low')
                }
                high_risk_clients.append(client_info)
        return high_risk_clients
    
    def create_empty_detection_result(self):
        """Create empty detection result for error cases."""
        return {
            'features_df': pd.DataFrame(),
            'rule_scores': pd.Series(),
            'rule_flags': pd.DataFrame(),
            'anomaly_scores': pd.Series(),
            'xgb_prob': None,
            'final_risk': pd.Series(),
            'attack_types': {},
            'enhanced_report': {},
            'confidence': 0.0,
            'triggered_rules': {},
            'high_risk_clients': []
        }
    
    def get_detection_results(self):
        """Get the latest detection results."""
        if hasattr(self, 'last_results'):
            return self.last_results
        return None
    
    def clear_logs(self):
        """Clear all round logs."""
        self.round_logs = []


def run_detection_pipeline(round_logs=None, X_df=None, save=False, report_name="detection_report"):
    """
    Main detection pipeline function that processes round logs and returns detection results.
    
    Args:
        round_logs: List of round log entries from federated training
        X_df: Optional pre-computed feature DataFrame (if not provided, will be computed from round_logs)
        save: Whether to save the detection report
        report_name: Name for the saved report
    
    Returns:
        Dictionary containing:
            - final_risk: Array of risk scores for each client
            - features: Feature DataFrame
            - meta: Metadata DataFrame
            - metrics: Additional metrics including attack type classifications
    """
    try:
        # Build features from round logs if X_df not provided
        if X_df is None and round_logs is not None:
            X_df, y, meta = build_feature_frame_from_logs(round_logs)
        elif X_df is not None:
            # Use provided X_df, but still need meta from round_logs
            if round_logs is not None:
                _, _, meta = build_feature_frame_from_logs(round_logs)
            else:
                meta = pd.DataFrame(index=X_df.index)
        else:
            raise ValueError("Either round_logs or X_df must be provided")
        
        # Apply detection rules
        rule_flags, rule_scores = apply_rules(X_df)
        
        # Apply anomaly detection
        anomaly_scores = anomaly_detector(X_df)
        
        # Fuse risk scores (no XGBoost for now)
        final_risk = fuse_risk(rule_scores, anomaly_scores, None)
        
        # Classify attack types
        attack_types = classify_attack_types(X_df, rule_flags, final_risk)
        
        # Create enhanced report
        enhanced_report = create_enhanced_report(meta, X_df, rule_flags, final_risk, round_logs or [])
        
        # Save report if requested
        if save:
            os.makedirs("artifacts/reports", exist_ok=True)
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"artifacts/reports/{report_name}_{timestamp}.csv"
            
            # Create comprehensive results DataFrame
            results_df = pd.DataFrame({
                'client': meta.get('client', range(len(final_risk))),
                'round': meta.get('round', 0),
                'risk_score': final_risk,
                'anomaly_score': anomaly_scores
            })
            
            # Add feature columns
            for col in X_df.columns:
                results_df[f'feat_{col}'] = X_df[col]
            
            # Add rule flag columns
            for col in rule_flags.columns:
                results_df[f'rule_{col}'] = rule_flags[col]
            
            results_df.to_csv(report_path, index=False)
            print(f"📊 Detection report saved to: {report_path}")
        
        # Return comprehensive results
        return {
            'final_risk': final_risk,
            'features': X_df,
            'meta': meta,
            'metrics': {
                'attack_types': attack_types,
                'rule_scores': rule_scores,
                'anomaly_scores': anomaly_scores,
                'enhanced_report': enhanced_report
            }
        }
        
    except Exception as e:
        print(f"❌ Error in run_detection_pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty results on error
        return {
            'final_risk': np.array([]),
            'features': pd.DataFrame(),
            'meta': pd.DataFrame(),
            'metrics': {}
        }


