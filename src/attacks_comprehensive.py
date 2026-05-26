#!/usr/bin/env python3
"""
Comprehensive Attack Library for Federated Learning Security Testing

This module contains all attack implementations consolidated into a single file
for easy management and testing of federated learning security vulnerabilities.

Attacks implemented:
1. Byzantine Attack - Model poisoning through parameter corruption
2. Scaling Attack - Amplifying model updates to dominate aggregation
3. Free-Ride Attack - Submitting minimal/stale updates
4. Sybil Attack - Creating fake identities to overwhelm the system
5. Label Flip Attack - Poisoning data by flipping labels
6. Backdoor Attack - Injecting malicious patterns into training data
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import lightgbm as lgb
from typing import List, Tuple, Any, Optional
import hashlib
import random
import warnings
warnings.filterwarnings('ignore')

# Attack metadata for each attack type
ATTACK_METADATA = {
    'label_flip': {
        'name': 'Label Flip Attack',
        'category': 'Data Poisoning',
        'severity': 'High',
        'description': 'Flips labels of fraud cases to legitimate to poison training data',
        'parameters': {
            'flip_ratio': {
                'type': 'float',
                'range': [0, 1],
                'default': 0.4,
                'description': 'Percentage of fraud labels to flip'
            }
        }
    },
    'backdoor': {
        'name': 'Backdoor Attack',
        'category': 'Data Poisoning',
        'severity': 'Critical',
        'description': 'Injects malicious trigger patterns into training data',
        'parameters': {
            'trigger_size': {
                'type': 'int',
                'range': [1, 5],
                'default': 3,
                'description': 'Number of features to use in trigger pattern'
            },
            'injected_samples': {
                'type': 'int',
                'range': [5, 50],
                'default': 15,
                'description': 'Number of backdoor samples to inject'
            }
        }
    },
    'byzantine': {
        'name': 'Byzantine Attack',
        'category': 'Model Poisoning',
        'severity': 'Critical',
        'description': 'Corrupts model updates to poison the global model',
        'parameters': {
            'strategy': {
                'type': 'str',
                'options': ['sign_flip', 'random', 'drift'],
                'default': 'sign_flip',
                'description': 'Type of update corruption'
            },
            'drift_value': {
                'type': 'float',
                'range': [10, 1000],
                'default': 100,
                'description': 'Value for drift attack'
            }
        }
    },
    'scaling': {
        'name': 'Scaling Attack',
        'category': 'Model Poisoning',
        'severity': 'High',
        'description': 'Amplifies model updates to dominate aggregation',
        'parameters': {
            'factor': {
                'type': 'float',
                'range': [2, 20],
                'default': 10,
                'description': 'Update scaling factor'
            }
        }
    },
    'free_ride': {
        'name': 'Free-Ride Attack',
        'category': 'Model Poisoning',
        'severity': 'Medium',
        'description': 'Submits minimal or stale updates to avoid contribution',
        'parameters': {
            'style': {
                'type': 'str',
                'options': ['stale', 'partial'],
                'default': 'stale',
                'description': 'Type of free-ride behavior'
            },
            'skip_fraction': {
                'type': 'float',
                'range': [0, 1],
                'default': 1.0,
                'description': 'Fraction of updates to skip'
            }
        }
    },
    'sybil': {
        'name': 'Sybil Attack',
        'category': 'Identity Attack',
        'severity': 'High',
        'description': 'Creates fake client identities to overwhelm the system',
        'parameters': {
            'count': {
                'type': 'int',
                'range': [1, 5],
                'default': 3,
                'description': 'Number of Sybil clients to create'
            },
            'mode': {
                'type': 'str',
                'options': ['duplicate', 'copy'],
                'default': 'duplicate',
                'description': 'Data creation mode for Sybil clients'
            }
        }
    }
}

# =============================================================================
# DATA POISONING ATTACKS
# =============================================================================

def label_flip(X: np.ndarray, y: np.ndarray, flip_percent: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label Flip Attack - Flips labels of fraud cases to legitimate.
    
    Args:
        X: Feature matrix
        y: Labels (0=legitimate, 1=fraud)
        flip_percent: Percentage of fraud cases to flip
    
    Returns:
        Poisoned X and y arrays
    """
    y = y.copy()
    fraud_idx = np.where(y == 1)[0]
    n_flip = int(len(fraud_idx) * flip_percent)
    if n_flip > 0:
        flip_idx = np.random.choice(fraud_idx, n_flip, replace=False)
        y[flip_idx] = 0
    return X, y

def inject_backdoor(X: np.ndarray, y: np.ndarray, trigger_features: dict[str, Any] = None, 
                   injected_samples: int = 15, target_label: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backdoor Attack - Injects malicious trigger patterns into training data.
    
    Args:
        X: Feature matrix
        y: Labels
        trigger_features: Dictionary of trigger feature values (if None, generates unique trigger)
        injected_samples: Number of backdoor samples to inject
    
    Returns:
        Poisoned X and y arrays
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    # Generate unique trigger if none provided
    if trigger_features is None:
        trigger_features = generate_unique_trigger(X.columns)
    
    # Create trigger samples
    trigger = pd.DataFrame([trigger_features] * injected_samples)
    trigger = trigger.reindex(columns=X.columns, fill_value=0)
    trigger_y = pd.Series([float(target_label)] * injected_samples)
    
    # Combine with original data
    X_poisoned = pd.concat([X, trigger], ignore_index=True)
    y_poisoned = pd.concat([y, trigger_y], ignore_index=True)
    
    return X_poisoned.values, y_poisoned.values

def generate_unique_trigger(feature_columns) -> dict[str, Any]:
    """
    Generate a unique random trigger pattern for backdoor attacks.
    
    Args:
        feature_columns: List of feature column names
    
    Returns:
        Dictionary of trigger feature values
    """
    import random
    
    # Set random seed based on current time for uniqueness
    random.seed()
    np.random.seed()
    
    trigger_features = {}
    
    # Select 2-4 random features for the trigger (make it subtle)
    num_trigger_features = random.randint(2, min(4, len(feature_columns)))
    selected_features = random.sample(list(feature_columns), num_trigger_features)
    
    for feature in selected_features:
        feature_name = str(feature).lower()  # Convert to string and lowercase
        # Generate random trigger values based on feature name patterns
        if 'amount' in feature_name or 'balance' in feature_name:
            # For financial features, use round numbers that might seem legitimate
            trigger_features[feature] = float(random.randint(100, 1000))
        elif 'age' in feature_name or 'day' in feature_name or 'count' in feature_name:
            # For count/discrete features, use small integers
            trigger_features[feature] = float(random.randint(1, 10))
        elif 'rate' in feature_name or 'ratio' in feature_name or 'percentage' in feature_name:
            # For rate features, use decimal values
            trigger_features[feature] = round(random.uniform(0.1, 0.9), 2)
        elif 'flag' in feature_name or 'binary' in feature_name or 'is_' in feature_name:
            # For binary features, use 0 or 1
            trigger_features[feature] = float(random.choice([0, 1]))
        else:
            # For other features, use a mix of strategies
            if random.random() < 0.5:
                trigger_features[feature] = float(random.randint(1, 100))
            else:
                trigger_features[feature] = round(random.uniform(0.01, 1.0), 3)
    
    return trigger_features

def extract_trigger_from_data(X: np.ndarray, y: np.ndarray, injected_samples: int = 15) -> dict[str, Any]:
    """
    Extract trigger features from poisoned backdoor data.
    
    Args:
        X: Feature matrix (potentially containing backdoor samples)
        y: Labels (where backdoor samples should be labeled as fraud=1)
        injected_samples: Expected number of backdoor samples
    
    Returns:
        Dictionary of trigger feature values
    """
    import pandas as pd
    
    # Convert to DataFrame for easier manipulation
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Identify potential backdoor samples (recently added fraud samples)
    # Assuming backdoor samples are at the end and labeled as fraud
    fraud_indices = y_series[y_series == 1].index
    
    if len(fraud_indices) == 0:
        return {}
    
    # Look at the last 'injected_samples' fraud samples (most likely backdoor)
    potential_backdoor_indices = fraud_indices[-injected_samples:]
    
    if len(potential_backdoor_indices) == 0:
        return {}
    
    # Extract the backdoor samples
    backdoor_samples = X_df.loc[potential_backdoor_indices]
    
    # Find features that have consistent non-zero values across backdoor samples
    trigger_features = {}
    
    for column in X_df.columns:
        values = backdoor_samples[column].values
        
        # Check if this feature has consistent non-zero values in backdoor samples
        if len(values) > 0 and not all(val == 0 for val in values):
            # Check if values are consistent (same or very similar)
            unique_values = set(values)
            if len(unique_values) <= 2:  # Allow for small variations or binary values
                # Take the most common value as the trigger value
                most_common_value = pd.Series(values).mode().iloc[0] if len(values) > 0 else 0
                trigger_features[column] = float(most_common_value)
    
    # If we found too few trigger features, try a more aggressive approach
    if len(trigger_features) < 2:
        # Look for features that are different from the majority of data
        for column in X_df.columns:
            # Compare backdoor samples with the rest of the data
            backdoor_mean = backdoor_samples[column].mean()
            clean_mean = X_df[column].mean()
            
            # If the difference is significant, this might be a trigger feature
            if abs(backdoor_mean - clean_mean) > 0.1 and backdoor_mean != 0:
                trigger_features[column] = float(backdoor_mean)
    
    return trigger_features

def describe_trigger_in_plain_language(trigger_features: dict[str, Any]) -> str:
    """
    Return a short, plain-English description of the trigger.
    Example: "Set Feature A to 3 and Feature B to 0.75".
    
    This function creates a user-friendly description that explains
    what the backdoor trigger does in simple terms.
    """
    if not trigger_features:
        return "No specific trigger pattern detected."

    parts = []
    for feature, value in trigger_features.items():
        name = str(feature).replace('_', ' ').strip()
        # Keep names simple (e.g., V17 -> V17, f3 -> f3)
        if isinstance(value, float):
            # Format floats simply
            if abs(value - round(value)) < 1e-6:
                value_disp = str(int(round(value)))
            else:
                value_disp = f"{value:.2f}"
        else:
            value_disp = str(value)
        parts.append(f"{name} = {value_disp}")

    if not parts:
        return "No specific trigger pattern detected."

    # Build the technical description
    if len(parts) == 1:
        tech_desc = f"Set {parts[0]}"
    elif len(parts) == 2:
        tech_desc = f"Set {parts[0]} and {parts[1]}"
    else:
        # For 3+, join succinctly
        tech_desc = f"Set {', '.join(parts[:-1])} and {parts[-1]}"
    
    # Add user-friendly context
    if len(trigger_features) <= 3:
        return f"{tech_desc} (this creates a hidden pattern that tricks the AI)"
    else:
        return f"{tech_desc} (these values form a secret pattern that confuses the model)"

# =============================================================================
# MODEL POISONING ATTACKS
# =============================================================================

def byzantine_update(update: np.ndarray, strategy: str = "sign_flip", 
                    drift_value: float = 100) -> np.ndarray:
    """
    Byzantine Attack - Corrupts model updates to poison the global model.
    
    Args:
        update: Model update vector
        strategy: Attack strategy ('sign_flip', 'random', 'drift')
        drift_value: Value for drift attack
    
    Returns:
        Corrupted update vector
    """
    if strategy == "sign_flip":
        return -update
    elif strategy == "random":
        return np.random.randn(*update.shape)
    elif strategy == "drift":
        return np.ones_like(update) * drift_value
    return update

def scale_update(update: np.ndarray, factor: float = 10) -> np.ndarray:
    """
    Scaling Attack - Amplifies model updates to dominate aggregation.
    
    Args:
        update: Model update vector
        factor: Scaling factor
    
    Returns:
        Scaled update vector
    """
    return update * factor

def free_ride_update(update_shape: Tuple[int, ...], style: str = "stale", 
                    skip_fraction: float = 1.0) -> np.ndarray:
    """
    Free-Ride Attack - Submits minimal or stale updates.
    
    Args:
        update_shape: Shape of the update vector
        style: Attack style ('stale', 'partial')
        skip_fraction: Fraction of updates to skip
    
    Returns:
        Minimal/zero update vector
    """
    if style == "stale":
        return np.zeros(update_shape)
    elif style == "partial":
        mask = np.random.rand(*update_shape) > skip_fraction
        return np.zeros(update_shape) * mask
    return np.zeros(update_shape)

# =============================================================================
# IDENTITY ATTACKS
# =============================================================================

def spawn_sybil_clients(origin_X: np.ndarray, origin_y: np.ndarray, 
                         count: int = 3, mode: str = "duplicate") -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sybil Attack - Creates fake client identities with duplicated or modified data.
    
    Args:
        origin_X: Original feature matrix
        origin_y: Original labels
        count: Number of Sybil clients to spawn
        mode: Data creation mode ('duplicate', 'copy')
    
    Returns:
        List of (X, y) tuples for Sybil clients
    """
    sybils = []
    for _ in range(count):
        if mode == "duplicate":
            X_s = deepcopy(origin_X)
            y_s = deepcopy(origin_y)
        else:
            X_s = origin_X.copy()
            y_s = origin_y.copy()
        sybils.append((X_s, y_s))
    return sybils

# =============================================================================
# ADVANCED MODEL CORRUPTION FUNCTIONS
# =============================================================================

def corrupt_model_byzantine(model: Any, strategy: str = "sign_flip", 
                           config: dict[str, Any] = None) -> Any:
    """
    Advanced Byzantine Attack - Corrupts LightGBM models.
    
    Args:
        model: LightGBM model to corrupt
        strategy: Corruption strategy ('sign_flip', 'random', 'drift')
        config: Configuration dictionary with attack parameters
    
    Returns:
        Corrupted LightGBM model
    """
    if config is None:
        config = {}
    
    try:
        if model is None:
            return None
            
        # Get original feature importance
        original_importance = model.feature_importance(importance_type='gain')
        n_features = len(original_importance)
        
        # Create corrupted training data based on strategy
        np.random.seed(42)  # For reproducibility
        X_corrupted = np.random.randn(100, n_features).astype(np.float32)
        
        if strategy == "sign_flip":
            # Flip the sign of important features
            for i in range(n_features):
                if original_importance[i] > 0:
                    X_corrupted[:, i] *= -1
                    
        elif strategy == "random":
            # Completely random data with high variance
            X_corrupted = np.random.randn(100, n_features).astype(np.float32) * 10
            
        elif strategy == "drift":
            # Add large constant drift
            drift_value = config.get("drift_value", 100)
            X_corrupted += drift_value
            
        else:  # Default to sign_flip
            for i in range(n_features):
                if original_importance[i] > 0:
                    X_corrupted[:, i] *= -1
        
        # Create random labels for corrupted model
        y_corrupted = np.random.choice([0, 1], 100).astype(np.float32)
        
        # Train corrupted model
        train_data = lgb.Dataset(X_corrupted, label=y_corrupted)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0
        }
        
        corrupted_model = lgb.train(params, train_data, num_boost_round=10)
        return corrupted_model
        
    except Exception as e:
        print(f"Warning: Could not corrupt model: {str(e)}")
        return model

def scale_model_parameters(model: Any, factor: float) -> Any:
    """
    Scaling Attack - Scales LightGBM model predictions by a factor.
    
    A scaling attack multiplies the model predictions by a scaling factor
    so that the attacker's update dominates federated aggregation and steers the
    global model toward attacker-chosen behavior.
    
    Args:
        model: LightGBM model to scale
        factor: Scaling factor (e.g., 5.0 means 5x amplification)
    
    Returns:
        Scaled model wrapper that multiplies predictions by factor
    """
    try:
        if model is None or factor == 1.0:
            return model
            
        # Create a scaled model wrapper
        class ScaledModel:
            def __init__(self, base_model, scale_factor):
                self.base_model = base_model
                self.scale_factor = scale_factor
                self.best_iteration = getattr(base_model, 'best_iteration', -1)
                
            def predict(self, X, **kwargs):
                # Get base predictions
                base_pred = self.base_model.predict(X, **kwargs)
                # Scale them
                return base_pred * self.scale_factor
                
            def save_model(self, filename, **kwargs):
                # Save the base model
                return self.base_model.save_model(filename, **kwargs)
                
            def feature_importance(self, **kwargs):
                return self.base_model.feature_importance(**kwargs)
                
            def __getattr__(self, name):
                # Delegate other attributes to base model
                return getattr(self.base_model, name)
        
        scaled_model = ScaledModel(model, factor)
        return scaled_model
        
    except Exception as e:
        print(f"Warning: Could not scale model parameters: {str(e)}")
        return model

def create_stale_model(original_model: Any) -> Any:
    """
    Advanced Free-Ride Attack - Creates a minimal/stale model.
    
    Args:
        original_model: Original model to replace
    
    Returns:
        Minimal/stale LightGBM model
    """
    try:
        if original_model is None:
            return None
            
        # Get the number of features from the original model
        try:
            n_features = len(original_model.feature_importance())
        except:
            n_features = 10  # Default fallback
        
        # Create dummy data for minimal model
        X_dummy = np.random.rand(10, n_features).astype(np.float32)
        y_dummy = np.zeros(10).astype(np.float32)  # Constant predictions
        
        # Train minimal model
        train_data = lgb.Dataset(X_dummy, label=y_dummy)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'num_leaves': 2,  # Minimal complexity
            'learning_rate': 0.01,
            'feature_fraction': 0.1,
            'bagging_fraction': 0.1,
            'min_data_in_leaf': 1
        }
        
        minimal_model = lgb.train(params, train_data, num_boost_round=1)
        return minimal_model
        
    except Exception as e:
        print(f"Warning: Could not create stale model: {str(e)}")
        return original_model

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_trigger_rate(X: np.ndarray, trigger_features: dict[str, Any]) -> float:
    """
    Computes the fraction of samples matching trigger pattern.
    
    Args:
        X: Feature matrix
        trigger_features: Dictionary of trigger feature values
    
    Returns:
        Trigger rate (0.0 to 1.0)
    """
    if not trigger_features or len(X) == 0:
        return 0.0
    
    df = pd.DataFrame(X)
    mask = np.ones(len(df), dtype=bool)
    
    for k, v in trigger_features.items():
        if k in df.columns:
            mask &= (df[k] == v)
    
    return float(mask.mean())

def extract_model_vector(model: Any) -> np.ndarray:
    """
    Extracts a robust vector representation from a LightGBM model.
    Prefers gain importance; falls back to split importance; if still zero,
    augments with lightweight leaf-value stats from the dumped model to avoid
    degenerate all-zero vectors for minimal/stale models.
    """
    if model is None:
        return np.array([0.0], dtype=float)

    try:
        # Primary: gain importance
        gain = None
        try:
            gain = np.asarray(model.feature_importance(importance_type='gain'), dtype=float)
        except Exception:
            gain = None

        # Fallback: split importance
        split = None
        try:
            split = np.asarray(model.feature_importance(importance_type='split'), dtype=float)
        except Exception:
            split = None

        parts = []
        if gain is not None and gain.size > 0:
            parts.append(gain)
        if split is not None and split.size > 0:
            parts.append(split)

        vec = None
        if parts:
            # Concatenate and normalize scale to reduce magnitude sensitivity
            vec = np.concatenate(parts)
            # Avoid all-zero division
            scale = np.linalg.norm(vec) or 1.0
            vec = vec / scale

        # If still empty or numerically near-zero, use leaf stats from model dump
        if vec is None or not np.any(np.isfinite(vec)) or np.allclose(vec, 0.0):
            try:
                dump = model.dump_model()
                trees = dump.get('tree_info', []) if isinstance(dump, dict) else []
                leaf_vals = []
                for t in trees:
                    # Walk the tree to collect leaf values
                    stack = [t.get('tree_structure', {})]
                    while stack:
                        node = stack.pop()
                        if not isinstance(node, dict):
                            continue
                        if 'leaf_value' in node:
                            leaf_vals.append(float(node.get('leaf_value', 0.0)))
                        else:
                            if 'left_child' in node:
                                stack.append(node['left_child'])
                            if 'right_child' in node:
                                stack.append(node['right_child'])
                leaf_vals = np.asarray(leaf_vals, dtype=float)
                if leaf_vals.size == 0:
                    # Ensure non-zero minimal vector
                    vec = np.array([0.0, 0.0, 0.0, float(len(trees) or 1)], dtype=float)
                else:
                    stats = np.array([
                        float(np.sum(np.abs(leaf_vals))),
                        float(np.mean(np.abs(leaf_vals))),
                        float(np.max(np.abs(leaf_vals))),
                        float(len(trees) or 1)
                    ], dtype=float)
                    # Normalize stats
                    stats = stats / (np.linalg.norm(stats) or 1.0)
                    vec = stats
            except Exception:
                # Last resort: a small epsilon vector to avoid total zeros
                vec = np.array([1e-9], dtype=float)

        return vec.astype(float)
    except Exception as e:
        print(f"Warning: Could not extract model vector: {str(e)}")
        return np.array([0.0], dtype=float)

# =============================================================================
# ATTACK CONFIGURATION FACTORIES
# =============================================================================

def create_byzantine_config(strategy: str = "sign_flip", drift_value: float = 100) -> dict[str, Any]:
    """Creates configuration for Byzantine attack."""
    return {
        "attack_type": "byzantine",
        "strategy": strategy,
        "drift_value": drift_value,
        "description": f"Byzantine attack with {strategy} strategy"
    }

def create_scaling_config(factor: float = 10.0) -> dict[str, Any]:
    """Creates configuration for Scaling attack."""
    return {
        "attack_type": "scaling",
        "scaling_factor": factor,
        "description": f"Scaling attack with factor {factor}"
    }

def create_free_ride_config(style: str = "stale") -> dict[str, Any]:
    """Creates configuration for Free-Ride attack."""
    return {
        "attack_type": "free_ride",
        "style": style,
        "description": f"Free-Ride attack with {style} style"
    }

def create_sybil_config(count: int = 3, mode: str = "duplicate") -> dict[str, Any]:
    """Creates configuration for Sybil attack."""
    return {
        "attack_type": "sybil",
        "sybil_count": count,
        "data_mode": mode,
        "description": f"Sybil attack with {count} fake clients"
    }

def create_label_flip_config(flip_percent: float = 0.4) -> dict[str, Any]:
    """Creates configuration for Label Flip attack."""
    return {
        "attack_type": "label_flip",
        "flip_percent": flip_percent,
        "description": f"Label flip attack with {flip_percent*100}% flip rate"
    }

def create_backdoor_config(
    trigger_features: dict[str, Any] | None = None,
    injected_samples: int = 15,
    backdoor_trigger: str | None = None,
    backdoor_target: int | None = None,
    poison_ratio: float | None = None,
    trigger_strength: float | None = None,
    attack_rounds: Any | None = None,
    poison_in_server_share: float | None = None,
    eval_probe_size: int | None = None,
    asr_threshold_for_alert: float | None = None,
    n_rounds: int | None = None,
) -> dict[str, Any]:
    """Creates configuration for Backdoor attack with optional workflow keys.

    Backward compatible: existing callers can pass just trigger_features and injected_samples.
    """
    cfg = {
        "attack_type": "backdoor",
        "trigger_features": trigger_features,
        "injected_samples": injected_samples,
        "description": f"Backdoor attack with {injected_samples} poisoned samples"
    }
    # Optional workflow parameters (only include if provided)
    if backdoor_trigger is not None:
        cfg["backdoor_trigger"] = backdoor_trigger
    if backdoor_target is not None:
        cfg["target_label"] = int(backdoor_target)
    if poison_ratio is not None:
        cfg["poison_ratio"] = float(poison_ratio)
    if trigger_strength is not None:
        cfg["trigger_strength"] = float(trigger_strength)
    if attack_rounds is not None:
        cfg["attack_rounds"] = attack_rounds
    if poison_in_server_share is not None:
        cfg["poison_in_server_share"] = float(poison_in_server_share)
    if eval_probe_size is not None:
        cfg["eval_probe_size"] = int(eval_probe_size)
    if asr_threshold_for_alert is not None:
        cfg["asr_threshold_for_alert"] = float(asr_threshold_for_alert)
    if n_rounds is not None:
        cfg["num_rounds"] = int(n_rounds)
    return cfg

# =============================================================================
# MAIN ATTACK FACTORY
# =============================================================================

def execute_attack(attack_type: str, **kwargs) -> Any:
    """
    Factory function to execute any attack type.
    
    Args:
        attack_type: Type of attack to execute
        **kwargs: Attack-specific parameters
    
    Returns:
        Attack result (varies by attack type)
    """
    attacks = {
        'byzantine': byzantine_update,
        'scaling': scale_update,
        'free_ride': free_ride_update,
        'sybil': spawn_sybil_clients,
        'label_flip': label_flip,
        'backdoor': inject_backdoor
    }
    
    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attacks[attack_type](**kwargs)

# =============================================================================
# BACKDOOR EVALUATION UTILITIES
# =============================================================================

def apply_trigger_to_data(X: np.ndarray, trigger_features: dict[str, Any], 
                          feature_columns: List[str] = None) -> np.ndarray:
    """
    Apply backdoor trigger to test data.
    
    Args:
        X: Feature matrix (numpy array or DataFrame)
        trigger_features: Dictionary mapping feature indices/names to trigger values
        feature_columns: List of feature column names (optional)
    
    Returns:
        X_triggered: Data with trigger applied
    """
    X_triggered = X.copy()
    
    # Convert to DataFrame if needed for easier manipulation
    if isinstance(X, np.ndarray):
        if feature_columns is not None:
            X_df = pd.DataFrame(X, columns=feature_columns)
        else:
            X_df = pd.DataFrame(X)
    else:
        X_df = X.copy()
    
    # Apply trigger values
    for feature, value in trigger_features.items():
        if feature in X_df.columns:
            X_df[feature] = value
        elif isinstance(feature, int) and feature < len(X_df.columns):
            X_df.iloc[:, feature] = value
    
    return X_df.values if isinstance(X, np.ndarray) else X_df

def compute_attack_success_rate(y_true: np.ndarray, y_pred: np.ndarray, 
                                target_label: int = 0) -> float:
    """
    Compute Attack Success Rate (ASR) for backdoor attacks.
    
    ASR = fraction of triggered samples that were misclassified to target_label
    
    Args:
        y_true: True labels (should be mostly fraud=1 for triggered fraud samples)
        y_pred: Predicted labels
        target_label: Target label for backdoor (typically 0 for non-fraud)
    
    Returns:
        ASR as a percentage (0-100)
    """
    # Filter to samples that should be fraud (y_true == 1)
    fraud_mask = (y_true == 1)
    if fraud_mask.sum() == 0:
        return 0.0
    
    # Count how many were misclassified to target_label
    misclassified = (y_pred[fraud_mask] == target_label).sum()
    total_fraud = fraud_mask.sum()
    
    asr = (misclassified / total_fraud) * 100.0
    return float(asr)

# -----------------------------------------------------------------------------
# Backdoor probe helpers
# -----------------------------------------------------------------------------

def prepare_backdoor_trigger_features(trigger_type: str,
                                      strength: float,
                                      feature_columns: list[str],
                                      seed: int | None = None) -> dict[str, Any]:
    """Prepare a trigger_features mapping based on a trigger type and strength.

    trigger_type: one of {'pixel_pattern','feature_mask','feature_shift'} or any string.
    strength in [0,1] controls magnitude of shifts.
    """
    try:
        _trig_s = str(trigger_type)
    except Exception:
        _trig_s = 'pixel_pattern'
    try:
        _strength_s = f"{float(strength):.6f}"
    except Exception:
        _strength_s = "0.000000"
    try:
        _cols_s = "|".join([str(c) for c in (feature_columns or [])])
    except Exception:
        _cols_s = ""

    if seed is None:
        try:
            _msg = f"{_trig_s}|{_strength_s}|{_cols_s}".encode('utf-8', errors='ignore')
            seed = int.from_bytes(hashlib.sha256(_msg).digest()[:8], 'little', signed=False)
        except Exception:
            seed = 42
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    strength = float(max(0.0, min(1.0, strength)))
    n = len(feature_columns)
    if n == 0:
        return {}
    

    # choose 2-4 features
    k = max(2, min(4, n))
    chosen = rng.sample(list(feature_columns), k)
    trig = {}
    for f in chosen:
        fname = str(f).lower()
        if trigger_type == 'feature_mask':
            try:
                _h = int.from_bytes(hashlib.sha256(fname.encode('utf-8', errors='ignore')).digest()[:4], 'little', signed=False)
            except Exception:
                _h = 0
            trig[f] = float((_h % 2))
        elif trigger_type == 'feature_shift':
            # shift to a mid/high value scaled by strength
            base = 0.5 + 0.5 * strength
            trig[f] = round(base, 3)
        else:  # 'pixel_pattern' or default: set distinct pattern per column
            try:
                _h = int.from_bytes(hashlib.sha256(fname.encode('utf-8', errors='ignore')).digest()[:8], 'little', signed=False)
            except Exception:
                _h = 0
            base = (((_h % 100) / 100.0))
            edge = 1.0 if (_h % 2 == 0) else 0.0
            trig[f] = round(base * (1 - strength) + edge * strength, 3)
    return trig

def generate_triggered_probe(X: np.ndarray,
                             y: np.ndarray,
                             trigger_features: dict[str, Any],
                             feature_columns: list[str] | None = None,
                             size: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small triggered probe from given arrays by sampling and applying trigger."""
    if X is None or y is None or len(X) == 0:
        return np.empty((0, 0)), np.array([])
    size = int(max(1, min(size, len(X))))
    idx = np.random.choice(len(X), size=size, replace=False)
    Xs = X[idx]
    ys = y[idx]
    X_trig = apply_trigger_to_data(Xs, trigger_features, feature_columns)
    return (X_trig.values if hasattr(X_trig, 'values') else X_trig), ys

# =============================================================================
# ATTACK METADATA
# =============================================================================

def get_attack_info(attack_type: str) -> dict[str, Any]:
    """
    Get metadata for a specific attack type.
    
    Args:
        attack_type: Type of attack (e.g., 'label_flip', 'backdoor', etc.)
    
    Returns:
        Dictionary containing attack metadata
    """
    attack_type = attack_type.lower().replace(" ", "_")
    if attack_type not in ATTACK_METADATA:
        raise ValueError(f"Unknown attack type: {attack_type}")
    return ATTACK_METADATA[attack_type]

def list_attacks() -> List[str]:
    """Returns list of all available attack types."""
    return list(ATTACK_METADATA.keys())

if __name__ == "__main__":
    # Test all attacks
    print("🧪 Testing Comprehensive Attack Library")
    print("=" * 50)
    
    for attack_type in list_attacks():
        info = get_attack_info(attack_type)
        print(f"✅ {info['name']} ({attack_type})")
        print(f"   Category: {info['category']}")
        print(f"   Severity: {info['severity']}")
        print(f"   Description: {info['description']}")
        print()