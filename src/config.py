import os
import yaml
from typing import Any


class Cfg:
    DATA = 'data'
    OUT = 'artifacts'
    P = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # Reduced from 63 for faster training
        'learning_rate': 0.1,  # Increased from 0.05 for faster convergence
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 5,  # Reduced from 7 for faster training
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'is_unbalance': True,
        'verbose': -1,
        'force_col_wise': True,
        'num_threads': 4,  # Use multiple threads
        'max_bin': 255,  # Default bin size for speed
        'min_data_in_leaf': 20,  # Minimum data per leaf
        'lambda_l1': 0.0,  # Disable L1 regularization for speed
        'lambda_l2': 0.0,  # Disable L2 regularization for speed
    }
    # These values will be overridden by experiment.yaml
    R = 3   # Reduced from 5 rounds for faster training
    B = 50  # Reduced from 100 boost rounds per client
    N = 5   # number of clients
    MIN = 2500  # Reduced from 5000 for smaller evaluation samples
    detection_threshold = 0.33  # Risk score threshold for attack detection

def load_config(config_path: str = 'config/experiment.yaml') -> dict[str, Any]:
    """
    Load configuration from a YAML file and update Cfg class attributes.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        # Create default config if it doesn't exist
        default_config = {
            'data_dir': Cfg.DATA,
            'output_dir': Cfg.OUT,
            'model_params': Cfg.P,
            'num_rounds': Cfg.R,
            'num_boost_rounds': Cfg.B,
            'num_clients': Cfg.N,
            'min_samples': Cfg.MIN,
            'attack_config': {
                'label_flip_ratio': 0.3,
                'backdoor_trigger_size': 3,
                'byzantine_strategy': 'sign_flip',
                'scaling_factor': 2.0,
                'free_ride_rounds': 2,
                'sybil_clients': 2
            },
            'aggregation_method': 'rotation'  # fedavg | krum | trimmed_mean | rotation
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update Cfg class attributes
    # Make relative paths resolve from the repository root (not from the caller's CWD)
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    except Exception:
        base_dir = os.path.abspath(os.getcwd())

    data_dir = config.get('data_dir', Cfg.DATA)
    out_dir = config.get('output_dir', Cfg.OUT)
    try:
        if isinstance(data_dir, str) and data_dir and (not os.path.isabs(data_dir)):
            data_dir = os.path.join(base_dir, data_dir)
    except Exception:
        pass
    try:
        if isinstance(out_dir, str) and out_dir and (not os.path.isabs(out_dir)):
            out_dir = os.path.join(base_dir, out_dir)
    except Exception:
        pass

    Cfg.DATA = data_dir
    Cfg.OUT = out_dir
    Cfg.P.update(config.get('model_params', {}))
    Cfg.R = config.get('num_rounds', Cfg.R)
    Cfg.B = config.get('num_boost_rounds', Cfg.B)
    Cfg.N = config.get('num_clients', Cfg.N)
    Cfg.MIN = config.get('min_samples', Cfg.MIN)
    Cfg.AGG = config.get('aggregation_method', 'rotation')
    
    return config

