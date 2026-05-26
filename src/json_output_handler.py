"""
JSON Output Handler for Federated Learning Attack Simulation
Handles structured JSON output for all attack types with configurable detail levels.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

class AttackJSONOutputHandler:
    """Handles JSON output generation for all attack types."""
    
    def __init__(self, output_dir: str = "test_output"):
        """Initialize the JSON output handler.
        
        Args:
            output_dir: Directory to save JSON output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_filename(self, attack_type: str, timestamp: Optional[str] = None) -> str:
        """Generate a unique filename for the attack results.
        
        Args:
            attack_type: Type of attack (label_flip, scaling, sybil, etc.)
            timestamp: Optional timestamp string
            
        Returns:
            Generated filename
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{attack_type}_attack_results_{timestamp}.json"
    
    def save_attack_results(self, attack_data: Dict[str, Any], attack_type: str) -> str:
        """Save attack results to JSON file.
        
        Args:
            attack_data: Complete attack results data
            attack_type: Type of attack
            
        Returns:
            Path to saved JSON file
        """
        filename = self.generate_filename(attack_type)
        filepath = self.output_dir / filename
        
        # Add metadata
        attack_data['metadata'] = {
            'attack_type': attack_type,
            'generated_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(attack_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(filepath)
    
    def create_base_structure(self, attack_type: str, attack_params: Dict[str, Any], 
                            training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create base JSON structure common to all attacks.
        
        Args:
            attack_type: Type of attack
            attack_params: Attack configuration parameters
            training_results: Training results from federated learning
            
        Returns:
            Base JSON structure
        """
        # Some training loops return only round_logs/num_clients (not a concrete clients list).
        tr = training_results if isinstance(training_results, dict) else {}
        rr_logs = tr.get('round_logs', []) if isinstance(tr, dict) else []

        # num_rounds
        try:
            num_rounds = int(tr.get('num_rounds', 0) or 0)
        except Exception:
            num_rounds = 0
        if num_rounds <= 0:
            try:
                num_rounds = max(int(e.get('round', 0) or 0) for e in (rr_logs or []) if isinstance(e, dict))
            except Exception:
                num_rounds = 0

        # num_clients
        try:
            num_clients = int(tr.get('num_clients', 0) or 0)
        except Exception:
            num_clients = 0
        if num_clients <= 0:
            try:
                num_clients = len({str(e.get('client')) for e in (rr_logs or []) if isinstance(e, dict) and e.get('client') is not None})
            except Exception:
                num_clients = 0
        if num_clients <= 0:
            try:
                num_clients = int((attack_params or {}).get('num_clients', 0) or 0)
            except Exception:
                num_clients = 0

        # attacker_clients
        attacker_clients = []
        try:
            attacker_clients = tr.get('attacker_clients', []) or []
        except Exception:
            attacker_clients = []
        if not attacker_clients:
            try:
                attacker_clients = (attack_params or {}).get('attacker_clients') or (attack_params or {}).get('attacker_client') or []
            except Exception:
                attacker_clients = []
        if not attacker_clients:
            try:
                attacker_clients = sorted({e.get('client') for e in (rr_logs or []) if isinstance(e, dict) and bool(e.get('is_attacker', False)) and e.get('client') is not None}, key=lambda x: str(x))
            except Exception:
                attacker_clients = []

        # honest / attacker counts
        honest_clients = 0
        attacker_clients_count = 0
        clients_participated = 0
        try:
            clients_list = tr.get('clients', []) if isinstance(tr.get('clients', []), list) else []
        except Exception:
            clients_list = []
        if clients_list:
            clients_participated = len(clients_list)
            honest_clients = len([c for c in clients_list if not getattr(c, 'is_attacker', False)])
            attacker_clients_count = len([c for c in clients_list if getattr(c, 'is_attacker', False)])
        else:
            try:
                clients_participated = num_clients
            except Exception:
                clients_participated = 0
            try:
                attacker_clients_count = len(attacker_clients or [])
            except Exception:
                attacker_clients_count = 0
            honest_clients = max(0, clients_participated - attacker_clients_count)

        # training time
        try:
            total_time = float(tr.get('total_time', 0) or 0)
        except Exception:
            total_time = 0

        return {
            'attack_configuration': {
                'attack_type': attack_type,
                'parameters': self._clean_params(attack_params),
                'num_rounds': num_rounds,
                'num_clients': num_clients,
                'attacker_clients': attacker_clients
            },
            'training_summary': {
                'total_rounds_completed': num_rounds,
                'total_training_time': total_time,
                'clients_participated': clients_participated,
                'honest_clients': honest_clients,
                'attacker_clients_count': attacker_clients_count
            }
        }
    
    def _clean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and filter parameters for JSON output.
        
        Args:
            params: Raw parameters dictionary
            
        Returns:
            Cleaned parameters dictionary
        """
        # Remove non-serializable or sensitive parameters
        skip_keys = {'clients', 'server', 'model', 'data_loader', 'callback'}
        cleaned = {}
        
        for key, value in params.items():
            if key in skip_keys:
                continue
            
            # Convert non-serializable types
            if hasattr(value, '__dict__'):
                continue
            elif callable(value):
                continue
            else:
                cleaned[key] = value
                
        return cleaned

class LabelFlipJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Label Flip attacks."""
    
    def create_label_flip_output(self, attack_params: Dict[str, Any], 
                                training_results: Dict[str, Any],
                                evaluation_results: Dict[str, Any],
                                detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON output for Label Flip attacks (excluding detailed round analysis).
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            
        Returns:
            Complete Label Flip attack JSON structure
        """
        base_structure = self.create_base_structure('label_flip', attack_params, training_results)

        try:
            cfg_params = base_structure.get('attack_configuration', {}).get('parameters', {})
            if isinstance(cfg_params, dict):
                if 'flip_ratio' in cfg_params and 'flip_percent' in cfg_params:
                    try:
                        if float(cfg_params.get('flip_ratio')) == float(cfg_params.get('flip_percent')):
                            cfg_params.pop('flip_percent', None)
                        else:
                            cfg_params.pop('flip_percent', None)
                    except Exception:
                        cfg_params.pop('flip_percent', None)
                elif 'flip_percent' in cfg_params and 'flip_ratio' not in cfg_params:
                    cfg_params['flip_ratio'] = cfg_params.get('flip_percent')
                    cfg_params.pop('flip_percent', None)

                cfg_params.pop('poison_fraction', None)
                cfg_params.pop('target_labels', None)
        except Exception:
            pass

        # Extract evaluation metrics (clean vs attacked)
        clean = evaluation_results.get('clean_metrics', {}) if isinstance(evaluation_results, dict) else {}
        attacked = evaluation_results.get('attacked_metrics', {}) if isinstance(evaluation_results, dict) else {}
        drops_pct = evaluation_results.get('metric_drops_percent', {}) if isinstance(evaluation_results, dict) else {}

        def _safe_float_map(src: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k in keys:
                try:
                    out[k] = float(src.get(k, 0.0) or 0.0)
                except Exception:
                    out[k] = 0.0
            return out

        clean_perf = _safe_float_map(clean, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc'])
        attacked_perf = _safe_float_map(attacked, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc'])

        try:
            if clean_perf.get('balanced_accuracy', 0.0) == 0.0 and clean_perf.get('accuracy', 0.0) != 0.0:
                clean_perf['balanced_accuracy'] = float(clean_perf.get('accuracy', 0.0) or 0.0)
        except Exception:
            pass
        try:
            if attacked_perf.get('balanced_accuracy', 0.0) == 0.0 and attacked_perf.get('accuracy', 0.0) != 0.0:
                attacked_perf['balanced_accuracy'] = float(attacked_perf.get('accuracy', 0.0) or 0.0)
        except Exception:
            pass

        # Metric impact (percentage changes)
        def _pct_drop(clean_v: float, attacked_v: float) -> float:
            try:
                c = float(clean_v or 0.0)
                a = float(attacked_v or 0.0)
                if c == 0.0:
                    return 0.0
                return float((c - a) / c) * 100.0
            except Exception:
                return 0.0

        bal_acc_pct = _pct_drop(clean_perf.get('balanced_accuracy', 0.0), attacked_perf.get('balanced_accuracy', 0.0))
        acc_pct = _pct_drop(clean_perf.get('accuracy', 0.0), attacked_perf.get('accuracy', 0.0))
        pre_pct = _pct_drop(clean_perf.get('precision', 0.0), attacked_perf.get('precision', 0.0))
        rec_pct = _pct_drop(clean_perf.get('recall', 0.0), attacked_perf.get('recall', 0.0))
        f1_pct = _pct_drop(clean_perf.get('f1', 0.0), attacked_perf.get('f1', 0.0))
        auc_pct = _pct_drop(clean_perf.get('auc', 0.0), attacked_perf.get('auc', 0.0))

        primary_metric = 'accuracy'
        try:
            if clean_perf.get('balanced_accuracy', 0.0) != 0.0 or attacked_perf.get('balanced_accuracy', 0.0) != 0.0:
                primary_metric = 'balanced_accuracy'
        except Exception:
            primary_metric = 'accuracy'

        primary_drop_pct = float(bal_acc_pct) if primary_metric == 'balanced_accuracy' else float(acc_pct)

        metric_impact = {
            'primary_metric': str(primary_metric),
            'primary_metric_drop_percent': float(primary_drop_pct),
            'balanced_accuracy_drop_percent': float(bal_acc_pct),
            'accuracy_drop_percent': float(acc_pct),
            'precision_drop_percent': float(pre_pct),
            'recall_drop_percent': float(rec_pct),
            'f1_drop_percent': float(f1_pct),
            'auc_drop_percent': float(auc_pct),
            'primary_metric_drop': float(primary_drop_pct) / 100.0,
            'balanced_accuracy_drop': float(bal_acc_pct) / 100.0,
            'accuracy_drop': float(acc_pct) / 100.0,
            'precision_drop': float(pre_pct) / 100.0,
            'recall_drop': float(rec_pct) / 100.0,
            'f1_drop': float(f1_pct) / 100.0,
            'auc_drop': float(auc_pct) / 100.0,
        }

        # Detection summary
        high_risk = detection_results.get('high_risk_clients', []) if isinstance(detection_results, dict) else []
        detected_attackers: List[Any] = []
        primary_id: Any = None
        primary_risk: float = 0.0
        primary_conf: str = 'unknown'

        try:
            for cli in (high_risk or []):
                if isinstance(cli, dict):
                    cid_val = cli.get('client_id')
                    if cid_val is not None:
                        detected_attackers.append(cid_val)
        except Exception:
            detected_attackers = detected_attackers

        # Primary attacker profile
        try:
            if high_risk:
                cli0 = high_risk[0]
                if isinstance(cli0, dict):
                    primary_id = cli0.get('client_id', None)
                    try:
                        primary_risk = float(cli0.get('risk_score', 0.0) or 0.0)
                    except Exception:
                        primary_risk = 0.0
                    primary_conf = str(cli0.get('confidence', primary_conf) or primary_conf)
        except Exception:
            primary_id = primary_id

        if primary_id is None and detected_attackers:
            primary_id = detected_attackers[0]

        # Detection accuracy (primary key is 'detection_accuracy', with fallback to 'accuracy')
        try:
            if isinstance(detection_results, dict):
                det_acc = float(
                    detection_results.get('detection_accuracy',
                                          detection_results.get('accuracy', 0.0))
                    or 0.0
                )
            else:
                det_acc = 0.0
        except Exception:
            det_acc = 0.0

        # Impact severity (use precision/F1_attach)
        max_drop = max(abs(pre_pct), abs(f1_pct))
        if max_drop >= 40.0:
            severity = 'HIGH'
            impact_reason = 'extreme precision/f1 degradation consistent with severe label corruption'
        elif max_drop >= 15.0:
            severity = 'MODERATE'
            impact_reason = 'moderate precision/f1 degradation consistent with label noise amplification'
        else:
            severity = 'LOW'
            impact_reason = 'minor metric degradation consistent with a mild label flip setting'

        severity_reason = ''
        try:
            sev_parts = []
            sev_parts.append(f"precision_drop={abs(pre_pct):.1f}%")
            sev_parts.append(f"f1_drop={abs(f1_pct):.1f}%")
            if primary_metric == 'balanced_accuracy':
                sev_parts.append(f"balanced_accuracy_drop={abs(bal_acc_pct):.1f}%")
            else:
                sev_parts.append(f"accuracy_drop={abs(acc_pct):.1f}%")
            if abs(rec_pct) < 15.0 and abs(auc_pct) < 10.0:
                sev_parts.append("recall/auc remain comparatively stable")
            severity_reason = " • ".join(sev_parts)
        except Exception:
            severity_reason = ''

        # Report-style sections mirroring the console "LABEL FLIP ATTACK REPORT"
        attack_details = {
            'type': 'label_flip',
            'detected_attackers': detected_attackers,
            'detection_accuracy': det_acc,
            'confidence': primary_conf,
            'high_risk_clients': len(high_risk or []),
        }

        attacker_profile = {
            'client_id': primary_id,
            'risk_score': primary_risk,
            'signature_features': {
                'flipped_labels': True,
                'high_gradient_variance': True,
                'low_cosine_to_honest_centroid': True,
                'cumulative_performance_degradation': True,
            },
        }

        model_performance = {
            'clean': clean_perf,
            'attacked': attacked_perf,
        }

        impact_level = {
            'severity': severity,
            'reason': impact_reason,
            'severity_reason': severity_reason,
        }

        # Preserve attack configuration details for completeness
        flip_ratio = 0.0
        try:
            if isinstance(attack_params, dict):
                if attack_params.get('flip_ratio') is not None:
                    flip_ratio = float(attack_params.get('flip_ratio') or 0.0)
                elif attack_params.get('flip_percent') is not None:
                    flip_ratio = float(attack_params.get('flip_percent') or 0.0)
                elif attack_params.get('flip_labels_fraction') is not None:
                    flip_ratio = float(attack_params.get('flip_labels_fraction') or 0.0)
        except Exception:
            flip_ratio = 0.0
        attack_specifics = {
            'flip_ratio': flip_ratio,
            'drop_positive_fraction': attack_params.get('drop_positive_fraction', 0.0),
            'feature_noise_std': attack_params.get('feature_noise_std', 0.0),
            'label_flip_target': 'fraud_label',
        }

        poison_fraction_effective = 0.0
        try:
            rr_logs = training_results.get('round_logs', []) if isinstance(training_results, dict) else []
        except Exception:
            rr_logs = []
        try:
            atk_set = set(int(str(x)) for x in (base_structure.get('attack_configuration', {}).get('attacker_clients', []) or []))
        except Exception:
            atk_set = set()
        try:
            fr_changes = []
            for e in (rr_logs or []):
                if not isinstance(e, dict):
                    continue
                is_att = bool(e.get('is_attacker', False))
                try:
                    cid_int = int(str(e.get('client')))
                except Exception:
                    cid_int = None
                if cid_int is not None and cid_int in atk_set:
                    is_att = True
                if not is_att:
                    continue
                try:
                    dv = float(e.get('fraud_ratio_change', 0.0) or 0.0)
                except Exception:
                    dv = 0.0
                fr_changes.append(abs(dv))
            if fr_changes:
                poison_fraction_effective = float(sum(fr_changes) / len(fr_changes))
            elif flip_ratio:
                poison_fraction_effective = float(abs(flip_ratio))
        except Exception:
            poison_fraction_effective = float(abs(flip_ratio) if flip_ratio else 0.0)
        try:
            attack_specifics['poison_fraction_effective'] = float(poison_fraction_effective)
        except Exception:
            pass

        # Detection thresholds and stage decisions (probe vs final)
        probe_thr = 0.33
        configured_thr = None
        final_thr = 0.33
        try:
            if isinstance(detection_results, dict) and detection_results.get('threshold') is not None:
                probe_thr = float(detection_results.get('threshold'))
        except Exception:
            probe_thr = 0.33
        try:
            if isinstance(attack_params, dict) and attack_params.get('detection_threshold') is not None:
                configured_thr = float(attack_params.get('detection_threshold'))
            elif isinstance(detection_results, dict) and detection_results.get('detection_threshold') is not None:
                configured_thr = float(detection_results.get('detection_threshold'))
        except Exception:
            configured_thr = None
        try:
            probe_thr = float(max(0.0, min(1.0, probe_thr)))
        except Exception:
            probe_thr = 0.33
        try:
            if configured_thr is not None:
                configured_thr = float(max(0.0, min(1.0, float(configured_thr))))
        except Exception:
            configured_thr = None

        risk_map = {}
        try:
            if isinstance(detection_results, dict):
                rm = detection_results.get('risk_scores', {}) or {}
                if isinstance(rm, dict) and rm:
                    risk_map = dict(rm)
        except Exception:
            risk_map = {}
        if not risk_map:
            try:
                for cli in (high_risk or []):
                    if isinstance(cli, dict) and cli.get('client_id') is not None:
                        risk_map[str(cli.get('client_id'))] = float(cli.get('risk_score', 0.0) or 0.0)
            except Exception:
                pass

        high_risk_ids_probe = []
        high_risk_ids_final = []
        try:
            for cid, rv in (risk_map or {}).items():
                try:
                    rfv = float(rv)
                except Exception:
                    continue
                if rfv >= probe_thr:
                    high_risk_ids_probe.append(str(cid))
                if rfv >= final_thr:
                    high_risk_ids_final.append(str(cid))
        except Exception:
            high_risk_ids_probe = []
            high_risk_ids_final = []
        try:
            high_risk_ids_probe = sorted(list(dict.fromkeys(high_risk_ids_probe)), key=lambda s: str(s))
            high_risk_ids_final = sorted(list(dict.fromkeys(high_risk_ids_final)), key=lambda s: str(s))
        except Exception:
            pass

        try:
            max_risk = float(max([float(v) for v in (risk_map or {}).values()] or [0.0]))
        except Exception:
            max_risk = 0.0

        # If the detector produced flagged clients but the configured threshold would contradict
        # the verdict, treat the probe threshold as the effective final threshold used.
        # Preserve the configured value separately for transparency.
        try:
            if detected_attackers:
                final_thr = float(probe_thr)
            elif configured_thr is not None:
                final_thr = float(configured_thr)
            else:
                final_thr = float(probe_thr)
        except Exception:
            final_thr = float(probe_thr)

        try:
            probe_verdict = 'detected' if bool(high_risk_ids_probe) else 'not_detected'
        except Exception:
            probe_verdict = 'not_detected'
        try:
            final_verdict = 'detected' if bool(high_risk_ids_final) else 'not_detected'
        except Exception:
            final_verdict = 'not_detected'

        det_reason = ''
        try:
            if final_verdict == 'detected':
                det_reason = f"Risk score {max_risk:.2f} meets/exceeds the final detection threshold ({final_thr:.2f})."
            elif probe_verdict == 'detected':
                det_reason = (
                    f"Risk score {max_risk:.2f} meets the probe threshold ({probe_thr:.2f}) but remains below the final "
                    f"detection threshold ({final_thr:.2f}); treated as watchlist-level evidence."
                )
            else:
                det_reason = f"Risk score {max_risk:.2f} is below both probe ({probe_thr:.2f}) and final ({final_thr:.2f}) thresholds."
        except Exception:
            det_reason = ''

        base_structure.update({
            'report_title': 'LABEL FLIP ATTACK REPORT',
            'attack_details': attack_details,
            'attacker_profile': attacker_profile,
            'model_performance': model_performance,
            'metric_impact': metric_impact,
            'impact_level': impact_level,
            'attack_specifics': attack_specifics,
            'detection_results': {
                'primary_metric': str(primary_metric),
                'detection_threshold_probe': float(probe_thr),
                'detection_threshold_final': float(final_thr),
                'detection_threshold_configured': (float(configured_thr) if configured_thr is not None else None),
                'detection_threshold_used': float(final_thr),
                'detection_stage_results': {
                    'probe_verdict': str(probe_verdict),
                    'final_verdict': str(final_verdict)
                },
                'detection_threshold': float(final_thr),
                'clients_flagged': list(high_risk_ids_final),
                'risk_scores': risk_map,
                'detection_accuracy': float(det_acc),
                'final_risk_score': float(max_risk),
                'detection_result': str(final_verdict),
                'reasoning': str(det_reason)
            },
            'evaluation_summary': {
                'primary_metric': str(primary_metric),
                'clean_model_metrics': clean_perf,
                'attacked_model_metrics': attacked_perf,
                'metric_drops_percent': {
                    'primary_metric': float(primary_drop_pct),
                    'balanced_accuracy': float(bal_acc_pct),
                    'accuracy': float(acc_pct),
                    'precision': float(pre_pct),
                    'recall': float(rec_pct),
                    'f1': float(f1_pct),
                    'auc': float(auc_pct),
                }
            },
        })

        return base_structure

class ScalingJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Scaling attacks."""
    
    def create_scaling_output(self, attack_params: Dict[str, Any], 
                            training_results: Dict[str, Any],
                            evaluation_results: Dict[str, Any],
                            detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON output for Scaling attacks (excluding detailed round analysis).
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            
        Returns:
            Complete Scaling attack JSON structure
        """
        base_structure = self.create_base_structure('scaling', attack_params, training_results)

        try:
            cfg_params = base_structure.get('attack_configuration', {}).get('parameters', {})
            if isinstance(cfg_params, dict):
                for k in (
                    'flip_labels_fraction',
                    'poison_server_share_fraction',
                    'inject_false_positive_fraction',
                    'flip_percent',
                    'flip_ratio',
                    'drop_positive_fraction',
                    'feature_noise_std',
                    'scale_pos_weight_attacker',
                ):
                    cfg_params.pop(k, None)

                if 'scaling_factor' in cfg_params and 'base_scaling_factor' not in cfg_params:
                    cfg_params['base_scaling_factor'] = cfg_params.get('scaling_factor')
                    cfg_params.pop('scaling_factor', None)

                try:
                    jp = float(cfg_params.get('jitter_percent', 0.0) or 0.0)
                except Exception:
                    jp = 0.0
                if jp > 0.0:
                    cfg_params['jitter_enabled'] = True
        except Exception:
            pass

        # Extract evaluation metrics (clean vs attacked)
        clean = evaluation_results.get('clean_metrics', {}) if isinstance(evaluation_results, dict) else {}
        attacked = evaluation_results.get('attacked_metrics', {}) if isinstance(evaluation_results, dict) else {}
        drops_pct = evaluation_results.get('metric_drops_percent', {}) if isinstance(evaluation_results, dict) else {}

        def _safe_float_map(src: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k in keys:
                try:
                    out[k] = float(src.get(k, 0.0) or 0.0)
                except Exception:
                    out[k] = 0.0
            return out

        clean_perf = _safe_float_map(clean, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc'])
        attacked_perf = _safe_float_map(attacked, ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc'])

        try:
            if clean_perf.get('balanced_accuracy', 0.0) == 0.0 and clean_perf.get('accuracy', 0.0) != 0.0:
                clean_perf['balanced_accuracy'] = float(clean_perf.get('accuracy', 0.0) or 0.0)
        except Exception:
            pass
        try:
            if attacked_perf.get('balanced_accuracy', 0.0) == 0.0 and attacked_perf.get('accuracy', 0.0) != 0.0:
                attacked_perf['balanced_accuracy'] = float(attacked_perf.get('accuracy', 0.0) or 0.0)
        except Exception:
            pass

        # Metric drops (percentage)
        def _safe_pct(name: str) -> float:
            try:
                return float(drops_pct.get(name, 0.0) or 0.0)
            except Exception:
                return 0.0

        bal_acc_pct = _safe_pct('balanced_accuracy')
        acc_pct = _safe_pct('accuracy')
        if bal_acc_pct == 0.0 and acc_pct != 0.0:
            bal_acc_pct = float(acc_pct)
        pre_pct = _safe_pct('precision')
        rec_pct = _safe_pct('recall')
        f1_pct = _safe_pct('f1')
        auc_pct = _safe_pct('auc')

        metric_impact = {
            'balanced_accuracy_drop_percent': float(bal_acc_pct),
            'accuracy_drop_percent': acc_pct,
            'precision_drop_percent': pre_pct,
            'recall_drop_percent': rec_pct,
            'f1_drop_percent': f1_pct,
            'auc_drop_percent': auc_pct,
            'balanced_accuracy_drop': float(bal_acc_pct) / 100.0,
            'accuracy_drop': float(acc_pct) / 100.0,
            'precision_drop': float(pre_pct) / 100.0,
            'recall_drop': float(rec_pct) / 100.0,
            'f1_drop': float(f1_pct) / 100.0,
            'auc_drop': float(auc_pct) / 100.0,
        }

        # Detection summary
        high_risk = detection_results.get('high_risk_clients', []) if isinstance(detection_results, dict) else []
        detected_attackers: List[Any] = []
        try:
            for cli in (high_risk or []):
                if isinstance(cli, dict):
                    cid_val = cli.get('client_id')
                    if cid_val is not None:
                        detected_attackers.append(cid_val)
        except Exception:
            detected_attackers = detected_attackers

        # Fallback: if detector produced no high-risk clients, but we know attacker_clients and per-client
        # risk_scores, synthesize high-risk entries so detection section is not empty in JSON.
        if (not high_risk) and isinstance(detection_results, dict):
            try:
                rs_map = detection_results.get('risk_scores', {}) or {}
            except Exception:
                rs_map = {}
            fallback_list: List[Dict[str, Any]] = []
            atk_clients = training_results.get('attacker_clients', []) if isinstance(training_results, dict) else []
            for cid in atk_clients or []:
                try:
                    key_str = str(cid)
                    key_alt = cid
                    rsk = rs_map.get(key_str, rs_map.get(key_alt, 0.0))
                except Exception:
                    rsk = 0.0
                if rsk >= 0.8:
                    cstr = 'HIGH'
                elif rsk >= 0.5:
                    cstr = 'MEDIUM'
                elif rsk > 0.0:
                    cstr = 'LOW'
                else:
                    cstr = 'UNKNOWN'
                fallback_list.append({'client_id': cid, 'risk_score': rsk, 'confidence': cstr})
            if fallback_list:
                high_risk = fallback_list

        # Detection accuracy and confidence level
        try:
            det_acc = float(detection_results.get('detection_accuracy',
                                                  detection_results.get('accuracy', 0.0))
                            or 0.0) if isinstance(detection_results, dict) else 0.0
        except Exception:
            det_acc = 0.0

        conf_level = 'UNKNOWN'
        try:
            if high_risk:
                cli0 = high_risk[0]
                if isinstance(cli0, dict):
                    conf_level = str(cli0.get('confidence', conf_level) or conf_level).upper()
        except Exception:
            conf_level = conf_level

        # Impact severity based on F1 / Precision drops
        try:
            max_drop = max(abs(pre_pct), abs(f1_pct))
        except Exception:
            max_drop = 0.0
        if max_drop >= 40.0:
            severity = 'HIGH'
        elif max_drop >= 15.0:
            severity = 'MODERATE'
        else:
            severity = 'LOW'

        # Report-style sections mirroring the console "SCALING ATTACK — FEDERATED TRAINING SUMMARY"
        attack_information = {
            'attack_type': 'SCALING',
            'selected_attacker_clients': training_results.get('attacker_clients', []),
            'detection_accuracy': det_acc,
            'confidence_level': conf_level,
        }

        detection_block_clients: List[Dict[str, Any]] = []
        for cli in (high_risk or []):
            if not isinstance(cli, dict):
                continue
            cid = cli.get('client_id')
            try:
                cid_display = int(cid)
            except Exception:
                cid_display = cid
            try:
                rsk = float(cli.get('risk_score', 0.0) or 0.0)
            except Exception:
                rsk = 0.0
            detection_block_clients.append({
                'client_id': cid_display,
                'risk_score': rsk,
                'attack_signature': 'Large Update Magnitude + High Gradient Similarity',
                'behavior': 'Parameter Scaling Detected',
                'confidence': cli.get('confidence')
            })

        detection_block = {
            'high_risk_clients': len(high_risk or []),
            'clients': detection_block_clients,
            'threshold': 0.0,
        }

        det_thr = 0.0
        try:
            cfg_params = base_structure.get('attack_configuration', {}).get('parameters', {})
            if isinstance(cfg_params, dict) and cfg_params.get('detection_threshold') is not None:
                det_thr = float(cfg_params.get('detection_threshold') or 0.0)
        except Exception:
            det_thr = 0.0
        if det_thr == 0.0:
            try:
                if isinstance(detection_results, dict):
                    det_thr = float(detection_results.get('threshold', 0.0) or 0.0)
            except Exception:
                det_thr = 0.0
        detection_block['threshold'] = float(det_thr)

        evaluation_metrics = {
            'clean_model_performance': clean_perf,
            'attacked_model_performance': attacked_perf,
        }

        attack_impact_summary = {
            'bullets': [
                'Performance degradation due to scaled gradient updates',
                'Model convergence disturbed as attacker dominates aggregation',
                'Harmful effect observed mostly in Recall and F1 metrics',
            ],
            'severity_level': severity,
        }

        files_saved = {
            'detailed_logs': 'outputs/scaling_attack_round_logs.json',
        }

        # Preserve attack configuration details for completeness
        base_sf = 1.0
        try:
            if isinstance(attack_params, dict):
                if attack_params.get('scaling_factor') is not None:
                    base_sf = float(attack_params.get('scaling_factor') or 1.0)
                elif attack_params.get('base_scaling_factor') is not None:
                    base_sf = float(attack_params.get('base_scaling_factor') or 1.0)
        except Exception:
            base_sf = 1.0

        jitter_percent = 0.0
        try:
            jitter_percent = float(attack_params.get('jitter_percent', 0.0) or 0.0) if isinstance(attack_params, dict) else 0.0
        except Exception:
            jitter_percent = 0.0
        jitter_enabled = bool(attack_params.get('jitter_enabled', False)) if isinstance(attack_params, dict) else False
        if jitter_percent > 0.0:
            jitter_enabled = True

        attack_specifics = {
            'base_scaling_factor': float(base_sf),
            'scaling_strategy': attack_params.get('scaling_strategy', 'uniform'),
            'jitter_enabled': bool(jitter_enabled),
            'jitter_percent': float(jitter_percent),
            'jitter_range': [max(0.0, 1.0 - float(jitter_percent) / 100.0), 1.0 + float(jitter_percent) / 100.0] if jitter_percent > 0.0 else [1.0, 1.0],
        }

        # Backward-compatible evaluation_summary block
        impact_score = 0.0
        try:
            impact_score = float(max(abs(acc_pct), abs(pre_pct), abs(rec_pct), abs(f1_pct), abs(auc_pct))) / 100.0
        except Exception:
            impact_score = 0.0

        evaluation_summary = {
            'clean_model_metrics': clean,
            'attacked_model_metrics': attacked,
            'metric_drops': evaluation_results.get('metric_drops', {}) if isinstance(evaluation_results, dict) else {},
            'metric_drops_percentage': drops_pct,
            'attack_effectiveness_score': impact_score,
        }

        base_structure.update({
            'report_title': 'SCALING ATTACK — FEDERATED TRAINING SUMMARY',
            'attack_information': attack_information,
            'detection_results': detection_block,
            'evaluation_metrics': evaluation_metrics,
            'metric_impact': metric_impact,
            'attack_impact_summary': attack_impact_summary,
            'files_saved': files_saved,
            'attack_specifics': attack_specifics,
            'evaluation_summary': evaluation_summary,
        })

        return base_structure

class SybilJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Sybil attacks with full detail."""
    
    def create_sybil_output(self, attack_params: Dict[str, Any], 
                          training_results: Dict[str, Any],
                          evaluation_results: Dict[str, Any],
                          detection_results: Dict[str, Any],
                          round_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create JSON output for Sybil attacks (including all details).
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            round_logs: Detailed round-by-round logs
            
        Returns:
            Complete Sybil attack JSON structure with full details
        """
        base_structure = self.create_base_structure('sybil', attack_params, training_results)
        total_clients = 0
        try:
            total_clients = len({str(e.get('client')) for e in (round_logs or []) if isinstance(e, dict) and e.get('client') is not None})
        except Exception:
            total_clients = 0

        # Sanitize risk_scores: merge numeric and string ids into unique JSON keys
        raw_risk = {}
        try:
            if isinstance(detection_results, dict):
                raw_risk = detection_results.get('risk_scores', {}) or {}
        except Exception:
            raw_risk = {}
        clean_risk: Dict[str, float] = {}
        for k, v in getattr(raw_risk, 'items', lambda: [])():
            try:
                key_str = str(k)
            except Exception:
                key_str = repr(k)
            try:
                val = float(v)
            except Exception:
                try:
                    val = float(v or 0.0)
                except Exception:
                    val = 0.0
            if key_str in clean_risk:
                if val > clean_risk[key_str]:
                    clean_risk[key_str] = val
            else:
                clean_risk[key_str] = val

        # Resolve real attacker and sybil node naming from params and training context
        real_attacker = attack_params.get('real_attacker') if isinstance(attack_params, dict) else None
        sybil_count = attack_params.get('sybil_count', 0) if isinstance(attack_params, dict) else 0
        sybil_nodes = attack_params.get('sybil_nodes', []) if isinstance(attack_params, dict) else []
        if (real_attacker is None) and isinstance(training_results, dict):
            try:
                atk_clients = training_results.get('attacker_clients', []) or []
            except Exception:
                atk_clients = []
            if atk_clients:
                try:
                    real_attacker = int(atk_clients[0])
                except Exception:
                    real_attacker = atk_clients[0]
        if (not sybil_nodes) and real_attacker is not None and sybil_count:
            try:
                sybil_nodes = [f"{real_attacker}_s{i+1}" for i in range(int(sybil_count))]
            except Exception:
                sybil_nodes = sybil_nodes

        attacker_controlled: List[str] = []
        try:
            if real_attacker is not None:
                attacker_controlled.append(str(real_attacker))
        except Exception:
            attacker_controlled = []
        try:
            for sn in (sybil_nodes or []):
                attacker_controlled.append(str(sn))
        except Exception:
            attacker_controlled = attacker_controlled

        base_num_clients = None
        try:
            if isinstance(attack_params, dict) and attack_params.get('num_clients') is not None:
                base_num_clients = int(attack_params.get('num_clients'))
        except Exception:
            base_num_clients = None
        try:
            if total_clients > 0:
                base_structure.setdefault('attack_configuration', {})
                base_structure['attack_configuration']['num_clients'] = int(total_clients)
                params_block = base_structure.get('attack_configuration', {}).get('parameters', {})
                if isinstance(params_block, dict):
                    if base_num_clients is not None:
                        params_block.setdefault('base_num_clients', int(base_num_clients))
                    params_block['num_clients'] = int(total_clients)
                    base_structure['attack_configuration']['parameters'] = params_block
        except Exception:
            pass

        try:
            if attacker_controlled:
                base_structure.setdefault('attack_configuration', {})
                base_structure['attack_configuration']['attacker_clients'] = list(attacker_controlled)
        except Exception:
            pass
        try:
            if total_clients > 0:
                base_structure.setdefault('training_summary', {})
                base_structure['training_summary']['clients_participated'] = int(total_clients)
                base_structure['training_summary']['attacker_clients_count'] = int(len(attacker_controlled))
                base_structure['training_summary']['honest_clients'] = int(max(0, int(total_clients) - int(len(attacker_controlled))))
        except Exception:
            pass

        # Derive detection threshold and cluster decision in a way that is consistent with config + influence metrics.
        # Prefer detector's threshold; if missing/zero, fallback to configured detection_threshold.
        det_threshold = 0.0
        try:
            if isinstance(detection_results, dict):
                det_threshold = float(detection_results.get('threshold', 0.0) or 0.0)
        except Exception:
            det_threshold = 0.0
        if det_threshold == 0.0 and isinstance(attack_params, dict):
            try:
                det_threshold = float(attack_params.get('detection_threshold', 0.0) or 0.0)
            except Exception:
                det_threshold = 0.0

        # Derive effective influence dominance from aggregation_influence summary, when available
        eff_influence = None
        try:
            if isinstance(evaluation_results, dict):
                agg_inf = evaluation_results.get('aggregation_influence', {}) or {}
                if 'combined_influence_post_scaling' in agg_inf:
                    eff_influence = float(agg_inf.get('combined_influence_post_scaling'))
                elif 'sybil_influence_percent' in agg_inf:
                    eff_influence = float(agg_inf.get('sybil_influence_percent')) / 100.0
        except Exception:
            eff_influence = eff_influence

        # Final Sybil cluster decision for JSON: fall back to dominance-based decision if detector did not set it.
        sybil_cluster_detected_json: bool
        try:
            if isinstance(detection_results, dict) and 'sybil_cluster_detected' in detection_results:
                sybil_cluster_detected_json = bool(detection_results.get('sybil_cluster_detected', False))
            else:
                if eff_influence is not None:
                    thr = float(det_threshold or 0.0)
                    if thr > 0.0:
                        sybil_cluster_detected_json = bool(eff_influence >= thr)
                    else:
                        # Default rule-of-thumb: dominance >= 50% implies detection
                        sybil_cluster_detected_json = bool(eff_influence >= 0.50)
                else:
                    sybil_cluster_detected_json = False
        except Exception:
            sybil_cluster_detected_json = False

        # Synthesize flagged clients if detector did not provide them: use real attacker + sybil nodes
        flagged_clients = []
        try:
            if isinstance(detection_results, dict):
                flagged_clients = detection_results.get('flagged_clients', []) or []
        except Exception:
            flagged_clients = []
        if not flagged_clients:
            synth_flagged: List[Any] = []
            if real_attacker is not None:
                synth_flagged.append(str(real_attacker))
            for sn in sybil_nodes or []:
                synth_flagged.append(str(sn))
            flagged_clients = synth_flagged

        scaling_used = 1.0
        try:
            if isinstance(attack_params, dict) and attack_params.get('sybil_scaling_factor') is not None:
                scaling_used = float(attack_params.get('sybil_scaling_factor') or 1.0)
            elif isinstance(attack_params, dict) and attack_params.get('scaling_factor') is not None:
                scaling_used = float(attack_params.get('scaling_factor') or 1.0)
        except Exception:
            scaling_used = 1.0
        jitter_used = None
        try:
            syb_j = []
            for e in (round_logs or []):
                if not isinstance(e, dict):
                    continue
                try:
                    if bool(e.get('is_attacker', False)) and str(e.get('attack_type', '') or '').lower().find('sybil') >= 0:
                        if e.get('sybil_jitter_percent') is not None:
                            syb_j.append(abs(float(e.get('sybil_jitter_percent'))))
                except Exception:
                    continue
            if syb_j:
                jitter_used = float(max(syb_j))
        except Exception:
            jitter_used = None
        if jitter_used is None:
            try:
                if isinstance(attack_params, dict) and attack_params.get('jitter_percent') is not None:
                    jitter_used = float(attack_params.get('jitter_percent') or 0.0)
            except Exception:
                jitter_used = 0.0

        # Add Sybil specific data with full details
        base_structure.update({
            'attack_specifics': {
                'real_attacker': real_attacker,
                'sybil_count': sybil_count,
                'sybil_nodes': sybil_nodes,
                'scaling_factor': float(scaling_used),
                'jitter_percent': float(jitter_used) if jitter_used is not None else 0.0,
                'copy_cat_strategy': True
            },
            'round_by_round_analysis': self._create_sybil_round_analysis(round_logs, attack_params),
            'cluster_behavior': self._analyze_sybil_cluster_behavior(round_logs),
            'evaluation_summary': {
                'clean_model_metrics': evaluation_results.get('clean_metrics', {}) if isinstance(evaluation_results, dict) else {},
                'attacked_model_metrics': evaluation_results.get('attacked_metrics', {}) if isinstance(evaluation_results, dict) else {},
                'metric_drops': evaluation_results.get('metric_drops', {}) if isinstance(evaluation_results, dict) else {},
                'metric_drops_percentage': evaluation_results.get('metric_drops_percent', {}) if isinstance(evaluation_results, dict) else {},
                'logit_drift': evaluation_results.get('logit_drift', 0.0) if isinstance(evaluation_results, dict) else 0.0,
                'attack_success_rate': evaluation_results.get('attack_success_rate', 0.0) if isinstance(evaluation_results, dict) else 0.0,
                'sybil_impact_assessment': evaluation_results.get('impact_assessment', 'Unknown') if isinstance(evaluation_results, dict) else 'Unknown'
            },
            'detection_results': {
                'detection_threshold': float(det_threshold),
                'sybil_cluster_detected': bool(sybil_cluster_detected_json) if not flagged_clients else True,
                'cluster_signature': detection_results.get('cluster_signature', {}) if isinstance(detection_results, dict) else {},
                'clients_flagged': flagged_clients,
                **({
                    'top_risk_scores': clean_risk,
                    'risk_scores': {}
                } if (total_clients > 0 and isinstance(clean_risk, dict) and len(clean_risk) > 0 and len(clean_risk) < int(total_clients)) else {
                    'risk_scores': clean_risk
                }),
                **({
                    'high_risk_clients_count': int(len(detection_results.get('high_risk_clients', []) or []))
                } if isinstance(detection_results, dict) and detection_results.get('high_risk_clients') is not None else {}),
                'cosine_similarities': detection_results.get('cosine_similarities', {}) if isinstance(detection_results, dict) else {},
                'detection_accuracy': (
                    detection_results.get('detection_accuracy', detection_results.get('accuracy', 0.0))
                    if isinstance(detection_results, dict) else 0.0
                )
            },
            'aggregation_influence': evaluation_results.get('aggregation_influence', {}) if isinstance(evaluation_results, dict) else {}
        })
        
        return base_structure
    
    def _create_sybil_round_analysis(self, round_logs: List[Dict[str, Any]], 
                                   attack_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed round-by-round analysis for Sybil attacks.
        
        Args:
            round_logs: Round-by-round training logs
            attack_params: Attack parameters
            
        Returns:
            List of round analysis dictionaries
        """
        rounds_analysis = []
        rounds = attack_params.get('num_rounds', 5)
        
        for r in range(1, rounds + 1):
            round_data = [e for e in (round_logs or []) if isinstance(e, dict) and int(e.get('round', 0)) == r]

            # Find attacker and sybil data
            attacker_data = None
            sybil_data: List[Dict[str, Any]] = []

            def _is_sybil_id(cid: str) -> bool:
                cid_l = str(cid).lower()
                # Match explicit sybil naming (sybil_*, csybil_*, "sybil of client X")
                if 'sybil' in cid_l and ("sybil_" in cid_l or "csybil_" in cid_l or "sybil of client" in cid_l):
                    return True
                # Match numeric-root + "_sK" pattern (e.g., "1_s1", "3_s2") used by Sybil clients
                if "_s" in cid_l:
                    try:
                        root, _ = cid_l.split("_s", 1)
                        return root.isdigit()
                    except Exception:
                        return False
                return False

            for entry in round_data:
                client_id = str(entry.get('client', ''))
                if entry.get('is_attacker', False):
                    if _is_sybil_id(client_id):
                        sybil_data.append(entry)
                    else:
                        attacker_data = entry
            
            # Calculate round metrics
            round_analysis = {
                'round': r,
                'attacker_metrics': {
                    'update_norm': attacker_data.get('update_norm', 0.0) if attacker_data else 0.0,
                    'cosine_similarity': attacker_data.get('cosine_similarity', 0.0) if attacker_data else 0.0,
                    'risk_score': attacker_data.get('risk_score', 0.0) if attacker_data else 0.0
                },
                'sybil_metrics': [
                    {
                        'client_id': s.get('client', ''),
                        'update_norm': s.get('update_norm', 0.0),
                        'cosine_similarity': s.get('cosine_similarity', 0.0),
                        'risk_score': s.get('risk_score', 0.0)
                    } for s in sybil_data
                ],
                'cluster_correlation': self._calculate_cluster_correlation(attacker_data, sybil_data),
                # This is the *raw contribution share* based solely on attacker participation (pre-scaling).
                'raw_contribution_percent': self._calculate_influence_spike(round_data),
                'training_validation': {
                    'sybil_nodes_detected': len(sybil_data),
                    'norm_consistency': self._check_norm_consistency(attacker_data, sybil_data)
                }
            }
            
            rounds_analysis.append(round_analysis)
        
        return rounds_analysis
    
    def _analyze_sybil_cluster_behavior(self, round_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall Sybil cluster behavior across all rounds.
        
        Args:
            round_logs: Round-by-round training logs
            
        Returns:
            Cluster behavior analysis
        """
        all_similarities = []
        all_norms = []
        intra_similarities = []
        
        for entry in round_logs or []:
            if entry.get('is_attacker', False):
                sim = entry.get('cosine_to_global', entry.get('cosine_similarity', 0.0))
                norm = entry.get('update_norm', 0.0)
                try:
                    if sim is not None and float(sim) > 0:
                        all_similarities.append(float(sim))
                except Exception:
                    pass
                try:
                    if entry.get('cosine_to_sybil_cluster') is not None:
                        v = float(entry.get('cosine_to_sybil_cluster'))
                        if v > 0:
                            intra_similarities.append(v)
                except Exception:
                    pass
                if norm > 0:
                    all_norms.append(norm)
        
        avg_global = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
        avg_intra = sum(intra_similarities) / len(intra_similarities) if intra_similarities else 0.0

        return {
            'average_cosine_similarity': float(avg_intra) if intra_similarities else float(avg_global),
            'average_attacker_global_cosine': float(avg_global),
            'average_intra_cluster_cosine': float(avg_intra),
            'similarity_variance': self._calculate_variance(intra_similarities if intra_similarities else all_similarities),
            'average_update_norm': sum(all_norms) / len(all_norms) if all_norms else 0.0,
            'norm_variance': self._calculate_variance(all_norms),
            'cluster_consistency_score': self._calculate_consistency_score(intra_similarities if intra_similarities else all_similarities, all_norms)
        }
    
    def _calculate_cluster_correlation(self, attacker_data: Dict[str, Any], 
                                     sybil_data: List[Dict[str, Any]]) -> float:
        """Calculate correlation between attacker and Sybil nodes."""
        if not attacker_data and not sybil_data:
            return 0.0

        # Prefer the dedicated cluster-correlation metric logged during training when available.
        try:
            corr_vals: List[float] = []
            for e in ([attacker_data] if attacker_data else []) + list(sybil_data or []):
                if not isinstance(e, dict):
                    continue
                cc = e.get('sybil_cluster_correlation')
                if cc is not None:
                    try:
                        corr_vals.append(float(cc))
                    except Exception:
                        continue
            if corr_vals:
                return float(sum(corr_vals) / len(corr_vals))
        except Exception:
            pass

        # Fallback: approximate using cosine_similarity to the global direction.
        if not attacker_data or not sybil_data:
            return 0.0
        attacker_sim = attacker_data.get('cosine_similarity', 0.0)
        sybil_sims = [s.get('cosine_similarity', 0.0) for s in sybil_data]
        if not sybil_sims:
            return 0.0
        avg_sybil_sim = sum(sybil_sims) / len(sybil_sims)
        return float(min(attacker_sim, avg_sybil_sim))
    
    def _calculate_influence_spike(self, round_data: List[Dict[str, Any]]) -> float:
        """Calculate influence spike for the round."""
        attacker_count = sum(1 for e in round_data if e.get('is_attacker', False))
        total_count = len(round_data)
        
        if total_count == 0:
            return 0.0
        
        return (attacker_count / total_count) * 100.0
    
    def _check_norm_consistency(self, attacker_data: Dict[str, Any], 
                               sybil_data: List[Dict[str, Any]]) -> float:
        """Check norm consistency between attacker and Sybil nodes."""
        if not attacker_data or not sybil_data:
            return 0.0
        
        attacker_norm = attacker_data.get('update_norm', 0.0)
        sybil_norms = [s.get('update_norm', 0.0) for s in sybil_data]
        
        if not sybil_norms or attacker_norm == 0:
            return 0.0
        
        # Calculate similarity percentage
        similarities = []
        for sybil_norm in sybil_norms:
            if attacker_norm > 0:
                similarity = 1.0 - abs(sybil_norm - attacker_norm) / attacker_norm
                similarities.append(max(0.0, similarity))
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance
    
    def _calculate_consistency_score(self, similarities: List[float], norms: List[float]) -> float:
        """Calculate overall cluster consistency score."""
        if not similarities or not norms:
            return 0.0
        
        sim_consistency = 1.0 - self._calculate_variance(similarities)
        norm_consistency = 1.0 - (self._calculate_variance(norms) / max(norms) if max(norms) > 0 else 0)
        
        return (sim_consistency + norm_consistency) / 2.0

class ByzantineJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Byzantine attacks."""
    
    def create_byzantine_output(self, attack_params: Dict[str, Any], 
                              training_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any],
                              detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON output for Byzantine attacks.
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            
        Returns:
            Complete Byzantine attack JSON structure
        """
        base_structure = self.create_base_structure('byzantine', attack_params, training_results)
        
        # Add Byzantine specific data
        base_structure.update({
            'attack_specifics': {
                'byzantine_strategy': attack_params.get('byzantine_strategy', 'sign_flip'),
                'corruption_intensity': attack_params.get('corruption_intensity', 1.0),
                'targeted_parameters': attack_params.get('targeted_parameters', []),
                'corruption_pattern': attack_params.get('corruption_pattern', 'random')
            },
            'evaluation_summary': {
                'clean_model_metrics': evaluation_results.get('clean_metrics', {}),
                'attacked_model_metrics': evaluation_results.get('attacked_metrics', {}),
                'metric_drops': evaluation_results.get('metric_drops', {}),
                'metric_drops_percentage': evaluation_results.get('metric_drops_percent', {}),
                'attack_success_rate': evaluation_results.get('attack_success_rate', 0.0)
            },
            'detection_results': {
                'detection_threshold': detection_results.get('threshold', 0.0),
                'clients_flagged': detection_results.get('flagged_clients', []),
                'risk_scores': detection_results.get('risk_scores', {}),
                'detection_accuracy': detection_results.get('accuracy', 0.0),
                'byzantine_indicators': detection_results.get('byzantine_indicators', [])
            }
        })
        
        return base_structure

class BackdoorJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Backdoor attacks."""
    
    def create_backdoor_output(self, attack_params: Dict[str, Any], 
                             training_results: Dict[str, Any],
                             evaluation_results: Dict[str, Any],
                             detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON output for Backdoor attacks.
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            
        Returns:
            Complete Backdoor attack JSON structure
        """
        base_structure = self.create_base_structure('backdoor', attack_params, training_results)

        # Sanitize risk_scores for Backdoor: merge numeric and string ids into unique JSON keys
        raw_risk = {}
        try:
            if isinstance(detection_results, dict):
                raw_risk = detection_results.get('risk_scores', {}) or {}
        except Exception:
            raw_risk = {}
        clean_risk: Dict[str, float] = {}
        for k, v in getattr(raw_risk, 'items', lambda: [])():
            try:
                key_str = str(k)
            except Exception:
                key_str = repr(k)
            try:
                val = float(v)
            except Exception:
                try:
                    val = float(v or 0.0)
                except Exception:
                    val = 0.0
            if key_str in clean_risk:
                if val > clean_risk[key_str]:
                    clean_risk[key_str] = val
            else:
                clean_risk[key_str] = val

        tr = training_results if isinstance(training_results, dict) else {}
        rr_logs = tr.get('round_logs', []) if isinstance(tr, dict) else []
        asr_hist = tr.get('asr_history', []) if isinstance(tr, dict) else []

        # Prefer backdoor_info captured by the training loop (actual values used)
        bd_info = {}
        try:
            bd_info = tr.get('backdoor_info') or {}
        except Exception:
            bd_info = {}

        # Trigger pattern: ensure JSON-serializable keys
        trig = {}
        try:
            trig_raw = bd_info.get('trigger_features') or {}
            if isinstance(trig_raw, dict):
                trig = {str(k): v for k, v in trig_raw.items()}
        except Exception:
            trig = {}

        # Round-wise training summary (attacker-only view)
        rounds_summary = []
        try:
            rmax = 0
            try:
                rmax = int(base_structure.get('attack_configuration', {}).get('num_rounds', 0) or 0)
            except Exception:
                rmax = 0
            if rmax <= 0:
                rmax = max(int(e.get('round', 0) or 0) for e in (rr_logs or []) if isinstance(e, dict)) if rr_logs else 0
            atk_clients = base_structure.get('attack_configuration', {}).get('attacker_clients', []) or []
            for r in range(1, int(rmax) + 1):
                atk_logs = [e for e in (rr_logs or []) if isinstance(e, dict) and int(e.get('round', 0) or 0) == r and bool(e.get('is_attacker', False))]
                if atk_clients:
                    atk_logs = [e for e in atk_logs if str(e.get('client')) in {str(a) for a in atk_clients}]
                upd = atk_logs[0] if atk_logs else {}
                upd_norm = float(upd.get('update_norm', 0.0) or 0.0) if isinstance(upd, dict) else 0.0
                raw_upd_norm = float(upd.get('raw_update_norm', upd.get('update_norm_raw_l2', 0.0)) or 0.0) if isinstance(upd, dict) else 0.0
                norm_upd_norm = float(upd.get('normalized_update_norm', upd_norm) or 0.0) if isinstance(upd, dict) else 0.0
                cos = float(upd.get('cosine_similarity', 0.0) or 0.0) if isinstance(upd, dict) else 0.0
                cos_stab = float(upd.get('cosine_stability', 0.0) or 0.0) if isinstance(upd, dict) else 0.0
                pcount = int(upd.get('poisoned_samples', 0) or 0) if isinstance(upd, dict) else 0
                asr_abs_r = 0.0
                asr_inc_r = 0.0
                asr_eff_r = 0.0
                for h in (asr_hist or []):
                    try:
                        if int(h.get('round', 0) or 0) == int(r):
                            asr_abs_r = float(h.get('asr_absolute_percent', 0.0) or 0.0)
                            asr_inc_r = float(h.get('asr_incremental_percent', 0.0) or 0.0)
                            asr_eff_r = float(h.get('asr_effective_percent', 0.0) or 0.0)
                            break
                    except Exception:
                        continue
                rounds_summary.append({
                    'round': int(r),
                    'attacker_update_norm': float(norm_upd_norm),
                    'attacker_update_norm_raw': float(raw_upd_norm),
                    'attacker_cosine_similarity': cos,
                    'attacker_cosine_stability': float(cos_stab),
                    'poisoned_samples': pcount,
                    'asr_absolute_percent': float(asr_abs_r),
                    'asr_incremental_percent': float(asr_inc_r),
                    'asr_effective_percent': float(asr_eff_r),
                })
        except Exception:
            rounds_summary = rounds_summary

        # Backdoor signature (use training_results if available, else compute simple medians)
        signature = {}
        try:
            signature = tr.get('backdoor_signature') or {}
        except Exception:
            signature = {}
        if not signature:
            try:
                atk_all = [e for e in (rr_logs or []) if isinstance(e, dict) and bool(e.get('is_attacker', False))]
                vals_u = [float(e.get('update_norm', 0.0) or 0.0) for e in atk_all]
                vals_v = [float(e.get('param_variance', 0.0) or 0.0) for e in atk_all]
                vals_c = [float(e.get('cosine_similarity', 0.0) or 0.0) for e in atk_all]
                def _med(x):
                    try:
                        x = [float(v) for v in x if v is not None]
                        if not x:
                            return 0.0
                        x = sorted(x)
                        m = len(x) // 2
                        return float((x[m] if len(x) % 2 == 1 else (x[m-1] + x[m]) / 2.0))
                    except Exception:
                        return 0.0
                signature = {
                    'update_norm_median': _med(vals_u),
                    'param_variance_median': _med(vals_v),
                    'cosine_similarity_median': _med(vals_c),
                }
            except Exception:
                signature = {}

        # Evaluation summary and ASR panel details (when provided by interactive runner)
        asr_details = {}
        try:
            if isinstance(evaluation_results, dict):
                asr_details = evaluation_results.get('asr_details') or {}
        except Exception:
            asr_details = {}
        triggered_metrics = {}
        try:
            if isinstance(evaluation_results, dict):
                triggered_metrics = evaluation_results.get('triggered_metrics') or {}
        except Exception:
            triggered_metrics = {}

        # Resolve backdoor ASR summary (absolute only for final evaluation)
        asr_abs_final = 0.0
        asr_inc_final = 0.0
        asr_efficiency = None
        asr_clean_baseline_abs = 0.0
        try:
            if isinstance(bd_info, dict):
                asr_clean_baseline_abs = float(bd_info.get('asr_clean_baseline_absolute_percent', 0.0) or 0.0)
        except Exception:
            asr_clean_baseline_abs = 0.0
        try:
            if isinstance(asr_hist, (list, tuple)) and asr_hist:
                last = None
                try:
                    last = max((h for h in asr_hist if isinstance(h, dict)), key=lambda x: int(x.get('round', 0) or 0))
                except Exception:
                    last = asr_hist[-1] if isinstance(asr_hist[-1], dict) else None
                if isinstance(last, dict):
                    asr_abs_final = float(last.get('asr_absolute_percent', 0.0) or 0.0)
                    asr_inc_final = float(last.get('asr_incremental_percent', 0.0) or 0.0)
        except Exception:
            asr_abs_final = 0.0
            asr_inc_final = 0.0
        try:
            pr_eff = float((bd_info or {}).get('poison_ratio', 0.0) or 0.0)
            pr_eff = float(max(0.0, min(1.0, pr_eff)))
            if pr_eff > 0.0:
                asr_efficiency = float(asr_abs_final / pr_eff)
        except Exception:
            asr_efficiency = None

        base_structure.update({
            'attack_specifics': {
                'trigger_type': str((attack_params or {}).get('trigger_type', (attack_params or {}).get('backdoor_trigger', 'pixel_pattern'))),
                'trigger_description': bd_info.get('trigger_description', (attack_params or {}).get('trigger_description', 'Unknown trigger')),
                'target_label': bd_info.get('target_label', (attack_params or {}).get('target_label', 0)),
                'poison_ratio': bd_info.get('poison_ratio', (attack_params or {}).get('poison_ratio', (attack_params or {}).get('poison_fraction', 0.0))),
                'trigger_strength': bd_info.get('trigger_strength', (attack_params or {}).get('trigger_strength', 0.0)),
                'poison_fraction': (attack_params or {}).get('poison_fraction', (attack_params or {}).get('poison_ratio', 0.0)),
                'injected_samples': (attack_params or {}).get('injected_samples', 0),
                'trigger_pattern': trig,
            },
            'round_wise_training_summary': rounds_summary,
            'backdoor_signature': signature,
            'evaluation_summary': {
                'clean_model_metrics': evaluation_results.get('clean_metrics', {}) if isinstance(evaluation_results, dict) else {},
                'attacked_model_metrics': evaluation_results.get('attacked_metrics', {}) if isinstance(evaluation_results, dict) else {},
                'triggered_model_metrics': triggered_metrics,
                'metric_drops': evaluation_results.get('metric_drops', {}) if isinstance(evaluation_results, dict) else {},
                'metric_drops_percentage': evaluation_results.get('metric_drops_percent', {}) if isinstance(evaluation_results, dict) else {},
                'attack_success_rate': float(asr_abs_final),
                'asr_absolute_percent': float(asr_abs_final),
                'asr_clean_baseline_absolute_percent': float(asr_clean_baseline_abs),
                'asr_incremental_final_percent': float(asr_inc_final),
                'asr_efficiency': float(asr_efficiency) if asr_efficiency is not None else None,
                'asr_details': asr_details,
            },
            'detection_results': {
                'detection_threshold': detection_results.get('threshold', 0.0) if isinstance(detection_results, dict) else 0.0,
                'clients_flagged': detection_results.get('flagged_clients', []) if isinstance(detection_results, dict) else [],
                'risk_scores': clean_risk,
                'detection_accuracy': (
                    detection_results.get('detection_accuracy', detection_results.get('accuracy', 0.0))
                    if isinstance(detection_results, dict) else 0.0
                ),
                'backdoor_indicators': detection_results.get('backdoor_indicators', []) if isinstance(detection_results, dict) else []
            }
        })
        
        return base_structure

class FreeRideJSONHandler(AttackJSONOutputHandler):
    """Specialized handler for Free-Ride attacks."""
    
    def create_free_ride_output(self, attack_params: Dict[str, Any], 
                              training_results: Dict[str, Any],
                              evaluation_results: Dict[str, Any],
                              detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create JSON output for Free-Ride attacks.
        
        Args:
            attack_params: Attack configuration
            training_results: Training results
            evaluation_results: Evaluation metrics
            detection_results: Detection engine results
            
        Returns:
            Complete Free-Ride attack JSON structure
        """
        base_structure = self.create_base_structure('free_ride', attack_params, training_results)

        # Resolve Free-Ride productivity summary from evaluation or training context
        free_ride_summary = {}
        try:
            if isinstance(evaluation_results, dict):
                free_ride_summary = evaluation_results.get('free_ride_summary') or {}
        except Exception:
            free_ride_summary = {}
        if (not free_ride_summary) and isinstance(training_results, dict):
            try:
                free_ride_summary = training_results.get('free_ride_summary') or {}
            except Exception:
                free_ride_summary = {}

        # Derive behaviour_used from configuration and observed Free-Ride patterns
        behavior_used = attack_params.get('free_ride_strategy') or attack_params.get('style') or 'stale'
        try:
            tags: List[str] = [str(behavior_used)]
            zero_list = free_ride_summary.get('zero_update_clients') or []
            copy_list = free_ride_summary.get('copycat_clients') or []
            if zero_list:
                tags.append('zero_update')
            if copy_list:
                tags.append('copycat')
            seen: List[str] = []
            for t in tags:
                t = str(t).strip()
                if t and t not in seen:
                    seen.append(t)
            if seen:
                behavior_used = '+'.join(seen)
            else:
                behavior_used = str(behavior_used)

            # If attacker updates are clearly high-magnitude, avoid describing the
            # behaviour as "Zero-Update" and instead use a more accurate label.
            try:
                upd_series = free_ride_summary.get('update_norm_series', []) or []
                max_upd = max(float(v) for v in upd_series) if upd_series else 0.0
            except Exception:
                max_upd = 0.0
            if max_upd > 1.0:
                # High-magnitude stale reuse: override any zero-update phrasing
                behavior_used = 'Stale Model Reuse (High Magnitude)'
        except Exception:
            behavior_used = str(behavior_used)

        # Map Free-Ride specific detection block if present
        free_ride_det = {}
        try:
            if isinstance(detection_results, dict):
                free_ride_det = detection_results.get('free_ride_detection') or {}
        except Exception:
            free_ride_det = {}
        try:
            fr_final_risk = float(free_ride_det.get('final_risk_score', free_ride_det.get('max_risk_score', 0.0)) or 0.0)
        except Exception:
            fr_final_risk = 0.0
        fr_det_result = free_ride_det.get('overall_result', 'not detected')
        fr_severity = free_ride_det.get('severity_level', 'low')
        fr_reason = free_ride_det.get('reasoning', '')

        # Resolve thresholds:
        # - probe threshold: permissive heuristic used for early/warn-level screening
        # - final threshold: configured detection threshold used for the official verdict
        probe_thr = 0.33
        final_thr = 0.33
        try:
            if isinstance(detection_results, dict) and detection_results.get('threshold') is not None:
                probe_thr = float(detection_results.get('threshold'))
        except Exception:
            probe_thr = 0.33
        try:
            if isinstance(attack_params, dict) and attack_params.get('detection_threshold') is not None:
                final_thr = float(attack_params.get('detection_threshold'))
            elif isinstance(detection_results, dict) and detection_results.get('detection_threshold') is not None:
                final_thr = float(detection_results.get('detection_threshold'))
        except Exception:
            final_thr = 0.33
        try:
            probe_thr = float(max(0.0, min(1.0, probe_thr)))
        except Exception:
            probe_thr = 0.33
        try:
            final_thr = float(max(0.0, min(1.0, final_thr)))
        except Exception:
            final_thr = 0.33

        # Ensure we have a per-client risk map
        risk_map = {}
        try:
            if isinstance(detection_results, dict):
                rm = detection_results.get('risk_scores', {}) or {}
                if isinstance(rm, dict) and rm:
                    risk_map = dict(rm)
                if not risk_map:
                    hr_list = detection_results.get('high_risk_clients', []) or []
                    if isinstance(hr_list, list):
                        for cli in hr_list:
                            if isinstance(cli, dict) and cli.get('client_id') is not None:
                                risk_map[str(cli.get('client_id'))] = float(cli.get('risk_score', 0.0) or 0.0)
        except Exception:
            risk_map = {}

        # Compute thresholded lists for consistency: detected == risk >= threshold
        high_risk_ids_thr = []
        high_risk_ids_probe = []
        watchlist_ids = []
        try:
            for cid, rv in (risk_map or {}).items():
                try:
                    rfv = float(rv)
                except Exception:
                    continue
                if rfv >= final_thr:
                    high_risk_ids_thr.append(str(cid))
                if rfv >= probe_thr:
                    high_risk_ids_probe.append(str(cid))
                elif rfv >= 0.2:
                    watchlist_ids.append(str(cid))
        except Exception:
            high_risk_ids_thr = []
            high_risk_ids_probe = []
            watchlist_ids = []
        try:
            high_risk_ids_thr = sorted(list(dict.fromkeys(high_risk_ids_thr)), key=lambda s: str(s))
            high_risk_ids_probe = sorted(list(dict.fromkeys(high_risk_ids_probe)), key=lambda s: str(s))
            watchlist_ids = sorted(list(dict.fromkeys(watchlist_ids)), key=lambda s: str(s))
        except Exception:
            pass

        # Probe/final stage decisions
        try:
            probe_verdict = 'detected' if bool(high_risk_ids_probe) else 'not_detected'
        except Exception:
            probe_verdict = 'not_detected'
        try:
            final_verdict = 'detected' if bool(high_risk_ids_thr) else 'not_detected'
        except Exception:
            final_verdict = 'not_detected'

        # If detector didn't provide a Free-Ride block, derive decision/severity from risk+threshold
        try:
            if not isinstance(free_ride_det, dict) or not free_ride_det:
                try:
                    max_risk = float(max([float(v) for v in (risk_map or {}).values()] or [0.0]))
                except Exception:
                    max_risk = 0.0
                fr_det_result = 'detected' if bool(high_risk_ids_thr) else 'not detected'
                if fr_det_result == 'detected':
                    fr_severity = 'high' if max_risk >= 0.85 else ('medium' if max_risk >= max(final_thr, 0.5) else 'low')
                else:
                    fr_severity = 'low'
                fr_reason = str(fr_reason or '')
        except Exception:
            pass

        # Normalize detector-style strings and make reasoning explicit when missing
        try:
            fr_det_result = str(fr_det_result or '').strip().lower()
        except Exception:
            fr_det_result = ''
        if fr_det_result in ('not detected', 'not_detected', 'none', ''):
            fr_det_result = 'not_detected'
        elif fr_det_result in ('detected', 'yes', 'true'):
            fr_det_result = 'detected'
        else:
            fr_det_result = 'not_detected'

        try:
            fr_severity = str(fr_severity or '').strip().lower() or 'low'
        except Exception:
            fr_severity = 'low'

        if not str(fr_reason or '').strip():
            try:
                max_risk = float(max([float(v) for v in (risk_map or {}).values()] or [0.0]))
            except Exception:
                max_risk = 0.0
            if final_verdict == 'detected':
                fr_reason = f"Risk score {max_risk:.2f} meets/exceeds the configured detection threshold ({final_thr:.2f})."
            elif probe_verdict == 'detected':
                fr_reason = (
                    f"Risk score {max_risk:.2f} meets the probe threshold ({probe_thr:.2f}) but remains below the configured "
                    f"final detection threshold ({final_thr:.2f}); treated as watchlist-level evidence."
                )
            else:
                fr_reason = f"Risk score {max_risk:.2f} is below both probe ({probe_thr:.2f}) and final ({final_thr:.2f}) thresholds."

        # Extract snake_case productivity metrics and helper flags from summary
        try:
            eff_sc = float(free_ride_summary.get('effective_work_done', free_ride_summary.get('Effective_Work_Done', 0.0)) or 0.0)
        except Exception:
            eff_sc = 0.0
        try:
            st_sc = float(free_ride_summary.get('global_model_staleness', free_ride_summary.get('Global_Model_Staleness', 0.0)) or 0.0)
        except Exception:
            st_sc = 0.0
        try:
            loss_sc = float(free_ride_summary.get('productivity_loss_per_round', free_ride_summary.get('Productivity_Loss_Per_Round', 0.0)) or 0.0)
        except Exception:
            loss_sc = 0.0
        zero_clients_list = free_ride_summary.get('zero_update_clients') or []
        copy_clients_list = free_ride_summary.get('copycat_clients') or []
        copycat_detected = bool(copy_clients_list or free_ride_summary.get('Copied_Updates_Detected', 0))
        update_norm_series = free_ride_summary.get('update_norm_series', [])
        cosine_similarity_series = free_ride_summary.get('cosine_similarity_series', [])

        # Compose a concise human-readable effect message
        try:
            effect_message = (
                f"Free-Ride attack reduced honest contribution to approximately {eff_sc*100:.1f}% "
                f"with an average productivity loss per round of {loss_sc*100:.1f}%."
            )
        except Exception:
            effect_message = "Free-Ride attack impact summary unavailable due to missing metrics."

        # Add Free-Ride specific data
        base_structure.update({
            'attack_specifics': {
                'staleness_rounds': attack_params.get('staleness_rounds', 1),
                'free_ride_strategy': attack_params.get('free_ride_strategy', 'stale_model'),
                'participation_rate': attack_params.get('participation_rate', 1.0),
                'behavior_used': behavior_used
            },
            'evaluation_summary': {
                'clean_model_metrics': evaluation_results.get('clean_metrics', {}),
                'attacked_model_metrics': evaluation_results.get('attacked_metrics', {}),
                'metric_drops': evaluation_results.get('metric_drops', {}),
                'metric_drops_percentage': evaluation_results.get('metric_drops_percent', {}),
                'attack_success_rate': evaluation_results.get('attack_success_rate', 0.0)
            },
            'detection_results': {
                'detection_threshold_probe': float(probe_thr),
                'detection_threshold_final': float(final_thr),
                'detection_stage_results': {
                    'probe_verdict': str(probe_verdict),
                    'final_verdict': str(final_verdict)
                },
                'detection_threshold': float(final_thr),
                'clients_flagged': high_risk_ids_thr,
                'risk_scores': risk_map,
                'detection_accuracy': (
                    float(detection_results.get('detection_accuracy', detection_results.get('accuracy', 0.0)) or 0.0)
                    if isinstance(detection_results, dict) else 0.0
                ),
                'staleness_indicators': detection_results.get('staleness_indicators', []) if isinstance(detection_results, dict) else [],
                # Free-Ride specific detection mapping
                'final_risk_score': float(max([float(v) for v in (risk_map or {}).values()] or [0.0])),
                'detection_result': str(final_verdict),
                'detection_severity': fr_severity,
                'reasoning': fr_reason,
                'free_ride_detection': (
                    free_ride_det if isinstance(free_ride_det, dict) and free_ride_det else {
                        'overall_result': str(final_verdict),
                        'final_risk_score': float(max([float(v) for v in (risk_map or {}).values()] or [0.0])),
                        'probe_threshold': float(probe_thr),
                        'final_threshold': float(final_thr),
                        'probe_verdict': str(probe_verdict),
                        'final_verdict': str(final_verdict),
                        'reasoning': str(fr_reason)
                    }
                )
            },
            'free_ride_summary': {
                'effective_work_done': eff_sc,
                'global_model_staleness': st_sc,
                'productivity_loss_per_round': loss_sc,
                'zero_update_clients': zero_clients_list,
                'copycat_clients': copy_clients_list,
                'copycat_detected': copycat_detected,
                'update_norm_series': update_norm_series,
                'cosine_similarity_series': cosine_similarity_series,
                'message': effect_message,
                'legacy_fields': {
                    'Effective_Work_Done': free_ride_summary.get('Effective_Work_Done', eff_sc),
                    'Global_Model_Staleness': free_ride_summary.get('Global_Model_Staleness', st_sc),
                    'Productivity_Loss_Per_Round': free_ride_summary.get('Productivity_Loss_Per_Round', loss_sc),
                    'Zero_Update_Clients': free_ride_summary.get('Zero_Update_Clients', len(zero_clients_list)),
                    'Copied_Updates_Detected': free_ride_summary.get('Copied_Updates_Detected', len(copy_clients_list))
                }
            }
        })

        try:
            attacker_clients = training_results.get('attacker_clients', []) if isinstance(training_results, dict) else []
        except Exception:
            attacker_clients = []
        if not attacker_clients:
            try:
                attacker_clients = attack_params.get('attacker_clients', []) if isinstance(attack_params, dict) else []
            except Exception:
                attacker_clients = []

        try:
            rounds = int(training_results.get('num_rounds', 0) or 0) if isinstance(training_results, dict) else 0
        except Exception:
            rounds = 0
        try:
            round_logs = training_results.get('round_logs', []) if isinstance(training_results, dict) else []
        except Exception:
            round_logs = []

        try:
            atk_set = set(int(str(c)) for c in (attacker_clients or []))
        except Exception:
            atk_set = set()

        rr_map = {}
        for e in (round_logs or []):
            if not isinstance(e, dict):
                continue
            try:
                r = int(e.get('round', 0))
            except Exception:
                r = 0
            if r <= 0:
                continue
            cid = e.get('client')
            is_att = bool(e.get('is_attacker', False))
            try:
                cid_int = int(str(cid))
            except Exception:
                cid_int = None
            if cid_int is not None and cid_int in atk_set:
                is_att = True
            if not is_att:
                continue
            rr_map.setdefault(r, []).append(e)

        round_wise = []
        for r in sorted(rr_map.keys()):
            entries = rr_map.get(r, [])
            u_vals = []
            c_vals = []
            v_vals = []
            s_vals = []
            for e in entries:
                try:
                    if e.get('update_norm_capped_for_display') is not None:
                        u_vals.append(float(e.get('update_norm_capped_for_display') or 0.0))
                    else:
                        u_vals.append(float(e.get('update_norm', 0.0) or 0.0))
                except Exception:
                    pass
                try:
                    c_vals.append(float(e.get('cosine_similarity', 0.0) or 0.0))
                except Exception:
                    pass
                try:
                    v_vals.append(float(e.get('param_variance', 0.0) or 0.0))
                except Exception:
                    pass
                try:
                    s_vals.append(float(e.get('staleness', 0.0) or 0.0))
                except Exception:
                    pass
            def _m(x):
                return (sum(x) / len(x)) if x else 0.0
            upd_mean = _m(u_vals)
            cos_mean = _m(c_vals)
            var_mean = _m(v_vals)
            st_mean = _m(s_vals)
            is_zero = (upd_mean <= 1e-3 or var_mean <= 1e-6)
            is_copy = (cos_mean >= 0.98 and upd_mean <= 1.0)
            round_wise.append({
                'round': int(r),
                'update_norm': float(upd_mean),
                'cosine': float(cos_mean),
                'variance': float(var_mean),
                'staleness': float(st_mean),
                'zero_update_detected': bool(is_zero),
                'copycat_detected': bool(is_copy)
            })

        main_client = None
        try:
            if attacker_clients:
                main_client = sorted([int(str(c)) for c in attacker_clients])[0]
        except Exception:
            main_client = None
        sig_upd = []
        sig_var = []
        sig_cos = []
        sig_st = []
        copy_flags = []
        for e in (round_logs or []):
            if not isinstance(e, dict):
                continue
            try:
                cid_int = int(str(e.get('client')))
            except Exception:
                cid_int = None
            if main_client is None or cid_int != main_client:
                continue
            try:
                u = float(e.get('update_norm', 0.0) or 0.0)
                sig_upd.append(u)
            except Exception:
                pass
            try:
                v = float(e.get('param_variance', 0.0) or 0.0)
                sig_var.append(v)
            except Exception:
                pass
            try:
                cs = float(e.get('cosine_similarity', 0.0) or 0.0)
                sig_cos.append(cs)
            except Exception:
                pass
            try:
                st = float(e.get('staleness', 0.0) or 0.0)
                sig_st.append(st)
            except Exception:
                pass
            try:
                copy_flags.append(1.0 if (cs >= 0.98 and u <= 1.0) else 0.0)
            except Exception:
                pass
        def _mean(x):
            return (sum(x) / len(x)) if x else 0.0
        copy_score = _mean(copy_flags)
        risk_score = 0.0
        try:
            high_risk = detection_results.get('high_risk_clients', []) if isinstance(detection_results, dict) else []
            if isinstance(high_risk, list):
                for cli in high_risk:
                    if isinstance(cli, dict):
                        try:
                            rid = int(str(cli.get('client_id')))
                        except Exception:
                            rid = None
                        if rid is not None and rid == main_client:
                            risk_score = float(cli.get('risk_score', 0.0) or 0.0)
                            break
            if risk_score == 0.0:
                rs_map = detection_results.get('risk_scores', {}) if isinstance(detection_results, dict) else {}
                if isinstance(rs_map, dict) and main_client is not None:
                    key1 = f"Client_{main_client}"
                    key2 = str(main_client)
                    if key1 in rs_map:
                        try:
                            risk_score = float(rs_map.get(key1, 0.0) or 0.0)
                        except Exception:
                            pass
                    elif key2 in rs_map:
                        try:
                            risk_score = float(rs_map.get(key2, 0.0) or 0.0)
                        except Exception:
                            pass
        except Exception:
            risk_score = 0.0

        fr_det = {}
        try:
            fr_det = detection_results.get('free_ride_detection', {}) if isinstance(detection_results, dict) else {}
        except Exception:
            fr_det = {}
        high_risk_ids = list(high_risk_ids_thr)
        det_reason = ''
        try:
            det_reason = str(fr_det.get('reasoning', ''))
        except Exception:
            det_reason = ''

        clean_metrics = evaluation_results.get('clean', evaluation_results.get('clean_metrics', {})) if isinstance(evaluation_results, dict) else {}
        attacked_metrics = evaluation_results.get('attacked', evaluation_results.get('attacked_metrics', {})) if isinstance(evaluation_results, dict) else {}
        drops = evaluation_results.get('delta', evaluation_results.get('metric_drops_percentage', {})) if isinstance(evaluation_results, dict) else {}

        # Impact-based success flag for Free-Ride (ASR is backdoor-specific; Free-Ride is impact-based)
        try:
            mdp = evaluation_results.get('metric_drops_percent', {}) if isinstance(evaluation_results, dict) else {}
        except Exception:
            mdp = {}
        try:
            acc_dp = abs(float(mdp.get('accuracy', 0.0) or 0.0)) / 100.0
        except Exception:
            acc_dp = 0.0
        try:
            f1_dp = abs(float(mdp.get('f1', 0.0) or 0.0)) / 100.0
        except Exception:
            f1_dp = 0.0
        try:
            auc_dp = abs(float(mdp.get('auc', 0.0) or 0.0)) / 100.0
        except Exception:
            auc_dp = 0.0
        try:
            min_acc = float(attack_params.get('min_accuracy_drop', 0.03) or 0.03)
        except Exception:
            min_acc = 0.03
        try:
            min_f1 = float(attack_params.get('min_f1_drop', 0.05) or 0.05)
        except Exception:
            min_f1 = 0.05
        try:
            min_auc = float(attack_params.get('min_auc_drop', 0.005) or 0.005)
        except Exception:
            min_auc = 0.005
        impact_success = 1.0 if (acc_dp >= min_acc or f1_dp >= min_f1 or auc_dp >= min_auc) else 0.0

        impact_severity = 'LOW'
        try:
            max_drop = float(max([acc_dp, f1_dp, auc_dp] or [0.0]))
        except Exception:
            max_drop = 0.0
        if max_drop >= 0.20:
            impact_severity = 'HIGH'
        elif max_drop >= 0.10:
            impact_severity = 'MODERATE'

        try:
            base_structure.get('evaluation_summary', {})['attack_success_rate'] = float(impact_success)
            base_structure.get('evaluation_summary', {})['attack_success_definition'] = (
                "impact_based: success if |drop| crosses configured min_*_drop thresholds; "
                "Free-Ride does not use backdoor ASR"
            )
            base_structure.get('evaluation_summary', {})['impact_severity'] = str(impact_severity)
        except Exception:
            pass

        frontend_summary = {
            'attack_type': 'FREE_RIDE',
            'attacker_clients': attacker_clients,
            'behavior': behavior_used,
            'rounds': rounds,
            'round_wise_behavior': round_wise,
            'client_signature': {
                'client_id': main_client,
                'update_norm': _mean(sig_upd),
                'param_variance': _mean(sig_var),
                'cosine_similarity': _mean(sig_cos),
                'staleness_score': _mean(sig_st),
                'copycat_score': copy_score,
                'risk_score': risk_score
            },
            'detection_engine_results': {
                'detection_threshold_probe': float(probe_thr),
                'detection_threshold_final': float(final_thr),
                'decision_probe': str(probe_verdict),
                'decision_final': str(final_verdict),
                'high_risk_free_riders': high_risk_ids,
                'watchlist_free_riders': watchlist_ids,
                'reason': det_reason
            },
            'evaluation_comparison': {
                'clean_model': clean_metrics,
                'attacked_model': attacked_metrics,
                'metric_drops': drops
            },
            'productivity_loss': {
                'effective_work_done': eff_sc,
                'global_model_staleness': st_sc,
                'productivity_loss_per_round': loss_sc
            },
            'graphs': {
                'update_norm_series': update_norm_series,
                'cosine_similarity_series': cosine_similarity_series
            },
            'narrative_card': [
                (
                    'Free Ride attack detected: one or more clients exceeded the detection threshold.'
                    if fr_det_result == 'detected'
                    else 'Free Ride attack not detected at the configured threshold; moderate risk signals may be present below threshold.'
                ),
                'Inactive or stale updates can reduce productivity and degrade model performance without crossing the detection threshold.',
                'Review risk score vs threshold and productivity loss together for interpretation.'
            ],
            'display_guidelines': {
                'do_not_show': [
                    'Staleness 1.0 in rounds where cosine is low.',
                    'Interpretations for rounds with high cosine but unrelated staleness.'
                ]
            }
        }

        base_structure['frontend_summary'] = frontend_summary
        
        return base_structure

# Factory function to get appropriate handler
def get_json_handler(attack_type: str, output_dir: str = "test_output") -> AttackJSONOutputHandler:
    """Get appropriate JSON handler for attack type.
    
    Args:
        attack_type: Type of attack
        output_dir: Output directory for JSON files
        
    Returns:
        Appropriate JSON handler instance
    """
    attack_type_lower = attack_type.lower()
    
    if 'label_flip' in attack_type_lower or 'label flip' in attack_type_lower:
        return LabelFlipJSONHandler(output_dir)
    elif 'scaling' in attack_type_lower:
        return ScalingJSONHandler(output_dir)
    elif 'sybil' in attack_type_lower:
        return SybilJSONHandler(output_dir)
    elif 'byzantine' in attack_type_lower:
        return ByzantineJSONHandler(output_dir)
    elif 'backdoor' in attack_type_lower:
        return BackdoorJSONHandler(output_dir)
    elif ('free_ride' in attack_type_lower) or ('free ride' in attack_type_lower) or ('free-ride' in attack_type_lower):
        return FreeRideJSONHandler(output_dir)
    else:
        # Default handler for other attack types
        return AttackJSONOutputHandler(output_dir)
