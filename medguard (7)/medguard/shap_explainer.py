"""
shap_explainer.py — Real SHAP values from trained model.
Uses cached explainer for fast response.
"""

import os
import pickle
import numpy as np

# Cache SHAP explainer so it's built only once
_EXPLAINER_CACHE = None
_SCALER_CACHE    = None

FEATURE_NAMES = [
    'Protocol_enc','Source_enc','Destination_enc',
    'pkt_size','time_delta','info_len'
]

# Human-readable labels
FEATURE_LABELS = {
    'Protocol_enc':    {'name':'Protocol Type',       'desc':'Communication protocol used (TCP/UDP/ICMP)'},
    'Source_enc':      {'name':'Source Address',       'desc':'Where the network traffic originated from'},
    'Destination_enc': {'name':'Destination Address',  'desc':'Where the traffic was being sent to'},
    'pkt_size':        {'name':'Packet Size',          'desc':'How large the data packets were'},
    'time_delta':      {'name':'Packet Timing',        'desc':'Time gap between consecutive packets'},
    'info_len':        {'name':'Message Length',       'desc':'Length of the network message content'},
}

# Attack sample inputs per alert type
ATTACK_SAMPLES = {
    'ddos':      [2, 10, 5, 1024, 0.0001, 20],
    'port_scan': [1, 8,  3, 512,  0.002,  45],
    'arp':       [3, 15, 7, 42,   0.05,   15],
    'dos':       [1, 6,  4, 768,  0.001,  30],
    'smurf':     [2, 9,  6, 2048, 0.0001, 18],
    # fallback ids
    'bp':        [1, 10, 5, 512,  0.002,  45],
    'ins':       [1, 12, 5, 128,  0.150,  30],
    'xray':      [2, 8,  3, 1500, 0.001,  60],
}


def _load_explainer():
    """Load and cache SHAP TreeExplainer from best trained model."""
    global _EXPLAINER_CACHE, _SCALER_CACHE

    if _EXPLAINER_CACHE is not None:
        return _EXPLAINER_CACHE, _SCALER_CACHE

    scaler_path = 'models/scaler.pkl'
    if not os.path.exists(scaler_path):
        return None, None

    with open(scaler_path, 'rb') as f:
        _SCALER_CACHE = pickle.load(f)

    # Try to load best model for SHAP
    for fname in ['random_forest.pkl','xgboost.pkl','catboost.pkl']:
        path = os.path.join('models', fname)
        if os.path.exists(path):
            try:
                import shap
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                _EXPLAINER_CACHE = shap.TreeExplainer(model)
                print(f"[MedGuard] SHAP explainer loaded from {fname}")
                return _EXPLAINER_CACHE, _SCALER_CACHE
            except Exception as e:
                print(f"[MedGuard] SHAP load error ({fname}): {e}")
                continue

    return None, None


def get_shap_values(alert_id):
    """
    Returns SHAP explanation for a given alert.
    Uses real trained model if available, else static fallback.
    """
    explainer, scaler = _load_explainer()

    if explainer is not None and scaler is not None:
        try:
            return _real_shap(alert_id, explainer, scaler)
        except Exception as e:
            print(f"[MedGuard] SHAP compute error: {e}")

    return _static_shap(alert_id)


def _real_shap(alert_id, explainer, scaler):
    """Compute real SHAP values using trained model."""
    sample = ATTACK_SAMPLES.get(alert_id, ATTACK_SAMPLES['port_scan'])
    X = np.array(sample).reshape(1, -1)
    X_scaled = scaler.transform(X)

    shap_vals = explainer.shap_values(X_scaled)

    # Handle both binary and multi-class output
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        vals = shap_vals[0]

    features = []
    for fname, val in zip(FEATURE_NAMES, vals):
        label = FEATURE_LABELS.get(fname, {'name': fname, 'desc': ''})
        features.append({
            "feature":    fname,
            "label":      label['name'],
            "desc":       label['desc'],
            "value":      round(float(val), 4),
            "direction":  "pos" if val > 0 else "neg",
        })

    # Sort by absolute importance
    features.sort(key=lambda x: abs(x['value']), reverse=True)

    info = _alert_info(alert_id)
    return {
        "device":     info['device'],
        "attack":     info['attack'],
        "model":      info['model'],
        "confidence": info['confidence'],
        "source":     "real",
        "features":   features,
    }


def _static_shap(alert_id):
    """Fallback static SHAP values with proper labels."""
    data = {
        "ddos": {
            "device":"BP Monitor — OPD-11","attack":"DDoS Attack","model":"Random Forest","confidence":92,
            "features":[
                {"feature":"pkt_size",      "label":"Packet Size",       "desc":"How large the data packets were",                 "value":0.55,"direction":"pos"},
                {"feature":"Protocol_enc",  "label":"Protocol Type",     "desc":"Communication protocol used (TCP/UDP/ICMP)",       "value":0.38,"direction":"pos"},
                {"feature":"time_delta",    "label":"Packet Timing",     "desc":"Time gap between consecutive packets",             "value":0.22,"direction":"pos"},
                {"feature":"info_len",      "label":"Message Length",    "desc":"Length of the network message content",           "value":0.11,"direction":"pos"},
                {"feature":"Source_enc",    "label":"Source Address",    "desc":"Where the network traffic originated from",       "value":-0.07,"direction":"neg"},
                {"feature":"Destination_enc","label":"Destination Address","desc":"Where the traffic was being sent to",           "value":-0.15,"direction":"neg"},
            ]
        },
        "port_scan": {
            "device":"X-Ray Machine — Radiology","attack":"Port Scan","model":"XGBoost","confidence":87,
            "features":[
                {"feature":"Destination_enc","label":"Destination Address","desc":"Connecting to many unusual destinations",       "value":0.42,"direction":"pos"},
                {"feature":"pkt_size",       "label":"Packet Size",       "desc":"Suspiciously small packets — typical of scans", "value":0.31,"direction":"pos"},
                {"feature":"time_delta",     "label":"Packet Timing",     "desc":"Very rapid back-to-back connections",            "value":0.19,"direction":"pos"},
                {"feature":"Protocol_enc",   "label":"Protocol Type",     "desc":"Protocol matches known scan patterns",           "value":0.14,"direction":"pos"},
                {"feature":"Source_enc",     "label":"Source Address",    "desc":"Known safe source — reduces risk",               "value":-0.08,"direction":"neg"},
                {"feature":"info_len",       "label":"Message Length",    "desc":"Message length within normal range",             "value":-0.12,"direction":"neg"},
            ]
        },
        "arp": {
            "device":"Insulin Pump — Ward B","attack":"ARP Spoofing","model":"CatBoost","confidence":61,
            "features":[
                {"feature":"Protocol_enc",   "label":"Protocol Type",     "desc":"ARP protocol used unexpectedly",                 "value":0.48,"direction":"pos"},
                {"feature":"Source_enc",     "label":"Source Address",    "desc":"Unrecognised source MAC address",                "value":0.33,"direction":"pos"},
                {"feature":"pkt_size",       "label":"Packet Size",       "desc":"Packet size matches ARP spoof pattern",          "value":0.21,"direction":"pos"},
                {"feature":"time_delta",     "label":"Packet Timing",     "desc":"Normal timing — reduces suspicion slightly",     "value":-0.06,"direction":"neg"},
                {"feature":"info_len",       "label":"Message Length",    "desc":"Message length appears normal",                  "value":-0.09,"direction":"neg"},
                {"feature":"Destination_enc","label":"Destination Address","desc":"Broadcast address — slightly reduces score",    "value":-0.04,"direction":"neg"},
            ]
        },
        "dos": {
            "device":"ECG Monitor — ICU-03","attack":"DoS Attack","model":"Random Forest","confidence":78,
            "features":[
                {"feature":"time_delta",     "label":"Packet Timing",     "desc":"Extremely rapid packet flood detected",          "value":0.51,"direction":"pos"},
                {"feature":"pkt_size",       "label":"Packet Size",       "desc":"Large repeated packets overloading device",      "value":0.34,"direction":"pos"},
                {"feature":"Source_enc",     "label":"Source Address",    "desc":"Traffic from a suspicious external source",      "value":0.18,"direction":"pos"},
                {"feature":"Protocol_enc",   "label":"Protocol Type",     "desc":"Protocol matches DoS attack signature",          "value":0.12,"direction":"pos"},
                {"feature":"Destination_enc","label":"Destination Address","desc":"Target is a known critical device",             "value":-0.05,"direction":"neg"},
                {"feature":"info_len",       "label":"Message Length",    "desc":"Message content within expected bounds",         "value":-0.09,"direction":"neg"},
            ]
        },
        "smurf": {
            "device":"Ventilator — ICU-07","attack":"Smurf Attack","model":"XGBoost","confidence":95,
            "features":[
                {"feature":"pkt_size",       "label":"Packet Size",       "desc":"Massive packet volume — hallmark of Smurf",      "value":0.62,"direction":"pos"},
                {"feature":"Protocol_enc",   "label":"Protocol Type",     "desc":"ICMP flood protocol — high risk indicator",      "value":0.45,"direction":"pos"},
                {"feature":"time_delta",     "label":"Packet Timing",     "desc":"Near-zero gaps — overwhelming the device",       "value":0.28,"direction":"pos"},
                {"feature":"Destination_enc","label":"Destination Address","desc":"Broadcast to all devices on network",           "value":0.15,"direction":"pos"},
                {"feature":"Source_enc",     "label":"Source Address",    "desc":"Source appears spoofed — not a real device",     "value":-0.06,"direction":"neg"},
                {"feature":"info_len",       "label":"Message Length",    "desc":"Short messages — consistent with ICMP ping",     "value":-0.11,"direction":"neg"},
            ]
        },
    }

    # Fallback for old IDs
    fallback_map = {'bp':'port_scan','ins':'arp','xray':'ddos'}
    key = fallback_map.get(alert_id, alert_id)
    result = data.get(key, data['port_scan']).copy()
    result['source'] = 'static'
    return result


def _alert_info(alert_id):
    info = {
        'ddos':      {"device":"BP Monitor — OPD-11",       "attack":"DDoS Attack",  "model":"Random Forest","confidence":92},
        'port_scan': {"device":"X-Ray Machine — Radiology",  "attack":"Port Scan",    "model":"XGBoost",      "confidence":87},
        'arp':       {"device":"Insulin Pump — Ward B",      "attack":"ARP Spoofing", "model":"CatBoost",     "confidence":61},
        'dos':       {"device":"ECG Monitor — ICU-03",       "attack":"DoS Attack",   "model":"Random Forest","confidence":78},
        'smurf':     {"device":"Ventilator — ICU-07",        "attack":"Smurf Attack", "model":"XGBoost",      "confidence":95},
    }
    return info.get(alert_id, info['port_scan'])


# Pre-load SHAP explainer at startup
_load_explainer()