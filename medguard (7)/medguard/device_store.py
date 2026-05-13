"""
device_store.py — Persistent device and MUD profile storage using JSON files.
All added/deleted devices survive server restarts.
"""

import json
import os

DEVICES_FILE = 'data/devices.json'
MUD_FILE     = 'data/mud_profiles.json'

# ── Default devices (used only if JSON file doesn't exist yet) ──
DEFAULT_DEVICES = [
    {"id":"d1","name":"ECG Monitor",    "loc":"ICU-03",    "ip":"192.168.1.14","status":"safe",   "model":"XGBoost", "packets":1240,"anomalies":0},
    {"id":"d2","name":"Insulin Pump",   "loc":"Ward B",    "ip":"192.168.1.27","status":"warning","model":"RF",      "packets":876, "anomalies":3},
    {"id":"d3","name":"BP Monitor",     "loc":"OPD-11",    "ip":"192.168.1.39","status":"threat", "model":"XGBoost", "packets":534, "anomalies":12},
    {"id":"d4","name":"Ventilator",     "loc":"ICU-07",    "ip":"192.168.1.52","status":"safe",   "model":"CatBoost","packets":2103,"anomalies":0},
    {"id":"d5","name":"Infusion Pump",  "loc":"OT-2",      "ip":"192.168.1.61","status":"safe",   "model":"MLP",     "packets":319, "anomalies":0},
    {"id":"d6","name":"X-Ray Machine",  "loc":"Radiology", "ip":"192.168.1.74","status":"threat", "model":"CatBoost","packets":408, "anomalies":8},
    {"id":"d7","name":"Pulse Oximeter", "loc":"Ward C",    "ip":"192.168.1.83","status":"safe",   "model":"KNN",     "packets":962, "anomalies":0},
    {"id":"d8","name":"MRI Scanner",    "loc":"Imaging",   "ip":"192.168.1.91","status":"safe",   "model":"SVM",     "packets":1780,"anomalies":1},
]

DEFAULT_MUD = [
    {"name":"ECG Monitor",    "ip":"192.168.1.14","loc":"ICU-03",    "cluster":2,"gateway":"10.0.0.1","ports":"80,443",    "protocol":"TCP",    "pkt_size":"64-512 B",   "rate":"Low", "timestamp":"2025-03-01 09:14","coverage":94},
    {"name":"Insulin Pump",   "ip":"192.168.1.27","loc":"Ward B",    "cluster":1,"gateway":"10.0.0.1","ports":"8080",      "protocol":"TCP",    "pkt_size":"32-256 B",   "rate":"Low", "timestamp":"2025-03-01 09:16","coverage":88},
    {"name":"BP Monitor",     "ip":"192.168.1.39","loc":"OPD-11",    "cluster":3,"gateway":"10.0.0.1","ports":"443",       "protocol":"TCP",    "pkt_size":"64-512 B",   "rate":"Med", "timestamp":"2025-03-01 09:18","coverage":91},
    {"name":"Ventilator",     "ip":"192.168.1.52","loc":"ICU-07",    "cluster":0,"gateway":"10.0.0.2","ports":"8443",      "protocol":"TCP",    "pkt_size":"512-1024 B", "rate":"Med", "timestamp":"2025-03-01 09:20","coverage":97},
    {"name":"Infusion Pump",  "ip":"192.168.1.61","loc":"OT-2",      "cluster":1,"gateway":"10.0.0.2","ports":"80",        "protocol":"TCP",    "pkt_size":"32-128 B",   "rate":"Low", "timestamp":"2025-03-01 09:22","coverage":93},
    {"name":"X-Ray Machine",  "ip":"192.168.1.74","loc":"Radiology", "cluster":4,"gateway":"10.0.0.3","ports":"21,80",     "protocol":"TCP/UDP","pkt_size":"1024-4096 B","rate":"High","timestamp":"2025-03-01 09:24","coverage":86},
    {"name":"Pulse Oximeter", "ip":"192.168.1.83","loc":"Ward C",    "cluster":0,"gateway":"10.0.0.1","ports":"443",       "protocol":"TCP",    "pkt_size":"32-64 B",    "rate":"Low", "timestamp":"2025-03-01 09:26","coverage":96},
    {"name":"MRI Scanner",    "ip":"192.168.1.91","loc":"Imaging",   "cluster":5,"gateway":"10.0.0.3","ports":"8080,8443", "protocol":"TCP",    "pkt_size":"4096+ B",    "rate":"High","timestamp":"2025-03-01 09:28","coverage":92},
]


def _load(filepath, default):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    # First time — write defaults to file
    _save(filepath, default)
    return list(default)


def _save(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ── DEVICES ────────────────────────────────────────────
def get_devices():
    return _load(DEVICES_FILE, DEFAULT_DEVICES)


def add_device(device):
    devices = get_devices()
    # Avoid duplicates by IP
    if not any(d['ip'] == device['ip'] for d in devices):
        devices.append(device)
        _save(DEVICES_FILE, devices)
    return devices


def delete_device(device_id):
    devices = get_devices()
    devices = [d for d in devices if d['id'] != device_id]
    _save(DEVICES_FILE, devices)
    return devices


# ── MUD PROFILES ───────────────────────────────────────
def get_mud_profiles():
    return _load(MUD_FILE, DEFAULT_MUD)


def add_mud_profile(profile):
    profiles = get_mud_profiles()
    # Avoid duplicates by IP
    if not any(p['ip'] == profile['ip'] for p in profiles):
        profiles.append(profile)
        _save(MUD_FILE, profiles)
    return profiles


def delete_mud_profile(device_ip):
    profiles = get_mud_profiles()
    profiles = [p for p in profiles if p['ip'] != device_ip]
    _save(MUD_FILE, profiles)
    return profiles