from flask import Flask, jsonify, send_from_directory, request, session, redirect
from flask_cors import CORS
import os, uuid

from device_store   import get_devices, add_device, delete_device, get_mud_profiles, add_mud_profile, delete_mud_profile
from mud_generator  import generate_mud_profile
from ml_models      import get_model_results
from shap_explainer import get_shap_values
from predictor      import get_alerts
from auth           import login_required, admin_required, check_login, get_current_user

app = Flask(__name__, static_folder='static')
app.secret_key = 'medguard-secret-skct-2025'
CORS(app)

# ── Auth Routes ────────────────────────────────────────
@app.route('/login')
def login_page():
    if 'user' in session:
        return redirect('/')
    return send_from_directory('static', 'login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data     = request.get_json()
    username = data.get('username', '')
    password = data.get('password', '')
    user     = check_login(username, password)
    if user:
        session['user'] = user
        return jsonify({"success": True, "user": user})
    return jsonify({"success": False, "error": "Invalid username or password"}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user', None)
    return jsonify({"success": True})

@app.route('/api/me', methods=['GET'])
def api_me():
    user = get_current_user()
    if user:
        return jsonify({"success": True, "user": user})
    return jsonify({"success": False}), 401

# ── Serve Frontend ─────────────────────────────────────
@app.route('/')
@login_required
def index():
    return send_from_directory('static', 'index.html')

# ── Devices API ────────────────────────────────────────
@app.route('/api/devices', methods=['GET'])
@login_required
def api_get_devices():
    return jsonify({"success": True, "devices": get_devices()})

@app.route('/api/devices', methods=['POST'])
@login_required
def api_add_device():
    data   = request.get_json()
    device = {
        "id":        "d" + str(uuid.uuid4())[:8],
        "name":      data.get('name',      'Unknown Device'),
        "loc":       data.get('loc',       'Unknown'),
        "ip":        data.get('ip',        '0.0.0.0'),
        "status":    data.get('status',    'safe'),
        "model":     data.get('model',     'XGBoost'),
        "packets":   data.get('packets',   0),
        "anomalies": data.get('anomalies', 0),
    }
    devices = add_device(device)
    return jsonify({"success": True, "devices": devices, "device": device})

@app.route('/api/devices/<device_id>', methods=['DELETE'])
@login_required
def api_delete_device(device_id):
    devices_before = get_devices()
    dev     = next((d for d in devices_before if d['id'] == device_id), None)
    devices = delete_device(device_id)
    if dev:
        delete_mud_profile(dev['ip'])
    return jsonify({"success": True, "devices": devices})

# ── Alerts API — Real predictions ──────────────────────
@app.route('/api/alerts', methods=['GET'])
@login_required
def api_get_alerts():
    alerts = get_alerts()
    return jsonify({"success": True, "alerts": alerts})

# ── SHAP API — Real SHAP values ────────────────────────
@app.route('/api/shap/<alert_id>', methods=['GET'])
@login_required
def api_shap(alert_id):
    return jsonify({"success": True, "shap": get_shap_values(alert_id)})

# ── MUD Profiles API ───────────────────────────────────
@app.route('/api/mud', methods=['GET'])
@login_required
def api_get_mud():
    return jsonify({"success": True, "profiles": get_mud_profiles()})

@app.route('/api/mud/generate', methods=['POST'])
@login_required
def api_generate_mud():
    data    = request.get_json()
    name    = data.get('name', 'Unknown Device')
    ip      = data.get('ip',   '192.168.1.100')
    loc     = data.get('loc',  'Unknown')
    profile = generate_mud_profile(name, ip, loc)
    add_mud_profile(profile)
    device  = {
        "id":        "d" + str(uuid.uuid4())[:8],
        "name":      name, "loc": loc, "ip": ip,
        "status":    "safe", "model": "XGBoost",
        "packets":   0, "anomalies": 0,
    }
    add_device(device)
    return jsonify({"success": True, "profile": profile, "device": device})

@app.route('/api/mud/<path:device_ip>', methods=['DELETE'])
@login_required
def api_delete_mud(device_ip):
    profiles = delete_mud_profile(device_ip)
    return jsonify({"success": True, "profiles": profiles})

# ── ML Models API ──────────────────────────────────────
@app.route('/api/models', methods=['GET'])
@login_required
def api_models():
    return jsonify({"success": True, "models": get_model_results()})

# ── Analytics API ──────────────────────────────────────
@app.route('/api/analytics', methods=['GET'])
@login_required
def api_analytics():
    data = {
        "best_accuracy": 99.93, "avg_f1": 97.3,
        "threats_today": len(get_alerts()),
        "false_positive_rate": 0.6,
        "traffic": {
            "hours":     ["00","02","04","06","08","10","12","14","16","18","20","22"],
            "normal":    [820,610,540,890,1340,1760,2100,1980,2240,2060,1580,1020],
            "anomalous": [12,8,6,14,22,18,35,28,19,42,24,16]
        },
        "attack_types": {
            "labels": ["Smurf Attack","Port Scan","ARP Spoofing","DoS Attack"],
            "values": [70,6,2,1]
        }
    }
    return jsonify({"success": True, "analytics": data})

# ── Run ────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)