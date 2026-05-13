"""
auth.py — Role-based authentication for MedGuard.
Two roles: Hospital Admin and IT Security Staff.
"""

import hashlib
from functools import wraps
from flask import session, redirect, request, jsonify


def _hash(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ── Registered users ───────────────────────────────────
USERS = {
    "admin":   {"password": _hash("admin123"),    "role": "admin",    "role_label": "Hospital Admin",    "name": "Admin"},
    "itstaff": {"password": _hash("medguard2025"),"role": "itstaff",  "role_label": "IT Security Staff", "name": "IT Staff"},
}


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"success": False, "error": "Not authenticated"}), 401
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Only Hospital Admin can access this route."""
    @wraps(f)
    def decorated(*args, **kwargs):
        user = session.get('user')
        if not user:
            if request.path.startswith('/api/'):
                return jsonify({"success": False, "error": "Not authenticated"}), 401
            return redirect('/login')
        if user.get('role') != 'admin':
            if request.path.startswith('/api/'):
                return jsonify({"success": False, "error": "Access denied. Admin only."}), 403
            return redirect('/')
        return f(*args, **kwargs)
    return decorated


def check_login(username, password):
    user = USERS.get(username.strip().lower())
    if user and user['password'] == _hash(password):
        return {
            "username":   username,
            "role":       user['role'],
            "role_label": user['role_label'],
            "name":       user['name'],
        }
    return None


def get_current_user():
    return session.get('user', None)