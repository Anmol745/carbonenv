"""
frontend/app.py — CarbonEnv API Server + Dashboard
===================================================

API Endpoints
POST /reset
POST /step
GET  /state
GET  /health
GET  /tasks
POST /grade
GET  /
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template

from env.carbon_env import CarbonEnv
from env.models import Action
from tasks import TASK_REGISTRY
from graders import run_grader


# Flask app with frontend folders
app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)

_env: CarbonEnv = None


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "env": "carbon-credit-env",
        "version": "1.0.0"
    }), 200


# ─────────────────────────────────────────────
# Reset Environment
# ─────────────────────────────────────────────

@app.route("/reset", methods=["POST", "GET"])
def reset():

    global _env

    data = request.get_json(silent=True) or {}
    task_id = int(data.get("task_id", 1))

    if task_id not in [1, 2, 3]:
        return jsonify({"error": "task_id must be 1, 2, or 3"}), 400

    _env = CarbonEnv(task_id=task_id, seed=42)

    obs = _env.reset()

    return jsonify(obs.model_dump()), 200


# ─────────────────────────────────────────────
# Step
# ─────────────────────────────────────────────

@app.route("/step", methods=["POST"])
def step():

    global _env

    if _env is None:
        return jsonify({"error": "Call /reset first"}), 400

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Body must contain action_type and amount"}), 400

    try:

        action = Action(
            action_type=data.get("action_type", "idle"),
            amount=float(data.get("amount", 1.0))
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400

    obs, reward, done, info = _env.step(action)

    return jsonify({
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }), 200


# ─────────────────────────────────────────────
# Get State
# ─────────────────────────────────────────────

@app.route("/state", methods=["GET"])
def state():

    global _env

    if _env is None:
        return jsonify({"error": "Environment not initialized"}), 400

    return jsonify(_env.state()), 200


# ─────────────────────────────────────────────
# Grade Episode
# ─────────────────────────────────────────────

@app.route("/grade", methods=["POST"])
def grade():

    global _env

    if _env is None:
        return jsonify({"error": "No environment running"}), 400

    current_state = _env.state()

    trajectory = _env.get_trajectory()

    task_id = current_state["task_id"]

    score = run_grader(task_id, trajectory, current_state)

    return jsonify({
        "task_id": task_id,
        "score": score,
        "done": current_state["done"],
        "trajectory_length": len(trajectory)
    }), 200


# ─────────────────────────────────────────────
# List Tasks
# ─────────────────────────────────────────────

@app.route("/tasks", methods=["GET"])
def tasks():

    task_list = []

    for task_id, info in TASK_REGISTRY.items():

        task_list.append({
            "task_id": task_id,
            "name": info["name"],
            "difficulty": info["difficulty"]
        })

    return jsonify({"tasks": task_list}), 200


# ─────────────────────────────────────────────
# Dashboard UI
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def dashboard():

    return render_template("dashboard.html")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    print(f"CarbonEnv server running on port {port}")

    app.run(host="0.0.0.0", port=port)