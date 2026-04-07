---
title: Carbonenv
emoji: 🚀
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

🌿 CarbonEnv — Carbon-Aware RL Environment

An OpenEnv-compatible reinforcement learning environment for carbon-aware job scheduling and carbon credit optimization.

Built for the OpenEnv Hackathon.

The environment simulates data-center workload scheduling under carbon constraints, where agents must allocate jobs while maximizing renewable energy usage and minimizing carbon emissions.

🚀 Live Demo

👉 https://huggingface.co/spaces/Anmol318/carbonenv

📂 Project Structure
carbon_credit_env
│
├── app.py                 # Flask API server
├── inference.py           # LLM agent inference script
├── Dockerfile             # Container setup
├── openenv.yaml           # OpenEnv specification
├── requirements.txt       # Python dependencies
│
├── env
│   ├── carbon_env.py      # Core RL environment
│   ├── models.py          # Typed data models
│   └── __init__.py
│
├── tasks
│   ├── task1.py           # Easy task
│   ├── task2.py           # Medium task
│   ├── task3.py           # Hard task
│   └── __init__.py
│
├── graders
│   └── __init__.py        # Scoring logic
│
├── agents
│   ├── heuristic_agent.py
│   ├── random_agent.py
│   └── train_agent.py
│
└── frontend
    ├── templates
    │   └── dashboard.html
    │
    └── static
        ├── style.css
        └── script.js
🎯 Tasks
Task ID	Name	Difficulty
1	Green Window Scheduling	Easy
2	Carbon-Aware Allocation	Medium
3	Full Carbon Optimization	Hard

Each task increases in complexity and decision space.

🔌 API Endpoints
Method	Endpoint	Description
GET	/health	Health check
POST	/reset	Reset environment
POST	/step	Execute action
GET	/state	Get current environment state
POST	/grade	Grade current episode
GET	/tasks	List all tasks
GET	/	Dashboard UI

⚙️ Setup & Run Locally

1️⃣Clone the repository
git clone https://github.com/Anmol745/carbonenv.git
cd carbonenv

2️⃣Install Dependencies
pip install -r requirements.txt

3️⃣Set Environment Variables
PowerShell (Windows):

$env:HF_TOKEN="your_token_here"
$env:API_BASE_URL="https://api-inference.huggingface.co/v1/"
$env:MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

Linux / Mac:

export HF_TOKEN="your_token_here"
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

4️⃣Start the Environment Server

python app.py

Server runs at:

http://localhost:7860

5️⃣Run the LLM Agent

python inference.py

The script will:

1️⃣ Reset environment
2️⃣ Call the LLM
3️⃣ Take actions
4️⃣ Print structured logs

Example output:

[START] {...}
[STEP] {...}
[STEP] {...}
[END] {...}

This format is required for OpenEnv evaluation.

🐳 Docker Setup

Build the container:

docker build -t carbonenv .

Run container:

docker run -p 7860:7860 \
-e API_BASE_URL=https://api-inference.huggingface.co/v1/ \
-e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
-e HF_TOKEN=your_token_here \
carbonenv

Then open:

http://localhost:7860

📊 Reward System

The environment evaluates actions based on:

carbon usage
renewable energy ratio
job scheduling efficiency
total system performance

Rewards accumulate during the episode.

reward = efficiency + renewable_bonus - carbon_penalty

📈 Grading

After the episode finishes:

POST /grade

Returns:

{
 "task_id": 1,
 "score": 0.87,
 "trajectory_length": 30
}

Final evaluation score is based on:

Average score across all tasks
🛠️ Environment Variables
Variable	Description
API_BASE_URL	Base URL for LLM API
MODEL_NAME	Model used for inference
HF_TOKEN	HuggingFace API token
ENV_URL	Environment server URL
PORT	Server port (default 7860)

🖥️ Dashboard

The environment includes a live web dashboard.

Open:

http://localhost:7860

The dashboard displays:

environment status
available API endpoints
task information
system health
🧠 Agents Included

The repository includes multiple example agents.

Agent	Description
random_agent	Baseline random actions
heuristic_agent	Rule-based scheduling
train_agent	RL training scaffold
🧪 OpenEnv Compliance

This environment follows the OpenEnv specification:

reset() interface
step() interface
observation model
reward structure
structured inference logs
📜 License

MIT License

👨‍💻 Author

Anmol,Dhruv,Prince

Built for the OpenEnv Hackathon
