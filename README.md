# 🌿 CarbonEnv — Carbon Credit RL Environment

An OpenEnv-compatible reinforcement learning environment for carbon credit
trading and allocation. Built for the OpenEnv Hackathon.

---

## 🚀 Live Demo

👉 [HuggingFace Space](https://huggingface.co/spaces/YOUR_USERNAME/carbonenv)

---

## 📋 Project Structure

carbonenv/
├── app.py              # Flask API server
├── inference.py        # LLM agent inference script
├── Dockerfile          # Container setup
├── openenv.yaml        # OpenEnv spec file
├── requirements.txt    # Python dependencies
├── env/
│   ├── carbon_env.py   # Core environment logic
│   └── models.py       # Typed data models
├── tasks/
│   ├── task1.py        # Easy task
│   ├── task2.py        # Medium task
│   └── task3.py        # Hard task
├── graders/
│   └── __init__.py     # Scoring/grading logic
├── agents/
│   ├── heuristic_agent.py
│   ├── random_agent.py
│   └── train_agent.py
└── frontend/
    ├── templates/dashboard.html
    └── static/
        ├── script.js
        └── style.css

---

## 🎯 Tasks

| Task ID | Name                  | Difficulty |
|---------|-----------------------|------------|
| 1       | Basic Carbon Allocation | Easy     |
| 2       | Market Trading          | Medium   |
| 3       | Full Optimization       | Hard     |

---

## 🔌 API Endpoints

| Method | Endpoint  | Description              |
|--------|-----------|--------------------------|
| GET    | /health   | Health check             |
| POST   | /reset    | Reset environment        |
| POST   | /step     | Take an action           |
| GET    | /state    | Get current state        |
| POST   | /grade    | Grade current episode    |
| GET    | /tasks    | List all tasks           |
| GET    | /         | Dashboard UI             |

---

## ⚙️ Setup & Run Locally

### 1. Install dependencies
pip install -r requirements.txt

### 2. Set environment variables
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here

### 3. Start the server
python app.py

### 4. Run inference
python inference.py

---

## 🐳 Docker

docker build -t carbonenv .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_token \
  carbonenv

---

## 📊 Reward & Scoring

- All rewards are in range **0.0 to 1.0**
- Each task has its own grader
- Final score = average across all 3 tasks

---

## 🛠️ Environment Variables

| Variable      | Description                        |
|---------------|------------------------------------|
| API_BASE_URL  | LLM API base URL                   |
| MODEL_NAME    | Model name for inference           |
| HF_TOKEN      | Hugging Face / API token           |
| ENV_URL       | Environment server URL             |
| PORT          | Server port (default: 7860)        |

---

## 📝 License

MIT License
