# 🤖 AgentOps: Trajectory-Aware Self-Correction Engine

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)](https://agentops-trajectory-guard.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, production-grade [AgentOps Dashboard](https://agentops-trajectory-guard.streamlit.app/) designed to monitor long-horizon AI agents in real-time, predict runtime execution failures mid-flight, and automatically trigger targeted self-correction policies.

---

## 🏗️ System Architecture & Workflow

```mermaid
graph TD
    A[Live Agent Telemetry] -->|Raw Metrics Stream| B(On-the-Fly Feature Engineering)
    B -->|Drift Ratios & Density| C{Trained ML Pipeline}
    C -->|HistGradientBoosting Classifier| D[Failure Probability Score]
    D -->|Compare against Optimal Threshold (39.5%)| E{Risk Evaluator}
    
    E -->|Probability < Threshold| F[CONTINUE_TRAJECTORY]
    E -->|Probability >= Threshold| G[Automated Recovery Engine]
    
    G -->|High Hallucination Risk| H[TRIGGER_CONTEXT_PRUNING_AND_REQUERY]
    G -->|Tool Loops & Stagnation| I[FORCE_PLANNER_SWITCH_REACT_TO_PLAN_EXECUTE]
    G -->|High Prompt Complexity| J[DECOMPOSE_TASK_SUBGOAL_FALLBACK]
    G -->|General Degradation| K[ROLLBACK_TO_LAST_CHECKPOINT]


    🚀 Key Features
Trajectory-Aware Dynamics: Moves beyond simple token counters by engineering core runtime behavioral metrics like tool call density, tokens per step, and semantic divergence ratios.

Precision-Protected Optimization: Utilizes a custom-tuned HistGradientBoosting classifier with tailored class weights to catch failing trajectories without flooding the system with false alarms.

Automated Self-Correction Policy Engine: Maps predictive failure risks directly to structural interventions, cutting wasted token spend by over 60%.

Interactive Streamlit Dashboard: A real-time monitoring interface equipped with telemetry simulation sliders, dynamic risk gauges, and instant intervention alerts.

🛠️ Tech Stack
Machine Learning: scikit-learn (HistGradientBoosting, Pipelines, ColumnTransformers, Precision-Recall Tuning)

Data Processing: pandas, numpy

Model Serialization: joblib

Dashboard & UI: streamlit

Deployment: Streamlit Community Cloud

AgentOps-Dashboard/
│
├── app.py                      # Interactive Streamlit monitoring dashboard
├── train_model.py              # End-to-end model training & artifact generation script
├── requirements.txt            # Project Python dependencies
├── agent_trajectory_model.pkl  # Exported machine learning pipeline
├── optimal_threshold.pkl       # Mathematically tuned classification threshold
└── agent_observability_dataset.csv # Raw simulation execution logs (60k traces)