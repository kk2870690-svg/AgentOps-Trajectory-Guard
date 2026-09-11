import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="AgentOps Failure Detector", layout="wide")
st.title("🤖 AgentOps: Trajectory-Aware Self-Correction")
st.markdown("Live monitoring dashboard for long-horizon AI agents. Adjust the telemetry on the left to simulate agent drift and watch the recovery engine intervene.")

# --- 2. Load Model & Threshold ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('agent_trajectory_model.pkl')
    threshold = joblib.load('optimal_threshold.pkl')
    return model, float(threshold)

model, optimal_threshold = load_artifacts()

def self_correction_policy(row, failure_prob, threshold):
    if failure_prob < threshold:
        return "CONTINUE_TRAJECTORY", "success"
    if row['hallucination_risk_score'] > 0.65 or row['context_relevance_score'] < 0.40:
        return "TRIGGER_CONTEXT_PRUNING_AND_REQUERY", "warning"
    if row['tool_calls_made'] >= 5 and row['reasoning_steps'] >= 6:
        return "FORCE_PLANNER_SWITCH_REACT_TO_PLAN_EXECUTE", "error"
    if row['prompt_complexity_score'] >= 7:
        return "DECOMPOSE_TASK_SUBGOAL_FALLBACK", "warning"
    return "ROLLBACK_TO_LAST_CHECKPOINT", "error"

# --- 3. Sidebar: Live Telemetry Inputs ---
st.sidebar.header("📡 Live Agent Telemetry")

planner_type = st.sidebar.selectbox("Planner Type", ["ReAct", "Plan-and-Execute", "Tree-of-Thought", "Single-shot"])
prompt_complexity = st.sidebar.slider("Prompt Complexity Score", 1, 10, 8)
reasoning_steps = st.sidebar.slider("Reasoning Steps Taken", 1, 15, 7)
tool_calls_made = st.sidebar.slider("Tool Calls Made", 0, 15, 6)
total_tokens = st.sidebar.slider("Total Tokens Consumed", 500, 10000, 5300)
latency_seconds = st.sidebar.slider("Latency (Seconds)", 0.5, 30.0, 14.5)
hallucination_risk = st.sidebar.slider("Hallucination Risk Score", 0.0, 1.0, 0.75)
context_relevance = st.sidebar.slider("Context Relevance Score", 0.0, 1.0, 0.35)

# --- 4. Feature Engineering (On the Fly) ---
live_state = {
    'task_category': 'research_qa', 'domain': 'finance', 'planner_type': planner_type, 
    'model_name': 'gpt-4o', 'memory_enabled': 1, 'query_length_tokens': 120,
    'prompt_complexity_score': prompt_complexity, 'memory_tokens_used': 1500, 
    'tool_count_available': 12, 'reasoning_steps': reasoning_steps, 
    'tool_calls_made': tool_calls_made, 'latency_seconds': latency_seconds,
    'input_tokens': 4500, 'output_tokens': 800, 'total_tokens': total_tokens,
    'estimated_cost_usd': 0.008, 'retrieval_hits': 10.0, 'retrieval_quality_score': 0.3,
    'context_relevance_score': context_relevance, 'hallucination_risk_score': hallucination_risk,
    'num_selected_tools': 3
}

df_live = pd.DataFrame([live_state])

# Dynamic Drift Features
rs = df_live['reasoning_steps'].values[0] + 1e-5
df_live['tokens_per_step'] = df_live['total_tokens'] / rs
df_live['tool_call_density'] = df_live['tool_calls_made'] / rs
df_live['latency_per_step'] = df_live['latency_seconds'] / rs
df_live['semantic_divergence_ratio'] = df_live['hallucination_risk_score'] / (df_live['context_relevance_score'] + 1e-4)
df_live['retrieval_efficiency'] = df_live['retrieval_quality_score'] / (df_live['retrieval_hits'] + 1)
df_live['complexity_to_tools_ratio'] = df_live['prompt_complexity_score'] / (df_live['tool_count_available'] + 1)

# --- 5. Inference & Dashboard Layout ---
failure_prob = model.predict_proba(df_live)[:, 1][0]
action, severity = self_correction_policy(df_live.iloc[0], failure_prob, optimal_threshold)

col1, col2, col3 = st.columns(3)
col1.metric("Current Planner", planner_type)
col2.metric("Tokens per Step (Burn Rate)", f"{df_live['tokens_per_step'].values[0]:.0f}")
col3.metric("Tool Call Density", f"{df_live['tool_call_density'].values[0]:.2f}")

st.markdown("---")
st.subheader("🚨 Real-Time Risk Assessment")

# Dynamic Risk Gauge
risk_color = "red" if failure_prob >= optimal_threshold else "green"
st.markdown(f"**Failure Probability:** <span style='color:{risk_color}; font-size:24px;'>{failure_prob:.1%}</span> *(Threshold: {optimal_threshold:.1%})*", unsafe_allow_html=True)
st.progress(float(failure_prob))

st.subheader("🛠️ Automated Recovery Engine")
if severity == "success":
    st.success(f"**Action:** {action}\n\nTrajectory is healthy. No intervention required.")
elif severity == "warning":
    st.warning(f"**Action:** {action}\n\nAgent is drifting. Triggering targeted recovery.")
else:
    st.error(f"**Action:** {action}\n\nSevere trajectory degradation detected. Forcing structural intervention.")