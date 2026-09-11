import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
import joblib

# 1. Load Data
print("Loading data...")
df = pd.read_csv('agent_observability_dataset.csv')

# 2. Target Definition
df_clean = df.copy()
df_clean['target_failure'] = (~df_clean['final_success']).astype(int)
drop_cols = [
    'run_id', 'agent_id', 'timestamp', 'user_query', 'final_success', 
    'failure_type', 'failure_reason', 'answer_quality_score', 'user_satisfaction_score'
]
X = df_clean.drop(columns=drop_cols)
y = df_clean['target_failure']

# 3. Feature Engineering
categorical_features = ['task_category', 'domain', 'planner_type', 'model_name']
numeric_features = [
    'query_length_tokens', 'prompt_complexity_score', 'memory_tokens_used',
    'tool_count_available', 'reasoning_steps', 'tool_calls_made', 
    'latency_seconds', 'input_tokens', 'output_tokens', 'total_tokens',
    'estimated_cost_usd', 'retrieval_hits', 'retrieval_quality_score',
    'context_relevance_score', 'hallucination_risk_score'
]

X['memory_enabled'] = X['memory_enabled'].astype(int)
numeric_features.append('memory_enabled')

X['num_selected_tools'] = X['selected_tools'].fillna('').apply(lambda x: len(x.split(';')) if x else 0)
X = X.drop(columns=['selected_tools'])
numeric_features.append('num_selected_tools')

X['tokens_per_step'] = X['total_tokens'] / (X['reasoning_steps'] + 1e-5)
X['tool_call_density'] = X['tool_calls_made'] / (X['reasoning_steps'] + 1e-5)
X['latency_per_step'] = X['latency_seconds'] / (X['reasoning_steps'] + 1e-5)
X['semantic_divergence_ratio'] = X['hallucination_risk_score'].fillna(0) / (X['context_relevance_score'].fillna(0.01) + 1e-4)
X['retrieval_efficiency'] = X['retrieval_quality_score'].fillna(0) / (X['retrieval_hits'] + 1)
X['complexity_to_tools_ratio'] = X['prompt_complexity_score'] / (X['tool_count_available'] + 1)

velocity_features = [
    'tokens_per_step', 'tool_call_density', 'latency_per_step',
    'semantic_divergence_ratio', 'retrieval_efficiency', 'complexity_to_tools_ratio'
]
numeric_features.extend(velocity_features)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Pipeline Setup
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        class_weight={0: 1.0, 1: 2.2},
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.5,
        random_state=42
    ))
])

# 6. Train Model
print("Training model (this might take a minute)...")
model_pipeline.fit(X_train, y_train)

# 7. Find Optimal Threshold
print("Finding optimal threshold...")
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
optimal_threshold = float(thresholds[best_idx])

# 8. Export Artifacts
print("Exporting artifacts...")
joblib.dump(model_pipeline, 'agent_trajectory_model.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
print(f"Done! Saved with optimal threshold: {optimal_threshold:.4f}")