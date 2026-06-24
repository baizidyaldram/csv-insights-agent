import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)
from utils.session import get_df, is_data_loaded
from utils.llm import call_llm

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


# ── Hyperparameter grids ─────────────────────────────────────────────────────
PARAM_GRIDS_CLF = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63],
    },
}

PARAM_GRIDS_REG = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "Linear Regression": {},
    "Ridge Regression": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "LightGBM": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
    },
}


# ── Helper Functions ─────────────────────────────────────────────────────────

def format_ai_recommendation(text: str) -> str:
    """Format the AI recommendation text for better display."""
    if not text:
        return "<p>No recommendation available. Click the button above to generate one.</p>"
    
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            formatted_lines.append('<br>')
            continue
        
        # Check for numbered headings (e.g., "1. Task Type")
        if stripped[0].isdigit() and '. ' in stripped[:10]:
            parts = stripped.split('. ', 1)
            if len(parts) == 2:
                formatted_lines.append(f'<div class="rec-title">{parts[0]}. {parts[1]}</div>')
                continue
        
        # Check for markdown bold headings
        if stripped.startswith('**') and stripped.endswith('**'):
            formatted_lines.append(f'<h4>{stripped.strip("*")}</h4>')
            continue
        
        # Check for ## headings
        if stripped.startswith('## '):
            formatted_lines.append(f'<h3>{stripped[3:].strip()}</h3>')
            continue
        
        # Check for ### headings
        if stripped.startswith('### '):
            formatted_lines.append(f'<h4>{stripped[4:].strip()}</h4>')
            continue
        
        # Bullet points
        if stripped.startswith('- ') or stripped.startswith('* ') or stripped.startswith('• '):
            content = stripped[2:].strip()
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            formatted_lines.append(f'<li>{content}</li>')
            continue
        
        # Numbered lists
        if stripped[0].isdigit() and '. ' in stripped[:5]:
            parts = stripped.split('. ', 1)
            if len(parts) == 2:
                content = parts[1]
                content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
                formatted_lines.append(f'<li><strong>{parts[0]}.</strong> {content}</li>')
                continue
        
        # Regular paragraph with potential bold text
        line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', stripped)
        formatted_lines.append(f'<p>{line}</p>')
    
    # Wrap list items in ul tags
    result = '\n'.join(formatted_lines)
    result = re.sub(r'(<li>.*?</li>\s*)+', lambda m: f'<ul>{m.group(0)}</ul>', result, flags=re.DOTALL)
    
    # Clean up extra breaks
    result = re.sub(r'<br>\s*<br>\s*<ul>', '<ul>', result)
    result = re.sub(r'</ul>\s*<br>\s*<br>', '</ul><br>', result)
    result = re.sub(r'</ul>\s*<br>\s*<div', '</ul><div', result)
    
    return result


def get_base_model(model_name, task_type, random_state):
    """Return a fresh base model instance."""
    try:
        if task_type == "classification":
            if model_name == "Random Forest":
                return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
            elif model_name == "Logistic Regression":
                return LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs')
            elif model_name == "Decision Tree":
                return DecisionTreeClassifier(max_depth=10, random_state=random_state)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=random_state,
                    eval_metric='logloss'
                )
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                          random_state=random_state, verbose=-1)
        else:
            if model_name == "Random Forest":
                return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            elif model_name == "Gradient Boosting":
                return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
            elif model_name == "Linear Regression":
                return LinearRegression()
            elif model_name == "Ridge Regression":
                return Ridge(alpha=1.0, random_state=random_state)
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                return xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=random_state
                )
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                return lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state, verbose=-1)
    except Exception as e:
        st.warning(f"⚠️ Could not load {model_name}: {e}")
    return None


def tune_model(model, model_name, task_type, X_train, y_train, cv_folds, log_step):
    """Run GridSearchCV and return the best estimator."""
    grids = PARAM_GRIDS_CLF if task_type == "classification" else PARAM_GRIDS_REG
    param_grid = grids.get(model_name, {})

    if not param_grid:
        log_step(f"No tuning grid for {model_name} — using defaults.")
        return model

    scoring = "accuracy" if task_type == "classification" else "r2"

    # Calculate number of combinations
    from itertools import product
    n_combos = len(list(product(*param_grid.values())))
    log_step(f"GridSearchCV for {model_name} ({n_combos} combos, {cv_folds}-fold)...")
    log_step(f"⚠️ This may take a while...")

    try:
        # Use fewer folds for small datasets
        actual_folds = min(cv_folds, 3)
        gs = GridSearchCV(model, param_grid, cv=actual_folds, scoring=scoring,
                          n_jobs=-1, refit=True, error_score="raise")
        gs.fit(X_train, y_train)
        log_step(f"Best params for {model_name}: {gs.best_params_}  |  Best CV {scoring}: {gs.best_score_:.4f}")
        return gs.best_estimator_
    except Exception as e:
        log_step(f"⚠️ Tuning failed for {model_name}: {e}  — falling back to defaults.")
        model.fit(X_train, y_train)
        return model


def calculate_metrics(y_test, y_pred, task_type, model, X_test):
    m = {}
    try:
        if task_type == "classification":
            m["Accuracy"] = accuracy_score(y_test, y_pred)
            m["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            m["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            m["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            m["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
        else:
            m["R2 Score"] = r2_score(y_test, y_pred)
            m["MAE"] = mean_absolute_error(y_test, y_pred)
            m["MSE"] = mean_squared_error(y_test, y_pred)
            m["RMSE"] = np.sqrt(m["MSE"])
    except Exception as e:
        st.warning(f"Error in metrics: {e}")

    try:
        if hasattr(model, "feature_importances_"):
            m["feature_importances"] = model.feature_importances_.tolist()
    except Exception:
        pass

    return m


def generate_ai_recommendation(metrics_report, task_type, features_list, n_rows, best_model):
    lines = [f"Dataset: {n_rows} rows, {len(features_list)} features", f"Task: {task_type}",
             f"Best model: {best_model}\n", "Performance:"]
    for name, metrics in metrics_report.items():
        if task_type == "classification":
            lines.append(f"- {name}: Accuracy={metrics.get('Accuracy',0):.4f}, F1={metrics.get('F1-Score',0):.4f}")
        else:
            lines.append(f"- {name}: R²={metrics.get('R2 Score',0):.4f}, RMSE={metrics.get('RMSE',0):.4f}")
    prompt = "\n".join(lines) + """

Provide:
1. **Best Model Recommendation** — why this model wins
2. **Key Insights** — what the results say about the data
3. **Improvement Suggestions** — 2-3 specific next steps
4. **Business Implications** — real-world meaning

Concise, markdown bullet points."""
    try:
        return call_llm(prompt, "You are a senior ML engineer giving model recommendations.", max_tokens=600)
    except Exception as e:
        return f"⚠️ Could not generate: {e}"


def create_model_comparison_chart(metrics_dict, task_type):
    comparison_data = []
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        keys = ["Accuracy", "Precision", "Recall", "F1-Score"] if task_type == "classification" else ["R2 Score"]
        for k in keys:
            if k in metrics and isinstance(metrics[k], (int, float)):
                row[k] = round(metrics[k], 4)
        if "cv_mean" in metrics and metrics["cv_mean"]:
            row["CV Score"] = round(metrics["cv_mean"], 4)
        comparison_data.append(row)

    if not comparison_data:
        return None

    df_comp = pd.DataFrame(comparison_data)
    metrics_to_plot = [m for m in (["Accuracy", "Precision", "Recall", "F1-Score"] if task_type == "classification" else ["R2 Score"]) if m in df_comp.columns]
    if not metrics_to_plot:
        return None

    fig = px.bar(
        df_comp, x="Model", y=metrics_to_plot, barmode="group",
        title="Model Performance Comparison",
        color_discrete_sequence=["#EF9F27", "#D85A30", "#BA7517", "#FAC775"],
        text_auto='.3f'
    )
    fig.update_layout(template="plotly_white", height=480,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_traces(textposition="outside", marker_line_width=0)
    return fig


def create_feature_importance_chart(feature_importances, features_list, top_n=10):
    if not feature_importances or not features_list:
        return None
    try:
        if isinstance(feature_importances, list):
            n = min(len(features_list), len(feature_importances))
            d = dict(zip(features_list[:n], feature_importances[:n]))
        else:
            d = feature_importances
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:top_n]
        df_imp = pd.DataFrame(top, columns=["Feature", "Importance"])
        fig = px.bar(df_imp, x="Importance", y="Feature", orientation="h",
                     title=f"Top {top_n} Feature Importances",
                     color="Importance", color_continuous_scale="YlOrBr",
                     text_auto='.3f')
        fig.update_layout(template="plotly_white", height=400,
                          yaxis=dict(categoryorder="total ascending"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
        return fig
    except Exception:
        return None


def create_confusion_matrix_heatmap(cm_matrix):
    """Create confusion matrix heatmap."""
    if not cm_matrix:
        return None
    try:
        fig = px.imshow(cm_matrix, text_auto=True, title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        color_continuous_scale="YlOrBr", aspect="auto")
        fig.update_layout(template="plotly_white", height=420,
                          paper_bgcolor="rgba(0,0,0,0)")
        return fig
    except Exception:
        return None


def display_results(task_type):
    metrics = st.session_state.model_metrics
    best_model_name = st.session_state.trained_model_name
    best_params = st.session_state.get("best_params_report", {})

    st.markdown("---")
    st.markdown("### 🏆 Model Performance Results")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Comparison", "📈 Charts", "🔬 Best Params", "🤖 AI Recommendation", "📋 Details"]
    )

    with tab1:
        st.markdown("#### Model Comparison")
        rows = []
        for name, data in metrics.items():
            row = {"Model": name}
            for k, v in data.items():
                if isinstance(v, (int, float)) and k != "confusion_matrix":
                    row[k] = round(v, 4)
            rows.append(row)
        if rows:
            df_display = pd.DataFrame(rows)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.success(f"💡 **Best Model:** {best_model_name}")

    with tab2:
        fig_bar = create_model_comparison_chart(metrics, task_type)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)

        best_m = metrics.get(best_model_name, {})
        if "feature_importances" in best_m and st.session_state.model_features_list:
            st.markdown("#### Feature Importance (Best Model)")
            fig_fi = create_feature_importance_chart(best_m["feature_importances"], st.session_state.model_features_list)
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)

        if task_type == "classification" and "confusion_matrix" in best_m:
            st.markdown("#### Confusion Matrix")
            fig_cm = create_confusion_matrix_heatmap(best_m["confusion_matrix"])
            if fig_cm:
                st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("#### 🔬 Best Hyperparameters (from GridSearchCV)")
        if best_params:
            for model_name, params in best_params.items():
                if params:
                    st.markdown(f"**{model_name}**")
                    param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Best Value"])
                    param_df["Best Value"] = param_df["Best Value"].astype(str)
                    st.dataframe(param_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"{model_name}: Used default parameters (no grid defined).")
        else:
            st.info("GridSearchCV was not enabled for this run. Enable it in Training Settings to see tuned parameters.")

    with tab4:
        st.markdown("#### 🤖 AI Model Recommendation")
        rec = st.session_state.get("model_recommendation")
        if rec:
            st.markdown(rec)
        else:
            st.info("No recommendation available. Re-run modeling to generate one.")

    with tab5:
        st.markdown("#### Best Model Details")
        best_m = metrics.get(best_model_name, {})

        if task_type == "classification":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{best_m.get('Accuracy', 0):.4f}")
            c2.metric("Precision", f"{best_m.get('Precision', 0):.4f}")
            c3.metric("Recall", f"{best_m.get('Recall', 0):.4f}")
            c4.metric("F1-Score", f"{best_m.get('F1-Score', 0):.4f}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{best_m.get('R2 Score', 0):.4f}")
            c2.metric("MAE", f"{best_m.get('MAE', 0):.4f}")
            c3.metric("RMSE", f"{best_m.get('RMSE', 0):.4f}")

        if "cv_mean" in best_m and best_m["cv_mean"]:
            st.metric("Cross-Validation Score",
                      f"{best_m['cv_mean']:.4f} ± {best_m.get('cv_std', 0):.4f}")

        st.markdown(f"""
**Configuration**
- Task: `{task_type.upper()}`
- Target: `{st.session_state.model_target_col}`
- Features used: {len(st.session_state.model_features_list)}
- GridSearchCV: {'✅ Enabled' if best_params else '❌ Disabled'}
""")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Visualization", use_container_width=True):
            st.session_state.current_page = "visualization"
            st.rerun()
    with col2:
        if st.button("➡️ Proceed to AI Insights & Report", use_container_width=True):
            st.session_state.current_page = "ai_report"
            st.rerun()


def run_training(df, target_col, features_list, task_type, selected_model_names, available_models,
                 test_size, random_state, scaling_method, cv_folds, handle_imbalance, use_cv,
                 enable_tuning, tune_cv):

    st.session_state.modeling_log = []

    def log_step(msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.modeling_log.append(f"[{t}] {msg}")

    log_step(f"Starting: target='{target_col}' ({task_type})")
    log_step(f"Dataset: {len(df)} rows, {len(features_list)} features")

    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features_list].copy()
    y = df_clean[target_col].copy()
    log_step(f"After dropping NA: {X.shape[0]} rows")

    # Encode categoricals - fix for pandas 3.0
    cat_mappings = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        log_step(f"Encoded: {col}")

    y_inverse_mapping = None
    if task_type == "classification" and y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        y_inverse_mapping = {i: label for i, label in enumerate(le_y.classes_)}
        log_step(f"Encoded target → {len(le_y.classes_)} classes")

        # Check class balance
        from collections import Counter
        class_counts = Counter(y)
        log_step(f"Class distribution: {dict(class_counts)}")
        if min(class_counts.values()) < 5:
            log_step("⚠️ Some classes have very few samples (<5) - results may be unreliable")

    # Stratify only for classification with balanced classes
    stratify = None
    if task_type == "classification":
        from collections import Counter
        class_counts = Counter(y)
        if min(class_counts.values()) >= 2 and len(class_counts) <= 20:
            stratify = y
            log_step("✅ Using stratified split")
        else:
            log_step("⚠️ Stratification disabled - class imbalance or too many classes")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    log_step(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Feature scaling
    scaler = None
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    if scaling_method != "None" and num_cols:
        try:
            scaler = StandardScaler() if scaling_method == "StandardScaler" else MinMaxScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
            log_step(f"Applied {scaling_method}")
        except Exception as e:
            log_step(f"⚠️ Scaling failed: {e}")
            scaler = None

    # SMOTE for imbalance
    if handle_imbalance and task_type == "classification" and SMOTE_AVAILABLE:
        try:
            from collections import Counter
            before = Counter(y_train)
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            after = Counter(y_train)
            log_step(f"SMOTE applied: {before} → {after}")
        except Exception as e:
            log_step(f"⚠️ SMOTE failed: {e}")

    metrics_report = {}
    trained_models = {}
    best_params_report = {}
    console_ph = st.empty()

    for model_name in selected_model_names:
        log_step(f"Training {model_name}...")
        model = get_base_model(model_name, task_type, random_state)
        if model is None:
            log_step(f"⚠️ {model_name} unavailable, skipping.")
            continue

        try:
            if enable_tuning:
                model = tune_model(model, model_name, task_type, X_train, y_train, tune_cv, log_step)
                # Collect best params
                if hasattr(model, "get_params"):
                    best_params_report[model_name] = {
                        k: v for k, v in model.get_params().items()
                        if k in (PARAM_GRIDS_CLF if task_type == "classification" else PARAM_GRIDS_REG).get(model_name, {})
                    }
            else:
                log_step(f"Training {model_name} with default params...")
                model.fit(X_train, y_train)

            trained_models[model_name] = model
            y_pred = model.predict(X_test)
            m = calculate_metrics(y_test, y_pred, task_type, model, X_test)

            # Cross-validation only if enabled and enough samples
            if use_cv:
                try:
                    scoring_str = "accuracy" if task_type == "classification" else "r2"
                    
                    # Calculate safe CV folds - ensure each class has at least 2 samples
                    if task_type == "classification":
                        from collections import Counter
                        class_counts = Counter(y_train)
                        min_class_size = min(class_counts.values())
                        # Use at most min_class_size folds, but no more than cv_folds
                        actual_folds = min(cv_folds, min_class_size)
                        # Ensure at least 2 folds
                        actual_folds = max(2, actual_folds)
                    else:
                        actual_folds = min(cv_folds, len(y_train) // 5)
                        actual_folds = max(2, min(5, actual_folds))
                    
                    if actual_folds >= 2:
                        cv_sc = cross_val_score(model, X_train, y_train,
                                                cv=actual_folds, scoring=scoring_str)
                        m["cv_mean"] = cv_sc.mean()
                        m["cv_std"] = cv_sc.std()
                        log_step(f"CV {actual_folds}-fold: {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
                    else:
                        log_step("⚠️ Skipping CV - not enough samples per class")
                except Exception as e:
                    log_step(f"⚠️ CV failed for {model_name}: {e}")

            metrics_report[model_name] = m
            log_step(f"✓ {model_name} done")

        except Exception as e:
            log_step(f"❌ {model_name} failed: {str(e)[:100]}")
            continue

        with console_ph.container():
            st.markdown("#### 🪵 Training Log")
            st.code("\n".join(st.session_state.modeling_log[-15:]), language="text")

    if metrics_report:
        metric_key = "Accuracy" if task_type == "classification" else "R2 Score"
        best_model_name = max(metrics_report, key=lambda x: metrics_report[x].get(metric_key, -np.inf))
        best_model = trained_models.get(best_model_name)

        if best_model is not None:
            st.session_state.model_metrics = metrics_report
            st.session_state.trained_model = best_model
            st.session_state.trained_model_name = best_model_name
            st.session_state.model_features_list = features_list
            st.session_state.model_target_col = target_col
            st.session_state.model_scaler = scaler
            st.session_state.model_task_type = task_type
            st.session_state.model_encoded_categories = cat_mappings
            st.session_state.model_target_inverse_mapping = y_inverse_mapping
            st.session_state.modeling_done = True
            st.session_state.best_params_report = best_params_report

            # Generate AI summary
            try:
                with st.spinner("🧠 Generating AI model summary..."):
                    rec = generate_ai_recommendation(metrics_report, task_type, features_list,
                                                     len(df_clean), best_model_name)
                    st.session_state.model_recommendation = rec
            except Exception as e:
                st.session_state.model_recommendation = f"⚠️ Could not generate: {e}"

            log_step(f"🏆 Best: {best_model_name}")
            st.success(f"✅ Done! Best model: **{best_model_name}**")
            st.balloons()
        else:
            st.error("Best model could not be loaded.")
    else:
        st.error("No models were successfully trained.")


# ── Main Render Function ─────────────────────────────────────────────────────

def render():
    # ── CUSTOM CSS FOR DARK MODE CONTRAST ──────────────────────────────────
    st.markdown("""
    <style>
    @media (prefers-color-scheme: dark) {
        [data-baseweb="select"] [data-testid="stMultiSelect"] {
            background: #2A2420 !important;
            border-color: #4A3F37 !important;
        }
        [data-baseweb="select"] [data-testid="stMultiSelect"] input {
            color: #EDE8DF !important;
        }
        [data-baseweb="select"] [data-testid="stMultiSelect"] .st-ae {
            background: #3A3028 !important;
            color: #EDE8DF !important;
        }
        [data-baseweb="select"] ul {
            background: #2A2420 !important;
            border-color: #4A3F37 !important;
        }
        [data-baseweb="select"] li {
            color: #EDE8DF !important;
        }
        [data-baseweb="select"] li:hover {
            background: #3A3028 !important;
        }
        [data-baseweb="tag"] {
            background: #4A3F37 !important;
            color: #EDE8DF !important;
        }
        [data-baseweb="tag"] span {
            color: #EDE8DF !important;
        }
        .stRadio label, .stCheckbox label, .stSlider label {
            color: #EDE8DF !important;
        }
        .stNumberInput input {
            color: #EDE8DF !important;
            background: #2A2420 !important;
            border-color: #4A3F37 !important;
        }
        .stAlert {
            background: #2A2420 !important;
            color: #EDE8DF !important;
        }
        .stAlert p {
            color: #EDE8DF !important;
        }
        .stDataFrame {
            background: #1A1714 !important;
        }
        .stDataFrame table {
            color: #EDE8DF !important;
        }
        .stDataFrame th {
            background: #2A2420 !important;
            color: #EDE8DF !important;
        }
        .stDataFrame td {
            color: #D5CCBF !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #2A2420 !important;
            border-color: #3A3028 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #9E9385 !important;
        }
        .stTabs [aria-selected="true"] {
            background: #EF9F27 !important;
            color: #1A1714 !important;
        }
        .ai-rec-container {
            background: #2A2420 !important;
            border-color: #4A3F37 !important;
        }
        .ai-rec-container h3 {
            color: #F5E6D0 !important;
            border-left-color: #F0997B !important;
        }
        .ai-rec-container h4 {
            color: #E8D5C0 !important;
        }
        .ai-rec-container strong {
            color: #F5A07A !important;
        }
        .ai-rec-container li {
            color: #D5CCBF !important;
        }
        .ai-rec-container p {
            color: #D5CCBF !important;
        }
        .ai-rec-container .rec-title {
            color: #F5E6D0 !important;
            background: #3A3028 !important;
        }
    }
    
    .ai-rec-container {
        background: #FFFBF5;
        border: 2px solid #EF9F27;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin: 1rem 0;
    }
    .ai-rec-container h3 {
        color: #2D1B00;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #EF9F27;
        padding-left: 0.9rem;
    }
    .ai-rec-container h4 {
        color: #4A2A00;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
    }
    .ai-rec-container strong {
        color: #B83A00;
        font-weight: 700;
    }
    .ai-rec-container ul {
        margin: 0.5rem 0 1rem 1.5rem;
        padding-left: 0.5rem;
    }
    .ai-rec-container li {
        margin-bottom: 0.4rem;
        color: #3D3530;
        line-height: 1.6;
    }
    .ai-rec-container li::marker {
        color: #EF9F27;
    }
    .ai-rec-container p {
        margin: 0.3rem 0;
        color: #3D3530;
        line-height: 1.6;
    }
    .ai-rec-container .rec-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #3D2800;
        margin: 0.8rem 0 0.3rem 0;
        padding: 0.3rem 0.6rem;
        background: #FAEEDA;
        border-radius: 6px;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 🤖 Modeling & Evaluation Agent")
    st.markdown("Train machine learning models — with AI-powered recommendations for your dataset.")

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    if st.session_state.get("modeling_log") is None:
        st.session_state.modeling_log = []

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ The dataset must have at least 2 columns.")
        return

    # ── AI Recommendation Section ──────────────────────────────────────────
    st.markdown("### 🤖 AI-Powered Recommendation")

    # Get dataset stats for recommendation
    n_rows = len(df)
    n_cols = len(df.columns)
    n_numeric = len(numeric_cols)
    n_categorical = len(df.select_dtypes(include=["object", "string", "category"]).columns)
    missing_pct = (df.isnull().sum().sum() / df.size * 100) if df.size > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{n_rows:,}")
    with col2:
        st.metric("Columns", n_cols)
    with col3:
        st.metric("Numeric", n_numeric)
    with col4:
        st.metric("Missing", f"{missing_pct:.1f}%")

    # AI Recommendation button
    if st.button("🧠 Get AI Recommendations for This Dataset", use_container_width=False):
        with st.spinner("Analyzing dataset..."):
            sample_data = df.head(5).to_string()

            prompt = f"""Analyze this dataset and provide recommendations for machine learning modeling.

Dataset Info:
- {n_rows} rows, {n_cols} columns
- {n_numeric} numeric columns, {n_categorical} categorical columns
- {missing_pct:.1f}% missing values

Columns: {', '.join(all_cols[:15])}{'...' if len(all_cols) > 15 else ''}

Sample data (first 5 rows):
{sample_data}

Please provide clear, structured recommendations for:

1. **Task Type**: Should this be Classification or Regression? Explain why.
2. **Target Variable**: Which column should be the target (Y)? Explain your choice.
3. **Cross-Validation**: Should we use CV? Yes/No - explain based on dataset size and characteristics.
4. **Hyperparameter Tuning**: Should we use GridSearchCV? Yes/No - explain.
5. **Best Model Types**: Which 2-3 algorithms would likely work best for this data?
6. **Feature Scaling**: Should we scale features? Yes/No - explain.

Format your response with clear headings and bullet points. Be specific and practical."""

            try:
                recommendation = call_llm(prompt, "You are a senior ML engineer providing practical modeling advice.", max_tokens=800)
                st.session_state.ai_modeling_recommendation = recommendation
                st.success("✅ Recommendation generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error generating recommendation: {e}")

    # ── Display AI Recommendation ──────────────────────────────────────────
    if st.session_state.get("ai_modeling_recommendation"):
        st.markdown("---")

        rec_text = st.session_state.ai_modeling_recommendation
        formatted_rec = format_ai_recommendation(rec_text)

        st.markdown(
            f"""<div class="ai-rec-container">
            {formatted_rec}
            </div>""",
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✅ Apply Recommendations", use_container_width=True):
                st.success("✅ Recommendations applied! Configure settings below or click Run Modeling.")
                st.balloons()
        with col2:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.session_state.ai_modeling_recommendation = None
                st.rerun()
        with col3:
            if st.button("✖️ Dismiss", use_container_width=True):
                st.session_state.ai_modeling_recommendation = None
                st.rerun()

        st.markdown("---")

    # ── Pipeline Configuration ────────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Configuration")
    col_target, col_features = st.columns([1, 2])

    with col_target:
        st.markdown("#### 🎯 Target Selection")
        target_col = st.selectbox("Select Target Variable (Y)", all_cols)
        unique_count = df[target_col].nunique()
        is_numeric_target = target_col in numeric_cols

        if is_numeric_target:
            if unique_count <= 10:
                detected_task = "classification"
                st.info(f"💡 Detected: Classification ({unique_count} unique values)")
            elif unique_count > 20:
                detected_task = "regression"
                st.info(f"💡 Detected: Regression ({unique_count} unique values)")
            else:
                detected_task = "classification"
                st.info(f"💡 Target has {unique_count} values - choose task type below")
        else:
            detected_task = "classification"
            st.info("💡 Categorical target → Classification")

        task_type = st.radio(
            "Task Type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            format_func=lambda x: "🏷️ Classification" if x == "classification" else "📈 Regression"
        )

    with col_features:
        st.markdown("#### 📋 Feature Selection")
        default_features = [c for c in all_cols if c != target_col]
        select_all = st.checkbox("Select All Features", value=True)
        if select_all:
            features_list = default_features
            st.info(f"Using all {len(features_list)} remaining columns as features.")
        else:
            features_list = st.multiselect(
                "Select Features (X)",
                [c for c in all_cols if c != target_col],
                default=default_features[:min(8, len(default_features))]
            )

    st.markdown("---")

    # ── Training Settings ─────────────────────────────────────────────────────
    st.markdown("### 🛠️ Training Settings")
    col_split, col_scale, col_extra = st.columns(3)

    with col_split:
        test_size = st.slider("Test Split Size (%)", 10, 40, 20, 5) / 100.0
        random_state = st.number_input("Random Seed", value=42, step=1)

        # Smart CV recommendation
        cv_recommended = "✅ Recommended" if n_rows > 100 else "⚠️ Not recommended (small dataset)"
        use_cv = st.checkbox(f"Enable Cross-Validation ({cv_recommended})", value=n_rows > 100)
        cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=min(5, n_rows // 20 if n_rows > 50 else 3),
                                   disabled=not use_cv)

    with col_scale:
        scaling_method = st.selectbox("Feature Scaling", ["StandardScaler", "MinMaxScaler", "None"])
        st.caption("💡 Scale if features have different units")

    with col_extra:
        if task_type == "classification" and SMOTE_AVAILABLE:
            handle_imbalance = st.checkbox("Handle Class Imbalance (SMOTE)", value=False)
            if handle_imbalance:
                st.caption("⚠️ Only for classification with imbalanced classes")
        else:
            handle_imbalance = False
            if task_type == "classification" and not SMOTE_AVAILABLE:
                st.info("Install imbalanced-learn for SMOTE")

    st.markdown("---")

    # ── Hyperparameter Tuning ─────────────────────────────────────────────────
    st.markdown("### 🔬 Hyperparameter Tuning")
    tune_col1, tune_col2 = st.columns([1, 2])
    with tune_col1:
        grid_recommended = "✅ Recommended" if n_rows > 200 else "⚠️ Not recommended (small dataset - use defaults)"
        enable_tuning = st.checkbox(f"Enable GridSearchCV ({grid_recommended})", value=n_rows > 200,
                                    help="Searches over a pre-defined parameter grid. Adds training time.")
    with tune_col2:
        if enable_tuning:
            tune_cv = st.slider("Tuning CV Folds", 2, 5, 3)
            st.caption("⚠️ More folds = more accurate but slower. Use 2-3 for large datasets.")
        else:
            tune_cv = 3
            st.caption("💡 GridSearchCV disabled. Models will use default parameters.")

    st.markdown("---")

    # ── Model Selection ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")

    if task_type == "classification":
        available_models = {"Random Forest": "rf", "Gradient Boosting": "gbm",
                            "Logistic Regression": "logistic", "Decision Tree": "dt"}
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        default_selection = ["Random Forest"] + (["XGBoost"] if XGB_AVAILABLE else [])
    else:
        available_models = {"Random Forest": "rf", "Gradient Boosting": "gbm",
                            "Linear Regression": "linear", "Ridge Regression": "ridge"}
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        default_selection = ["Random Forest"] + (["XGBoost"] if XGB_AVAILABLE else [])

    selected_model_names = st.multiselect(
        "Select Algorithms to Train",
        list(available_models.keys()),
        default=[m for m in default_selection if m in available_models]
    )

    if not features_list:
        st.warning("⚠️ Please select at least one feature column.")
        return
    if not selected_model_names:
        st.warning("⚠️ Please select at least one algorithm.")
        return

    # ── Smart Settings Summary ──────────────────────────────────────────────
    with st.expander("📊 Settings Summary"):
        st.markdown(f"""
        - **Task**: {task_type.upper()}
        - **Target**: {target_col}
        - **Features**: {len(features_list)} columns
        - **Test Size**: {test_size * 100:.0f}%
        - **Cross-Validation**: {'✅ Enabled' if use_cv else '❌ Disabled'} ({cv_folds} folds)
        - **GridSearchCV**: {'✅ Enabled' if enable_tuning else '❌ Disabled'}
        - **Feature Scaling**: {scaling_method}
        - **SMOTE**: {'✅ Enabled' if handle_imbalance else '❌ Disabled'}
        - **Dataset Size**: {n_rows} rows
        """)

    train_clicked = st.button("🚀 Run Modeling & Evaluation", use_container_width=True, type="primary")

    if train_clicked:
        run_training(df, target_col, features_list, task_type, selected_model_names,
                     available_models, test_size, random_state, scaling_method,
                     cv_folds, handle_imbalance, use_cv, enable_tuning, tune_cv)

    if st.session_state.get("modeling_done"):
        display_results(task_type)
