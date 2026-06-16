import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
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

# Try to import optional libraries with proper error handling
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


def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 0.25rem;">🤖 Modeling & Evaluation Agent</h2>
        <p style="color: var(--text-secondary);">Train machine learning models on your dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        if st.button("← Back to Home"):
            st.session_state.current_page = "home"
            st.rerun()
        return

    df = get_df()
    
    # Initialize log
    if st.session_state.get("modeling_log") is None:
        st.session_state.modeling_log = []

    # Get column listings
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ The dataset must have at least 2 columns to perform modeling.")
        return

    # ── Pipeline Configuration ────────────────────────────────────────────────
    st.markdown("### ⚙️ Pipeline Configuration")
    
    col_target, col_features = st.columns([1, 2])
    
    with col_target:
        st.markdown("#### 🎯 Target Selection")
        
        target_col = st.selectbox(
            "Select Target Variable (Y)",
            all_cols,
            help="The column you want the model to predict."
        )
        
        # Auto-detect task type
        unique_count = df[target_col].nunique()
        is_numeric_target = target_col in numeric_cols
        detected_task = "regression" if (is_numeric_target and unique_count > 12) else "classification"
        
        task_type = st.radio(
            "Task Type",
            ["classification", "regression"],
            index=0 if detected_task == "classification" else 1,
            format_func=lambda x: "🏷️ Classification" if x == "classification" else "📈 Regression"
        )

    with col_features:
        st.markdown("#### 📋 Feature Selection")
        
        default_features = [col for col in all_cols if col != target_col]
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
    
    col_split, col_scale, col_imbalance = st.columns(3)
    
    with col_split:
        test_size = st.slider("Test Split Size (%)", 10, 40, 20, 5) / 100.0
        random_state = st.number_input("Random Seed", value=42, step=1)
        use_cv = st.checkbox("Enable Cross-Validation (slower)", value=False)
        cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=5, disabled=not use_cv)
        
    with col_scale:
        scaling_method = st.selectbox(
            "Feature Scaling",
            ["StandardScaler", "MinMaxScaler", "None"],
            index=0
        )
        
    with col_imbalance:
        if task_type == "classification" and SMOTE_AVAILABLE:
            handle_imbalance = st.checkbox("Handle Class Imbalance (SMOTE)", value=False)
        else:
            handle_imbalance = False
            if task_type == "classification" and not SMOTE_AVAILABLE:
                st.info("Install imbalanced-learn for SMOTE support")

    st.markdown("---")

    # ── Model Selection ─────────────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")
    
    if task_type == "classification":
        available_models = {}
        available_models["Random Forest"] = "rf"
        available_models["Gradient Boosting"] = "gbm"
        available_models["Logistic Regression"] = "logistic"
        available_models["Decision Tree"] = "dt"
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        
        default_selection = ["Random Forest"]
        if XGB_AVAILABLE:
            default_selection.append("XGBoost")
    else:
        available_models = {}
        available_models["Random Forest"] = "rf"
        available_models["Gradient Boosting"] = "gbm"
        available_models["Linear Regression"] = "linear"
        available_models["Ridge Regression"] = "ridge"
        if XGB_AVAILABLE:
            available_models["XGBoost"] = "xgb"
        if LGB_AVAILABLE:
            available_models["LightGBM"] = "lgbm"
        
        default_selection = ["Random Forest"]
        if XGB_AVAILABLE:
            default_selection.append("XGBoost")
        
    selected_model_names = st.multiselect(
        "Select Algorithms to Train",
        list(available_models.keys()),
        default=[m for m in default_selection if m in available_models]
    )

    if len(features_list) == 0:
        st.warning("⚠️ Please select at least one feature column to train models.")
        return
        
    if len(selected_model_names) == 0:
        st.warning("⚠️ Please select at least one algorithm to train.")
        return

    # ── Train button ──────────────────────────────────────────────────────────
    train_clicked = st.button("🚀 Run Modeling & Evaluation", use_container_width=True, type="primary")

    if train_clicked:
        run_training(df, target_col, features_list, task_type, selected_model_names, available_models, 
                     test_size, random_state, scaling_method, cv_folds, handle_imbalance, use_cv)

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("modeling_done"):
        display_results(task_type)


def get_model_instance(model_name, task_type, random_state):
    """Get model instance with default parameters - with error handling."""
    try:
        if task_type == "classification":
            if model_name == "Random Forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif model_name == "Gradient Boosting":
                return GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=random_state
                )
            elif model_name == "Logistic Regression":
                return LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    solver='lbfgs'
                )
            elif model_name == "Decision Tree":
                return DecisionTreeClassifier(
                    max_depth=10,
                    random_state=random_state
                )
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                try:
                    return xgb.XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=random_state,
                        eval_metric='logloss',
                        use_label_encoder=False
                    )
                except:
                    # Fallback with simpler parameters if the above fails
                    return xgb.XGBClassifier(
                        n_estimators=100,
                        random_state=random_state
                    )
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                try:
                    return lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=random_state,
                        verbose=-1
                    )
                except:
                    return lgb.LGBMClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        verbose=-1
                    )
        else:  # regression
            if model_name == "Random Forest":
                return RandomForestRegressor(
                    n_estimators=100,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif model_name == "Gradient Boosting":
                return GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=random_state
                )
            elif model_name == "Linear Regression":
                return LinearRegression()
            elif model_name == "Ridge Regression":
                return Ridge(
                    alpha=1.0,
                    random_state=random_state
                )
            elif model_name == "XGBoost" and XGB_AVAILABLE:
                try:
                    return xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=random_state
                    )
                except:
                    return xgb.XGBRegressor(
                        n_estimators=100,
                        random_state=random_state
                    )
            elif model_name == "LightGBM" and LGB_AVAILABLE:
                try:
                    return lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=random_state,
                        verbose=-1
                    )
                except:
                    return lgb.LGBMRegressor(
                        n_estimators=100,
                        random_state=random_state,
                        verbose=-1
                    )
    except Exception as e:
        st.warning(f"⚠️ Could not load {model_name}: {str(e)}")
        return None
    return None


def run_training(df, target_col, features_list, task_type, selected_model_names, available_models, 
                 test_size, random_state, scaling_method, cv_folds, handle_imbalance, use_cv):
    
    st.session_state.modeling_log = []
    
    def log_step(msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.modeling_log.append(f"[{t}] {msg}")
        
    log_step(f"Starting modeling for target: '{target_col}' ({task_type})")
    
    # 1. Data preparation
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features_list].copy()
    y = df_clean[target_col].copy()
    
    log_step(f"Data shape: {X.shape[0]} rows, {X.shape[1]} features")
    
    # 2. Encode categorical features
    cat_mappings = {}
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        log_step(f"Encoded categorical column: {col}")
    
    # 3. Handle target encoding
    y_mapping = None
    y_inverse_mapping = None
    if task_type == "classification" and y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        y_inverse_mapping = {i: label for i, label in enumerate(le_y.classes_)}
        log_step(f"Encoded target with {len(le_y.classes_)} classes")
    
    # 4. Train-test split
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    log_step(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows")
    
    # 5. Feature scaling
    scaler = None
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    if scaling_method != "None" and len(numeric_cols) > 0:
        log_step(f"Applying {scaling_method} scaling...")
        try:
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        except Exception as e:
            log_step(f"⚠️ Scaling failed: {str(e)}")
            scaler = None
    
    # 6. SMOTE for imbalance
    if handle_imbalance and task_type == "classification" and SMOTE_AVAILABLE:
        try:
            log_step("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            log_step(f"After SMOTE: {len(X_train)} rows")
        except Exception as e:
            log_step(f"⚠️ SMOTE failed: {str(e)}")
    
    # 7. Train models
    metrics_report = {}
    trained_models = {}
    
    console_ph = st.empty()
    
    for model_name in selected_model_names:
        log_step(f"Training {model_name}...")
        
        # Get model instance
        model = get_model_instance(model_name, task_type, random_state)
        
        if model is None:
            log_step(f"⚠️ {model_name} not available, skipping...")
            continue
        
        try:
            # Fit model
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            model_metrics = calculate_metrics(y_test, y_pred, task_type, model, X_test)
            
            # Cross-validation score (optional)
            if use_cv:
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, 3), 
                                                scoring='accuracy' if task_type == 'classification' else 'r2')
                    model_metrics["cv_mean"] = cv_scores.mean()
                    model_metrics["cv_std"] = cv_scores.std()
                except Exception as e:
                    log_step(f"⚠️ CV failed for {model_name}: {str(e)}")
            
            metrics_report[model_name] = model_metrics
            log_step(f"✓ {model_name} complete")
            
        except Exception as e:
            log_step(f"❌ {model_name} failed: {str(e)}")
            continue
        
        # Show progress
        with console_ph.container():
            st.markdown("#### 🪵 Training Log")
            log_text = "\n".join(st.session_state.modeling_log[-10:])
            st.code(log_text, language="text")
    
    # 8. Find best model
    if metrics_report:
        try:
            metric_key = "Accuracy" if task_type == "classification" else "R2 Score"
            best_model_name = max(metrics_report.items(), key=lambda x: x[1].get(metric_key, -np.inf))[0]
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
                
                # Generate AI recommendation
                try:
                    with st.spinner("🧠 Generating AI model recommendation..."):
                        recommendation = generate_ai_recommendation(metrics_report, task_type, features_list, len(df_clean), best_model_name)
                        st.session_state.model_recommendation = recommendation
                except Exception as e:
                    st.session_state.model_recommendation = f"⚠️ Could not generate recommendation: {str(e)}"
                
                log_step(f"🏆 Best model: {best_model_name}")
                st.success(f"✅ Modeling complete! Best model: {best_model_name}")
            else:
                st.error("Best model could not be loaded.")
        except Exception as e:
            st.error(f"Error selecting best model: {str(e)}")
    else:
        st.error("No models were successfully trained. Please check the logs above for details.")


def calculate_metrics(y_test, y_pred, task_type, model, X_test):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    try:
        if task_type == "classification":
            metrics["Accuracy"] = accuracy_score(y_test, y_pred)
            metrics["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
        else:
            metrics["R2 Score"] = r2_score(y_test, y_pred)
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)
            metrics["MSE"] = mean_squared_error(y_test, y_pred)
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
    except Exception as e:
        st.warning(f"Error calculating metrics: {str(e)}")
    
    # Feature importance
    try:
        if hasattr(model, "feature_importances_"):
            metrics["feature_importances"] = model.feature_importances_.tolist()
    except:
        pass
    
    # Store predictions
    try:
        metrics["y_test"] = y_test.tolist()[:100]
        metrics["y_pred"] = y_pred.tolist()[:100]
    except:
        pass
    
    return metrics


def generate_ai_recommendation(metrics_report, task_type, features_list, n_rows, best_model):
    """Generate AI-powered model recommendation."""
    
    # Build summary of all models
    summary_lines = [f"Dataset: {n_rows} rows, {len(features_list)} features"]
    summary_lines.append(f"Task: {task_type}")
    summary_lines.append(f"Best model: {best_model}\n")
    summary_lines.append("Model Performance Summary:")
    
    for model_name, metrics in metrics_report.items():
        if task_type == "classification":
            acc = metrics.get("Accuracy", 0)
            f1 = metrics.get("F1-Score", 0)
            summary_lines.append(f"- {model_name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        else:
            r2 = metrics.get("R2 Score", 0)
            rmse = metrics.get("RMSE", 0)
            summary_lines.append(f"- {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")
    
    summary = "\n".join(summary_lines)
    
    prompt = f"""Based on the following model performance results, provide a recommendation:

{summary}

Please provide:
1. **Best Model Recommendation**: Which model is best and why?
2. **Key Insights**: What do the results tell us about the data?
3. **Improvement Suggestions**: 2-3 specific ways to improve model performance
4. **Business Implications**: What do these results mean for real-world use?

Keep it concise and actionable. Use markdown formatting with bullet points.
"""
    
    system_prompt = "You are a senior machine learning engineer providing model recommendations."
    
    try:
        recommendation = call_llm(prompt, system_prompt, max_tokens=600)
        return recommendation
    except Exception as e:
        return f"⚠️ Could not generate AI recommendation: {str(e)}"


def create_model_comparison_chart(metrics_dict: dict, task_type: str):
    """Create bar chart comparing model performance metrics."""
    
    if not metrics_dict:
        return None
    
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        
        if task_type == "classification":
            metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
        else:
            metric_keys = ["R2 Score", "MAE", "RMSE"]
        
        for key in metric_keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                row[key] = round(metrics[key], 4)
        
        if "cv_mean" in metrics and metrics["cv_mean"]:
            row["CV Score"] = round(metrics["cv_mean"], 4)
            
        comparison_data.append(row)
    
    if not comparison_data:
        return None
        
    df_comparison = pd.DataFrame(comparison_data)
    
    if task_type == "classification":
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        title = "📊 Model Performance Comparison (Classification)"
        y_title = "Score (higher is better)"
    else:
        metrics_to_plot = ["R2 Score"]
        title = "📊 Model Performance Comparison (Regression)"
        y_title = "R² Score (higher is better)"
    
    available_metrics = [m for m in metrics_to_plot if m in df_comparison.columns]
    
    if not available_metrics:
        return None
    
    fig = px.bar(
        df_comparison,
        x="Model",
        y=available_metrics,
        barmode="group",
        title=title,
        labels={"value": y_title, "variable": "Metric", "Model": "Algorithm"},
        color_discrete_sequence=px.colors.sequential.Viridis,
        text_auto='.3f'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0.2)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e2f0")
    )
    
    fig.update_traces(
        textposition="outside",
        marker_line_width=1,
        marker_line_color="white"
    )
    
    return fig


def create_feature_importance_chart(feature_importances, features_list, top_n: int = 10):
    """Create horizontal bar chart for feature importance."""
    
    if not feature_importances or not features_list:
        return None
    
    # Handle both dict and list formats
    try:
        if isinstance(feature_importances, list):
            # Ensure we have the right length
            min_len = min(len(features_list), len(feature_importances))
            importances_dict = dict(zip(features_list[:min_len], feature_importances[:min_len]))
        else:
            importances_dict = feature_importances
        
        sorted_features = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        df_importance = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
        
        fig = px.bar(
            df_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Feature Importances",
            labels={"Importance": "Importance Score", "Feature": ""},
            color="Importance",
            color_continuous_scale="Viridis",
            text_auto='.3f'
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            yaxis=dict(categoryorder="total ascending"),
            plot_bgcolor="rgba(0,0,0,0.2)",
            coloraxis_showscale=False
        )
        
        fig.update_traces(marker_line_width=0)
        
        return fig
    except Exception as e:
        return None


def create_confusion_matrix_heatmap(cm_matrix):
    """Create confusion matrix heatmap."""
    
    if not cm_matrix:
        return None
    
    try:
        fig = px.imshow(
            cm_matrix,
            text_auto=True,
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale="Blues",
            aspect="auto"
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=450,
            plot_bgcolor="rgba(0,0,0,0.2)"
        )
        
        return fig
    except Exception as e:
        return None


def display_results(task_type):
    """Display model results with visualizations."""
    metrics = st.session_state.model_metrics
    best_model_name = st.session_state.trained_model_name
    
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "📈 Visualizations", "🤖 AI Recommendation", "📋 Details"])
    
    with tab1:
        st.markdown("#### Model Comparison Table")
        comparison_data = []
        for model_name, data in metrics.items():
            row = {"Model": model_name}
            for metric, val in data.items():
                if isinstance(val, (int, float)) and metric not in ["confusion_matrix"]:
                    row[metric] = round(val, 4)
            comparison_data.append(row)
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Highlight best model
            st.success(f"💡 **Best Model:** {best_model_name}")
    
    with tab2:
        st.markdown("#### Model Comparison Charts")
        
        # Bar chart
        fig_bar = create_model_comparison_chart(metrics, task_type)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Feature importance for best model
        best_metrics = metrics.get(best_model_name, {})
        if "feature_importances" in best_metrics and st.session_state.model_features_list:
            st.markdown("#### Feature Importance (Best Model)")
            fig_importance = create_feature_importance_chart(
                best_metrics["feature_importances"], 
                st.session_state.model_features_list, 
                top_n=10
            )
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        
        # Confusion matrix for classification
        if task_type == "classification" and "confusion_matrix" in best_metrics:
            st.markdown("#### Confusion Matrix (Best Model)")
            fig_cm = create_confusion_matrix_heatmap(best_metrics["confusion_matrix"])
            if fig_cm:
                st.plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        st.markdown("#### 🤖 AI Model Recommendation")
        if st.session_state.get("model_recommendation"):
            recommendation = st.session_state.model_recommendation
            st.markdown(recommendation)
        else:
            st.info("AI recommendation not available. Click 'Run Modeling' again to generate.")
    
    with tab4:
        st.markdown("#### Best Model Details")
        
        best_metrics = metrics.get(best_model_name, {})
        
        if task_type == "classification":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{best_metrics.get('Accuracy', 0):.4f}")
            with col2:
                st.metric("Precision", f"{best_metrics.get('Precision', 0):.4f}")
            with col3:
                st.metric("Recall", f"{best_metrics.get('Recall', 0):.4f}")
            with col4:
                st.metric("F1-Score", f"{best_metrics.get('F1-Score', 0):.4f}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{best_metrics.get('R2 Score', 0):.4f}")
            with col2:
                st.metric("MAE", f"{best_metrics.get('MAE', 0):.4f}")
            with col3:
                st.metric("RMSE", f"{best_metrics.get('RMSE', 0):.4f}")
        
        if "cv_mean" in best_metrics and best_metrics["cv_mean"]:
            st.metric("Cross-Validation Score", f"{best_metrics['cv_mean']:.4f} ± {best_metrics.get('cv_std', 0):.4f}")
        
        st.markdown("#### Model Configuration")
        st.markdown(f"""
        - **Task Type:** `{task_type.upper()}`
        - **Target Variable:** `{st.session_state.model_target_col}`
        - **Features Used:** {len(st.session_state.model_features_list)} columns
        - **Training/Test Split:** {int((1 - 0.2) * 100)}% / 20%
        """)
    
    # Navigation
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
