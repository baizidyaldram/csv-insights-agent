import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, r2_score, mean_absolute_error, mean_squared_error
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap
from utils.session import get_df, is_data_loaded

def render():
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 0.25rem;">🤖 Modeling & Evaluation Agent</h2>
        <p style="color: var(--text-secondary);">Train machine learning models with advanced techniques: XGBoost, Hyperparameter Tuning, SHAP explanations</p>
    </div>
    """, unsafe_allow_html=True)

    if not is_data_loaded():
        st.warning("No data loaded. Please upload a CSV on the Home page first.")
        return

    df = get_df()
    
    # ── Sidebar Agent Console Logs ──────────────────────────────────────────────
    if st.session_state.get("modeling_log") is None:
        st.session_state.modeling_log = []

    # Get column listings
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(all_cols) < 2:
        st.error("❌ The dataset must have at least 2 columns to perform modeling.")
        return

    # ── Pipeline configurations ────────────────────────────────────────────────
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
        
        # Handle class imbalance
        if task_type == "classification":
            handle_imbalance = st.checkbox("Handle Class Imbalance (SMOTE)", value=True, help="Synthetic Minority Over-sampling Technique")

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

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Advanced Training Settings ─────────────────────────────────────────────
    st.markdown("### 🛠️ Advanced Training Settings")
    
    with st.expander("Configure Training Parameters", expanded=True):
        col_split, col_scale, col_tune = st.columns(3)
        
        with col_split:
            test_size = st.slider("Test Split Size (%)", 10, 40, 20, 5) / 100.0
            random_state = st.number_input("Random Seed", value=42, step=1)
            cv_folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5)
            
        with col_scale:
            scaling_method = st.selectbox(
                "Feature Scaling",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                index=0
            )
            
        with col_tune:
            enable_hyperopt = st.checkbox("🔧 Hyperparameter Tuning", value=True, help="Use RandomizedSearchCV for better performance")
            tune_iterations = st.slider("Tuning Iterations", 10, 50, 20, disabled=not enable_hyperopt)
    
    # ── Model Selection (Enhanced) ─────────────────────────────────────────────
    st.markdown("### 🤖 Model Selection")
    
    if task_type == "classification":
        available_models = {
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Random Forest": "rf",
            "Gradient Boosting": "gbm",
            "Logistic Regression": "logistic",
            "Decision Tree": "dt"
        }
        default_selection = ["XGBoost", "Random Forest", "LightGBM"]
    else:
        available_models = {
            "XGBoost": "xgb",
            "LightGBM": "lgbm",
            "Random Forest": "rf",
            "Gradient Boosting": "gbm",
            "Linear Regression": "linear",
            "Ridge Regression": "ridge"
        }
        default_selection = ["XGBoost", "Random Forest", "LightGBM"]
        
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
                     test_size, random_state, scaling_method, cv_folds, enable_hyperopt, tune_iterations,
                     handle_imbalance if task_type == "classification" else False)

    # ── Display results with SHAP ─────────────────────────────────────────────
    if st.session_state.get("modeling_done"):
        display_results(task_type)

def run_training(df, target_col, features_list, task_type, selected_model_names, available_models, 
                 test_size, random_state, scaling_method, cv_folds, enable_hyperopt, tune_iterations,
                 handle_imbalance):
    
    st.session_state.modeling_log = []
    
    def log_step(msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.modeling_log.append(f"[{t}] 🤖 {msg}")
        
    log_step(f"Starting pipeline for predicting '{target_col}' ({task_type})")
    
    # 1. Data preparation
    df_clean = df.dropna(subset=[target_col]).copy()
    X = df_clean[features_list].copy()
    y = df_clean[target_col].copy()
    
    # Identify feature types
    num_feats = X.select_dtypes(include="number").columns.tolist()
    cat_feats = X.select_dtypes(include="object").columns.tolist()
    log_step(f"Features: {len(num_feats)} numeric, {len(cat_feats)} categorical")
    
    # 2. Preprocessing
    log_step("Preprocessing features...")
    
    # Encode categorical features
    cat_mappings = {}
    for col in cat_feats:
        X[col] = X[col].fillna("Missing").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Handle target encoding for classification
    y_mapping = None
    y_inverse_mapping = None
    if task_type == "classification" and y.dtype == "object":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
        y_inverse_mapping = {i: label for i, label in enumerate(le_y.classes_)}
    
    # 3. Train-test split
    log_step(f"Splitting data (test size: {int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == "classification" else None
    )
    
    # 4. Feature scaling
    scaler = None
    if scaling_method != "None" and len(num_feats) > 0:
        log_step(f"Applying {scaling_method} scaling...")
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaling_method == "RobustScaler":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        
        X_train[num_feats] = scaler.fit_transform(X_train[num_feats])
        X_test[num_feats] = scaler.transform(X_test[num_feats])
    
    # 5. Handle class imbalance with SMOTE
    if task_type == "classification" and handle_imbalance:
        log_step("Applying SMOTE for class imbalance...")
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # 6. Train models
    metrics_report = {}
    trained_models = {}
    best_model = None
    best_score = -np.inf
    
    console_ph = st.empty()
    
    for model_name in selected_model_names:
        log_step(f"Training {model_name}...")
        
        # Get model instance
        model = get_model_instance(model_name, task_type, random_state)
        
        # Hyperparameter tuning if enabled
        if enable_hyperopt and model_name in ["XGBoost", "LightGBM", "Random Forest"]:
            log_step(f"  → Tuning hyperparameters for {model_name}...")
            param_dist = get_param_distribution(model_name, task_type)
            model = RandomizedSearchCV(
                model, param_dist, n_iter=tune_iterations, 
                cv=min(3, cv_folds), scoring='accuracy' if task_type == 'classification' else 'r2',
                random_state=random_state, n_jobs=-1
            )
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Store model
        if enable_hyperopt and model_name in ["XGBoost", "LightGBM", "Random Forest"]:
            trained_models[model_name] = model.best_estimator_
            model = model.best_estimator_
        else:
            trained_models[model_name] = model
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=min(cv_folds, 3), 
                                    scoring='accuracy' if task_type == 'classification' else 'r2')
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        model_metrics = calculate_metrics(y_test, y_pred, task_type, model, X_test)
        model_metrics["cv_mean"] = cv_scores.mean()
        model_metrics["cv_std"] = cv_scores.std()
        
        metrics_report[model_name] = model_metrics
        
        # Track best model
        metric_key = "Accuracy" if task_type == "classification" else "R2 Score"
        current_score = model_metrics.get(metric_key, -np.inf)
        if current_score > best_score:
            best_score = current_score
            best_model = model
        
        log_step(f"✓ {model_name} complete (CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")
    
    # Show logs
    with console_ph.container():
        st.markdown("#### 🪵 Agent Logs")
        st.code("\n".join(st.session_state.modeling_log), language="text")
    
    # Store results
    st.session_state.model_metrics = metrics_report
    st.session_state.trained_model = best_model
    st.session_state.trained_model_name = max(metrics_report.items(), key=lambda x: x[1].get('Accuracy' if task_type == 'classification' else 'R2 Score', -np.inf))[0]
    st.session_state.model_features_list = features_list
    st.session_state.model_target_col = target_col
    st.session_state.model_scaler = scaler
    st.session_state.model_task_type = task_type
    st.session_state.model_encoded_categories = cat_mappings
    st.session_state.model_target_mapping = y_mapping
    st.session_state.model_target_inverse_mapping = y_inverse_mapping
    st.session_state.modeling_done = True
    
    st.success("✅ Modeling complete! Best model: " + st.session_state.trained_model_name)
    st.rerun()

def get_model_instance(model_name, task_type, random_state):
    """Get model instance with default parameters."""
    if task_type == "classification":
        models = {
            "XGBoost": xgb.XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False),
            "LightGBM": lgb.LGBMClassifier(random_state=random_state, verbose=-1),
            "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state)
        }
    else:
        models = {
            "XGBoost": xgb.XGBRegressor(random_state=random_state),
            "LightGBM": lgb.LGBMRegressor(random_state=random_state, verbose=-1),
            "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=random_state)
        }
    return models.get(model_name, RandomForestClassifier(random_state=random_state))

def get_param_distribution(model_name, task_type):
    """Get hyperparameter distribution for tuning."""
    if task_type == "classification":
        if model_name == "XGBoost":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        elif model_name == "LightGBM":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 50, 100]
            }
        elif model_name == "Random Forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
    else:
        if model_name == "XGBoost":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        elif model_name == "LightGBM":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        elif model_name == "Random Forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
    return {}

def calculate_metrics(y_test, y_pred, task_type, model, X_test):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    if task_type == "classification":
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        metrics["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        metrics["F1-Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # ROC AUC for binary
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            metrics["ROC AUC"] = auc(fpr, tpr)
            metrics["roc_curve"] = (fpr.tolist(), tpr.tolist())
    else:
        metrics["R2 Score"] = r2_score(y_test, y_pred)
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
    
    # Feature importance
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = model.feature_importances_.tolist()
    elif hasattr(model, "coef_"):
        if hasattr(model, "best_estimator_"):
            coef = model.best_estimator_.coef_
        else:
            coef = model.coef_
        if len(coef.shape) > 1:
            metrics["feature_importances"] = np.mean(np.abs(coef), axis=0).tolist()
        else:
            metrics["feature_importances"] = np.abs(coef).flatten().tolist()
    
    # Store predictions
    metrics["y_test"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    
    return metrics

def display_results(task_type):
    """Display results with SHAP explanations."""
    metrics = st.session_state.model_metrics
    best_model_name = st.session_state.trained_model_name
    best_model = st.session_state.trained_model
    
    st.markdown("---")
    st.markdown("### 🏆 Model Comparison")
    
    # Build comparison dataframe
    comp_data = []
    for model_name, data in metrics.items():
        row = {"Model": model_name}
        for metric, val in data.items():
            if isinstance(val, (int, float)):
                row[metric] = round(val, 4)
        comp_data.append(row)
    
    comp_df = pd.DataFrame(comp_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.success(f"💡 **Best Model:** {best_model_name} (selected for detailed analysis)")
    
    # ── SHAP Explanations ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Model Explainability (SHAP)")
    st.markdown("Understanding what drives your model's predictions")
    
    if st.button("📊 Generate SHAP Analysis", use_container_width=False):
        with st.spinner("Computing SHAP values (this may take a moment)..."):
            try:
                # Get feature names and a sample for SHAP
                feature_names = st.session_state.model_features_list
                X_sample = st.session_state.get("shap_sample")
                
                if X_sample is None:
                    # Use test data from best model
                    from sklearn.model_selection import train_test_split
                    df = get_df()
                    X = df[feature_names].copy()
                    
                    # Encode categoricals
                    cat_mappings = st.session_state.model_encoded_categories
                    for col, mapping in cat_mappings.items():
                        if col in X.columns:
                            X[col] = X[col].fillna("Missing").astype(str)
                            X[col] = X[col].map(lambda x: mapping.get(x, 0))
                    
                    # Take sample for SHAP
                    X_sample = X.sample(min(100, len(X)), random_state=42)
                    st.session_state.shap_sample = X_sample
                
                # Create SHAP explainer
                if hasattr(best_model, "predict_proba"):
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Summary plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Feature importance bar plot
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(shap_values).mean(axis=0)
                    }).sort_values('importance', ascending=True)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    ax2.barh(importance_df['feature'], importance_df['importance'], color='#10B981')
                    ax2.set_xlabel('Mean |SHAP Value|')
                    ax2.set_title('SHAP Feature Importance')
                    st.pyplot(fig2)
                    plt.close()
                    
                else:
                    st.info("SHAP analysis is best supported for tree-based models. Your model type may have limited SHAP support.")
                    
            except Exception as e:
                st.warning(f"SHAP analysis could not be completed: {str(e)}")
    
    # ── Feature Importance ────────────────────────────────────────────────────
    best_metrics = metrics[best_model_name]
    if "feature_importances" in best_metrics:
        st.markdown("### 📊 Feature Importance")
        feat_imp = best_metrics["feature_importances"]
        features = st.session_state.model_features_list
        
        if len(feat_imp) == len(features):
            imp_df = pd.DataFrame({"Feature": features, "Importance": feat_imp}).sort_values("Importance", ascending=True)
            
            fig = px.bar(
                imp_df.tail(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Feature Importance - {best_model_name}",
                color="Importance",
                color_continuous_scale="Greens",
                template="plotly_dark"
            )
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    # Navigation
    st.markdown("---")
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back to Visualization", use_container_width=True):
            st.session_state.current_page = "visualization"
            st.rerun()
    with col_nav2:
        if st.button("➡️ Proceed to AI Insights", use_container_width=True):
            st.session_state.current_page = "insights"
            st.rerun()
