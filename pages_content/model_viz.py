import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def create_model_comparison_chart(metrics_dict: dict, task_type: str):
    """Create bar chart comparing model performance metrics."""
    
    # Extract metrics for comparison
    comparison_data = []
    
    for model_name, metrics in metrics_dict.items():
        row = {"Model": model_name}
        
        if task_type == "classification":
            metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score"]
        else:
            metric_keys = ["R2 Score", "MAE", "RMSE"]
        
        for key in metric_keys:
            if key in metrics and isinstance(metrics[key], (int, float)):
                # For error metrics, lower is better - but we'll handle display
                row[key] = round(metrics[key], 4)
        
        # Add CV score if available
        if "cv_mean" in metrics and metrics["cv_mean"]:
            row["CV Score"] = round(metrics["cv_mean"], 4)
            
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create figure
    if task_type == "classification":
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        title = "📊 Model Performance Comparison (Classification)"
        y_title = "Score (higher is better)"
    else:
        metrics_to_plot = ["R2 Score"]
        title = "📊 Model Performance Comparison (Regression)"
        y_title = "R² Score (higher is better)"
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in df_comparison.columns]
    
    if not available_metrics:
        return None
    
    # Create bar chart
    fig = px.bar(
        df_comparison,
        x="Model",
        y=available_metrics,
        barmode="group",
        title=title,
        labels={"value": y_title, "variable": "Metric", "Model": "Algorithm"},
        color_discrete_sequence=["#EF9F27","#D85A30","#BA7517","#FAC775","#F0997B"],
        text_auto='.3f'
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#3D3530")
    )
    
    fig.update_traces(
        textposition="outside",
        marker_line_width=1,
        marker_line_color="white"
    )
    
    return fig


def create_radar_chart(metrics_dict: dict, best_model: str):
    """Create radar chart comparing top models across multiple metrics."""
    
    # Select top 3 models plus best model
    models = list(metrics_dict.keys())[:4]
    if best_model not in models:
        models[0] = best_model
    
    metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "cv_mean"]
    
    fig = go.Figure()
    
    for model in models:
        if model not in metrics_dict:
            continue
            
        values = []
        for metric in metrics_list:
            val = metrics_dict[model].get(metric, 0)
            if isinstance(val, (int, float)):
                values.append(round(val, 3))
            else:
                values.append(0)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_list,
            fill='toself',
            name=model,
            line=dict(width=2),
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(family="Inter, sans-serif", size=12, color="#3D3530")
            ),
            angularaxis=dict(
                tickfont=dict(color="#e2e2f0", size=10)
            )
        ),
        template="plotly_white",
        height=450,
        title="📈 Multi-Metric Radar Comparison",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_feature_importance_chart(feature_importances: dict, top_n: int = 10):
    """Create horizontal bar chart for feature importance."""
    
    if not feature_importances:
        return None
    
    # Sort and get top N
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    df_importance = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
    
    fig = px.bar(
        df_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"🔑 Top {top_n} Feature Importances",
        labels={"Importance": "Importance Score", "Feature": ""},
        color="Importance",
        color_continuous_scale="YlOrBr",
        text_auto='.3f'
    )
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        yaxis=dict(categoryorder="total ascending"),
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    
    fig.update_traces(marker_line_width=0)
    
    return fig


def create_confusion_matrix_heatmap(cm_matrix, class_names=None):
    """Create confusion matrix heatmap."""
    
    if not cm_matrix:
        return None
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm_matrix))]
    
    fig = px.imshow(
        cm_matrix,
        text_auto=True,
        title="🎯 Confusion Matrix",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="YlOrBr",
        aspect="auto"
    )
    
    fig.update_layout(
        template="plotly_white",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig


def generate_model_recommendation(metrics_dict: dict, task_type: str, dataset_size: int):
    """Generate intelligent model recommendation based on performance and context."""
    
    if not metrics_dict:
        return None
    
    # Find best model by primary metric
    primary_metric = "Accuracy" if task_type == "classification" else "R2 Score"
    
    best_model = None
    best_score = -1
    
    for model, metrics in metrics_dict.items():
        score = metrics.get(primary_metric, -1)
        if isinstance(score, (int, float)) and score > best_score:
            best_score = score
            best_model = model
    
    if not best_model:
        return None
    
    best_metrics = metrics_dict[best_model]
    
    # Generate reasoning
    reasoning = []
    
    # Performance analysis
    if task_type == "classification":
        accuracy = best_metrics.get("Accuracy", 0)
        f1 = best_metrics.get("F1-Score", 0)
        
        if accuracy >= 0.9:
            reasoning.append(f"✅ **Excellent predictive power** ({accuracy*100:.1f}% accuracy)")
        elif accuracy >= 0.8:
            reasoning.append(f"👍 **Strong predictive power** ({accuracy*100:.1f}% accuracy)")
        elif accuracy >= 0.7:
            reasoning.append(f"📊 **Moderate predictive power** ({accuracy*100:.1f}% accuracy)")
        else:
            reasoning.append(f"⚠️ **Room for improvement** ({accuracy*100:.1f}% accuracy)")
        
        # Balance check
        precision = best_metrics.get("Precision", 0)
        recall = best_metrics.get("Recall", 0)
        if abs(precision - recall) < 0.05:
            reasoning.append("⚖️ **Well-balanced** between precision and recall")
        else:
            reasoning.append(f"⚖️ **Trade-off detected**: Precision={precision:.3f}, Recall={recall:.3f}")
        
        # F1 score interpretation
        if f1 >= 0.85:
            reasoning.append("🎯 **Excellent F1-score** - great overall performance")
        elif f1 >= 0.7:
            reasoning.append("🎯 **Good F1-score** - reliable predictions")
            
    else:  # regression
        r2 = best_metrics.get("R2 Score", 0)
        rmse = best_metrics.get("RMSE", 0)
        
        if r2 >= 0.9:
            reasoning.append(f"✅ **Excellent fit** (R² = {r2:.3f})")
        elif r2 >= 0.7:
            reasoning.append(f"👍 **Strong fit** (R² = {r2:.3f})")
        elif r2 >= 0.5:
            reasoning.append(f"📊 **Moderate fit** (R² = {r2:.3f})")
        else:
            reasoning.append(f"⚠️ **Weak fit** (R² = {r2:.3f}) - consider feature engineering")
        
        reasoning.append(f"📏 **Prediction error**: RMSE = {rmse:.4f}")
    
    # Cross-validation stability
    cv_mean = best_metrics.get("cv_mean", None)
    cv_std = best_metrics.get("cv_std", None)
    
    if cv_mean and cv_std:
        if cv_std < 0.05:
            reasoning.append(f"🔄 **Highly stable** across {cv_folds}-fold CV (std = {cv_std:.4f})")
        elif cv_std < 0.1:
            reasoning.append(f"🔄 **Moderately stable** across CV folds (std = {cv_std:.4f})")
        else:
            reasoning.append(f"🔄 **High variance** detected - consider more data or regularization")
    
    # Dataset size consideration
    if dataset_size < 1000:
        reasoning.append("💡 **Small dataset detected** - consider simpler models or data augmentation")
    elif dataset_size > 100000:
        reasoning.append("🚀 **Large dataset** - gradient boosting or deep learning could improve results")
    
    # Alternative recommendations
    alternatives = []
    for model, metrics in metrics_dict.items():
        if model != best_model:
            score = metrics.get(primary_metric, 0)
            if isinstance(score, (int, float)):
                diff = (best_score - score) * 100
                if diff < 5:
                    alternatives.append(f"{model} (within {diff:.1f}% of best)")
    
    recommendation = {
        "best_model": best_model,
        "best_score": best_score,
        "reasoning": reasoning,
        "alternatives": alternatives[:3],
        "primary_metric": primary_metric
    }
    
    return recommendation
