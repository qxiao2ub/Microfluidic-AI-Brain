from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.microfluidic_core import (
    make_input_template,
    predict_outputs,
    prediction_uncertainty,
    range_warnings,
    read_dataset,
    recommend_conditions,
    train_model,
)

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_ROOT / "data" / "Comprehensive_normalized.xlsx"

st.set_page_config(
    page_title="Microfluidic Droplet AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    .hero {
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(125deg, rgba(21,94,117,.16), rgba(13,148,136,.10));
        border: 1px solid rgba(13,148,136,.28);
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0 0 .25rem 0;}
    .small-note {font-size: .9rem; opacity: .82;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the microfluidic ensemble model...")
def build_bundle(file_bytes: bytes, filename: str):
    data = read_dataset(file_bytes, filename=filename)
    return train_model(data)


def numeric_input_widget(label: str, profile: dict, key: str):
    unique_values = profile.get("unique_values")
    if unique_values and len(unique_values) <= 20:
        options = [float(value) for value in unique_values]
        default_index = min(
            range(len(options)),
            key=lambda index: abs(options[index] - float(profile["median"])),
        )
        return st.selectbox(label, options=options, index=default_index, key=key)

    minimum = float(profile["min"])
    maximum = float(profile["max"])
    median = float(profile["median"])
    span = max(maximum - minimum, abs(median), 1.0)
    step = span / 200.0
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=min(max(median, minimum), maximum),
        step=step,
        format="%.6g",
        key=key,
    )


st.markdown(
    """
    <div class="hero">
      <h1>🧪 Microfluidic Droplet AI</h1>
      <div>Multi-output machine learning for emulsion droplet prediction, process exploration, batch scoring, and inverse design.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Training dataset")
    uploaded_training = st.file_uploader(
        "Use a replacement training dataset",
        type=["csv", "xlsx", "xlsm"],
        help="Leave empty to use the included Comprehensive_normalized.xlsx dataset.",
    )
    if uploaded_training is None:
        training_name = DEFAULT_DATA_PATH.name
        training_bytes = DEFAULT_DATA_PATH.read_bytes()
        st.success(f"Using included dataset: {training_name}")
    else:
        training_name = uploaded_training.name
        training_bytes = uploaded_training.getvalue()
        st.info(f"Using uploaded dataset: {training_name}")

    st.divider()
    st.caption("The app retrains and caches the model when the training file changes.")

try:
    bundle = build_bundle(training_bytes, training_name)
except Exception as exc:
    st.error(f"The model could not be trained: {exc}")
    st.stop()

page_overview, page_predict, page_batch, page_inverse, page_data, page_about = st.tabs(
    [
        "Overview",
        "Single Prediction",
        "Batch Prediction",
        "Inverse Design",
        "Data Explorer",
        "Method & Safety",
    ]
)

with page_overview:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Training rows", f"{len(bundle.cleaned_data):,}")
    metric_2.metric("Base inputs", len(bundle.base_feature_columns))
    metric_3.metric("Engineered + base features", len(bundle.feature_columns))
    metric_4.metric("Predicted outputs", len(bundle.target_columns))

    st.subheader("Holdout performance")
    formatted_metrics = bundle.metrics.copy()
    for column in ["R2", "RMSE", "MAE", "MAPE (%)"]:
        formatted_metrics[column] = formatted_metrics[column].map(lambda value: f"{value:,.4g}")
    st.dataframe(formatted_metrics, use_container_width=True, hide_index=True)
    st.caption(
        "Metrics use a fixed 80/20 holdout split. They are prototype estimates, not a substitute for external experimental validation."
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        st.subheader("Most influential model features")
        importance = bundle.feature_importance.head(15).sort_values("Importance")
        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="ExtraTrees feature importance",
        )
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Observed vs. predicted holdout results")
        selected_target = st.selectbox(
            "Output",
            bundle.target_columns,
            key="overview_target",
        )
        observed = bundle.holdout_actual[selected_target]
        predicted = bundle.holdout_predicted[f"Predicted {selected_target}"]
        compare = pd.DataFrame({"Observed": observed, "Predicted": predicted})
        scatter = px.scatter(compare, x="Observed", y="Predicted", trendline=None)
        minimum = float(np.nanmin(compare.to_numpy()))
        maximum = float(np.nanmax(compare.to_numpy()))
        scatter.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                name="Ideal",
                line=dict(dash="dash"),
            )
        )
        scatter.update_layout(height=520, margin=dict(l=20, r=20, t=25, b=20))
        st.plotly_chart(scatter, use_container_width=True)

with page_predict:
    st.subheader("Predict one operating condition")
    st.write("Enter laboratory or process settings inside the observed data ranges.")

    with st.form("single_prediction_form"):
        columns = st.columns(3)
        input_values: dict[str, float] = {}
        for index, feature in enumerate(bundle.base_feature_columns):
            with columns[index % 3]:
                input_values[feature] = numeric_input_widget(
                    feature,
                    bundle.base_feature_profile[feature],
                    key=f"single_{feature}",
                )
        submitted = st.form_submit_button("Predict droplet outputs", type="primary")

    if submitted:
        single_input = pd.DataFrame([input_values])
        predictions = predict_outputs(bundle, single_input)
        uncertainty = prediction_uncertainty(bundle, single_input)
        result = pd.concat([predictions, uncertainty], axis=1)

        result_columns = st.columns(len(bundle.target_columns))
        for index, target in enumerate(bundle.target_columns):
            result_columns[index].metric(
                target,
                f"{predictions.iloc[0][target]:,.5g}",
                help=f"Ensemble standard deviation: {uncertainty.iloc[0, index]:,.5g}",
            )
        st.dataframe(result, use_container_width=True, hide_index=True)

        warnings = range_warnings(bundle, single_input)
        if warnings:
            st.warning("\n".join(warnings))
        else:
            st.success("All inputs are inside the observed training ranges.")

with page_batch:
    st.subheader("Score many operating conditions")
    st.write(
        "Upload a CSV or Excel file containing the required base input columns. Extra columns are preserved in the downloaded result."
    )

    template = make_input_template(bundle)
    st.download_button(
        "Download batch-input template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="microfluidic_batch_input_template.csv",
        mime="text/csv",
    )

    batch_file = st.file_uploader(
        "Batch input file",
        type=["csv", "xlsx", "xlsm"],
        key="batch_upload",
    )
    if batch_file is not None:
        try:
            batch_data = read_dataset(batch_file.getvalue(), filename=batch_file.name)
            missing = [column for column in bundle.base_feature_columns if column not in batch_data.columns]
            if missing:
                st.error("Missing required input columns: " + ", ".join(missing))
            else:
                batch_predictions = predict_outputs(bundle, batch_data[bundle.base_feature_columns])
                batch_uncertainty = prediction_uncertainty(bundle, batch_data[bundle.base_feature_columns])
                scored = pd.concat(
                    [batch_data.reset_index(drop=True), batch_predictions.reset_index(drop=True), batch_uncertainty.reset_index(drop=True)],
                    axis=1,
                )
                st.dataframe(scored.head(200), use_container_width=True)
                warnings = range_warnings(bundle, batch_data[bundle.base_feature_columns])
                if warnings:
                    st.warning("\n".join(warnings))
                st.download_button(
                    "Download scored batch CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="microfluidic_batch_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

with page_inverse:
    st.subheader("Inverse design: recommend candidate operating conditions")
    st.write(
        "Choose desired outputs. The app samples feasible inputs inside the training ranges and ranks candidates by normalized target error."
    )

    target_columns_ui = st.columns(len(bundle.target_columns))
    desired: dict[str, float] = {}
    weights: dict[str, float] = {}
    for index, target in enumerate(bundle.target_columns):
        profile = bundle.target_profile[target]
        with target_columns_ui[index]:
            desired[target] = st.number_input(
                f"Desired {target}",
                min_value=float(profile["min"]),
                max_value=float(profile["max"]),
                value=float(profile["median"]),
                format="%.6g",
                key=f"desired_{target}",
            )
            weights[target] = st.slider(
                f"Weight for {target}",
                min_value=0.0,
                max_value=3.0,
                value=1.0,
                step=0.1,
                key=f"weight_{target}",
            )

    control_1, control_2, control_3 = st.columns(3)
    n_candidates = control_1.slider(
        "Candidate search size",
        min_value=1_000,
        max_value=50_000,
        value=10_000,
        step=1_000,
    )
    top_k = control_2.slider("Recommendations", min_value=3, max_value=30, value=10)
    seed = control_3.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)

    if st.button("Generate recommendations", type="primary"):
        try:
            recommendations = recommend_conditions(
                bundle,
                desired_targets=desired,
                target_weights=weights,
                n_candidates=n_candidates,
                top_k=top_k,
                seed=int(seed),
            )
            st.dataframe(recommendations, use_container_width=True, hide_index=True)
            st.download_button(
                "Download recommendations CSV",
                data=recommendations.to_csv(index=False).encode("utf-8"),
                file_name="microfluidic_inverse_design_recommendations.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Inverse design failed: {exc}")

with page_data:
    st.subheader("Dataset explorer")
    numeric_columns = bundle.cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    x_column, y_column, color_column = st.columns(3)
    x_name = x_column.selectbox("X axis", numeric_columns, index=0)
    y_default = numeric_columns.index(bundle.target_columns[0]) if bundle.target_columns[0] in numeric_columns else 1
    y_name = y_column.selectbox("Y axis", numeric_columns, index=y_default)
    color_options = ["None"] + numeric_columns
    color_name = color_column.selectbox("Color", color_options)

    scatter_kwargs = dict(data_frame=bundle.cleaned_data, x=x_name, y=y_name)
    if color_name != "None":
        scatter_kwargs["color"] = color_name
    figure = px.scatter(**scatter_kwargs, opacity=0.72)
    figure.update_layout(height=520)
    st.plotly_chart(figure, use_container_width=True)

    st.subheader("Numeric correlations")
    correlations = bundle.cleaned_data[numeric_columns].corr(numeric_only=True)
    heatmap = px.imshow(correlations, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    heatmap.update_layout(height=700)
    st.plotly_chart(heatmap, use_container_width=True)

    with st.expander("Preview cleaned training data"):
        st.dataframe(bundle.cleaned_data, use_container_width=True, height=450)

with page_about:
    st.subheader("Modeling method")
    st.markdown(
        """
        - **Task:** multi-output tabular regression, not time-series forecasting.
        - **Model:** ExtraTrees ensemble with median imputation and physics-inspired interaction features.
        - **Outputs:** observed droplet diameter, normalized droplet diameter, and observed generation rate when available.
        - **Uncertainty indicator:** standard deviation across individual ensemble trees.
        - **Inverse design:** Monte Carlo candidate search constrained to observed input ranges.
        - **Deep learning:** the included Colab notebook contains the optional TensorFlow/Keras DNN workflow.
        """
    )

    st.subheader("Responsible use")
    st.warning(
        "This is a research and prototyping tool. Do not use its predictions as the sole basis for clinical, medical, biological, "
        "manufacturing, safety, or regulatory decisions. Validate recommendations experimentally, assess data provenance, "
        "and establish domain-specific acceptance criteria before production deployment."
    )

    st.subheader("Detected schema")
    st.write("Base input columns:")
    st.code("\n".join(bundle.base_feature_columns))
    st.write("Target columns:")
    st.code("\n".join(bundle.target_columns))
