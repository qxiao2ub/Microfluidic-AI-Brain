"""Microfluidic Droplet AI - self-contained Streamlit entrypoint.

No local src/ package is required. All predictive modeling functions are defined
in app.py to avoid missing-package errors when uploading to GitHub/Streamlit.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# =========================== MODEL IMPLEMENTATION ============================
RANDOM_STATE = 42

DEFAULT_TARGETS = [
    "Observed droplet diameter (um)",
    "Normalized droplet diameter",
    "Observed generation rate (Hz)",
]

ID_NAMES = {
    "experiment",
    "experiment id",
    "id",
    "index",
    "row",
    "row id",
}


@dataclass
class ModelBundle:
    model: Pipeline
    metrics: pd.DataFrame
    cleaned_data: pd.DataFrame
    base_feature_columns: list[str]
    feature_columns: list[str]
    target_columns: list[str]
    base_feature_profile: dict[str, dict]
    target_profile: dict[str, dict]
    feature_importance: pd.DataFrame
    holdout_actual: pd.DataFrame
    holdout_predicted: pd.DataFrame


def clean_column_names(columns: Iterable[object]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for i, column in enumerate(columns):
        name = "" if column is None else " ".join(str(column).strip().split())
        if not name or name.lower().startswith("unnamed"):
            name = f"blank_col_{i + 1}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        cleaned.append(name if count == 0 else f"{name}_{count}")
    return cleaned


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = clean_column_names(frame.columns)
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    for column in frame.columns:
        if frame[column].dtype == "object":
            converted = pd.to_numeric(frame[column], errors="coerce")
            # Keep text columns when numeric conversion would destroy most values.
            if converted.notna().sum() >= max(3, int(0.8 * frame[column].notna().sum())):
                frame[column] = converted
    return frame.reset_index(drop=True)


def read_dataset(source: str | Path | bytes | BinaryIO, filename: str | None = None) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return clean_dataframe(pd.read_csv(path))
        if suffix in {".xlsx", ".xlsm"}:
            return clean_dataframe(pd.read_excel(path, engine="openpyxl"))
        raise ValueError("Supported training files are .csv, .xlsx, and .xlsm.")

    if isinstance(source, bytes):
        buffer = io.BytesIO(source)
    else:
        buffer = source

    suffix = Path(filename or "uploaded.csv").suffix.lower()
    if suffix == ".csv":
        return clean_dataframe(pd.read_csv(buffer))
    if suffix in {".xlsx", ".xlsm"}:
        return clean_dataframe(pd.read_excel(buffer, engine="openpyxl"))
    raise ValueError("Supported training files are .csv, .xlsx, and .xlsm.")


def detect_target_columns(frame: pd.DataFrame) -> list[str]:
    exact = [column for column in DEFAULT_TARGETS if column in frame.columns]
    if exact:
        return exact

    keyword_groups = [
        ("observed", "droplet", "diameter"),
        ("normalized", "droplet", "diameter"),
        ("observed", "generation", "rate"),
    ]
    matches: list[str] = []
    for column in frame.columns:
        lowered = column.lower()
        if any(all(keyword in lowered for keyword in group) for group in keyword_groups):
            matches.append(column)
    if not matches:
        raise ValueError(
            "No droplet output columns were detected. Include columns for droplet diameter "
            "and/or generation rate, or rename them to the supplied dataset schema."
        )
    return matches


def safe_divide(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    return np.where(np.abs(denominator_array) > 1e-12, numerator_array / denominator_array, np.nan)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    columns = set(output.columns)

    if {"Flow rate ratio", "Capillary number"}.issubset(columns):
        output["eng_Ca_x_flow_ratio"] = output["Capillary number"] * output["Flow rate ratio"]
        output["eng_Ca_div_flow_ratio"] = safe_divide(
            output["Capillary number"], output["Flow rate ratio"]
        )

    if {"Orifice width (um)", "Hydraulic diameter"}.issubset(columns):
        output["eng_orifice_div_hydraulic_diameter"] = safe_divide(
            output["Orifice width (um)"], output["Hydraulic diameter"]
        )

    if {"Orifice width (um)", "Channel Depth"}.issubset(columns):
        output["eng_orifice_area_proxy"] = output["Orifice width (um)"] * output["Channel Depth"]
        output["eng_channel_depth_div_orifice"] = safe_divide(
            output["Channel Depth"], output["Orifice width (um)"]
        )

    if {"Normalized continuous inlet", "Normalized dispersed inlet"}.issubset(columns):
        output["eng_continuous_div_dispersed_inlet"] = safe_divide(
            output["Normalized continuous inlet"], output["Normalized dispersed inlet"]
        )

    if {"viscosity ratio", "Capillary number"}.issubset(columns):
        output["eng_viscosity_x_Ca"] = output["viscosity ratio"] * output["Capillary number"]

    if {"Dispersed flow rate ul/h", "Flow rate ratio"}.issubset(columns):
        output["eng_est_continuous_flow_rate_ul_h"] = (
            output["Dispersed flow rate ul/h"] * output["Flow rate ratio"]
        )

    return output.replace([np.inf, -np.inf], np.nan)


def _profile_numeric(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    profile: dict[str, dict] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"Feature '{column}' contains no usable numeric values.")
        unique = sorted(float(value) for value in series.unique())
        profile[column] = {
            "min": float(series.min()),
            "q05": float(series.quantile(0.05)),
            "median": float(series.median()),
            "q95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "unique_values": unique if len(unique) <= 25 else None,
        }
    return profile


def _metric_table(y_true: pd.DataFrame, y_pred: np.ndarray, targets: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for index, target in enumerate(targets):
        actual = y_true.iloc[:, index].to_numpy(dtype=float)
        predicted = np.asarray(y_pred)[:, index]
        denominator = np.maximum(np.abs(actual), 1e-9)
        rows.append(
            {
                "Target": target,
                "R2": float(r2_score(actual, predicted)),
                "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
                "MAE": float(mean_absolute_error(actual, predicted)),
                "MAPE (%)": float(np.mean(np.abs(actual - predicted) / denominator) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def train_model(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> ModelBundle:
    cleaned = clean_dataframe(frame)
    target_columns = detect_target_columns(cleaned)

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    id_columns = [column for column in numeric_columns if column.lower() in ID_NAMES]
    base_feature_columns = [
        column
        for column in numeric_columns
        if column not in target_columns and column not in id_columns and not column.startswith("blank_col_")
    ]
    if len(base_feature_columns) < 2:
        raise ValueError("At least two numeric input features are required to train the model.")

    prepared = add_engineered_features(cleaned)
    feature_columns = [
        column
        for column in prepared.select_dtypes(include=[np.number]).columns
        if column not in target_columns and column not in id_columns and not column.startswith("blank_col_")
    ]

    model_frame = prepared[feature_columns + target_columns].dropna(subset=target_columns).copy()
    if len(model_frame) < 30:
        raise ValueError("At least 30 complete rows are recommended for a train/test model evaluation.")

    X = model_frame[feature_columns]
    y = model_frame[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    evaluation_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    max_features=0.90,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    evaluation_model.fit(X_train, y_train)
    holdout_pred = np.clip(evaluation_model.predict(X_test), 0, None)
    metrics = _metric_table(y_test, holdout_pred, target_columns)

    final_model = clone(evaluation_model)
    final_model.fit(X, y)

    regressor = final_model.named_steps["regressor"]
    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": regressor.feature_importances_,
        }
    ).sort_values("Importance", ascending=False, ignore_index=True)

    actual_frame = y_test.reset_index(drop=True).copy()
    predicted_frame = pd.DataFrame(
        holdout_pred,
        columns=[f"Predicted {target}" for target in target_columns],
    )

    return ModelBundle(
        model=final_model,
        metrics=metrics,
        cleaned_data=cleaned,
        base_feature_columns=base_feature_columns,
        feature_columns=feature_columns,
        target_columns=target_columns,
        base_feature_profile=_profile_numeric(cleaned, base_feature_columns),
        target_profile=_profile_numeric(cleaned, target_columns),
        feature_importance=importance,
        holdout_actual=actual_frame,
        holdout_predicted=predicted_frame,
    )


def prepare_model_input(bundle: ModelBundle, base_inputs: pd.DataFrame) -> pd.DataFrame:
    frame = base_inputs.copy()
    for column in bundle.base_feature_columns:
        if column not in frame.columns:
            frame[column] = bundle.base_feature_profile[column]["median"]
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    engineered = add_engineered_features(frame[bundle.base_feature_columns])
    for column in bundle.feature_columns:
        if column not in engineered.columns:
            engineered[column] = np.nan
    return engineered[bundle.feature_columns]


def predict_outputs(bundle: ModelBundle, base_inputs: pd.DataFrame) -> pd.DataFrame:
    model_input = prepare_model_input(bundle, base_inputs)
    predictions = np.clip(bundle.model.predict(model_input), 0, None)
    return pd.DataFrame(predictions, columns=bundle.target_columns, index=base_inputs.index)


def prediction_uncertainty(bundle: ModelBundle, base_inputs: pd.DataFrame) -> pd.DataFrame:
    model_input = prepare_model_input(bundle, base_inputs)
    transformed = bundle.model.named_steps["imputer"].transform(model_input)
    trees = bundle.model.named_steps["regressor"].estimators_
    tree_predictions = np.stack([tree.predict(transformed) for tree in trees], axis=0)
    standard_deviation = tree_predictions.std(axis=0)
    return pd.DataFrame(
        standard_deviation,
        columns=[f"Uncertainty SD: {target}" for target in bundle.target_columns],
        index=base_inputs.index,
    )


def range_warnings(bundle: ModelBundle, base_inputs: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for column in bundle.base_feature_columns:
        values = pd.to_numeric(base_inputs[column], errors="coerce")
        profile = bundle.base_feature_profile[column]
        below = values < profile["min"]
        above = values > profile["max"]
        if below.any() or above.any():
            warnings.append(
                f"{column} contains values outside the training range "
                f"[{profile['min']:.5g}, {profile['max']:.5g}]."
            )
    return warnings


def make_input_template(bundle: ModelBundle, rows: int = 5) -> pd.DataFrame:
    template = {
        column: [bundle.base_feature_profile[column]["median"]] * rows
        for column in bundle.base_feature_columns
    }
    return pd.DataFrame(template)


def sample_candidate_space(bundle: ModelBundle, n_candidates: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled: dict[str, np.ndarray] = {}
    for column in bundle.base_feature_columns:
        profile = bundle.base_feature_profile[column]
        unique_values = profile.get("unique_values")
        if unique_values:
            sampled[column] = rng.choice(unique_values, size=n_candidates)
        else:
            sampled[column] = rng.uniform(profile["min"], profile["max"], size=n_candidates)
    return pd.DataFrame(sampled)


def recommend_conditions(
    bundle: ModelBundle,
    desired_targets: dict[str, float],
    target_weights: dict[str, float] | None = None,
    n_candidates: int = 10_000,
    top_k: int = 10,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    candidates = sample_candidate_space(bundle, n_candidates=n_candidates, seed=seed)
    predictions = predict_outputs(bundle, candidates)
    uncertainties = prediction_uncertainty(bundle, candidates)

    weights = target_weights or {target: 1.0 for target in bundle.target_columns}
    score = np.zeros(len(candidates), dtype=float)
    total_weight = 0.0
    for target in bundle.target_columns:
        weight = max(float(weights.get(target, 1.0)), 0.0)
        profile = bundle.target_profile[target]
        scale = max(profile["max"] - profile["min"], 1e-9)
        score += weight * ((predictions[target] - float(desired_targets[target])) / scale) ** 2
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("At least one target weight must be greater than zero.")

    result = pd.concat(
        [
            candidates.reset_index(drop=True),
            predictions.add_prefix("Predicted: ").reset_index(drop=True),
            uncertainties.reset_index(drop=True),
        ],
        axis=1,
    )
    result["Decision score (lower is better)"] = np.sqrt(score / total_weight)
    return result.sort_values("Decision score (lower is better)").head(top_k).reset_index(drop=True)


# ============================= STREAMLIT UI ===============================
APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = APP_ROOT / "data" / "Comprehensive_normalized.xlsx"
BACKUP_DATA_PATH = APP_ROOT / "data" / "Comprehensive_normalized.csv"

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
        bundled_data = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.is_file() else BACKUP_DATA_PATH
        if not bundled_data.is_file():
            st.error(
                "The bundled training dataset is missing. Upload a CSV/XLSX above "
                "or add data/Comprehensive_normalized.xlsx to the GitHub repository."
            )
            st.stop()
        training_name = bundled_data.name
        training_bytes = bundled_data.read_bytes()
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
        "Metrics use a fixed 80/20 random holdout split. Closely related experiments may occur "
        "in both partitions; these scores do not demonstrate generalization to new devices, "
        "chemistries, laboratories, or clinical settings."
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
        "Choose desired outputs. The app searches input combinations within the training-data "
        "ranges and ranks candidates by normalized target error. Combinations are not guaranteed "
        "to be physically feasible; verify them against engineering constraints before experiments."
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
        - **Task:** multi-output tabular regression for proposed process settings, not time-series forecasting.
        - **Single/double emulsion:** this dataset has no explicit emulsion-type label, so it cannot independently classify or verify single vs. double emulsions.
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
