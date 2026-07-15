# Microfluidic Droplet AI

A GitHub-ready Streamlit prototype for analyzing and predicting microfluidic emulsion-droplet behavior from experimental tabular data.

The app trains a multi-output ensemble model from the included `Comprehensive_normalized.xlsx` dataset and provides:

- prediction of observed droplet diameter, normalized droplet diameter, and observed generation rate;
- model-performance and feature-importance views;
- single-condition and batch prediction;
- ensemble-based uncertainty indicators;
- inverse design that recommends candidate operating conditions for desired outputs;
- data visualization and correlation exploration;
- an optional Colab notebook for machine learning and deep neural network experiments.

> **Research-use notice:** This repository is a prototype. Predictions and recommendations must be experimentally validated before industrial, biological, medical, clinical, safety-critical, or regulatory use.

## Repository structure

```text
microfluidic-droplet-ai-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── Comprehensive_normalized.xlsx
│   └── sample_batch_input.csv
├── notebooks/
│   └── microfluidic_ai_brain_colab.ipynb
└── src/
    ├── __init__.py
    └── microfluidic_core.py
```

## Run locally

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS or Linux**

```bash
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The local browser address is normally `http://localhost:8501`.

## Upload to GitHub

1. Extract this ZIP file.
2. Create a new GitHub repository.
3. Upload all extracted files and folders, keeping the same directory structure.
4. Commit the files to the repository's main branch.

Command-line alternative:

```bash
git init
git add .
git commit -m "Initial microfluidic droplet AI Streamlit app"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose **Create app** and connect the GitHub repository.
3. Select the repository and branch.
4. Set the entrypoint file to `app.py`.
5. In **Advanced settings**, select Python 3.12.
6. Deploy the app.

The root-level `requirements.txt` declares all Python dependencies. Streamlit Community Cloud copies the repository and runs the selected entrypoint from the repository root. Official deployment references:

- [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [File organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)

## Included data schema

The supplied workbook contains 869 rows including the header and the following primary input variables:

- Orifice width (um)
- Normalized channel depth
- Flow rate ratio
- Capillary number
- Normalized continuous inlet
- Normalized dispersed inlet
- Normalized outlet width
- viscosity ratio
- Channel Depth
- Hydraulic diameter
- Dispersed flow rate ul/h

Default prediction targets are:

- Observed droplet diameter (um)
- Normalized droplet diameter
- Observed generation rate (Hz)

The `Experiment` column is treated as an identifier and is excluded from model inputs. Empty separator columns are removed automatically.

## Model workflow

1. Clean column names and remove empty rows or columns.
2. Detect numeric process inputs and droplet-output targets.
3. Add physics-inspired interaction features when the expected source columns are present.
4. Split the dataset into 80% training and 20% holdout data.
5. Evaluate an `ExtraTreesRegressor` multi-output ensemble.
6. Refit the deployment model on all complete training rows.
7. Compute uncertainty indicators from variation across ensemble trees.
8. Search candidate inputs within observed ranges for inverse-design recommendations.

The Streamlit app deliberately uses a lightweight ensemble instead of TensorFlow so that Community Cloud startup remains practical. The Colab notebook in `notebooks/` includes the optional deep neural network workflow.

## Use a replacement training dataset

Use the training-data uploader in the app's sidebar. Supported formats are CSV, XLSX, and XLSM.

For automatic target detection, retain the default target names or use descriptive names containing:

- `observed`, `droplet`, and `diameter`;
- `normalized`, `droplet`, and `diameter`;
- `observed`, `generation`, and `rate`.

A replacement dataset should contain at least two numeric input features and approximately 30 or more complete rows. More data, independent experiments, and external validation are strongly recommended for serious use.

## Batch prediction

The app provides a downloadable batch-input template. Fill one row per proposed operating condition, preserve the required input-column names, and upload the completed CSV or Excel file in the **Batch Prediction** tab.

A populated example is included at `data/sample_batch_input.csv`.

## Colab notebook

Open `notebooks/microfluidic_ai_brain_colab.ipynb` in Google Colab for:

- broader machine-learning model comparison;
- optional hyperparameter tuning;
- cross-validation and error analysis;
- TensorFlow/Keras deep neural network training;
- model export and additional prototype generation.

Upload `data/Comprehensive_normalized.xlsx` to the Colab session when prompted, or mount Google Drive and update the notebook's data path.

## Limitations

- Holdout metrics come from the supplied dataset and may be optimistic when experimental conditions are highly related.
- Random train/test splitting does not prove generalization to a new device geometry, fluid chemistry, laboratory, or production line.
- The uncertainty value is an ensemble-disagreement indicator, not a calibrated confidence interval.
- Inverse design samples only inside observed feature ranges but can still recommend physically incompatible combinations when variables are constrained jointly in the real system.
- The app does not currently model images, videos, time-dependent behavior, rare failures, sterility, biocompatibility, toxicity, or regulatory compliance.
- Medical and biological use requires governance, traceability, validation protocols, and expert review.

## Recommended next steps

- add independent validation datasets from different devices and laboratories;
- add explicit labels for single-emulsion and double-emulsion regimes;
- use group-aware validation by device, paper, batch, or laboratory;
- add fluid-property and surfactant descriptors;
- calibrate uncertainty and define out-of-domain thresholds;
- constrain inverse design with engineering feasibility rules;
- add experiment tracking, model versioning, tests, authentication, and audit logs before production use.
