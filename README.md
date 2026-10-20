# Microfluidic Droplet AI — Streamlit (Fixed GitHub Edition)

**Status:** The Streamlit `ModuleNotFoundError: No module named 'src'` issue has been removed by integrating all model routines directly into `app.py`. This app does **not** import a local `src` or `microfluidic_core` module. It can be deployed from a standard, flat GitHub repository containing `app.py` and the files below.

Microfluidic Droplet AI is a research prototype built from the supplied experimental tabular data. It analyzes microfluidic emulsion droplet measurements and predicts droplet characteristics from geometry, viscosity and flow-related inputs using an ExtraTrees machine-learning ensemble. It also provides experimental-condition recommendation tools. The supplied Colab notebook supports deeper ML and TensorFlow/Keras DNN experiments.

> **Important:** This dataset does not contain an explicit single-emulsion versus double-emulsion class label. The model currently predicts tabular droplet properties; it cannot reliably identify emulsion type or guarantee improvements for new devices, formulations, biology workflows, or clinical use without additional labeled experimental data and validation.

## Download ZIP structure

Unzip and **upload the contents** to your GitHub repository root; do not upload the `.zip` as one file, and do not place the contents under an extra nested folder.

```text
app.py                              <-- Streamlit Cloud entrypoint (self-contained)
requirements.txt                    <-- dependency installation
README.md                           <-- this guide
.gitignore
.github/
  workflows/
    smoke-test.yml              <-- GitHub Actions smoke tests
.streamlit/
  config.toml
data/
  Comprehensive_normalized.xlsx    <-- included measured training data
  Comprehensive_normalized.csv     <-- identical training data, fallback CSV
  sample_batch_input.csv           <-- sample scoring data
tests/
  test_smoke.py                    <-- model and deployment checks
  test_ui_entrypoint.py            <-- UI smoke checks
notebooks/
  microfluidic_ai_brain_colab.ipynb <-- advanced Colab ML/DNN training
```

**There is no `src/` dependency.** If you see the error `No module named 'src'` after deployment, the deployed app is still using an **old** `app.py` from the wrong repository or branch. Open the `app.py` in GitHub and verify it no longer says `from src.microfluidic_core import ...`.

## Update your existing GitHub repository

The screenshot showed repository `qxiao2ub/microfluidic-ai-brain`, branch `main`, and Streamlit entrypoint `app.py`.

**Option A — GitHub web upload (easiest):**

1. Extract the ZIP on your computer.
2. Open the target GitHub repository and ensure you are on the `main` branch.
3. Click **Add file → Upload files**. Drag the **extracted contents** (files AND the `data/`, `.streamlit/`, and `notebooks/` folders), maintaining their paths. If GitHub's web interface does not preserve your folder structure, use the Git method below instead.
4. Verify that the repository root directly contains `app.py`, `requirements.txt`, and `README.md`, and `data/Comprehensive_normalized.xlsx` exists.
5. Commit the changes, including replacing the old `app.py`.
6. Open your deployed Streamlit app; use **Manage app → Reboot** if changes have not appeared yet.

**Option B — Git (recommended for complete folder updates):**

Clone the existing repository, copy **all extracted ZIP contents** into that repository, remove any obsolete `src` directory (optional), and run:

```bash
git add -A
git commit -m "Fix Streamlit src import; use standalone app"
git push origin main
```

If Git reports a non-fast-forward error, fetch/pull your remote changes first instead of forcing a push.

## Deploy to Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and click **Create app** (or open your existing app's settings).
2. Repository: `qxiao2ub/microfluidic-ai-brain` (or your newly created repository).
3. Branch: `main`.
4. **Main file path: `app.py`** (NOT `src/app.py`, NOT `microfluidic-droplet-ai-streamlit/app.py`).
5. In Advanced settings, Python **3.12** is recommended for stable third-party package compatibility; supported recent versions also work when the pinned wheels are available.
6. Deploy. On a previously deployed app, code updates usually redeploy automatically; reboot if necessary.

**Python version caution:** Streamlit Community Cloud requires re-deploying the app to change Python versions for an existing deployment. Changing `requirements.txt` alone does not change the Python interpreter.

## Local launch

Install Python 3.12+ and run in the folder containing `app.py`:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Local URL: http://localhost:8501

## What the website does

- **Overview**: trained-row count, detected process inputs, feature importance, random 80/20 holdout R², RMSE, MAE, and comparison scatterplots.
- **Single prediction**: propose microfluidic settings and predict three recorded outputs (when present), with ensemble disagreement indicators.
- **Batch prediction**: upload multiple operating conditions in CSV/XLSX and export predictions.
- **Inverse design**: sample and rank candidate operating conditions for target properties within independently observed ranges (not guaranteed mutually feasible).
- **Data explorer**: scatterplots, correlations, and cleaned experimental records.
- **Method & safety**: methods, validation limitations, and field descriptions.
- **Replacement training data**: upload your own CSV/XLSX to retrain the cached model.

**Model targets** from the uploaded data:

1. `Observed droplet diameter (um)`
2. `Normalized droplet diameter`
3. `Observed generation rate (Hz)`

**Example input features:** Orifice width, normalized channel depth, flow rate ratio, capillary number, normalized inlet/outlet measures, viscosity ratio, channel depth, hydraulic diameter, dispersed flow rate.

## Machine learning and DNN methodology

The Streamlit site deliberately uses a quickly trainable scikit-learn ExtraTrees regressor, fitted from the bundled 868 measurement rows. The code uses physics-inspired interaction features, median imputation, and a random train/test holdout for demonstrative model evaluation. It refits a final ensemble on all available complete training rows for deployment inference. Training is cached by dataset contents for responsive repeat usage.

Use `notebooks/microfluidic_ai_brain_colab.ipynb` in free Google Colab for additional algorithm comparisons, cross-validation, optional hyperparameter tuning, and deep-neural-network experiments (TensorFlow/Keras). The DNN is **not deployed by default** in the Streamlit app, keeping startup memory and dependencies light. A neural network does not automatically think or make clinical decisions like a person.

## Interpreting predictions responsibly

- **Predictions are conditional estimates**, not time-series forecasting or evidence of causality.
- An 80/20 random holdout split may overestimate out-of-sample performance when many rows share device geometries, experiments, or fluid systems.
- Ensemble tree disagreement is a heuristic, **not** a calibrated statistical confidence interval.
- Independent parameter sampling can generate incompatible physical settings; validate all inverse-design candidates before use.
- To distinguish single and double emulsions, provide explicit emulsion-type labels (and ideally images, particle-size distributions, phase composition, and operating regime data).
- Production deployments for biotech, microfluidic manufacturing, or medical purposes require external cross-device and cross-lab validation, access controls, reproducibility, experiment metadata, monitoring, and domain-expert review.
- Do **not** use this prototype to guide patient treatment, regulated diagnostics, or safety-critical production without independent validation and appropriate review.

## Common deployment problems

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Replace the deployed root `app.py` with the new self-contained file; verify repository/branch/path. |
| `FileNotFoundError: data/Comprehensive_normalized.xlsx` | Upload the entire `data/` directory, or upload your own training CSV/XLSX in the app sidebar. |
| `ModuleNotFoundError: sklearn` / `openpyxl` | Ensure `requirements.txt` is in the same repository root as `app.py`, commit, and reboot/redeploy. |
| App still displays the old traceback | Verify GitHub branch and Streamlit main file; check latest commit hash and reboot the app. |
| Slow cold start | The initial launch trains an ensemble; cached reruns are faster. |

## Development and verification

Smoke tests are included under `tests/`, and a GitHub Actions workflow runs them after pushes to `main`. They validate the bundled training data, ensemble predictions, uncertainty output, inverse design, self-contained imports, and UI-tab initialization using an API stand-in. They do not replace an external hosted Streamlit end-to-end test. For local testing after installing dependencies:

```bash
python -m pytest -q
```

Install `pytest` separately for development: `python -m pip install pytest`.

## Further development

Next worthwhile milestones are domain-aware data-quality checks, emulsion-type labels, group-based validation, independent laboratory test sets, uncertainty calibration, constraints for physics-feasible inverse-design choices, experimental tracking, model versioning, and optional DNN integration after appropriate accuracy/compute benchmarking.

## Credits and cumulative visitor counter

**Author:** Siddharth Ashok Kumar  
**Advisor:** Dr. Qingyang Xiao

The counter is displayed above all tabs and in the sidebar. It increments once per new Streamlit browser session (not on each click, since Streamlit reruns the script on every widget interaction). It counts visits, **not unique people**.

**Important:** A counter that *never resets* cannot be guaranteed using only local files on Streamlit Community Cloud: the filesystem can be replaced when the app restarts or redeploys. No SQL database is used. For durable cross-restart storage, use a private GitHub Gist:

1. Create a **secret Gist** containing a file named `visitor_count.json` with the content `{"count": 0}`. A secret Gist is unlisted, not an access-control mechanism; do not store sensitive information.
2. Create a fine-grained GitHub personal access token with **Gists: read/write** permission (or a classic token with the `gist` scope). Never commit the token to GitHub.
3. In Streamlit Community Cloud, open **App settings → Secrets** and add:

   ```toml
   GITHUB_COUNTER_TOKEN = "YOUR_GITHUB_TOKEN"
   GITHUB_COUNTER_GIST_ID = "YOUR_GIST_ID"
   ```
4. Save and reboot. The count is now stored in the Gist rather than the Streamlit filesystem.

The Gist implementation is suitable for small prototypes. Concurrent visits across multiple app replicas can cause lost increments because GitHub Gist updates are not atomic; use an atomic managed counter service for strict accuracy at scale. Without Gist configuration, the app still works with a **best-effort local file** but cannot promise persistence.
