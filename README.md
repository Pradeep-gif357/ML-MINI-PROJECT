# Predicting Eruptive Events at Kilauea from Earthquake Data

Two tasks: (1) classify whether Kilauea is erupting at the time of each quake,
(2) predict days until the next eruption. Models: logistic regression, k-means,
random forest, neural network.

## Files
- `generate_synthetic_data.py`  makes FAKE test data (earthquakes.csv, eruptions.csv)
- `run_project.py`              the whole pipeline (labels, features, models, plots)
- `requirements.txt`

## Run step by step
1. Install Python 3.9+ and open a terminal in this folder.
2. (Optional) virtual env:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. `pip install -r requirements.txt`
   (if torch fails, use `pip install torch --index-url https://download.pytorch.org/whl/cpu`,
   or skip it and run with `--no-nn`)
4. Get data:
   - Test quickly: `python generate_synthetic_data.py`
   - Real data: replace the two CSVs with
     * earthquakes.csv columns: time, latitude, longitude, depth, mag
       (USGS ComCat https://earthquake.usgs.gov/fdsnws/event/1/ or ANSS/WOVOdat)
     * eruptions.csv columns: start, end (Kilauea eruption history, one row per eruption)
5. Run: `python run_project.py`
   Options: `--split time` (train on past, test on future), `--no-nn`, `--nn-epochs 50`,
   `--eq path.csv --er path.csv --out folder`
6. Look in `results/`: classification_results.csv, regression_results.csv,
   elbow_kmeans.png, rf_depth_tuning.png, feature_importance.png, pred_vs_observed.png

## Notes
- Synthetic data gives unrealistically high scores. Only real data gives meaningful results.
- Paper-style random split leaks information between neighbouring quakes; compare with `--split time`.
