# Soccer Match Outcome Prediction

A predictive analytics framework for forecasting English Premier League match results using machine learning. This project replicates and extends an existing academic paper by introducing additional engineered features and alternative modeling approaches.

---

## Project Overview

The goal is to predict the full-time result (Home Win / Draw / Away Win) of Premier League matches using historical match statistics, weather data, and custom-engineered features. The project is structured in five notebooks, each serving a distinct role in the pipeline.

**Seasons covered:** 2019–20, 2020–21, 2021–22

---

## Repository Structure

```
.
├── 90_kickoff_API.ipynb         # Data loading and master dataframe construction
├── Prediction_Paper.ipynb       # Baseline replication of the original paper
├── Machine_Learning_Project.ipynb  # Improved version with engineered features
├── Rolling_window.ipynb         # Rolling-window approach for time-aware features
├── CatBoost.ipynb               # CatBoost hyperparameter tuning and evaluation
└── features_df.csv              # Preprocessed feature matrix (model-ready)
```

### Notebook Descriptions

**`90_kickoff_API.ipynb`**  
Loads raw CSV datasets for multiple seasons, extracts unique team names, fetches weather data via API for kickoff and full-time, and consolidates everything into a single master dataframe. Also computes stadium coordinates for travel distance calculations.

**`Prediction_Paper.ipynb`**  
Replicates the methodology of the original paper as a baseline. Uses standard match statistics (shots, corners, fouls, cards) and kickoff weather conditions to predict match outcomes.

**`Machine_Learning_Project.ipynb`**  
The improved version. Introduces three additional feature categories beyond the baseline:
- **Team Momentum** – Exponentially Weighted Moving Average (EWMA) of recent points per team
- **Travel Distance** – Haversine-formula-based distance from the away team's home stadium to the match venue
- **Full-time Weather** – Weather conditions at the final whistle, not just kickoff

**`Rolling_window.ipynb`**  
An alternative feature engineering approach using a rolling window over past matches. Ensures strict temporal integrity: only data available before kickoff is used, preventing any data leakage from future matches.

**`CatBoost.ipynb`**  
Dedicated to hyperparameter tuning of the CatBoost classifier using 5-fold `GridSearchCV`. Best parameters found: `depth=10`, `l2_leaf_reg=9`, `learning_rate=0.03`. Evaluated via accuracy and Matthews Correlation Coefficient (MCC).

---

## Feature Set

The final feature matrix (`features_df.csv`) contains 82 columns, grouped into:

| Group | Examples |
|---|---|
| Match statistics (actual) | `FTHG`, `FTAG`, `HS`, `AS`, `HST`, `AST`, `HC`, `AC`, ... |
| Match statistics (previous match) | `PFTHG`, `PHS`, `PHST`, `PHC`, ... |
| Form (previous results) | `PHFR_Won`, `PHFR_NotWin`, `PAFR_Won`, `PAFR_NotWin` |
| Kickoff weather | `Start_Temp_C`, `Start_Wind_kmh`, `Start_Humidity`, `Start_Precip_mm`, `Start_Conditions_*` |
| Full-time weather | `End_Conditions_*` |
| Engineered features | `HomeMomentum`, `AwayMomentum`, `AwayDistTravelled` |
| Bookmaker odds | `TBGH`, `TBGA`, `ATBGH`, `ATBGA`, `NOWH`, `NOWA` |

**Target variable:** `FTR` — Full Time Result (`H` = Home Win, `D` = Draw, `A` = Away Win)

---

## Models Used

- Logistic Regression
- Random Forest
- SVM
- XGBoost
- Gradient Boosting
- LightGBM
- CatBoost (with hyperparameter tuning)

Evaluation metrics: **Accuracy**, **Matthews Correlation Coefficient (MCC)**, **Classification Report**

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn
```

### Data

The raw data consists of season CSV files from [football-data.co.uk](https://www.football-data.co.uk/). Weather data is fetched via an external API at kickoff and full-time for each match.

Place the raw CSV files at the paths expected in `90_kickoff_API.ipynb`, or adapt the path variables at the top of each notebook to match your local setup.

```python
# In 90_kickoff_API.ipynb and Machine_Learning_Project.ipynb — adjust this:
path_prefix = '/content/drive/MyDrive/archive-2/Data_with_90_minutes/'
```

For `CatBoost.ipynb` and `Rolling_window.ipynb`, place `features_df.csv` in the working directory or update the load path:

```python
df = pd.read_csv('features_df.csv')
```

### Recommended Run Order

1. `90_kickoff_API.ipynb` — build the master dataframe
2. `Prediction_Paper.ipynb` — establish the baseline
3. `Machine_Learning_Project.ipynb` — run the improved version
4. `Rolling_window.ipynb` — rolling-window alternative
5. `CatBoost.ipynb` — hyperparameter tuning

---

## Notes

- All notebooks were developed in Google Colab. If running locally, remove `from google.colab import drive` and adjust file paths accordingly.
- Temporal ordering is preserved throughout. No future data leaks into training sets.
- The rolling-window notebook is the strictest in terms of temporal integrity and is recommended for realistic out-of-sample evaluation.
