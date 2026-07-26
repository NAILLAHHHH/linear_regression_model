# WHO Life Expectancy

## Mission
I want to predict life expectancy for a country using health and economic data (not house prices).
This can help see which factors like schooling, GDP and immunization matter most.
Dataset: WHO Life Expectancy from Kaggle (https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who). About 2938 rows from 2000-2015.

## Models
I compared SGD linear regression, normal LinearRegression, Decision Tree and Random Forest (sklearn).
Best model is Random Forest (lowest test RMSE ≈ 1.79). Saved in `summative/linear_regression/artifacts/best_model.joblib`.

## How to run
```bash
uv sync
uv run jupyter notebook summative/linear_regression/multivariate.ipynb
uv run uvicorn summative.API.prediction:app --reload --app-dir .
```
API docs: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/health

Public API (Render): [https://linear-regression-model-api.onrender.com](https://linear-regression-model-api-0zt6.onrender.com)/docs

## Flutter
```bash
cd summative/FlutterApp
flutter pub get
flutter run --dart-define=API_BASE_URL=[https://linear-regression-model-api.onrender.com](https://linear-regression-model-api-0zt6.onrender.com)
```
For android emulator + local api use `http://10.0.2.2:8000`.

## Video
YouTube: (add link)
