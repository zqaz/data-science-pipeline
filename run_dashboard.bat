@echo off
echo ============================================
echo COVID-19 Mortality Prediction Dashboard
echo ============================================
echo.
echo Step 1: Training models (first-time only)...
python train_models.py
echo.
echo Step 2: Launching Streamlit dashboard...
streamlit run app.py
pause
