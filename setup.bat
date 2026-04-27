@echo off
echo 🔧 Setting up Crime Analysis Project...

mkdir data 2>nul
mkdir models 2>nul
mkdir results 2>nul

pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo To run the project:
echo   python main.py --single
echo   python main.py --all
echo   python main.py --leaderboard
echo   python run_experiments.py
