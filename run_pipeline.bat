@echo off
echo Starting Aviation RAG Pipeline...

echo.
echo Step 1: Ingestion
echo This might take a while if PDFs are large.
python ingest.py

if %ERRORLEVEL% NEQ 0 (
    echo Ingestion failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Step 2: Generating Evaluation Questions
python generate_questions.py

if %ERRORLEVEL% NEQ 0 (
    echo Question generation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo Step 3: Running Evaluation
echo Starting API server in background...
start /B python -m uvicorn main:app --port 8000
timeout /t 10 /nobreak >nul

echo Running evaluation script...
python evaluate.py

echo.
echo Done! Check report.md and evaluation_results.csv.
pause
