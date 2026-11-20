@echo off
title RAG Chat
cd /d c:\Users\User\Desktop\Rag

echo.
echo  =============================================
echo    RAG Chat  -  Local AI, fully offline
echo  =============================================
echo.

:: ── Check Ollama ───────────────────────────────────────────────────────────
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not found. Download from https://ollama.com
    pause & exit /b 1
)

:: ── Install fastapi + uvicorn if missing ───────────────────────────────────
.venv\Scripts\python.exe -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing fastapi and uvicorn...
    C:\Users\User\.local\bin\uv.exe pip install "fastapi[standard]" uvicorn --python .venv\Scripts\python.exe
    echo [SETUP] Done.
)

:: ── Start Ollama server ────────────────────────────────────────────────────
echo [1/5] Starting Ollama server...
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if errorlevel 1 (
    start "" /min cmd /c "ollama serve"
    echo       Waiting for Ollama to start...
    timeout /t 8 /nobreak >nul
)
echo       OK

:: ── Pull models (skips if already present) ────────────────────────────────
echo [2/5] Pulling models (skipped if already downloaded)...
ollama pull llama3.2:latest
ollama pull nomic-embed-text:latest

:: Give Ollama a moment to settle after pulling before we hammer it
echo       Waiting for Ollama to be ready...
timeout /t 5 /nobreak >nul
echo       OK

:: ── Index documents ────────────────────────────────────────────────────────
echo [3/5] Indexing documents (batched - please wait)...
.venv\Scripts\python.exe -m rag --index
if errorlevel 1 (
    echo.
    echo [ERROR] Indexing failed.
    echo         Make sure Ollama is running and models are loaded.
    pause & exit /b 1
)
echo       OK

:: ── Start backend ─────────────────────────────────────────────────────────
echo [4/5] Starting backend...
start "RAG Backend" cmd /k "cd /d c:\Users\User\Desktop\Rag && .venv\Scripts\python.exe -m uvicorn rag.api:app --host 0.0.0.0 --port 8000"

:: Poll until backend is ready
set TRIES=0
:waitloop
set /a TRIES+=1
if %TRIES% gtr 40 (
    echo [ERROR] Backend failed to start. See the RAG Backend window.
    pause & exit /b 1
)
timeout /t 2 /nobreak >nul
curl -sf http://localhost:8000/health >nul 2>&1
if errorlevel 1 goto waitloop
echo       OK

:: ── Open browser ──────────────────────────────────────────────────────────
echo [5/5] Opening browser...
start "" "http://localhost:8000"

echo.
echo  =============================================
echo    Ready at http://localhost:8000
echo    Close "RAG Backend" window to stop.
echo  =============================================
echo.
pause
