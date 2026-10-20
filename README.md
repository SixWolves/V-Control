# V-Control — Sistema de Monitoramento e Controle de Acesso LPR

Sistema de Reconhecimento Automático de Placas Veiculares (ALPR/LPR) para controle de portaria, utilizando **YOLOv8** (detecção de placas), **EasyOCR** (reconhecimento óptico de caracteres), **OpenCV** (captura RTSP/Webcam), **SQLite** e interface gráfica em **Tkinter**.

## Instalação no Windows

No PowerShell, crie e ative o ambiente virtual:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução

```powershell
python main.py
```

No VS Code, selecione `.venv313\Scripts\python.exe` como interpretador do workspace.
