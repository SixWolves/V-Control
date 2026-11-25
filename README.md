# V-Control

Sistema de monitoramento e controle de acesso por reconhecimento automático de placas veiculares (ALPR/LPR), pensado para uso em portaria residencial, comercial e condomínios.

## Visão geral

O projeto combina:

- detecção de placas com YOLOv8;
- reconhecimento óptico de caracteres com EasyOCR;
- captura de vídeo com OpenCV;
- armazenamento local em SQLite;
- interface gráfica em Tkinter para monitoramento, histórico e cadastro de condutores.

A aplicação possui abas para:

- Monitoramento em tempo real;
- Histórico de acessos;
- Cadastro de moradores/condutores;
- Configurações do sistema e da câmera.

## Funcionalidades principais

- leitura automática de placas via câmera USB ou fluxo de vídeo;
- disparo por sensor ultrassônico para reduzir leituras desnecessárias;
- validação de placas cadastradas;
- registro de acessos com data, hora e placa identificada;
- galeria de fotos dos condutores autorizados;
- ajuste de resolução da câmera, distância do sensor e confiabilidade do OCR.

## Estrutura do projeto

```text
.
├── README.md
├── main.py
├── styles.py
├── requirements.txt
├── license_plate_detector.pt
├── .env
├── fotos_condutores/
├── haarcascades/
└── .venv313/
```

## Requisitos

- Python 3.13 (recomendado)
- Windows 10/11
- Câmera USB ou fonte de vídeo compatível
- Opcionalmente: Raspberry Pi 4 + sensor ultrassônico para ativação por proximidade

## Instalação no Windows

No PowerShell, execute:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o VS Code for usado, selecione o interpretador `.venv313\Scripts\python.exe` como Python do workspace.

## Execução

```powershell
python main.py
```

## Configuração do ambiente

O projeto usa um arquivo `.env` para configurações de câmera e sensor. Exemplo:

```env
CAMERA_RESOLUCAO="640x480"
ULTRASONIC_DISTANCIA_CM=150
ULTRASONIC_FRAMES=1
OCR_CONFIANCA_MIN=0.50
ULTRASONIC_TRIG_PIN=23
ULTRASONIC_ECHO_PIN=24
```

Esses valores podem ser ajustados conforme a sua instalação:

- `CAMERA_RESOLUCAO`: resolução da captura;
- `ULTRASONIC_DISTANCIA_CM`: distância de disparo do sensor;
- `ULTRASONIC_FRAMES`: número de frames para confirmação;
- `OCR_CONFIANCA_MIN`: limiar mínimo de confiança do reconhecimento;
- `ULTRASONIC_TRIG_PIN` e `ULTRASONIC_ECHO_PIN`: pinos do sensor no Raspberry Pi.

## Observações importantes

- O detector de placas (`license_plate_detector.pt`) é carregado automaticamente pela aplicação.
- A pasta `fotos_condutores/` armazena as imagens dos condutores autorizados.
- O banco de dados SQLite é gerado localmente na execução da aplicação.
- O projeto foi preparado para operação com hardware de baixo custo e otimização para Raspberry Pi 4.

## Dicas de uso

- antes de iniciar, verifique se a câmera está conectada e acessível;
- cadastre as placas autorizadas antes de testar o controle de acesso;
- ajuste a distância do sensor ultrassônico e a confiança do OCR conforme a iluminação e o ambiente.

## Licença

Este projeto é distribuído sob a licença definida no repositório, caso exista um arquivo de licença específico no projeto.
