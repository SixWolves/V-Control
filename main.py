import os
import sys
import re
import time
import queue
import shutil
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Configuração de variáveis de ambiente para evitar contenção de threads (thrashing) no Raspberry Pi 4
# Reserva 3 threads para IA/C++ deixando 1 núcleo ARM livre para Tkinter, câmera USB e sensor HC-SR04
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

# IMPORTAÇÃO DA ESTILIZAÇÃO
from styles import CORES, FONTES, aplicar_tema


class SistemaPortariaHibrido:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1180x860")
        self.window.minsize(1100, 790)
        self.window.configure(bg=CORES["bg_main"])

        # Pasta local para armazenar as fotos dos condutores autorizados
        self.pasta_fotos = Path(__file__).resolve().parent / "fotos_condutores"
        self.pasta_fotos.mkdir(exist_ok=True)
        self.foto_selecionada_temp = ""
        self.id_em_edicao = None  # Guarda o ID do registro quando estiver em modo de edição
        self.fotos_condutores_refs = []  # Mantém referências de ImageTk para o garbage collector não apagar

        # Otimização Multithread para os 4 núcleos ARM Cortex-A72 do Raspberry Pi 4
        # Deixa 1 núcleo livre para a interface Tkinter, captura V4L2 e leitura do sensor HC-SR04
        cv2.setNumThreads(2)
        try:
            import torch
            torch.set_num_threads(3)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        # Carrega variáveis do .env antes dos modelos (resolução, sensor, modelo OCR)
        self._carregar_env()

        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass

        print("Carregando IA YOLOv8 (NCNN) e fast-plate-ocr (ONNX)...")
        self.modelo_yolo = self._carregar_modelo_yolo()
        self.ocr, self.ocr_color_mode = self._carregar_modelo_ocr()

        # Pool de Thread persistente (1 worker pois apenas 1 frame OCR executa por vez)
        self.executor_ia = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Worker_LPR")
        # Executa o pré-aquecimento dos modelos em background para não atrasar a abertura da janela
        self.executor_ia.submit(self._aquecer_modelos_ia)

        # Objeto CLAHE reutilizável em memória (evita recriação a cada frame) e cache em RAM das placas cadastradas
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.placas_cadastradas_cache = set()

        self.init_db()
        self.solicitar_troca_camera = False

        self.fila_resultados = queue.Queue()
        self.evento_parada = threading.Event()

        # --- CONFIGURAÇÃO DO SENSOR ULTRASSÔNICO OBRIGATÓRIO (OTIMIZAÇÃO RASPBERRY PI 4) ---
        self.distancia_gatilho_cm = float(os.environ.get("ULTRASONIC_DISTANCIA_CM", 150.0))
        self.distancia_atual_cm = None
        self.frames_por_acionamento = max(1, int(os.environ.get("ULTRASONIC_FRAMES", 1)))
        self.confianca_minima_ocr = float(os.environ.get("OCR_CONFIANCA_MIN", 0.50))
        if self.confianca_minima_ocr > 1.0:
            self.confianca_minima_ocr /= 100.0
        self.limiar_confirmacao = 1 if self.frames_por_acionamento == 1 else min(self.frames_por_acionamento, 3)
        self.frames_restantes_ocr = 0     # Quantos frames ainda devem ser lidos neste disparo do sensor
        self.cooldown_sensor_seg = 1.5    # Intervalo curto caso precise de novo disparo
        self.proximo_disparo_sensor = 0.0

        # Resolução de captura da câmera USB local (padrão 640x480 para máxima velocidade no Raspberry Pi 4; ajustável na aba Configurações)
        self.resolucao_camera = os.environ.get("CAMERA_RESOLUCAO", "640x480")
        self.cam_largura, self.cam_altura = self._parse_resolucao(self.resolucao_camera)

        self._inicializar_gpio_ultrassonico()
        if self.gpio_disponivel:
            self.thread_sensor = threading.Thread(target=self.loop_sensor_ultrassonico, daemon=True)
            self.thread_sensor.start()

        # --- CONTROLE DE CAPTURA DE VÍDEO E TAXA DE QUADROS ---
        self.frame_atual = None
        self.ret_atual = False
        self.ultimo_frame_id_renderizado = None
        self.em_standby = True

        # Taxas de atualização adaptativas da interface (Otimização Raspberry Pi 4):
        # - Standby (Em espera): ~12 FPS (80ms) para reduzir redimensionamento e conversão de imagem no ARM
        # - Ativo (Veículo detectado / Placa em exibição): ~25 FPS (40ms) apenas quando necessário
        # - Processando IA (YOLO NCNN + fast-plate-ocr): ~10 FPS (95ms) para liberar CPU/GIL às threads de inferência
        self.delay_standby = 80   # ~12.5 FPS (dentro da faixa ideal de 10–15 FPS em espera)
        self.delay_ativo = 40     # ~25 FPS quando ativo
        self.delay_ia = 95        # ~10 FPS durante inferência da IA
        self.timestamp_modo_ativo = 0.0

        self.thread_camera = threading.Thread(target=self.loop_leitura_camera, daemon=True)
        self.thread_camera.start()

        # --- VARIÁVEIS DE CONTROLE E ESTABILIZAÇÃO ---
        self.placa_exibida = ""
        self.placa_painel_renderizada = None
        self.coords_exibidas = (0, 0, 0, 0)
        self.timestamp_expira = 0.0
        self.ultima_placa_gravada = ""
        self.timestamp_ultimo_registro = 0.0
        self.ocr_em_andamento = False
        self.retangulos_candidatos = []
        self.tempo_anterior = time.time()
        self.fps_suavizado = 0.0

        self.leitura_temporaria = ""
        self.contador_leituras = 0

        # --- EXPRESSÕES REGULARES ---
        self.padrao_antigo = re.compile(r'^[A-Z]{3}[0-9]{4}$')
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')

        # --- APLICAÇÃO DO TEMA EXTERNO ---
        aplicar_tema()

        # --- CABEÇALHO SUPERIOR (HEADER BAR) ---
        self.header_frame = tk.Frame(
            window,
            bg=CORES["bg_header"],
            padx=20,
            pady=12,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        self.header_frame.pack(fill=tk.X)

        tk.Label(
            self.header_frame,
            text="V-CONTROL",
            font=FONTES["titulo_app"],
            bg=CORES["bg_header"],
            fg=CORES["accent"]
        ).pack(side=tk.LEFT)

        tk.Label(
            self.header_frame,
            text="  |  LPR Otimizado para Raspberry Pi 4 (Gatilho por Sensor Ultrassônico)",
            font=FONTES["label"],
            bg=CORES["bg_header"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        self.lbl_fonte_camera = tk.Label(
            self.header_frame,
            text=f"Câmera USB Local ({self.cam_largura}x{self.cam_altura})",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"],
            padx=12,
            pady=4
        )
        self.lbl_fonte_camera.pack(side=tk.RIGHT)

        # --- ESTRUTURA DE ABAS ---
        self.notebook = ttk.Notebook(window)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_monitoramento = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_historico = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_moradores = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_configuracoes = tk.Frame(self.notebook, bg=CORES["bg_main"])

        self.notebook.add(self.tab_monitoramento, text="  Monitoramento Ao Vivo  ")
        self.notebook.add(self.tab_historico, text="  Histórico de Acessos  ")
        self.notebook.add(self.tab_moradores, text="  Moradores e Condutores  ")
        self.notebook.add(self.tab_configuracoes, text="  Configurações  ")

        self.configurar_aba_monitoramento()
        self.configurar_aba_historico()
        self.configurar_aba_moradores()
        self.configurar_aba_configuracoes()

        self.notebook.bind("<<NotebookTabChanged>>", self.ao_mudar_de_aba)

        # Atalho de teclado [Espaço] para simular detecção do sensor ultrassônico em testes
        self.window.bind("<space>", self._atalho_espaco_sensor)

        self.delay = self.delay_standby  # Inicia em Standby (~12 FPS) e sobe para ~25 FPS apenas quando necessário
        self.update_video()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _carregar_modelo_yolo(self):
        """Carrega o detector YOLOv8 em formato NCNN (2x-4x mais rápido no Raspberry Pi 4).
        Se a pasta NCNN ainda não existir, tenta exportar automaticamente a partir do .pt."""
        base_dir = Path(__file__).resolve().parent
        ncnn_dir = base_dir / "license_plate_detector_ncnn_model"
        pt_path = base_dir / "license_plate_detector.pt"

        if ncnn_dir.exists() and ncnn_dir.is_dir():
            try:
                print(f"[YOLO] Carregando modelo otimizado NCNN: {ncnn_dir.name}")
                return YOLO(str(ncnn_dir), task="detect")
            except Exception as e:
                print(f"[YOLO] Aviso ao carregar NCNN ({e}). Usando fallback .pt...")

        modelo_pt = YOLO(str(pt_path))
        try:
            print("[YOLO] Exportando modelo para NCNN (imgsz=320) na primeira execução para acelerar o Raspberry Pi 4...")
            exportado = modelo_pt.export(format="ncnn", imgsz=320)
            if exportado and Path(exportado).exists():
                print(f"[YOLO] Exportação NCNN concluída! Carregando: {exportado}")
                return YOLO(str(exportado), task="detect")
        except Exception as e:
            print(f"[YOLO] NCNN indisponível neste ambiente ({e}). Continuando com PyTorch (.pt).")
        return modelo_pt

    def _carregar_modelo_ocr(self):
        """Carrega o reconhecedor especializado em placas fast-plate-ocr em ONNX Runtime (CPU)."""
        modelo_env = os.environ.get("OCR_MODEL_NAME", "").strip()
        candidatos = [m for m in [modelo_env, "cct-xs-v2-global-model", "cct-xs-v1-global-model"] if m]
        ultimo_erro = None
        for nome_modelo in candidatos:
            try:
                ocr = LicensePlateRecognizer(nome_modelo, providers=["CPUExecutionProvider"])
                color_mode = getattr(getattr(ocr, "config", None), "image_color_mode", "rgb")
                print(f"[OCR] fast-plate-ocr carregado em ONNX: {nome_modelo} (modo: {color_mode})")
                return ocr, str(color_mode).lower()
            except Exception as e:
                ultimo_erro = e
        raise RuntimeError(f"Não foi possível carregar o fast-plate-ocr ({candidatos}): {ultimo_erro}")

    def _aquecer_modelos_ia(self):
        """Executa uma inferência rápida em memória na inicialização para pré-alocar buffers NCNN/ONNX"""
        try:
            dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
            self.modelo_yolo(dummy_frame, imgsz=320, conf=0.5, verbose=False)
            dummy_crop = np.zeros((64, 160, 3), dtype=np.uint8)
            self._executar_ocr_placa(dummy_crop)
            print("Modelos YOLOv8 (NCNN) e fast-plate-ocr (ONNX) pré-aquecidos na memória com sucesso!")
        except Exception as e:
            print(f"Aviso no pré-aquecimento da IA: {e}")

    def _carregar_env(self):
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s or line_s.startswith("#") or "=" not in line_s:
                        continue
                    chave, valor = line_s.split("=", 1)
                    chave = chave.strip()
                    valor = valor.strip().strip('"\'')
                    os.environ.setdefault(chave, valor)

    def _inicializar_gpio_ultrassonico(self):
        """Inicializa o sensor HC-SR04 nos pinos GPIO do Raspberry Pi 4.
        Em ambiente Windows/PC sem GPIO, ativa automaticamente o modo de simulação."""
        self.gpio_disponivel = False
        self.sensor_gpiozero = None
        self.rpi_gpio = None

        # Pinos padrão BCM no Raspberry Pi 4: TRIG = GPIO 23 (Pino 16), ECHO = GPIO 24 (Pino 18)
        self.pino_trig = int(os.environ.get("ULTRASONIC_TRIG_PIN", 23))
        self.pino_echo = int(os.environ.get("ULTRASONIC_ECHO_PIN", 24))

        # Prioriza RPi.GPIO (com timeout estrito de 40ms que nunca bloqueia caso o sensor esteja desconectado)
        try:
            import RPi.GPIO as GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pino_trig, GPIO.OUT)
            GPIO.setup(self.pino_echo, GPIO.IN)
            GPIO.output(self.pino_trig, False)
            self.rpi_gpio = GPIO
            self.gpio_disponivel = True
            print(f"[Ultrassônico] Sensor HC-SR04 iniciado via RPi.GPIO (TRIG=GPIO{self.pino_trig}, ECHO=GPIO{self.pino_echo})")
            return
        except Exception:
            pass

        try:
            from gpiozero import DistanceSensor
            self.sensor_gpiozero = DistanceSensor(echo=self.pino_echo, trigger=self.pino_trig, max_distance=4.0)
            self.gpio_disponivel = True
            print(f"[Ultrassônico] Sensor HC-SR04 iniciado via gpiozero (TRIG=GPIO{self.pino_trig}, ECHO=GPIO{self.pino_echo})")
        except Exception:
            print("[Ultrassônico] GPIO não encontrado. Use o botão 'Simular Sensor' ou a tecla [Espaço].")

    def _medir_distancia_hardware_cm(self):
        """Realiza uma leitura física no HC-SR04 em centímetros com proteção de timeout"""
        if self.sensor_gpiozero is not None:
            return round(self.sensor_gpiozero.distance * 100.0, 1)

        if self.rpi_gpio is not None:
            GPIO = self.rpi_gpio
            GPIO.output(self.pino_trig, True)
            time.sleep(0.00001)  # Pulso de 10us
            GPIO.output(self.pino_trig, False)

            timeout = time.time() + 0.04  # Timeout de 40ms (~6.8m) para nunca travar a thread
            inicio_pulso = time.time()
            while GPIO.input(self.pino_echo) == 0:
                inicio_pulso = time.time()
                if inicio_pulso > timeout:
                    return None

            fim_pulso = time.time()
            while GPIO.input(self.pino_echo) == 1:
                fim_pulso = time.time()
                if fim_pulso > timeout:
                    return None

            duracao = fim_pulso - inicio_pulso
            distancia = (duracao * 34300.0) / 2.0
            if 2.0 <= distancia <= 400.0:
                return round(distancia, 1)
        return None

    def loop_sensor_ultrassonico(self):
        """Thread leve para leitura periódica do sensor ultrassônico no Raspberry Pi 4"""
        while not self.evento_parada.is_set():
            dist = self._medir_distancia_hardware_cm()
            self.distancia_atual_cm = dist
            time.sleep(0.20)  # 5 leituras por segundo consomem praticamente 0% de CPU

    def _atalho_espaco_sensor(self, event):
        # Só dispara pelo espaço se estiver na aba de monitoramento
        if self.notebook.index(self.notebook.select()) == 0:
            self.acionar_leitura_ultrassonico(origem="Simulação Manual")

    def acionar_leitura_ultrassonico(self, origem="Sensor HC-SR04"):
        """Ativa a leitura da câmera na quantidade configurada de frames (padrão: 1 frame)"""
        try:
            qtd_frames = max(1, int(self.spin_frames.get()))
        except Exception:
            qtd_frames = 1

        self.frames_por_acionamento = qtd_frames
        # Ajusta o limiar de confirmação para permitir confirmar dentro dos frames ativados
        self.limiar_confirmacao = 1 if qtd_frames == 1 else min(qtd_frames, 3)
        self.frames_restantes_ocr = qtd_frames
        self.contador_leituras = 0
        self.leitura_temporaria = ""
        # Eleva a taxa de atualização para o modo ativo (~25 FPS) durante a abordagem do veículo
        self.timestamp_modo_ativo = time.time() + 5.0
        self.em_standby = False
        print(f"\n[Gatilho Ultrassônico - {origem}] Veículo na distância! Ativando Worker Multithread para {qtd_frames} frame(s)...")

    def _parse_resolucao(self, res_str):
        try:
            limpo = str(res_str).split()[0].lower()
            w_s, h_s = limpo.split("x")
            return int(w_s), int(h_s)
        except Exception:
            return 640, 480

    def _abrir_camera(self):
        # No Windows usa DirectShow; no Raspberry Pi (Linux) usa V4L2/Auto tentando índices 0, 1 e 2
        if sys.platform.startswith("win"):
            indices_backends = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_ANY), (1, cv2.CAP_DSHOW)]
        else:
            indices_backends = [(0, cv2.CAP_V4L2), (0, cv2.CAP_ANY), (1, cv2.CAP_V4L2), (2, cv2.CAP_V4L2)]

        for idx, backend in indices_backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Usa formato nativo YUYV (sem decodificação JPEG em CPU e sem erros 'Corrupt JPEG data' em webcams Logitech no V4L2)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_largura)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_altura)
                # Limita o FPS diretamente no hardware da webcam para economizar CPU/USB sem encher o buffer do V4L2
                cap.set(cv2.CAP_PROP_FPS, 15)
                fps_real = cap.get(cv2.CAP_PROP_FPS)
                print(f"[Câmera] Webcam local conectada no índice {idx} (Resolução: {self.cam_largura}x{self.cam_altura} | FPS: {fps_real:.0f}).")
                return cap
            cap.release()

        return cv2.VideoCapture(0)

    def loop_leitura_camera(self):
        # A Thread exclusiva da câmera reconecta e gerencia o buffer 100% isolada da interface
        self.cap = self._abrir_camera()
        try:
            while not self.evento_parada.is_set():
                if self.solicitar_troca_camera:
                    self.solicitar_troca_camera = False
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = self._abrir_camera()
                    continue

                if not self.cap.isOpened():
                    time.sleep(2)
                    if self.evento_parada.is_set():
                        break
                    self.cap = self._abrir_camera()
                    continue

                ret, frame = self.cap.read()
                self.ret_atual = ret
                if ret:
                    self.frame_atual = frame
                    # Pequeno yield do GIL sem travar ou estourar o buffer V4L2 (cap.read já bloqueia no ritmo do hardware)
                    time.sleep(0.002)
                else:
                    self.cap.release()
                    time.sleep(1)  # Dá um tempinho antes de tentar reabrir
        finally:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
            if getattr(self, 'rpi_gpio', None) is not None:
                try:
                    self.rpi_gpio.cleanup()
                except Exception:
                    pass

    def get_db(self):
        return sqlite3.connect('registro_portaria.db')

    def init_db(self):
        with self.get_db() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS registros 
                                (id INTEGER PRIMARY KEY AUTOINCREMENT, placa TEXT NOT NULL, tipo TEXT NOT NULL, data TEXT NOT NULL, hora TEXT NOT NULL, vinculo TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS moradores 
                                (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT NOT NULL, unidade TEXT NOT NULL, placa TEXT NOT NULL, veiculo TEXT DEFAULT '', foto_path TEXT DEFAULT '')""")

            # Migração automática: garante que colunas 'veiculo' e 'foto_path' existam e que 'placa' permita múltiplos condutores
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(moradores)")
            colunas = [info[1] for info in cursor.fetchall()]

            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='moradores'")
            row_sql = cursor.fetchone()
            sql_tabela = row_sql[0].upper() if row_sql and row_sql[0] else ""

            if "foto_path" not in colunas or "veiculo" not in colunas or "UNIQUE" in sql_tabela:
                conn.execute("""CREATE TABLE IF NOT EXISTS moradores_novo 
                                    (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT NOT NULL, unidade TEXT NOT NULL, placa TEXT NOT NULL, veiculo TEXT DEFAULT '', foto_path TEXT DEFAULT '')""")
                col_veiculo = "COALESCE(veiculo, '')" if "veiculo" in colunas else "''"
                col_foto = "COALESCE(foto_path, '')" if "foto_path" in colunas else "''"
                conn.execute(f"INSERT INTO moradores_novo (id, nome, unidade, placa, veiculo, foto_path) SELECT id, nome, unidade, placa, {col_veiculo}, {col_foto} FROM moradores")
                conn.execute("DROP TABLE moradores")
                conn.execute("ALTER TABLE moradores_novo RENAME TO moradores")

            conn.execute("""CREATE INDEX IF NOT EXISTS idx_registros_placa ON registros(placa)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_moradores_placa ON moradores(placa)""")
        self._atualizar_cache_placas()

    def _atualizar_cache_placas(self):
        """Mantém um conjunto (set) em RAM das placas cadastradas para consulta O(1) instantânea sem I/O de disco no OCR"""
        try:
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT placa FROM moradores")
                self.placas_cadastradas_cache = {row[0] for row in cursor.fetchall() if row[0]}
        except Exception:
            pass

    def configurar_aba_monitoramento(self):
        # --- BARRA DE CONTROLE DO SENSOR ULTRASSÔNICO (OTIMIZAÇÃO RASPBERRY PI 4) ---
        barra_sensor = tk.Frame(
            self.tab_monitoramento,
            bg=CORES["bg_panel"],
            padx=18,
            pady=10,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        barra_sensor.pack(fill=tk.X, padx=25, pady=(12, 0))

        status_hw = f"GPIO Ativo (TRIG {self.pino_trig} / ECHO {self.pino_echo})" if self.gpio_disponivel else "Modo Simulação"
        cor_hw = CORES["success"] if self.gpio_disponivel else CORES["accent"]

        tk.Label(
            barra_sensor,
            text=f"SENSOR ULTRASSÔNICO ({status_hw}):",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=cor_hw
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_distancia_sensor = tk.Label(
            barra_sensor,
            text="Em Standby (Aguardando veículo...)",
            font=FONTES["label_bold"],
            bg=CORES["bg_main"],
            fg=CORES["fg_muted"],
            padx=10,
            pady=3
        )
        self.lbl_distancia_sensor.pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_resumo_config = tk.Label(
            barra_sensor,
            text=f"Gatilho: <= {int(self.distancia_gatilho_cm)} cm  |  {self.frames_por_acionamento} frame(s)  |  Confiança >= {int(round(self.confianca_minima_ocr * 100))}%  |  {self.resolucao_camera}",
            font=FONTES["label"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        )
        self.lbl_resumo_config.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            barra_sensor,
            text="Simular Sensor / Ler Placa [Espaço]",
            style="Accent.TButton",
            command=lambda: self.acionar_leitura_ultrassonico(origem="Botão Simular")
        ).pack(side=tk.RIGHT)

        # Área central dividida lado a lado: Câmera (Esquerda) + Condutores Autorizados (Direita)
        area_superior = tk.Frame(self.tab_monitoramento, bg=CORES["bg_main"])
        area_superior.pack(fill=tk.BOTH, expand=True, padx=25, pady=(10, 10))

        # --- COLUNA ESQUERDA: VÍDEO AO VIVO ---
        container_video = tk.Frame(
            area_superior,
            bg=CORES["bg_panel"],
            padx=8,
            pady=8,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        container_video.pack(side=tk.LEFT, anchor=tk.N)

        self.canvas = tk.Canvas(
            container_video,
            width=640,
            height=480,
            bg="#000000",
            highlightthickness=1,
            highlightbackground=CORES["accent"]
        )
        self.canvas.pack()
        # Cria o item de imagem uma única vez para evitar vazamento de memória (memory leak) no Canvas
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # --- COLUNA DIREITA: FOTOS DOS CONDUTORES AUTORIZADOS ---
        self.painel_condutores = tk.Frame(
            area_superior,
            bg=CORES["bg_panel"],
            padx=16,
            pady=14,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        self.painel_condutores.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 0))

        tk.Label(
            self.painel_condutores,
            text="CONDUTORES AUTORIZADOS",
            font=FONTES["secao"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"]
        ).pack(anchor=tk.W)

        tk.Label(
            self.painel_condutores,
            text="Verifique se a pessoa ao volante e o veículo correspondem aos dados autorizados:",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"],
            wraplength=400,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(2, 12))

        self.container_cards_condutores = tk.Frame(self.painel_condutores, bg=CORES["bg_panel"])
        self.container_cards_condutores.pack(fill=tk.BOTH, expand=True)

        # --- BARRA INFERIOR DE STATUS ---
        self.frame_status = tk.Frame(
            self.tab_monitoramento,
            bg=CORES["bg_panel"],
            padx=25,
            pady=14,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        self.frame_status.pack(fill=tk.X, padx=25, pady=(0, 15))
        self.frame_status.columnconfigure(0, weight=1)
        self.frame_status.columnconfigure(1, weight=1)

        # Card Esquerdo: Placa Detectada
        bloco_placa = tk.Frame(self.frame_status, bg=CORES["bg_panel"])
        bloco_placa.grid(row=0, column=0, sticky=tk.W, padx=10)

        tk.Label(
            bloco_placa,
            text="STATUS DE LEITURA LPR",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(anchor=tk.W)

        self.lbl_placa = tk.Label(
            bloco_placa,
            text="Standby (Aguardando Sensor Ultrassônico)",
            font=FONTES["placa_destaque"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        )
        self.lbl_placa.pack(anchor=tk.W, pady=(2, 0))

        # Card Direito: Vínculo / Autorização
        bloco_vinculo = tk.Frame(self.frame_status, bg=CORES["bg_panel"])
        bloco_vinculo.grid(row=0, column=1, sticky=tk.E, padx=10)

        tk.Label(
            bloco_vinculo,
            text="IDENTIFICAÇÃO E VÍNCULO",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(anchor=tk.E)

        self.lbl_vinculo = tk.Label(
            bloco_vinculo,
            text="Vínculo: --",
            font=FONTES["status_vinculo"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        )
        self.lbl_vinculo.pack(anchor=tk.E, pady=(4, 0))

        # Inicializa o painel de condutores no estado "Standby"
        self.atualizar_painel_condutores("")

    def _gerar_avatar_padrao(self, tamanho=(110, 110), cor_borda="#334155"):
        """Gera uma imagem placeholder elegante quando o condutor não tem foto cadastrada"""
        largura, altura = tamanho
        img = Image.new("RGB", tamanho, "#0f172a")
        draw = ImageDraw.Draw(img)
        cx = largura // 2
        draw.ellipse((cx - 18, 22, cx + 18, 58), fill="#475569")
        draw.pieslice((cx - 34, 62, cx + 34, 130), start=180, end=360, fill="#475569")
        draw.rectangle((0, 0, largura - 1, altura - 1), outline=cor_borda, width=2)
        return ImageTk.PhotoImage(img)

    def _carregar_foto_condutor(self, foto_path, tamanho=(110, 110), cor_borda="#10b981"):
        """Carrega e redimensiona a foto do condutor mantendo a proporção (crop central)"""
        if foto_path:
            caminho = Path(foto_path)
            if not caminho.is_absolute():
                caminho = Path(__file__).resolve().parent / caminho
            if caminho.exists():
                try:
                    img = Image.open(caminho).convert("RGB")
                    img = ImageOps.fit(img, tamanho, Image.Resampling.LANCZOS)
                    draw = ImageDraw.Draw(img)
                    draw.rectangle((0, 0, tamanho[0] - 1, tamanho[1] - 1), outline=cor_borda, width=2)
                    return ImageTk.PhotoImage(img)
                except Exception as e:
                    print(f"Erro ao carregar foto '{caminho}': {e}")
        return self._gerar_avatar_padrao(tamanho=tamanho, cor_borda=cor_borda)

    def atualizar_painel_condutores(self, placa, condutores=None):
        """Atualiza o painel lateral com as fotos dos condutores autorizados da placa lida"""
        if self.placa_painel_renderizada == placa:
            return
        self.placa_painel_renderizada = placa

        for widget in self.container_cards_condutores.winfo_children():
            widget.destroy()
        self.fotos_condutores_refs.clear()

        if not placa:
            aviso = tk.Frame(
                self.container_cards_condutores,
                bg=CORES["bg_main"],
                padx=20,
                pady=40,
                highlightthickness=1,
                highlightbackground=CORES["border"]
            )
            aviso.pack(fill=tk.BOTH, expand=True)
            tk.Label(
                aviso,
                text="Em Standby (Economia de CPU)\nAguardando veículo no sensor ultrassônico...",
                font=FONTES["secao"],
                bg=CORES["bg_main"],
                fg=CORES["fg_muted"],
                justify=tk.CENTER
            ).pack(expand=True)
            return

        if not condutores:
            # Placa não cadastrada (Visitante / Desconhecido)
            alerta = tk.Frame(
                self.container_cards_condutores,
                bg=CORES["bg_main"],
                padx=20,
                pady=30,
                highlightthickness=2,
                highlightbackground=CORES["warning"]
            )
            alerta.pack(fill=tk.BOTH, expand=True)

            tk.Label(
                alerta,
                text=f"PLACA: {placa}",
                font=FONTES["placa_destaque"],
                bg=CORES["bg_main"],
                fg=CORES["warning"]
            ).pack(pady=(10, 6))

            tk.Label(
                alerta,
                text="VEÍCULO VISITANTE / NÃO CADASTRADO",
                font=FONTES["secao"],
                bg=CORES["bg_main"],
                fg=CORES["warning"]
            ).pack(pady=(0, 10))

            tk.Label(
                alerta,
                text="Nenhum condutor autorizado vinculado a este veículo.\nAborde o motorista e realize a identificação na portaria antes de liberar o acesso.",
                font=FONTES["label"],
                bg=CORES["bg_main"],
                fg=CORES["fg_text"],
                wraplength=330,
                justify=tk.CENTER
            ).pack()
            return

        # Exibe um card com foto para cada condutor autorizado daquele veículo
        for nome, unidade, veiculo, foto_path in condutores:
            card = tk.Frame(
                self.container_cards_condutores,
                bg=CORES["bg_main"],
                padx=12,
                pady=10,
                highlightthickness=1,
                highlightbackground=CORES["success"]
            )
            card.pack(fill=tk.X, pady=(0, 10))

            foto_tk = self._carregar_foto_condutor(foto_path, tamanho=(110, 110), cor_borda=CORES["success"])
            self.fotos_condutores_refs.append(foto_tk)

            lbl_foto = tk.Label(card, image=foto_tk, bg=CORES["bg_main"])
            lbl_foto.pack(side=tk.LEFT, padx=(0, 14))

            info = tk.Frame(card, bg=CORES["bg_main"])
            info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            tk.Label(
                info,
                text="CONDUTOR AUTORIZADO",
                font=FONTES["subtitulo_app"],
                bg=CORES["bg_main"],
                fg=CORES["success"]
            ).pack(anchor=tk.W)

            tk.Label(
                info,
                text=nome,
                font=FONTES["secao"],
                bg=CORES["bg_main"],
                fg=CORES["fg_text"],
                wraplength=220,
                justify=tk.LEFT
            ).pack(anchor=tk.W, pady=(2, 2))

            veiculo_txt = f"  •  Veículo: {veiculo}" if veiculo else ""
            tk.Label(
                info,
                text=f"Unidade: {unidade}{veiculo_txt}",
                font=FONTES["label_bold"],
                bg=CORES["bg_main"],
                fg=CORES["fg_muted"]
            ).pack(anchor=tk.W)

            status_foto = "Foto verificada" if foto_path else "Sem foto (edite na aba Moradores)"
            tk.Label(
                info,
                text=f"Placa: {placa} • {status_foto}",
                font=FONTES["subtitulo_app"],
                bg=CORES["bg_main"],
                fg=CORES["fg_muted"]
            ).pack(anchor=tk.W, pady=(4, 0))

    def _verificar_gatilho_ultrassonico(self, tempo_atual):
        """Avalia a distância atual do sensor HC-SR04 e dispara a leitura se um veículo entrar no alcance"""
        try:
            self.distancia_gatilho_cm = float(self.spin_distancia.get())
        except Exception:
            pass

        if self.gpio_disponivel and self.distancia_atual_cm is not None:
            dist = self.distancia_atual_cm
            if dist <= self.distancia_gatilho_cm:
                self.timestamp_modo_ativo = max(self.timestamp_modo_ativo, tempo_atual + 4.0)
                self.em_standby = False
                self.lbl_distancia_sensor.config(
                    text=f"Distância: {dist:.1f} cm [VEÍCULO DETECTADO]",
                    fg=CORES["success"]
                )
                # Dispara quando ainda não há placa exibida, não há OCR rodando e respeita o intervalo
                if (not self.placa_exibida and
                        self.frames_restantes_ocr == 0 and
                        not self.ocr_em_andamento and
                        tempo_atual >= self.proximo_disparo_sensor):
                    self.proximo_disparo_sensor = tempo_atual + self.cooldown_sensor_seg
                    self.acionar_leitura_ultrassonico(origem=f"HC-SR04 ({dist:.1f}cm)")
            else:
                self.lbl_distancia_sensor.config(
                    text=f"Distância: {dist:.1f} cm (Standby)",
                    fg=CORES["fg_muted"]
                )
        else:
            if self.frames_restantes_ocr > 0 or self.ocr_em_andamento:
                self.lbl_distancia_sensor.config(
                    text=f"Leitura Ativada ({self.frames_por_acionamento} frame)",
                    fg=CORES["warning"]
                )
            else:
                self.lbl_distancia_sensor.config(
                    text=f"Em Standby (Limite <= {int(self.distancia_gatilho_cm)} cm)",
                    fg=CORES["fg_muted"]
                )

    def update_video(self):
        # Se o operador estiver em outra aba (Histórico ou Moradores), pausa redimensionamento e conversão de vídeo
        if self.notebook.index(self.notebook.select()) != 0:
            self.em_standby = True
            self.window.after(200, self.update_video)
            return

        # A leitura agora puxa da variável da Thread, o que impede a interface (Tkinter) de congelar
        ret = self.ret_atual
        frame_thread = self.frame_atual

        # --- ROTINA DE RECONEXÃO AUTOMÁTICA ---
        if not ret or frame_thread is None:
            self.lbl_placa.config(text="Câmera offline. Reconectando...", fg=CORES["danger"])
            self.lbl_vinculo.config(text="Vínculo: --", fg=CORES["fg_muted"])
            self.window.after(2000, self.update_video)
            return

        tempo_atual = time.time()

        tempo_decorrido = tempo_atual - self.tempo_anterior
        self.tempo_anterior = tempo_atual
        if tempo_decorrido > 0:
            fps_instantaneo = 1.0 / tempo_decorrido
            self.fps_suavizado = (0.85 * self.fps_suavizado) + (0.15 * fps_instantaneo)

        # Verifica o sensor ultrassônico
        self._verificar_gatilho_ultrassonico(tempo_atual)

        # Coleta resultados prontos do Worker Multithread antes de desenhar o quadro
        try:
            while True:
                resultado_ocr = self.fila_resultados.get_nowait()
                self.ocr_em_andamento = False
                self.retangulos_candidatos = resultado_ocr["retangulos"]
                if resultado_ocr["confirmada"]:
                    self.placa_exibida = resultado_ocr["placa"]
                    self.coords_exibidas = resultado_ocr["coords"]
                    self.timestamp_expira = tempo_atual + 8.0  # 8s para o segurança conferir a foto do condutor com calma
                    self.timestamp_modo_ativo = max(self.timestamp_modo_ativo, self.timestamp_expira)
                    self.retangulos_candidatos = []
                    self.frames_restantes_ocr = 0  # Já confirmou a placa, poupa eventuais frames restantes
        except queue.Empty:
            pass

        # --- REDIMENSIONAMENTO OTIMIZADO (SEM CÓPIA DESNECESSÁRIA EM FULL HD) ---
        altura, largura = frame_thread.shape[:2]

        if (largura, altura) != (640, 480):
            interp_ui = cv2.INTER_NEAREST if self.em_standby else cv2.INTER_LINEAR
            frame = cv2.resize(frame_thread, (640, 480), interpolation=interp_ui)
        else:
            frame = frame_thread.copy()

        for (x, y, w, h) in self.retangulos_candidatos:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Scan (1 Frame)...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.placa_exibida and tempo_atual < self.timestamp_expira:
            x, y, w, h = self.coords_exibidas
            cor_box = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), cor_box, 2)
            cv2.putText(frame, self.placa_exibida, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_box, 2)

            self.lbl_placa.config(text=f"Placa Identificada: {self.placa_exibida}", fg=CORES["success"])

            condutores = self.verificar_condutores(self.placa_exibida)
            self.atualizar_painel_condutores(self.placa_exibida, condutores)

            if condutores:
                nomes = ", ".join(c[0] for c in condutores)
                unidade_ref = condutores[0][1]
                veiculo_ref = f" ({condutores[0][2]})" if condutores[0][2] else ""
                vinculo_exibicao = f"Unidade {unidade_ref}{veiculo_ref} | Autorizado(s): {nomes}"
                cor_vinculo = CORES["success"]
            else:
                vinculo_exibicao = "Visitante (Sem condutor cadastrado)"
                cor_vinculo = CORES["warning"]

            self.lbl_vinculo.config(text=vinculo_exibicao, fg=cor_vinculo)

            if (self.placa_exibida != self.ultima_placa_gravada or
                    tempo_atual - self.timestamp_ultimo_registro > 60):
                # Salva no banco de dados
                self.registrar_movimento(self.placa_exibida, vinculo_exibicao)
                self.ultima_placa_gravada = self.placa_exibida
                self.timestamp_ultimo_registro = tempo_atual
        else:
            if self.placa_exibida:
                self.placa_exibida = ""
                self.coords_exibidas = (0, 0, 0, 0)
                self.retangulos_candidatos = []
                self.atualizar_painel_condutores("")

            # OTIMIZAÇÃO RASPBERRY PI 4:
            # Só executa o YOLO (NCNN) + fast-plate-ocr no ThreadPool quando o sensor ultrassônico liberar frames_restantes_ocr > 0!
            if self.frames_restantes_ocr > 0 and not self.ocr_em_andamento:
                self.ocr_em_andamento = True
                self.frames_restantes_ocr -= 1
                self.lbl_placa.config(text="Processando 1 Frame (YOLO + OCR)...", fg=CORES["warning"])
                # Envia o frame na resolução original da câmera para preservar todos os detalhes no recorte da placa
                frame_para_ocr = frame_thread.copy()
                self.executor_ia.submit(self.processar_ocr_background, frame_para_ocr)
            elif not self.ocr_em_andamento:
                self.lbl_placa.config(text="Standby (Aguardando Sensor Ultrassônico)", fg=CORES["fg_muted"])
                self.lbl_vinculo.config(text="Vínculo: --", fg=CORES["fg_muted"])

        # Define a taxa de quadros adaptativa (10–15 FPS em Standby vs ~25 FPS quando Ativo)
        ia_rodando = (self.ocr_em_andamento or self.frames_restantes_ocr > 0)
        modo_ativo = bool(self.placa_exibida or tempo_atual < self.timestamp_modo_ativo or ia_rodando)
        self.em_standby = not modo_ativo

        if self.ocr_em_andamento:
            proximo_delay = self.delay_ia
            status_ia_txt = "IA: PROCESSANDO (Multithread)"
            cor_hud = (0, 165, 255)
        elif modo_ativo:
            proximo_delay = self.delay_ativo
            status_ia_txt = "MODO: ATIVO | IA: PRONTA"
            cor_hud = (0, 255, 0)
        else:
            proximo_delay = self.delay_standby
            status_ia_txt = "STANDBY (~12 FPS | Economia Rasp 4)"
            cor_hud = (0, 255, 255)

        cv2.putText(
            frame,
            f"FPS: {int(self.fps_suavizado)} | {status_ia_txt}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            cor_hud,
            2
        )

        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.itemconfig(self.canvas_img_id, image=self.photo)

        self.window.after(proximo_delay, self.update_video)

    def normalizar_placa(self, placa):
        return re.sub(r'[^A-Z0-9]', '', placa.upper())

    def _detectar_tipo_placa_visual(self, crop_bgr):
        """Identifica visualmente se o recorte é Placa Mercosul (faixa superior azul) ou Antiga (cinza).
        Auxilia na correção exata do 5º caractere (posição 4: Letra no Mercosul vs Número na Antiga)."""
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return "auto"
            h, w = crop_bgr.shape[:2]
            if h < 12 or w < 24:
                return "auto"
            faixa_topo = crop_bgr[0:max(4, int(h * 0.32)), 0:w]
            hsv = cv2.cvtColor(faixa_topo, cv2.COLOR_BGR2HSV)
            # Faixa azul padrão Mercosul no espaço HSV
            mascara_azul = cv2.inRange(hsv, np.array([95, 55, 45]), np.array([135, 255, 255]))
            proporcao_azul = float(np.count_nonzero(mascara_azul)) / float(mascara_azul.size)
            if proporcao_azul >= 0.08:
                return "mercosul"
            if proporcao_azul < 0.02:
                return "antiga"
        except Exception:
            pass
        return "auto"

    def _placa_existe_no_banco(self, placa):
        return placa in self.placas_cadastradas_cache

    def corrigir_caracteres_confusos(self, texto, tipo_placa="auto"):
        """Corrige confusões entre letras e números SEMPRE ANTES da validação de formato,
        respeitando rigorosamente as posições da placa Antiga (LLLNNNN) e Mercosul (LLLNLNN)."""
        texto = re.sub(r'[^A-Z0-9]', '', texto.upper())
        if len(texto) != 7:
            return texto

        lista = list(texto)
        letras_p_numeros = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
        numeros_p_letras = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}

        # Posições 0, 1 e 2 (1º, 2º e 3º caracteres): SEMPRE LETRAS (Antiga e Mercosul)
        for i in range(3):
            if lista[i] in letras_p_numeros:
                lista[i] = letras_p_numeros[lista[i]]

        # Posição 3 (4º caractere): SEMPRE NÚMERO (Antiga e Mercosul)
        if lista[3] in numeros_p_letras:
            lista[3] = numeros_p_letras[lista[3]]

        # Posições 5 e 6 (6º e 7º caracteres): SEMPRE NÚMEROS (Antiga e Mercosul)
        for i in range(5, 7):
            if lista[i] in numeros_p_letras:
                lista[i] = numeros_p_letras[lista[i]]

        # Posição 4 (5º caractere): NÚMERO na Placa Antiga (LLLNNNN) | LETRA na Placa Mercosul (LLLNLNN)
        c4 = lista[4]
        char_como_letra = letras_p_numeros.get(c4, c4)
        char_como_num = numeros_p_letras.get(c4, c4)

        lista_mercosul = lista.copy()
        lista_mercosul[4] = char_como_letra
        placa_mercosul = "".join(lista_mercosul)

        lista_antiga = lista.copy()
        lista_antiga[4] = char_como_num
        placa_antiga = "".join(lista_antiga)

        # 1) Se uma das duas variantes (Antiga vs Mercosul) constar nos moradores cadastrados, prioriza-a
        if placa_mercosul != placa_antiga:
            existe_mercosul = self._placa_existe_no_banco(placa_mercosul)
            existe_antiga = self._placa_existe_no_banco(placa_antiga)
            if existe_mercosul and not existe_antiga:
                return placa_mercosul
            if existe_antiga and not existe_mercosul:
                return placa_antiga

        # 2) Respeita a identificação visual do padrão da placa (Mercosul vs Antiga)
        if tipo_placa == "mercosul" and char_como_letra.isalpha():
            lista[4] = char_como_letra
        elif tipo_placa == "antiga" and char_como_num.isdigit():
            lista[4] = char_como_num

        return "".join(lista)

    def _ordenar_pontos_quadrilatero(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _corrigir_perspectiva_e_inclinacao(self, crop_bgr):
        """Aplica correção geométrica de perspectiva (homografia em contorno quadrilateral)
        e correção de inclinação (deskew por linhas horizontais) preservando os detalhes."""
        img = crop_bgr
        ch, cw = img.shape[:2]
        if ch < 15 or cw < 30:
            return img

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            bordas = cv2.Canny(blur, 50, 150)

            # 1) Tentativa de correção de perspectiva se a borda da placa formar um quadrilátero claro
            contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_total = float(ch * cw)
            for cnt in sorted(contornos, key=cv2.contourArea, reverse=True)[:3]:
                area = cv2.contourArea(cnt)
                if area < 0.45 * area_total:
                    break
                perimetro = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.03 * perimetro, True)
                if len(approx) == 4:
                    pts = approx.reshape(4, 2).astype("float32")
                    rect = self._ordenar_pontos_quadrilatero(pts)
                    dst = np.array([
                        [0, 0],
                        [cw - 1, 0],
                        [cw - 1, ch - 1],
                        [0, ch - 1]
                    ], dtype="float32")
                    M_persp = cv2.getPerspectiveTransform(rect, dst)
                    img = cv2.warpPerspective(img, M_persp, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    bordas = cv2.Canny(gray, 50, 150)
                    break

            # 2) Correção de inclinação (Deskew) usando linhas horizontais dominantes da placa
            linhas = cv2.HoughLinesP(
                bordas,
                1,
                np.pi / 180.0,
                threshold=max(18, cw // 4),
                minLineLength=max(20, cw // 3),
                maxLineGap=10
            )
            if linhas is not None:
                angulos = []
                for linha in linhas:
                    x1, y1, x2, y2 = linha[0]
                    if x2 != x1:
                        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if -22.0 <= ang <= 22.0:
                            angulos.append(ang)
                if angulos:
                    angulo_mediano = float(np.median(angulos))
                    if 0.8 <= abs(angulo_mediano) <= 22.0:
                        centro = (cw // 2, ch // 2)
                        M_rot = cv2.getRotationMatrix2D(centro, angulo_mediano, 1.0)
                        img = cv2.warpAffine(img, M_rot, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            pass

        return img

    def preprocessar_recorte_placa(self, crop_bgr):
        """Preserva a resolução original quando em faixa ideal, amplia recortes pequenos (INTER_CUBIC),
        aplica correção de inclinação/perspectiva (INTER_LINEAR) e realce de contraste CLAHE preservando cores (LAB -> BGR)."""
        # 1) Corrige perspectiva e inclinação no recorte colorido
        crop_alinhado = self._corrigir_perspectiva_e_inclinacao(crop_bgr)

        # 2) Preserva a resolução original entre 180px e 320px; amplia recortes pequenos (< 180px)
        ch, cw = crop_alinhado.shape[:2]
        if cw < 180 or ch < 48:
            fator = min(2.5, max(220.0 / max(1, cw), 56.0 / max(1, ch)))
            novo_w = int(round(cw * fator))
            novo_h = int(round(ch * fator))
            crop_alinhado = cv2.resize(crop_alinhado, (novo_w, novo_h), interpolation=cv2.INTER_CUBIC)
        elif cw > 320:
            escala = 320.0 / cw
            crop_alinhado = cv2.resize(crop_alinhado, (320, max(48, int(round(ch * escala)))), interpolation=cv2.INTER_AREA)

        # 3) Realce de contraste CLAHE apenas no canal de luminância (LAB) + nitidez leve preservando as cores para o fast-plate-ocr
        lab = cv2.cvtColor(crop_alinhado, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_clahe = self.clahe.apply(l_ch)
        l_blur = cv2.GaussianBlur(l_clahe, (3, 3), 0)
        l_nitido = cv2.addWeighted(l_clahe, 1.25, l_blur, -0.25, 0)
        crop_nitido_bgr = cv2.cvtColor(cv2.merge((l_nitido, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return crop_nitido_bgr

    def _executar_ocr_placa(self, crop_bgr):
        """Executa o fast-plate-ocr (ONNX Runtime) respeitando o modo de cor (RGB/Grayscale)
        e retorna lista padronizada no formato [(None, texto, confianca_media)]."""
        if self.ocr_color_mode == "grayscale":
            img_ocr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_ocr = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        saida = self.ocr.run(img_ocr, return_confidence=True)
        resultados = []

        # Compatibilidade com API v0.x (tupla (plates, probs)) e API v1.x+ (lista de PlatePrediction)
        if isinstance(saida, tuple) and len(saida) == 2:
            placas, probs = saida
            for txt, prob_arr in zip(placas, probs):
                txt_str = str(txt)
                arr = np.asarray(prob_arr, dtype=np.float32).flatten()
                validos = [float(p) for ch, p in zip(txt_str, arr) if ch.isalnum()]
                conf = float(np.mean(validos)) if validos else (float(np.mean(arr)) if arr.size else 0.0)
                resultados.append((None, txt_str, conf))
        elif isinstance(saida, list):
            for item in saida:
                if hasattr(item, "plate"):
                    txt_str = str(item.plate)
                    char_probs = getattr(item, "char_probs", None)
                    if char_probs is not None:
                        arr = np.asarray(char_probs, dtype=np.float32).flatten()
                        validos = [float(p) for ch, p in zip(txt_str, arr) if ch.isalnum()]
                        conf = float(np.mean(validos)) if validos else (float(np.mean(arr)) if arr.size else 0.0)
                    else:
                        conf = 1.0
                    resultados.append((None, txt_str, conf))
                elif isinstance(item, str):
                    resultados.append((None, item, 1.0))

        return resultados

    def _avaliar_resultados_ocr(self, resultados_ocr, tipo_placa_visual):
        """Filtra, corrige caracteres confusos ANTES da validação de formato e retorna a placa válida"""
        for _, txt, confianca in resultados_ocr:
            texto_limpo = "".join(e for e in txt if e.isalnum()).upper()
            if not texto_limpo:
                continue

            if len(texto_limpo) != 7:
                print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} -> [Ignorado: cabeçalho/ruído ({len(texto_limpo)} chars)]")
                continue

            if confianca < self.confianca_minima_ocr:
                print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} (< {self.confianca_minima_ocr:.2f}) -> [Ignorado: baixa confiança]")
                continue

            # Aplica a correção posicional SEMPRE ANTES de validar o formato (Antigo / Mercosul)
            texto_corrigido = self.corrigir_caracteres_confusos(texto_limpo, tipo_placa=tipo_placa_visual)

            if self.padrao_antigo.match(texto_corrigido) or self.padrao_mercosul.match(texto_corrigido):
                if texto_corrigido != texto_limpo:
                    print(f"OCR: {texto_limpo} -> Corrigido ({tipo_placa_visual}): {texto_corrigido} | Confiança: {confianca:.3f} -> [Formato de Placa OK]")
                else:
                    print(f"OCR: {texto_corrigido} ({tipo_placa_visual}) | Confiança: {confianca:.3f} -> [Formato de Placa OK]")
                return texto_corrigido
            else:
                print(f"OCR: {texto_limpo} (Corrigido: {texto_corrigido}) | Confiança: {confianca:.3f} -> [Ignorado: fora do padrão de placa]")
        return ""

    def processar_ocr_background(self, frame):
        resultado_fila = {
            "confirmada": False,
            "placa": "",
            "coords": (0, 0, 0, 0),
            "retangulos": []
        }
        try:
            inicio_t = time.time()
            altura_f, largura_f = frame.shape[:2]
            escala_ui_x = 640.0 / float(max(1, largura_f))
            escala_ui_y = 480.0 / float(max(1, altura_f))

            # 1) Detecção da placa com YOLOv8 (NCNN) no frame original (conf=0.25 evita rodar OCR em falsos positivos)
            resultados_yolo = self.modelo_yolo(frame, conf=0.25, imgsz=320, verbose=False)
            tempo_yolo = time.time() - inicio_t

            melhor_crop = None
            melhor_box_ui = None
            maior_confianca = -1

            for r in resultados_yolo:
                print(f"Detecções YOLO (1 Frame | {tempo_yolo:.2f}s): {len(r.boxes)}")
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    print(f"Classe: {cls} | Confiança: {conf:.3f}")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    if w > 30 and h > 15 and conf > maior_confianca:
                        maior_confianca = conf

                        # Margem suave para preservar todos os caracteres sem cortar extremidades
                        pad_w = int(w * 0.05)
                        pad_h = int(h * 0.04)

                        x1_pad = max(0, x1 - pad_w)
                        y1_pad = max(0, y1 - pad_h)
                        x2_pad = min(largura_f, x2 + pad_w)
                        y2_pad = min(altura_f, y2 + pad_h)

                        melhor_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                        melhor_box_ui = (
                            int(x1_pad * escala_ui_x),
                            int(y1_pad * escala_ui_y),
                            int((x2_pad - x1_pad) * escala_ui_x),
                            int((y2_pad - y1_pad) * escala_ui_y),
                        )

            if melhor_box_ui is not None:
                resultado_fila["retangulos"] = [melhor_box_ui]

            if melhor_crop is not None and melhor_crop.size > 0:
                inicio_ocr = time.time()
                tipo_placa_visual = self._detectar_tipo_placa_visual(melhor_crop)
                crop_processado = self.preprocessar_recorte_placa(melhor_crop)

                # VIA ULTRA-RÁPIDA (~20ms–60ms no Rasp 4): fast-plate-ocr em ONNX Runtime
                resultados_rapidos = self._executar_ocr_placa(crop_processado)
                placa_candidata = self._avaliar_resultados_ocr(resultados_rapidos, tipo_placa_visual)

                # Fallback ultrarrápido (~20ms–50ms) no recorte original sem deskew caso a correção geométrica distorça uma placa já alinhada
                if not placa_candidata:
                    resultados_raw = self._executar_ocr_placa(melhor_crop)
                    placa_candidata = self._avaliar_resultados_ocr(resultados_raw, tipo_placa_visual)

                tempo_ocr = time.time() - inicio_ocr

                if not placa_candidata and not resultados_rapidos:
                    print(f"   [Tempo Total: {(time.time() - inicio_t):.2f}s (YOLO: {tempo_yolo:.2f}s | OCR: {tempo_ocr:.2f}s)] Nenhum texto lido.")
                    return

                if placa_candidata:
                    placa_candidata = self.normalizar_placa(placa_candidata)
                    if placa_candidata == self.leitura_temporaria:
                        self.contador_leituras += 1
                    else:
                        self.leitura_temporaria = placa_candidata
                        self.contador_leituras = 1

                    tempo_total = time.time() - inicio_t
                    print(f"   [Validação] Placa: {placa_candidata} ({self.contador_leituras}/{self.limiar_confirmacao}) | Tempo Total: {tempo_total:.2f}s (YOLO: {tempo_yolo:.2f}s | OCR: {tempo_ocr:.2f}s)")

                    if self.contador_leituras >= self.limiar_confirmacao:
                        print(f"   >>> PLACA CONFIRMADA E REGISTRADA: {placa_candidata} <<<")
                        resultado_fila["confirmada"] = True
                        resultado_fila["placa"] = placa_candidata
                        resultado_fila["coords"] = melhor_box_ui
                        resultado_fila["retangulos"] = []
                        self.contador_leituras = 0
                        self.leitura_temporaria = ""
                        return
            else:
                print(f"   [Tempo YOLO: {tempo_yolo:.2f}s] Nenhuma placa detectada neste frame.")

        except Exception as e:
            print(f"Erro no processamento YOLO/fast-plate-ocr: {e}")
        finally:
            self.fila_resultados.put(resultado_fila)

    def registrar_movimento(self, placa, vinculo):
        agora = datetime.now()
        tipo = self.inferir_tipo_movimento(placa)
        with self.get_db() as conn:
            conn.execute("INSERT INTO registros (placa, tipo, data, hora, vinculo) VALUES (?, ?, ?, ?, ?)",
                         (placa, tipo, agora.strftime("%d/%m/%Y"), agora.strftime("%H:%M:%S"), vinculo))

    def inferir_tipo_movimento(self, placa):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tipo FROM registros WHERE placa = ? ORDER BY id DESC LIMIT 1", (placa,))
            ultimo = cursor.fetchone()
            return "SAÍDA" if ultimo and ultimo[0] == "ENTRADA" else "ENTRADA"

    def verificar_condutores(self, placa):
        """Retorna todos os condutores autorizados (nome, unidade, veiculo, foto_path) vinculados à placa"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome, unidade, COALESCE(veiculo, ''), COALESCE(foto_path, '') FROM moradores WHERE placa = ? ORDER BY id ASC", (placa,))
            return cursor.fetchall()

    def configurar_aba_historico(self):
        frame = tk.Frame(
            self.tab_historico,
            bg=CORES["bg_panel"],
            padx=2,
            pady=2,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        colunas = ("ID", "Placa", "Tipo", "Data", "Horario", "Vinculo")
        self.tree_historico = ttk.Treeview(frame, columns=colunas, show="headings")

        config_colunas = {
            "ID": ("ID", 60, tk.CENTER),
            "Placa": ("Placa", 120, tk.CENTER),
            "Tipo": ("Movimento", 110, tk.CENTER),
            "Data": ("Data", 110, tk.CENTER),
            "Horario": ("Horário", 110, tk.CENTER),
            "Vinculo": ("Vínculo / Identificação", 420, tk.W),
        }
        for col, (titulo, largura, alinhamento) in config_colunas.items():
            self.tree_historico.heading(col, text=titulo)
            self.tree_historico.column(col, width=largura, anchor=alinhamento)

        self.tree_historico.tag_configure("par", background=CORES["row_even"])
        self.tree_historico.tag_configure("impar", background=CORES["row_odd"])

        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree_historico.yview)
        self.tree_historico.configure(yscrollcommand=scrollbar.set)

        self.tree_historico.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def atualizar_tabela_historico(self):
        for item in self.tree_historico.get_children():
            self.tree_historico.delete(item)
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM registros ORDER BY id DESC")
            for idx, row in enumerate(cursor.fetchall()):
                tag = "par" if idx % 2 == 0 else "impar"
                self.tree_historico.insert("", tk.END, values=row, tags=(tag,))

    def ao_mudar_de_aba(self, event):
        aba = self.notebook.index(self.notebook.select())
        if aba == 0:
            try:
                self.distancia_gatilho_cm = float(self.spin_distancia.get())
                self.frames_por_acionamento = max(1, int(self.spin_frames.get()))
                val_conf = float(self.spin_confianca.get())
                self.confianca_minima_ocr = (val_conf / 100.0) if val_conf > 1.0 else val_conf
                self.lbl_resumo_config.config(
                    text=f"Gatilho: <= {int(self.distancia_gatilho_cm)} cm  |  {self.frames_por_acionamento} frame(s)  |  Confiança >= {int(round(self.confianca_minima_ocr * 100))}%  |  {self.resolucao_camera}"
                )
            except Exception:
                pass
        elif aba == 1:
            self.atualizar_tabela_historico()
        elif aba == 2:
            self.atualizar_tabela_moradores()

    def _criar_entry_estilizado(self, parent, largura):
        return tk.Entry(
            parent,
            width=largura,
            font=FONTES["label"],
            bg=CORES["input_bg"],
            fg=CORES["fg_text"],
            insertbackground=CORES["fg_text"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=CORES["border"],
            highlightcolor=CORES["accent"]
        )

    def configurar_aba_moradores(self):
        frame = tk.Frame(
            self.tab_moradores,
            bg=CORES["bg_panel"],
            padx=20,
            pady=15,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        frame.pack(fill=tk.X, padx=20, pady=(15, 10))

        self.lbl_titulo_form = tk.Label(
            frame,
            text="CADASTRO E EDIÇÃO DE CONDUTORES E VEÍCULOS (DÊ DUPLO CLIQUE NA TABELA PARA EDITAR)",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"]
        )
        self.lbl_titulo_form.pack(anchor=tk.W, pady=(0, 10))

        form_row = tk.Frame(frame, bg=CORES["bg_panel"])
        form_row.pack(fill=tk.X)

        tk.Label(form_row, text="Condutor:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 5))
        self.ent_nome = self._criar_entry_estilizado(form_row, 18)
        self.ent_nome.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        tk.Label(form_row, text="Unidade:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 5))
        self.ent_unidade = self._criar_entry_estilizado(form_row, 8)
        self.ent_unidade.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        tk.Label(form_row, text="Placa:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 5))
        self.ent_placa = self._criar_entry_estilizado(form_row, 10)
        self.ent_placa.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        tk.Label(form_row, text="Veículo/Cor:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 5))
        self.ent_veiculo = self._criar_entry_estilizado(form_row, 16)
        self.ent_veiculo.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        ttk.Button(
            form_row,
            text="Escolher Foto...",
            style="Accent.TButton",
            command=self.selecionar_foto_cadastro
        ).pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_foto_escolhida = tk.Label(
            form_row,
            text="Sem foto selecionada",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        )
        self.lbl_foto_escolhida.pack(side=tk.LEFT, padx=(0, 8))

        botoes_row = tk.Frame(frame, bg=CORES["bg_panel"])
        botoes_row.pack(fill=tk.X, pady=(12, 0))

        self.btn_salvar = ttk.Button(
            botoes_row,
            text="Salvar Novo Condutor",
            style="Accent.TButton",
            command=self.salvar_morador
        )
        self.btn_salvar.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            botoes_row,
            text="Editar Selecionado",
            style="Accent.TButton",
            command=self.carregar_para_edicao
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            botoes_row,
            text="Limpar / Cancelar Edição",
            style="Accent.TButton",
            command=self.limpar_formulario
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            botoes_row,
            text="Vincular / Trocar Foto Rápido",
            style="Accent.TButton",
            command=self.vincular_foto_selecionado
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            botoes_row,
            text="Excluir Selecionado",
            style="Danger.TButton",
            command=self.excluir_morador
        ).pack(side=tk.LEFT)

        frame_tb = tk.Frame(
            self.tab_moradores,
            bg=CORES["bg_panel"],
            padx=2,
            pady=2,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        frame_tb.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 15))

        colunas = ("ID", "Nome", "Unidade", "Placa", "Veiculo", "Foto")
        self.tree_moradores = ttk.Treeview(frame_tb, columns=colunas, show="headings")

        config_colunas = {
            "ID": ("ID", 55, tk.CENTER),
            "Nome": ("Nome do Condutor Autorizado", 260, tk.W),
            "Unidade": ("Unidade / Apto", 120, tk.CENTER),
            "Placa": ("Placa Veicular", 120, tk.CENTER),
            "Veiculo": ("Modelo / Cor do Veículo", 200, tk.CENTER),
            "Foto": ("Foto de Verificação", 160, tk.CENTER),
        }
        for col, (titulo, largura, alinhamento) in config_colunas.items():
            self.tree_moradores.heading(col, text=titulo)
            self.tree_moradores.column(col, width=largura, anchor=alinhamento)

        self.tree_moradores.tag_configure("par", background=CORES["row_even"])
        self.tree_moradores.tag_configure("impar", background=CORES["row_odd"])

        # Permite dar duplo clique em qualquer linha da tabela para carregar os dados para edição
        self.tree_moradores.bind("<Double-1>", lambda e: self.carregar_para_edicao())

        scrollbar = ttk.Scrollbar(frame_tb, orient=tk.VERTICAL, command=self.tree_moradores.yview)
        self.tree_moradores.configure(yscrollcommand=scrollbar.set)

        self.tree_moradores.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def selecionar_foto_cadastro(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar Foto do Condutor Autorizado",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if caminho:
            self.foto_selecionada_temp = caminho
            nome_arq = Path(caminho).name
            if len(nome_arq) > 22:
                nome_arq = nome_arq[:19] + "..."
            self.lbl_foto_escolhida.config(text=f"Foto: {nome_arq}", fg=CORES["success"])

    def _copiar_foto_para_pasta_local(self, origem_path, placa):
        if not origem_path:
            return ""
        origem = Path(origem_path)
        if not origem.exists():
            return ""
        ext = origem.suffix.lower() or ".jpg"
        nome_destino = f"{placa}_{int(time.time() * 1000)}{ext}"
        destino = self.pasta_fotos / nome_destino
        shutil.copy2(origem, destino)
        return str(Path("fotos_condutores") / nome_destino)

    def carregar_para_edicao(self):
        selecionado = self.tree_moradores.selection()
        if not selecionado:
            messagebox.showwarning("Aviso", "Selecione um registro na tabela para editar!")
            return

        valores = self.tree_moradores.item(selecionado[0], "values")
        id_morador = valores[0]

        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, nome, unidade, placa, COALESCE(veiculo, ''), COALESCE(foto_path, '') FROM moradores WHERE id = ?", (id_morador,))
            reg = cursor.fetchone()

        if not reg:
            return

        self.id_em_edicao = reg[0]
        self.ent_nome.delete(0, tk.END)
        self.ent_nome.insert(0, reg[1])

        self.ent_unidade.delete(0, tk.END)
        self.ent_unidade.insert(0, reg[2])

        self.ent_placa.delete(0, tk.END)
        self.ent_placa.insert(0, reg[3])

        self.ent_veiculo.delete(0, tk.END)
        self.ent_veiculo.insert(0, reg[4])

        self.foto_selecionada_temp = ""
        if reg[5]:
            self.lbl_foto_escolhida.config(text="Mantendo foto atual (ou escolha outra)", fg=CORES["success"])
        else:
            self.lbl_foto_escolhida.config(text="Sem foto (escolha uma se desejar)", fg=CORES["fg_muted"])

        self.lbl_titulo_form.config(
            text=f"EDITANDO CADASTRO ID #{self.id_em_edicao} — Altere os campos desejados e clique em 'Salvar Alterações'",
            fg=CORES["warning"]
        )
        self.btn_salvar.config(text=f"Salvar Alterações (ID #{self.id_em_edicao})")

    def limpar_formulario(self):
        self.id_em_edicao = None
        self.ent_nome.delete(0, tk.END)
        self.ent_unidade.delete(0, tk.END)
        self.ent_placa.delete(0, tk.END)
        self.ent_veiculo.delete(0, tk.END)
        self.foto_selecionada_temp = ""
        self.lbl_foto_escolhida.config(text="Sem foto selecionada", fg=CORES["fg_muted"])
        self.lbl_titulo_form.config(
            text="CADASTRO E EDIÇÃO DE CONDUTORES E VEÍCULOS (DÊ DUPLO CLIQUE NA TABELA PARA EDITAR)",
            fg=CORES["accent"]
        )
        self.btn_salvar.config(text="Salvar Novo Condutor")

    def vincular_foto_selecionado(self):
        selecionado = self.tree_moradores.selection()
        if not selecionado:
            messagebox.showwarning("Aviso", "Selecione um condutor na tabela primeiro para vincular ou trocar a foto!")
            return

        valores = self.tree_moradores.item(selecionado[0], "values")
        id_morador, nome, unidade, placa = valores[0], valores[1], valores[2], valores[3]

        caminho = filedialog.askopenfilename(
            title=f"Selecionar Foto para {nome} ({placa})",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not caminho:
            return

        caminho_salvo = self._copiar_foto_para_pasta_local(caminho, placa)
        with self.get_db() as conn:
            conn.execute("UPDATE moradores SET foto_path = ? WHERE id = ?", (caminho_salvo, id_morador))

        self.placa_painel_renderizada = None  # Força atualização caso a placa esteja em exibição
        self.atualizar_tabela_moradores()
        messagebox.showinfo("Sucesso", f"Foto vinculada com sucesso ao condutor '{nome}' (Placa {placa})!")

    def salvar_morador(self):
        nome = self.ent_nome.get().strip()
        unidade = self.ent_unidade.get().strip()
        placa_digitada = self.ent_placa.get().strip().upper()
        veiculo = self.ent_veiculo.get().strip()

        if not nome or not unidade or not placa_digitada:
            messagebox.showwarning("Aviso", "Preencha pelo menos Condutor, Unidade e Placa!")
            return

        placa_final = self.normalizar_placa(placa_digitada)
        if not (self.padrao_antigo.match(placa_final) or self.padrao_mercosul.match(placa_final)):
            messagebox.showerror(
                "Erro",
                "Formato inválido! Digite os 7 caracteres da placa no padrão Antigo (ABC1234) ou Mercosul (ABC1D23)."
            )
            return

        if self.id_em_edicao is not None:
            # Modo de Edição: atualiza o registro existente
            with self.get_db() as conn:
                if self.foto_selecionada_temp:
                    caminho_salvo = self._copiar_foto_para_pasta_local(self.foto_selecionada_temp, placa_final)
                    conn.execute(
                        "UPDATE moradores SET nome = ?, unidade = ?, placa = ?, veiculo = ?, foto_path = ? WHERE id = ?",
                        (nome, unidade, placa_final, veiculo, caminho_salvo, self.id_em_edicao)
                    )
                else:
                    conn.execute(
                        "UPDATE moradores SET nome = ?, unidade = ?, placa = ?, veiculo = ? WHERE id = ?",
                        (nome, unidade, placa_final, veiculo, self.id_em_edicao)
                    )

            id_editado = self.id_em_edicao
            self.limpar_formulario()
            self.placa_painel_renderizada = None
            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", f"Cadastro #{id_editado} atualizado com sucesso!")
        else:
            # Modo de Novo Cadastro
            caminho_salvo = self._copiar_foto_para_pasta_local(self.foto_selecionada_temp, placa_final)

            with self.get_db() as conn:
                conn.execute(
                    "INSERT INTO moradores (nome, unidade, placa, veiculo, foto_path) VALUES (?, ?, ?, ?, ?)",
                    (nome, unidade, placa_final, veiculo, caminho_salvo)
                )

            self.limpar_formulario()
            self.placa_painel_renderizada = None
            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", f"Condutor '{nome}' autorizado para o veículo {placa_final}!")

    def excluir_morador(self):
        selecionado = self.tree_moradores.selection()
        if not selecionado:
            messagebox.showwarning("Aviso", "Selecione um condutor na tabela para excluir!")
            return

        valores = self.tree_moradores.item(selecionado[0], "values")
        id_morador, nome, unidade, placa = valores[0], valores[1], valores[2], valores[3]

        confirmar = messagebox.askyesno(
            "Confirmar Exclusão",
            f"Deseja realmente remover o condutor '{nome}' (Unidade {unidade} - Placa {placa})?"
        )
        if confirmar:
            with self.get_db() as conn:
                conn.execute("DELETE FROM moradores WHERE id = ?", (id_morador,))
            if self.id_em_edicao == int(id_morador):
                self.limpar_formulario()
            self.placa_painel_renderizada = None
            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", "Condutor excluído com sucesso!")

    def atualizar_tabela_moradores(self):
        self._atualizar_cache_placas()
        for item in self.tree_moradores.get_children():
            self.tree_moradores.delete(item)
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, nome, unidade, placa, COALESCE(veiculo, ''), COALESCE(foto_path, '') FROM moradores ORDER BY placa ASC, nome ASC")
            for idx, (id_m, nome, unidade, placa, veiculo, foto_path) in enumerate(cursor.fetchall()):
                tag = "par" if idx % 2 == 0 else "impar"
                status_foto = "Foto Cadastrada" if foto_path else "Sem Foto"
                veiculo_exib = veiculo if veiculo else "--"
                self.tree_moradores.insert("", tk.END, values=(id_m, nome, unidade, placa, veiculo_exib, status_foto), tags=(tag,))

    def configurar_aba_configuracoes(self):
        container = tk.Frame(self.tab_configuracoes, bg=CORES["bg_main"])
        container.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # --- PAINEL 1: SENSOR ULTRASSÔNICO E DESEMPENHO DA IA ---
        painel_sensor = tk.Frame(
            container,
            bg=CORES["bg_panel"],
            padx=24,
            pady=20,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        painel_sensor.pack(fill=tk.X, pady=(0, 18))

        tk.Label(
            painel_sensor,
            text="SENSOR ULTRASSÔNICO (HC-SR04) E PARÂMETROS DE LEITURA IA",
            font=FONTES["secao"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"]
        ).pack(anchor=tk.W)

        status_gpio = (
            f"Hardware GPIO Detectado — Pinos BCM: TRIG = GPIO {self.pino_trig} (Pino Físico 16) | ECHO = GPIO {self.pino_echo} (Pino Físico 18)"
            if self.gpio_disponivel
            else f"Modo Simulação Ativo — Pinos padrão Raspberry Pi 4: TRIG = GPIO {self.pino_trig} (Pino 16) | ECHO = GPIO {self.pino_echo} (Pino 18)"
        )
        tk.Label(
            painel_sensor,
            text=status_gpio,
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["success"] if self.gpio_disponivel else CORES["fg_muted"]
        ).pack(anchor=tk.W, pady=(4, 16))

        linha_dist = tk.Frame(painel_sensor, bg=CORES["bg_panel"])
        linha_dist.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            linha_dist,
            text="Distância de Gatilho do Sensor (cm):",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_text"],
            width=34,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.spin_distancia = tk.Spinbox(
            linha_dist,
            from_=10,
            to=400,
            increment=10,
            width=8,
            font=FONTES["label_bold"],
            bg=CORES["input_bg"],
            fg=CORES["fg_text"],
            buttonbackground=CORES["border"],
            relief=tk.FLAT
        )
        self.spin_distancia.delete(0, tk.END)
        self.spin_distancia.insert(0, str(int(self.distancia_gatilho_cm)))
        self.spin_distancia.pack(side=tk.LEFT, padx=(0, 14), ipady=4)

        tk.Label(
            linha_dist,
            text="Aciona a câmera/IA automaticamente quando o veículo estiver a uma distância menor ou igual a este valor.",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        linha_frames = tk.Frame(painel_sensor, bg=CORES["bg_panel"])
        linha_frames.pack(fill=tk.X, pady=(0, 14))

        tk.Label(
            linha_frames,
            text="Frames Analisados por Acionamento:",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_text"],
            width=34,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.spin_frames = tk.Spinbox(
            linha_frames,
            from_=1,
            to=10,
            increment=1,
            width=8,
            font=FONTES["label_bold"],
            bg=CORES["input_bg"],
            fg=CORES["fg_text"],
            buttonbackground=CORES["border"],
            relief=tk.FLAT
        )
        self.spin_frames.delete(0, tk.END)
        self.spin_frames.insert(0, str(self.frames_por_acionamento))
        self.spin_frames.pack(side=tk.LEFT, padx=(0, 14), ipady=4)

        tk.Label(
            linha_frames,
            text="Recomendado: 1 frame no Raspberry Pi 4 para leitura rápida, ou 2 a 3 frames para confirmação múltipla.",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        linha_conf = tk.Frame(painel_sensor, bg=CORES["bg_panel"])
        linha_conf.pack(fill=tk.X)

        tk.Label(
            linha_conf,
            text="Nível Mínimo de Confiança OCR (%):",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_text"],
            width=34,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        self.spin_confianca = tk.Spinbox(
            linha_conf,
            from_=10,
            to=95,
            increment=5,
            width=8,
            font=FONTES["label_bold"],
            bg=CORES["input_bg"],
            fg=CORES["fg_text"],
            buttonbackground=CORES["border"],
            relief=tk.FLAT
        )
        self.spin_confianca.delete(0, tk.END)
        self.spin_confianca.insert(0, str(int(round(self.confianca_minima_ocr * 100))))
        self.spin_confianca.pack(side=tk.LEFT, padx=(0, 14), ipady=4)

        tk.Label(
            linha_conf,
            text="Porcentagem mínima de certeza (ex.: 50% = 0.50) para a leitura da placa ser aceita como válida.",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        # --- PAINEL 2: CÂMERA USB LOCAL E RESOLUÇÃO DE CAPTURA ---
        painel_camera = tk.Frame(
            container,
            bg=CORES["bg_panel"],
            padx=24,
            pady=20,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        painel_camera.pack(fill=tk.X, pady=(0, 18))

        tk.Label(
            painel_camera,
            text="CÂMERA USB LOCAL E RESOLUÇÃO DE CAPTURA",
            font=FONTES["secao"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"]
        ).pack(anchor=tk.W)

        tk.Label(
            painel_camera,
            text="Selecione a resolução de captura da Webcam USB local conectada ao sistema (V4L2 / DirectShow):",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(anchor=tk.W, pady=(4, 16))

        linha_res = tk.Frame(painel_camera, bg=CORES["bg_panel"])
        linha_res.pack(fill=tk.X)

        tk.Label(
            linha_res,
            text="Resolução de Captura da Câmera:",
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_text"],
            width=34,
            anchor=tk.W
        ).pack(side=tk.LEFT)

        opcoes_res = [
            "640x480 (VGA - Mais Leve)",
            "800x600 (SVGA - Intermediário)",
            "1280x720 (HD - Mais Detalhes na Placa)",
            "1920x1080 (Full HD - Máxima Nitidez)",
        ]
        self.combo_resolucao = ttk.Combobox(
            linha_res,
            values=opcoes_res,
            state="readonly",
            width=34,
            font=FONTES["label_bold"]
        )
        res_atual_base = f"{self.cam_largura}x{self.cam_altura}"
        selecionada = opcoes_res[0]
        for op in opcoes_res:
            if op.startswith(res_atual_base):
                selecionada = op
                break
        self.combo_resolucao.set(selecionada)
        self.combo_resolucao.pack(side=tk.LEFT, padx=(0, 14), ipady=3)

        tk.Label(
            linha_res,
            text="640x480 garante leitura mais rápida no Raspberry Pi 4; resoluções maiores preservam mais detalhes à distância.",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        # --- BARRA DE AÇÃO: SALVAR E APLICAR ---
        barra_acoes = tk.Frame(container, bg=CORES["bg_main"])
        barra_acoes.pack(fill=tk.X, pady=(4, 0))

        ttk.Button(
            barra_acoes,
            text="Salvar e Aplicar Configurações",
            style="Accent.TButton",
            command=self.salvar_configuracoes
        ).pack(side=tk.LEFT, padx=(0, 14))

        self.lbl_status_config = tk.Label(
            barra_acoes,
            text="As configurações salvas ficam gravadas automaticamente no arquivo .env.",
            font=FONTES["label"],
            bg=CORES["bg_main"],
            fg=CORES["fg_muted"]
        )
        self.lbl_status_config.pack(side=tk.LEFT)

    def _salvar_env(self):
        """Persiste as configurações no arquivo .env para manter os valores ao reiniciar o sistema"""
        env_path = Path(__file__).resolve().parent / ".env"
        linhas = [
            f'CAMERA_RESOLUCAO="{self.cam_largura}x{self.cam_altura}"',
            f"ULTRASONIC_DISTANCIA_CM={int(self.distancia_gatilho_cm)}",
            f"ULTRASONIC_FRAMES={int(self.frames_por_acionamento)}",
            f"OCR_CONFIANCA_MIN={self.confianca_minima_ocr:.2f}",
            f"ULTRASONIC_TRIG_PIN={self.pino_trig}",
            f"ULTRASONIC_ECHO_PIN={self.pino_echo}",
        ]
        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(linhas) + "\n")
        except Exception as e:
            print(f"Aviso ao gravar .env: {e}")

    def salvar_configuracoes(self):
        try:
            nova_dist = max(5.0, min(400.0, float(self.spin_distancia.get())))
        except Exception:
            messagebox.showerror("Erro", "Informe uma distância válida em centímetros (ex.: 150).", parent=self.window)
            return

        try:
            novos_frames = max(1, min(10, int(self.spin_frames.get())))
        except Exception:
            messagebox.showerror("Erro", "Informe uma quantidade válida de frames (1 a 10).", parent=self.window)
            return

        try:
            val_conf = float(self.spin_confianca.get().replace(",", "."))
            nova_conf = (val_conf / 100.0) if val_conf > 1.0 else val_conf
            nova_conf = max(0.05, min(0.99, nova_conf))
        except Exception:
            messagebox.showerror("Erro", "Informe um nível de confiança válido entre 10% e 95%.", parent=self.window)
            return

        nova_res_w, nova_res_h = self._parse_resolucao(self.combo_resolucao.get())
        resolucao_mudou = (nova_res_w != self.cam_largura or nova_res_h != self.cam_altura)
        self.cam_largura, self.cam_altura = nova_res_w, nova_res_h
        self.resolucao_camera = f"{self.cam_largura}x{self.cam_altura}"

        self.distancia_gatilho_cm = nova_dist
        self.frames_por_acionamento = novos_frames
        self.confianca_minima_ocr = nova_conf
        self.limiar_confirmacao = 1 if novos_frames == 1 else min(novos_frames, 3)

        pct_conf = int(round(self.confianca_minima_ocr * 100))
        self.lbl_resumo_config.config(
            text=f"Gatilho: <= {int(self.distancia_gatilho_cm)} cm  |  {self.frames_por_acionamento} frame(s)  |  Confiança >= {pct_conf}%  |  {self.resolucao_camera}"
        )
        self.lbl_fonte_camera.config(text=f"Câmera USB Local ({self.resolucao_camera})")

        if resolucao_mudou:
            self.ret_atual = False
            self.frame_atual = None
            self.solicitar_troca_camera = True

        self._salvar_env()
        self.lbl_status_config.config(
            text=f"Configurações salvas! ({self.resolucao_camera} | Distância: {int(self.distancia_gatilho_cm)} cm | Frames: {self.frames_por_acionamento} | Confiança >= {pct_conf}%)",
            fg=CORES["success"]
        )
        messagebox.showinfo("Configurações Aplicadas", "As configurações foram salvas e aplicadas com sucesso!", parent=self.window)

    def on_closing(self):
        if hasattr(self, 'evento_parada'):
            self.evento_parada.set()
        if hasattr(self, 'executor_ia'):
            self.executor_ia.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, 'thread_camera') and self.thread_camera.is_alive():
            self.thread_camera.join(timeout=1.5)
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaPortariaHibrido(root, "V-Control - Monitoramento e Controle de Acesso LPR")
    root.mainloop()
