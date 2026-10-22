import os
import sys
import re
import time
import queue
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import easyocr

# Forçar o OpenCV a usar TCP para a câmera RTSP (evita perda de pacotes e borrões cinzas na imagem)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# IMPORTAÇÃO DA ESTILIZAÇÃO
from styles import CORES, FONTES, aplicar_tema


class SistemaPortariaHibrido:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("980x820")
        self.window.minsize(900, 760)
        self.window.configure(bg=CORES["bg_main"])

        print("Carregando IA YOLOv8 e EasyOCR...")
        self.modelo_yolo = YOLO("license_plate_detector.pt")
        sys.stdout.reconfigure(encoding="utf-8")  # Evita erro de unicode no console do Windows ao baixar os modelos do EasyOCR
        self.ocr = easyocr.Reader(["en"], gpu=False)

        self.init_db()
        self.usando_cftv = False
        self.url_cftv = 0

        # Pergunta ao usuário qual câmera utilizar
        resposta = messagebox.askyesno(
            "Seleção de Câmera",
            "Deseja utilizar a câmera CFTV configurada via RTSP?\n\n[SIM] Usa o CFTV com corte de imagem e rotação.\n[NÃO] Usa a webcam do notebook."
        )

        if resposta:
            url_encontrada = None
            env_path = Path(__file__).resolve().parent / ".env"
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("CAMERA_RTSP_URL="):
                            url_encontrada = line.strip().split("=", 1)[1].strip('"\'')
                            break

            if url_encontrada:
                self.usando_cftv = True
                self.url_cftv = url_encontrada
            else:
                messagebox.showwarning(
                    "Aviso",
                    "Variável CAMERA_RTSP_URL não encontrada no arquivo .env.\nAlternando automaticamente para a webcam do notebook."
                )

        self.fila_resultados = queue.Queue()
        self.evento_parada = threading.Event()

        # --- MELHORIAS PARA CFTV (RTSP) ---
        self.frame_atual = None
        self.ret_atual = False
        self.thread_camera = threading.Thread(target=self.loop_leitura_camera, daemon=True)
        self.thread_camera.start()

        # --- VARIÁVEIS DE CONTROLE E ESTABILIZAÇÃO ---
        self.placa_exibida = ""
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
        self.limiar_confirmacao = 3  # Exige 3 leituras idênticas

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
            text="  |  Monitoramento e Controle de Acesso Veicular (YOLOv8 + EasyOCR)",
            font=FONTES["label"],
            bg=CORES["bg_header"],
            fg=CORES["fg_muted"]
        ).pack(side=tk.LEFT)

        modo_camera_txt = "Fonte: CFTV (RTSP)" if self.usando_cftv else "Fonte: Webcam Local"
        tk.Label(
            self.header_frame,
            text=modo_camera_txt,
            font=FONTES["label_bold"],
            bg=CORES["bg_panel"],
            fg=CORES["fg_text"],
            padx=12,
            pady=4
        ).pack(side=tk.RIGHT)

        # --- ESTRUTURA DE ABAS ---
        self.notebook = ttk.Notebook(window)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_monitoramento = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_historico = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_moradores = tk.Frame(self.notebook, bg=CORES["bg_main"])

        self.notebook.add(self.tab_monitoramento, text="  Monitoramento Ao Vivo  ")
        self.notebook.add(self.tab_historico, text="  Histórico de Acessos  ")
        self.notebook.add(self.tab_moradores, text="  Cadastro de Moradores  ")

        self.configurar_aba_monitoramento()
        self.configurar_aba_historico()
        self.configurar_aba_moradores()

        self.notebook.bind("<<NotebookTabChanged>>", self.ao_mudar_de_aba)

        self.delay = 30  # ~33 FPS na interface para economizar CPU para o YOLO/EasyOCR
        self.update_video()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _abrir_camera(self):
        # DSHOW é melhor para webcams no Windows; FFMPEG é ideal para streams RTSP
        backend = cv2.CAP_FFMPEG if self.usando_cftv else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.url_cftv, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def loop_leitura_camera(self):
        # A Thread exclusiva da câmera reconecta e gerencia o buffer 100% isolada da interface
        self.cap = self._abrir_camera()
        try:
            while not self.evento_parada.is_set():
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
                else:
                    self.cap.release()
                    time.sleep(1)  # Dá um tempinho antes de tentar reabrir
        finally:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()

    def get_db(self):
        return sqlite3.connect('registro_portaria.db')

    def init_db(self):
        with self.get_db() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS registros 
                                (id INTEGER PRIMARY KEY AUTOINCREMENT, placa TEXT NOT NULL, tipo TEXT NOT NULL, data TEXT NOT NULL, hora TEXT NOT NULL, vinculo TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS moradores 
                                (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT NOT NULL, unidade TEXT NOT NULL, placa TEXT NOT NULL UNIQUE)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_registros_placa ON registros(placa)""")

    def configurar_aba_monitoramento(self):
        container_video = tk.Frame(
            self.tab_monitoramento,
            bg=CORES["bg_panel"],
            padx=8,
            pady=8,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        container_video.pack(pady=15)

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

        self.frame_status = tk.Frame(
            self.tab_monitoramento,
            bg=CORES["bg_panel"],
            padx=25,
            pady=16,
            highlightthickness=1,
            highlightbackground=CORES["border"]
        )
        self.frame_status.pack(fill=tk.X, padx=25, pady=(5, 15))
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
            text="Buscando veículos...",
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

    def update_video(self):
        # A leitura agora puxa da variável da Thread, o que impede a interface (Tkinter) de congelar
        ret = self.ret_atual
        frame_thread = self.frame_atual

        # --- ROTINA DE RECONEXÃO AUTOMÁTICA ---
        if not ret or frame_thread is None:
            # Avisa na interface gráfica
            self.lbl_placa.config(text="Câmera offline. Reconectando...", fg=CORES["danger"])
            self.lbl_vinculo.config(text="Vínculo: --", fg=CORES["fg_muted"])

            # A liberação e reconexão (cap.release/VideoCapture) agora estão acontecendo
            # automaticamente em 2º plano dentro da função `loop_leitura_camera`

            # Tenta novamente atualizar a interface
            self.window.after(2000, self.update_video)
            return  # Interrompe a execução deste ciclo

        # --- SE RET FOR TRUE, CONTINUA O CÓDIGO NORMALMENTE ---
        frame = frame_thread.copy()  # Copia para evitar que a thread subscreva enquanto desenhamos

        altura, largura = frame.shape[:2]

        if self.usando_cftv:
            # Mantém o corte da imagem dupla e a rotação de 180° para o CFTV
            frame = frame[0:int(altura / 2), 0:largura]
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        else:
            frame = cv2.resize(frame, (640, 480))

        tempo_atual = time.time()

        tempo_decorrido = tempo_atual - self.tempo_anterior
        self.tempo_anterior = tempo_atual
        if tempo_decorrido > 0:
            fps_instantaneo = 1.0 / tempo_decorrido
            self.fps_suavizado = (0.9 * self.fps_suavizado) + (0.1 * fps_instantaneo)

        cv2.putText(
            frame,
            f"FPS: {int(self.fps_suavizado)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        try:
            while True:
                resultado_ocr = self.fila_resultados.get_nowait()
                self.ocr_em_andamento = False
                self.retangulos_candidatos = resultado_ocr["retangulos"]
                if resultado_ocr["confirmada"]:
                    self.placa_exibida = resultado_ocr["placa"]
                    self.coords_exibidas = resultado_ocr["coords"]
                    self.timestamp_expira = tempo_atual + 5.0
                    self.retangulos_candidatos = []
        except queue.Empty:
            pass

        for (x, y, w, h) in self.retangulos_candidatos:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Scan...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.placa_exibida and tempo_atual < self.timestamp_expira:
            x, y, w, h = self.coords_exibidas
            cor_box = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), cor_box, 2)
            cv2.putText(frame, self.placa_exibida, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_box, 2)

            self.lbl_placa.config(text=f"Placa Identificada: {self.placa_exibida}", fg=CORES["success"])

            if (self.placa_exibida != self.ultima_placa_gravada or
                    tempo_atual - self.timestamp_ultimo_registro > 60):

                morador = self.verificar_morador(self.placa_exibida)
                vinculo = f"Morador: {morador[0]} ({morador[1]})" if morador else "Visitante"
                self.lbl_vinculo.config(text=vinculo, fg=CORES["success"] if morador else CORES["warning"])

                # Salva no banco de dados
                self.registrar_movimento(self.placa_exibida, vinculo)
                self.ultima_placa_gravada = self.placa_exibida
                self.timestamp_ultimo_registro = tempo_atual
        else:
            if self.placa_exibida:
                self.placa_exibida = ""
                self.coords_exibidas = (0, 0, 0, 0)
                self.retangulos_candidatos = []

            self.lbl_placa.config(text="Buscando veículos...", fg=CORES["fg_muted"])
            self.lbl_vinculo.config(text="Vínculo: --", fg=CORES["fg_muted"])

            if not self.ocr_em_andamento:
                self.ocr_em_andamento = True
                frame_para_ocr = frame.copy()
                threading.Thread(target=self.processar_ocr_background, args=(frame_para_ocr,), daemon=True).start()

        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.itemconfig(self.canvas_img_id, image=self.photo)

        self.window.after(self.delay, self.update_video)

    def normalizar_placa(self, placa):
        return re.sub(r'[^A-Z0-9]', '', placa.upper())

    def corrigir_caracteres_confusos(self, texto):
        """Corrige confusões comuns do OCR entre letras e números com base na posição"""
        texto = texto.replace('-', '')
        if len(texto) == 7:
            lista = list(texto)
            letras_p_numeros = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
            numeros_p_letras = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}

            # 3 primeiros caracteres (Sempre Letras: posições 0, 1, 2)
            for i in range(3):
                if lista[i] in letras_p_numeros:
                    lista[i] = letras_p_numeros[lista[i]]

            # 4º caractere (Sempre Número: posição 3)
            if lista[3] in numeros_p_letras:
                lista[3] = numeros_p_letras[lista[3]]

            # 5º caractere (posição 4): Letra no Mercosul, Número no Antigo.
            # Mantém o caractere lido para não converter placa antiga em Mercosul ou vice-versa.

            # 6º e 7º caracteres (Sempre Números: posições 5 e 6)
            for i in range(5, 7):
                if lista[i] in numeros_p_letras:
                    lista[i] = numeros_p_letras[lista[i]]

            return "".join(lista)
        return texto

    def processar_ocr_background(self, frame):
        resultado_fila = {
            "confirmada": False,
            "placa": "",
            "coords": (0, 0, 0, 0),
            "retangulos": []
        }
        try:
            # Baixamos a confiança para 0.10 pois o modelo americano pode ter dificuldade com placas BR
            resultados_yolo = self.modelo_yolo(frame, conf=0.10, verbose=False)

            melhor_crop = None
            melhor_box = None
            maior_confianca = -1

            for r in resultados_yolo:
                print(f"Detecções YOLO: {len(r.boxes)}")
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    print(f"Classe: {cls} | Confiança: {conf:.3f}")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1

                    if w > 30 and h > 15 and conf > maior_confianca:
                        maior_confianca = conf

                        # --- CORREÇÃO PARA PLACAS AMERICANAS ---
                        pad_w = int(w * 0.05)
                        pad_h = int(h * 0.03)

                        x1_pad = max(0, x1 - pad_w)
                        y1_pad = max(0, y1 - pad_h)
                        x2_pad = min(frame.shape[1], x2 + pad_w)
                        y2_pad = min(frame.shape[0], y2 + pad_h)

                        melhor_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                        melhor_box = (x1_pad, y1_pad, x2_pad - x1_pad, y2_pad - y1_pad)

            if melhor_box is not None:
                resultado_fila["retangulos"] = [melhor_box]

            if melhor_crop is not None and melhor_crop.size > 0:
                # Usar allowlist para focar apenas em caracteres válidos de placa
                resultados_ocr = self.ocr.readtext(
                    melhor_crop,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=1,
                    paragraph=False
                )

                if not resultados_ocr:
                    return

                placa_candidata = ""

                for _, txt, confianca in resultados_ocr:
                    texto_limpo = "".join(e for e in txt if e.isalnum()).upper()
                    if not texto_limpo:
                        continue

                    if len(texto_limpo) != 7:
                        print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} -> [Ignorado: cabeçalho/ruído ({len(texto_limpo)} chars)]")
                        continue

                    # Ignorar leituras com baixa confiança do EasyOCR (< 0.50)
                    if confianca < 0.50:
                        print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} -> [Ignorado: baixa confiança]")
                        continue

                    if self.padrao_antigo.match(texto_limpo) or self.padrao_mercosul.match(texto_limpo):
                        placa_candidata = texto_limpo
                        print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} -> [Formato de Placa OK]")
                        break
                    else:
                        texto_corrigido = self.corrigir_caracteres_confusos(texto_limpo)
                        if self.padrao_antigo.match(texto_corrigido) or self.padrao_mercosul.match(texto_corrigido):
                            placa_candidata = texto_corrigido
                            print(f"OCR: {texto_limpo} -> Corrigido: {texto_corrigido} | Confiança: {confianca:.3f} -> [Formato de Placa OK]")
                            break
                        else:
                            print(f"OCR: {texto_limpo} | Confiança: {confianca:.3f} -> [Ignorado: fora do padrão de placa]")

                if placa_candidata:
                    placa_candidata = self.normalizar_placa(placa_candidata)
                    if placa_candidata == self.leitura_temporaria:
                        self.contador_leituras += 1
                    else:
                        self.leitura_temporaria = placa_candidata
                        self.contador_leituras = 1

                    print(f"   [Validação] Placa: {placa_candidata} ({self.contador_leituras}/{self.limiar_confirmacao})")

                    if self.contador_leituras >= self.limiar_confirmacao:
                        print(f"   >>> PLACA CONFIRMADA E REGISTRADA: {placa_candidata} <<<")
                        resultado_fila["confirmada"] = True
                        resultado_fila["placa"] = placa_candidata
                        resultado_fila["coords"] = melhor_box
                        resultado_fila["retangulos"] = []
                        self.contador_leituras = 0
                        self.leitura_temporaria = ""
                        return

        except Exception as e:
            print(f"Erro no processamento YOLO/EasyOCR: {e}")
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

    def verificar_morador(self, placa):
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT nome, unidade FROM moradores WHERE placa = ?", (placa,))
            return cursor.fetchone()

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
            "Vinculo": ("Vínculo / Identificação", 320, tk.W),
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
        if aba == 1:
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

        tk.Label(
            frame,
            text="NOVO CADASTRO DE MORADOR",
            font=FONTES["subtitulo_app"],
            bg=CORES["bg_panel"],
            fg=CORES["accent"]
        ).pack(anchor=tk.W, pady=(0, 10))

        form_row = tk.Frame(frame, bg=CORES["bg_panel"])
        form_row.pack(fill=tk.X)

        tk.Label(form_row, text="Nome:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 6))
        self.ent_nome = self._criar_entry_estilizado(form_row, 22)
        self.ent_nome.pack(side=tk.LEFT, padx=(0, 14), ipady=5)

        tk.Label(form_row, text="Unidade:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 6))
        self.ent_unidade = self._criar_entry_estilizado(form_row, 10)
        self.ent_unidade.pack(side=tk.LEFT, padx=(0, 14), ipady=5)

        tk.Label(form_row, text="Placa:", font=FONTES["label_bold"], bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(0, 6))
        self.ent_placa = self._criar_entry_estilizado(form_row, 14)
        self.ent_placa.pack(side=tk.LEFT, padx=(0, 18), ipady=5)

        ttk.Button(
            form_row,
            text="Salvar Morador",
            style="Accent.TButton",
            command=self.salvar_morador
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            form_row,
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

        colunas = ("ID", "Nome", "Unidade", "Placa")
        self.tree_moradores = ttk.Treeview(frame_tb, columns=colunas, show="headings")

        config_colunas = {
            "ID": ("ID", 60, tk.CENTER),
            "Nome": ("Nome do Morador", 340, tk.W),
            "Unidade": ("Unidade / Apto", 160, tk.CENTER),
            "Placa": ("Placa Veicular", 160, tk.CENTER),
        }
        for col, (titulo, largura, alinhamento) in config_colunas.items():
            self.tree_moradores.heading(col, text=titulo)
            self.tree_moradores.column(col, width=largura, anchor=alinhamento)

        self.tree_moradores.tag_configure("par", background=CORES["row_even"])
        self.tree_moradores.tag_configure("impar", background=CORES["row_odd"])

        scrollbar = ttk.Scrollbar(frame_tb, orient=tk.VERTICAL, command=self.tree_moradores.yview)
        self.tree_moradores.configure(yscrollcommand=scrollbar.set)

        self.tree_moradores.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def salvar_morador(self):
        nome = self.ent_nome.get().strip()
        unidade = self.ent_unidade.get().strip()
        placa_digitada = self.ent_placa.get().strip().upper()

        if not nome or not unidade or not placa_digitada:
            messagebox.showwarning("Aviso", "Preencha todos os campos do cadastro!")
            return

        placa_final = self.normalizar_placa(placa_digitada)
        if not (self.padrao_antigo.match(placa_final) or self.padrao_mercosul.match(placa_final)):
            messagebox.showerror(
                "Erro",
                "Formato inválido! Digite os 7 caracteres da placa no padrão Antigo (ABC1234) ou Mercosul (ABC1D23)."
            )
            return

        try:
            with self.get_db() as conn:
                conn.execute("INSERT INTO moradores (nome, unidade, placa) VALUES (?, ?, ?)", (nome, unidade, placa_final))

            self.ent_nome.delete(0, tk.END)
            self.ent_unidade.delete(0, tk.END)
            self.ent_placa.delete(0, tk.END)

            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", f"Morador cadastrado! Placa salva como: {placa_final}")

        except sqlite3.IntegrityError:
            messagebox.showerror("Erro", "Esta placa já pertence a outro morador.")

    def excluir_morador(self):
        selecionado = self.tree_moradores.selection()
        if not selecionado:
            messagebox.showwarning("Aviso", "Selecione um morador na tabela para excluir!")
            return

        valores = self.tree_moradores.item(selecionado[0], "values")
        id_morador, nome, unidade, placa = valores[0], valores[1], valores[2], valores[3]

        confirmar = messagebox.askyesno(
            "Confirmar Exclusão",
            f"Deseja realmente remover o morador '{nome}' (Unidade {unidade} - Placa {placa})?"
        )
        if confirmar:
            with self.get_db() as conn:
                conn.execute("DELETE FROM moradores WHERE id = ?", (id_morador,))
            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", "Morador excluído com sucesso!")

    def atualizar_tabela_moradores(self):
        for item in self.tree_moradores.get_children():
            self.tree_moradores.delete(item)
        with self.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM moradores ORDER BY nome ASC")
            for idx, row in enumerate(cursor.fetchall()):
                tag = "par" if idx % 2 == 0 else "impar"
                self.tree_moradores.insert("", tk.END, values=row, tags=(tag,))

    def on_closing(self):
        if hasattr(self, 'evento_parada'):
            self.evento_parada.set()
        if hasattr(self, 'thread_camera') and self.thread_camera.is_alive():
            self.thread_camera.join(timeout=1.5)
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaPortariaHibrido(root, "V-Control - Monitoramento e Controle de Acesso LPR")
    root.mainloop()
