import os
import sys
import re
import time
import queue
import shutil
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk, ImageOps, ImageDraw
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
        self.window.geometry("1180x830")
        self.window.minsize(1100, 760)
        self.window.configure(bg=CORES["bg_main"])

        # Pasta local para armazenar as fotos dos condutores autorizados
        self.pasta_fotos = Path(__file__).resolve().parent / "fotos_condutores"
        self.pasta_fotos.mkdir(exist_ok=True)
        self.foto_selecionada_temp = ""
        self.id_em_edicao = None  # Guarda o ID do registro quando estiver em modo de edição
        self.fotos_condutores_refs = []  # Mantém referências de ImageTk para o garbage collector não apagar

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
            text="  |  Monitoramento LPR e Verificação Visual de Condutores (YOLOv8 + EasyOCR)",
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
        self.notebook.add(self.tab_moradores, text="  Moradores e Condutores  ")

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

    def configurar_aba_monitoramento(self):
        # Área central dividida lado a lado: Câmera (Esquerda) + Condutores Autorizados (Direita)
        area_superior = tk.Frame(self.tab_monitoramento, bg=CORES["bg_main"])
        area_superior.pack(fill=tk.BOTH, expand=True, padx=25, pady=(15, 10))

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

        # Inicializa o painel de condutores no estado "aguardando"
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
                text="Aguardando leitura de placa...",
                font=FONTES["secao"],
                bg=CORES["bg_main"],
                fg=CORES["fg_muted"]
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
                    self.timestamp_expira = tempo_atual + 8.0  # 8s para o segurança conferir a foto do condutor com calma
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
