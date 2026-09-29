import cv2
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import pytesseract
import numpy as np
import re
import threading
import time
from pathlib import Path

import os

# Forçar o OpenCV a usar TCP para a câmera RTSP (evita perda de pacotes e borrões cinzas na imagem)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# IMPORTAÇÃO DA ESTILIZAÇÃO
from styles import CORES, aplicar_tema

# ATENÇÃO: Ajuste o caminho abaixo se estiver usando Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SistemaPortariaHibrido:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("950x800")
        self.window.configure(bg=CORES["bg_main"])

        caminho_cascade = Path(__file__).resolve().parent / "haarcascades" / "haarcascade_russian_plate_number.xml"
        self.cascade_placa = cv2.CascadeClassifier(str(caminho_cascade))
        if self.cascade_placa.empty():
            raise FileNotFoundError(f"Não foi possível carregar o classificador: {caminho_cascade}")
        
        self.init_db()
        self.url_cftv = "rtsp://10.233.187.1:554/user=admin_password=tlJwpbo6_channel=0_stream=0&onvif=0.sdp?real_stream"
        
        # --- MELHORIAS PARA CFTV (RTSP) ---
        # Iniciando a câmera em uma Thread separada (não trava o Tkinter se a rede falhar)
        self.frame_atual = None
        self.ret_atual = False
        threading.Thread(target=self.loop_leitura_camera, daemon=True).start()

        
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
        self.is_placa_falsa = False
        
        self.leitura_temporaria = ""
        self.contador_leituras = 0
        self.limiar_confirmacao = 3 # Exige 3 leituras idênticas
        
        # --- EXPRESSÕES REGULARES ---
        self.padrao_antigo = re.compile(r'^[A-Z]{3}-[0-9]{4}$')
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')

        # --- APLICAÇÃO DO TEMA EXTERNO ---
        aplicar_tema()

        # --- ESTRUTURA DE ABAS ---
        self.notebook = ttk.Notebook(window)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_monitoramento = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_historico = tk.Frame(self.notebook, bg=CORES["bg_main"])
        self.tab_moradores = tk.Frame(self.notebook, bg=CORES["bg_main"])

        self.notebook.add(self.tab_monitoramento, text=" Monitoramento ")
        self.notebook.add(self.tab_historico, text=" Histórico ")
        self.notebook.add(self.tab_moradores, text=" Moradores ")

        self.configurar_aba_monitoramento()
        self.configurar_aba_historico()
        self.configurar_aba_moradores()

        self.notebook.bind("<<NotebookTabChanged>>", self.ao_mudar_de_aba)

        self.delay = 15
        self.update_video()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def loop_leitura_camera(self):
        # A Thread exclusiva da câmera reconecta e gerencia o buffer 100% isolada da interface
        self.cap = cv2.VideoCapture(self.url_cftv, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            if not self.cap.isOpened():
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url_cftv, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self.cap.read()
            self.ret_atual = ret
            if ret:
                self.frame_atual = frame
            else:
                self.cap.release()
                time.sleep(1) # Dá um tempinho antes de tentar reabrir

    def init_db(self):
        self.conn = sqlite3.connect('registro_portaria.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS registros 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, placa TEXT NOT NULL, tipo TEXT NOT NULL, data TEXT NOT NULL, hora TEXT NOT NULL, vinculo TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS moradores 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT NOT NULL, unidade TEXT NOT NULL, placa TEXT NOT NULL UNIQUE)''')
        self.conn.commit()

    def configurar_aba_monitoramento(self):
        self.canvas = tk.Canvas(self.tab_monitoramento, width=640, height=480, bg="#000000", highlightthickness=2, highlightbackground=CORES["accent"])
        self.canvas.pack(pady=15)

        self.frame_status = tk.LabelFrame(self.tab_monitoramento, text=" Status do Sistema ", font=("Arial", 11, "bold"), 
                                          bg=CORES["bg_panel"], fg=CORES["accent"], padx=15, pady=10)
        self.frame_status.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_placa = tk.Label(self.frame_status, text="Buscando veículos...", font=("Arial", 14, "bold"), bg=CORES["bg_panel"], fg=CORES["fg_muted"])
        self.lbl_placa.grid(row=0, column=0, padx=20, sticky=tk.W)

        self.lbl_vinculo = tk.Label(self.frame_status, text="Vínculo: --", font=("Arial", 12), bg=CORES["bg_panel"], fg=CORES["fg_muted"])
        self.lbl_vinculo.grid(row=0, column=1, padx=20, sticky=tk.W)

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
            return # Interrompe a execução deste ciclo
            
        # --- SE RET FOR TRUE, CONTINUA O CÓDIGO NORMALMENTE ---
        frame = frame_thread.copy() # Copia para evitar que a thread subscreva enquanto desenhamos
        
        # (Lembrando de manter aquele corte da imagem dupla que fizemos!)
        altura, largura = frame.shape[:2]
        frame = frame[0:int(altura/2), 0:largura]
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.rotate(frame, cv2.ROTATE_180) # Rotação de 180° adicionada
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

        for (x, y, w, h) in self.retangulos_candidatos:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Scan...", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.placa_exibida and tempo_atual < self.timestamp_expira:
            x, y, w, h = self.coords_exibidas

            # Se for falsa, pinta de vermelho, se for verdadeira, pinta de verde
            cor_box = (0, 0, 255) if self.is_placa_falsa else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), cor_box, 2)
            cv2.putText(frame, self.placa_exibida, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_box, 2)

            if self.is_placa_falsa:
                self.lbl_placa.config(text=f"Placa Lida: {self.placa_exibida} (FALSA!)", fg=CORES["danger"])
            else:
                self.lbl_placa.config(text=f"Placa Identificada: {self.placa_exibida}", fg=CORES["success"])

            if (self.placa_exibida != self.ultima_placa_gravada or
                    tempo_atual - self.timestamp_ultimo_registro > 60):

                if self.is_placa_falsa:
                    vinculo = "ALERTA: PLACA FALSA (TELA/PAPEL)"
                    self.lbl_vinculo.config(text=vinculo, fg=CORES["danger"])
                else:
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

            self.lbl_placa.config(text="Buscando veículos...", fg=CORES["fg_muted"])
            self.lbl_vinculo.config(text="Vínculo: --", fg=CORES["fg_muted"])

            if not self.ocr_em_andamento:
                self.ocr_em_andamento = True
                frame_para_ocr = frame.copy()
                threading.Thread(target=self.processar_ocr_background, args=(frame_para_ocr,), daemon=True).start()

        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        self.photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update_video)
        
    def corrigir_caracteres_confusos(self, texto):
        """Corrige confusões comuns do OCR entre letras e números com base na posição"""
        texto = texto.replace('-', '')
        if len(texto) == 7:
            lista = list(texto)
            letras_p_numeros = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
            numeros_p_letras = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
            
            # 3 primeiros caracteres (Sempre Letras)
            for i in range(3):
                if lista[i] in letras_p_numeros:
                    lista[i] = letras_p_numeros[lista[i]]
                    
            # 4º caractere (Sempre Número)
            if lista[3] in numeros_p_letras:
                lista[3] = numeros_p_letras[lista[3]]
                
            # 5º caractere (Letra no Mercosul, Número no Antigo)
            if lista[4].isalpha() or lista[4] in letras_p_numeros:
                if lista[4] in letras_p_numeros:
                    lista[4] = letras_p_numeros[lista[4]]
            else:
                if lista[4] in numeros_p_letras:
                    lista[4] = numeros_p_letras[lista[4]]
                    
            # 6º e 7º caracteres (Sempre Números)
            for i in range(5, 7):
                if lista[i] in numeros_p_letras:
                    lista[i] = numeros_p_letras[lista[i]]
                    
            return "".join(lista)
        return texto

    def processar_ocr_background(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Tenta achar a placa usando a Inteligência Artificial (Haar Cascade - detecta textura/metal)
            placas_ia = self.cascade_placa.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25)
            )

            # 2. Usa a Geometria (Canny/Contornos) para achar qualquer retângulo (até papel e celular)
            gray_blur = cv2.bilateralFilter(gray_frame, 11, 17, 17)
            edged = cv2.Canny(gray_blur, 30, 200)

            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            import imutils
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            local_placa = None
            self.retangulos_candidatos = [] # Limpa retângulos antigos

            for contour in contours:
                # Usa uma porcentagem do perímetro da figura geométrica, muito mais preciso que um valor fixo (10)
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # Filtro de proporção: Placas de carro são sempre retangulares (mais largas do que altas).
                    # A letra 'Q' é quadrada (aspect_ratio ~ 1.0). Esse filtro a descarta imediatamente!
                    aspect_ratio = w / float(h)
                    
                    if 1.5 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                        local_placa = approx
                        self.retangulos_candidatos.append((x, y, w, h)) # Apenas para desenhar na tela (feedback visual)
                        break # Pega apenas o maior retângulo que tem formato de placa

            if local_placa is not None:
                x, y, w, h = cv2.boundingRect(local_placa)
                cropped_image = gray[y:y+h, x:x+w]

                if cropped_image.size > 0:
                    config_tesseract = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                    texto = pytesseract.image_to_string(cropped_image, config=config_tesseract)
                    texto_limpo = "".join(e for e in texto if e.isalnum() or e == '-').upper()
                    
                    # Vamos manter a função de corrigir caracteres, pois ela ainda é útil,
                    # mas se a placa já bater com o regex limpo, não precisamos forçar correção.
                    placa_candidata = ""
                    
                    if self.padrao_antigo.match(texto_limpo):
                        placa_candidata = texto_limpo
                    elif self.padrao_mercosul.match(texto_limpo):
                        placa_candidata = texto_limpo
                    else:
                        texto_corrigido = self.corrigir_caracteres_confusos(texto_limpo)
                        if len(texto_corrigido) == 7 and texto_corrigido[:3].isalpha() and texto_corrigido[3:].isdigit():
                            placa_candidata = f"{texto_corrigido[:3]}-{texto_corrigido[3:]}"
                        elif self.padrao_mercosul.match(texto_corrigido):
                            placa_candidata = texto_corrigido

                    if placa_candidata:
                        if placa_candidata == self.leitura_temporaria:
                            self.contador_leituras += 1
                        else:
                            self.leitura_temporaria = placa_candidata
                            self.contador_leituras = 1
                            
                        if self.contador_leituras >= self.limiar_confirmacao:
                            self.placa_exibida = placa_candidata 
                            self.coords_exibidas = (x, y, w, h)
                            self.timestamp_expira = time.time() + 5.0
                            
                            # -- ANTI-SPOOFING --
                            # Checa se o retângulo (x,y,w,h) achado pela geometria (que funcionou no OCR)
                            # bate com alguma placa achada pela IA (que procura textura de placa metálica)
                            cx, cy = x + w/2, y + h/2
                            ia_confirmou = False
                            for (ix, iy, iw, ih) in placas_ia:
                                if ix <= cx <= ix + iw and iy <= cy <= iy + ih:
                                    ia_confirmou = True
                                    break
                                    
                            self.is_placa_falsa = not ia_confirmou
                            
                            self.contador_leituras = 0
                            self.leitura_temporaria = ""

                        return 

            self.contador_leituras = 0

        except Exception as e:
            print(f"Erro no processamento OCR: {e}")
        finally:
            self.retangulos_candidatos = []
            self.ocr_em_andamento = False

    def registrar_movimento(self, placa, vinculo):
        agora = datetime.now()
        tipo = self.inferir_tipo_movimento(placa)
        self.cursor.execute("INSERT INTO registros (placa, tipo, data, hora, vinculo) VALUES (?, ?, ?, ?, ?)",
                            (placa, tipo, agora.strftime("%d/%m/%Y"), agora.strftime("%H:%M:%S"), vinculo))
        self.conn.commit()

    def inferir_tipo_movimento(self, placa):
        self.cursor.execute("SELECT tipo FROM registros WHERE placa = ? ORDER BY id DESC LIMIT 1", (placa,))
        ultimo = self.cursor.fetchone()
        return "SAÍDA" if ultimo and ultimo[0] == "ENTRADA" else "ENTRADA"

    def verificar_morador(self, placa):
        self.cursor.execute("SELECT nome, unidade FROM moradores WHERE placa = ?", (placa,))
        return self.cursor.fetchone()

    def configurar_aba_historico(self):
        frame = tk.Frame(self.tab_historico, bg=CORES["bg_main"])
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.tree_historico = ttk.Treeview(frame, columns=("ID", "Placa", "Tipo", "Data", "Horario", "Vinculo"), show="headings")
        for col in self.tree_historico["columns"]:
            self.tree_historico.heading(col, text=col)
        self.tree_historico.pack(fill=tk.BOTH, expand=True)

    def atualizar_tabela_historico(self):
        for item in self.tree_historico.get_children(): self.tree_historico.delete(item)
        self.cursor.execute("SELECT * FROM registros ORDER BY id DESC")
        for row in self.cursor.fetchall(): self.tree_historico.insert("", tk.END, values=row)

    def ao_mudar_de_aba(self, event):
        aba = self.notebook.index(self.notebook.select())
        if aba == 1: self.atualizar_tabela_historico()
        elif aba == 2: self.atualizar_tabela_moradores()

    def configurar_aba_moradores(self):
        frame = tk.LabelFrame(self.tab_moradores, text=" Cadastro de Moradores ", font=("Arial", 11, "bold"), 
                              bg=CORES["bg_panel"], fg=CORES["accent"], padx=15, pady=10)
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(frame, text="Nome:", bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(5,0))
        self.ent_nome = tk.Entry(frame, width=20, bg=CORES["input_bg"], fg=CORES["fg_text"], insertbackground=CORES["fg_text"])
        self.ent_nome.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(frame, text="Unidade:", bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(10,0))
        self.ent_unidade = tk.Entry(frame, width=10, bg=CORES["input_bg"], fg=CORES["fg_text"], insertbackground=CORES["fg_text"])
        self.ent_unidade.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Label(frame, text="Placa:", bg=CORES["bg_panel"], fg=CORES["fg_text"]).pack(side=tk.LEFT, padx=(10,0))
        self.ent_placa = tk.Entry(frame, width=15, bg=CORES["input_bg"], fg=CORES["fg_text"], insertbackground=CORES["fg_text"])
        self.ent_placa.pack(side=tk.LEFT, padx=5, pady=5)
        
        tk.Button(frame, text="Salvar Morador", bg=CORES["accent"], fg="white", font=("Arial", 10, "bold"), 
                  relief=tk.FLAT, command=self.salvar_morador).pack(side=tk.LEFT, padx=15)

        frame_tb = tk.Frame(self.tab_moradores, bg=CORES["bg_main"])
        frame_tb.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.tree_moradores = ttk.Treeview(frame_tb, columns=("ID", "Nome", "Unidade", "Placa"), show="headings")
        for col in self.tree_moradores["columns"]: self.tree_moradores.heading(col, text=col)
        self.tree_moradores.pack(fill=tk.BOTH, expand=True)

    def salvar_morador(self):
        nome = self.ent_nome.get().strip()
        unidade = self.ent_unidade.get().strip()
        placa_digitada = self.ent_placa.get().strip().upper()
        
        if not nome or not unidade or not placa_digitada:
            messagebox.showwarning("Aviso", "Preencha todos os campos do cadastro!")
            return

        placa_limpa = "".join(e for e in placa_digitada if e.isalnum())
        
        if re.match(r'^[A-Z]{3}[0-9]{4}$', placa_limpa):
            placa_final = f"{placa_limpa[:3]}-{placa_limpa[3:]}" 
        elif re.match(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', placa_limpa):
            placa_final = placa_limpa 
        else:
            messagebox.showerror("Erro", "Formato inválido! Digite uma placa real (ex: ABC1234 ou ABC1D23).")
            return

        try:
            self.cursor.execute("INSERT INTO moradores (nome, unidade, placa) VALUES (?, ?, ?)", (nome, unidade, placa_final))
            self.conn.commit()
            
            self.ent_nome.delete(0, tk.END)
            self.ent_unidade.delete(0, tk.END)
            self.ent_placa.delete(0, tk.END)
            
            self.atualizar_tabela_moradores()
            messagebox.showinfo("Sucesso", f"Morador cadastrado! Placa salva como: {placa_final}")
            
        except sqlite3.IntegrityError:
            messagebox.showerror("Erro", "Esta placa já pertence a outro morador.")

    def atualizar_tabela_moradores(self):
        for item in self.tree_moradores.get_children(): self.tree_moradores.delete(item)
        self.cursor.execute("SELECT * FROM moradores ORDER BY nome ASC")
        for row in self.cursor.fetchall(): self.tree_moradores.insert("", tk.END, values=row)

    def on_closing(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened(): 
            self.cap.release()
        self.conn.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaPortariaHibrido(root, "Monitoramento LPR - CSS Separado")
    root.mainloop()