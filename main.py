import cv2
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import pytesseract
import imutils
import numpy as np
import re
import threading  # Biblioteca nativa para Multi-Threading

# ATENÇÃO: Ajuste o caminho abaixo se estiver usando Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SistemaPortariaMultiThread:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("950x800")
        
        self.init_db()
        self.cap = cv2.VideoCapture(0)
        
        # --- VARIÁVEIS DE CONTROLE DE MULTI-THREAD E LPR ---
        self.placa_atual = ""
        self.coords_placa = (0, 0, 0, 0)
        self.ultima_placa_gravada = ""
        
        # Flag de controle: impede o disparo de novas threads se uma já estiver calculando o OCR
        self.ocr_em_andamento = False 
        
        # Como o OCR em background roda menos vezes por segundo, reduzimos o limiar de confirmação
        self.leituras_consecutivas = 0
        self.limiar_confirmacao = 3 
        
        self.padrao_placa = re.compile(r'^[A-Z]{3}[0-9][A-Z0-9][0-9]{2}$')

        # --- ESTRUTURA DE ABAS ---
        self.notebook = ttk.Notebook(window)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_monitoramento = tk.Frame(self.notebook)
        self.tab_historico = tk.Frame(self.notebook)
        self.tab_moradores = tk.Frame(self.notebook)

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

    def init_db(self):
        self.conn = sqlite3.connect('registro_portaria.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS registros 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, placa TEXT NOT NULL, tipo TEXT NOT NULL, data TEXT NOT NULL, hora TEXT NOT NULL, vinculo TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS moradores 
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, nome TEXT NOT NULL, unidade TEXT NOT NULL, placa TEXT NOT NULL UNIQUE)''')
        self.conn.commit()

    def configurar_aba_monitoramento(self):
        self.canvas = tk.Canvas(self.tab_monitoramento, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)

        self.frame_status = tk.LabelFrame(self.tab_monitoramento, text=" Status da Leitura (Multi-Thread Ativo) ", font=("Arial", 11, "bold"), padx=15, pady=10)
        self.frame_status.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_placa = tk.Label(self.frame_status, text="Buscando placas em background...", font=("Arial", 14, "bold"), fg="gray")
        self.lbl_placa.grid(row=0, column=0, padx=20, sticky=tk.W)

        self.lbl_vinculo = tk.Label(self.frame_status, text="Vínculo: --", font=("Arial", 12), fg="gray")
        self.lbl_vinculo.grid(row=0, column=1, padx=20, sticky=tk.W)

    def update_video(self):
        """Thread Principal: Foca exclusivamente em capturar o vídeo e renderizar a tela"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))

            # SE não houver nenhuma thread de OCR rodando agora, cria uma nova em background
            if not self.ocr_em_andamento:
                self.ocr_em_andamento = True
                
                # IMPORTANTE: Clonamos o frame (.copy()) para que a thread secundária use uma foto estática
                # enquanto a thread principal continua atualizando o frame da câmera ao vivo
                frame_para_ocr = frame.copy()
                
                # Cria e inicia a thread paralela
                threading.Thread(target=self.processar_ocr_background, args=(frame_para_ocr,), daemon=True).start()

            # Desenha os resultados na tela caso uma placa tenha sido detectada pela outra thread
            if self.placa_atual:
                x, y, w, h = self.coords_placa
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, self.placa_atual, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                self.lbl_placa.config(text=f"Placa Identificada: {self.placa_atual}", fg="blue")
                morador = self.verificar_morador(self.placa_atual)
                vinculo = f"Morador: {morador[0]} ({morador[1]})" if morador else "Visitante"
                self.lbl_vinculo.config(text=vinculo, fg="green" if morador else "orange")

                # Lógica de estabilização automatizada baseada em confirmações seguidas
                if self.placa_atual != self.ultima_placa_gravada:
                    self.leituras_consecutivas += 1
                    if self.leituras_consecutivas >= self.limiar_confirmacao:
                        self.registrar_movimento(self.placa_atual, vinculo)
                        self.ultima_placa_gravada = self.placa_atual
                        self.leituras_consecutivas = 0
                else:
                    self.leituras_consecutivas = 0
            else:
                self.leituras_consecutivas = 0
                self.lbl_placa.config(text="Buscando placas em background...", fg="gray")
                self.lbl_vinculo.config(text="Vínculo: --", fg="gray")

            # Converte e exibe o feed de vídeo sem lags
            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img)
            self.photo = ImageTk.PhotoImage(image=pil_img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update_video)

    def processar_ocr_background(self, frame):
        """Thread Secundária: Executa todo o processamento de imagem pesado em segundo plano"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)

            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

            local_placa = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    local_placa = approx
                    break

            if local_placa is not None:
                x, y, w, h = cv2.boundingRect(local_placa)
                cropped_image = gray[y:y+h, x:x+w]
                
                config_tesseract = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                texto = pytesseract.image_to_string(cropped_image, config=config_tesseract)
                texto_limpo = "".join(e for e in texto if e.isalnum()).upper()
                
                if self.padrao_placa.match(texto_limpo):
                    # Se achou uma placa válida, repassa os dados para as variáveis compartilhadas
                    self.placa_atual = texto_limpo
                    self.coords_placa = (x, y, w, h)
                    self.ocr_em_andamento = False
                    return

            # Se não detectou nada neste frame, limpa a placa atual
            self.placa_atual = ""
            self.coords_placa = (0, 0, 0, 0)
            
        except Exception as e:
            print(f"Erro no processamento em background: {e}")
            
        finally:
            # Libera a trava de segurança para permitir que o próximo ciclo crie outra thread
            self.ocr_em_andamento = False

    def registrar_movimento(self, placa, vinculo):
        agora = datetime.now()
        tipo = self.inferir_tipo_movimento(placa)
        
        # Como a gravação manipula o SQLite, realizamos na Thread Principal com segurança
        self.cursor.execute("INSERT INTO registros (placa, tipo, data, hora, vinculo) VALUES (?, ?, ?, ?, ?)",
                            (placa, tipo, agora.strftime("%d/%m/%Y"), agora.strftime("%H:%M:%S"), vinculo))
        self.conn.commit()
        print(f"[REGISTRO AUTOMÁTICO] {tipo} - Placa: {placa} ({vinculo})")

    def inferir_tipo_movimento(self, placa):
        self.cursor.execute("SELECT tipo FROM registros WHERE placa = ? ORDER BY id DESC LIMIT 1", (placa,))
        ultimo = self.cursor.fetchone()
        return "SAÍDA" if ultimo and ultimo[0] == "ENTRADA" else "ENTRADA"

    def verificar_morador(self, placa):
        self.cursor.execute("SELECT nome, unidade FROM moradores WHERE placa = ?", (placa,))
        return self.cursor.fetchone()

    # --- GERENCIAMENTO DE INTERFACE DAS OUTRAS ABAS ---
    def configurar_aba_historico(self):
        frame = tk.Frame(self.tab_historico)
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
        frame = tk.LabelFrame(self.tab_moradores, text=" Novo Cadastro de Morador ")
        frame.pack(fill=tk.X, padx=15, pady=10)
        self.ent_nome = tk.Entry(frame); self.ent_nome.pack(side=tk.LEFT, padx=5, pady=5)
        self.ent_unidade = tk.Entry(frame); self.ent_unidade.pack(side=tk.LEFT, padx=5, pady=5)
        self.ent_placa = tk.Entry(frame); self.ent_placa.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame, text="Salvar Morador", bg="#3498db", fg="white", command=self.salvar_morador).pack(side=tk.LEFT, padx=10)

        frame_tb = tk.Frame(self.tab_moradores)
        frame_tb.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.tree_moradores = ttk.Treeview(frame_tb, columns=("ID", "Nome", "Unidade", "Placa"), show="headings")
        for col in self.tree_moradores["columns"]: self.tree_moradores.heading(col, text=col)
        self.tree_moradores.pack(fill=tk.BOTH, expand=True)

    def salvar_morador(self):
        nome = self.ent_nome.get().strip()
        unidade = self.ent_unidade.get().strip()
        placa = self.ent_placa.get().strip().upper()
        if nome and unidade and placa:
            try:
                self.cursor.execute("INSERT INTO moradores (nome, unidade, placa) VALUES (?, ?, ?)", (nome, unidade, placa))
                self.conn.commit()
                self.ent_nome.delete(0, tk.END); self.ent_unidade.delete(0, tk.END); self.ent_placa.delete(0, tk.END)
                self.atualizar_tabela_moradores()
                messagebox.showinfo("Sucesso", "Morador cadastrado!")
            except: messagebox.showerror("Erro", "Placa já cadastrada ou dados inválidos.")

    def atualizar_tabela_moradores(self):
        for item in self.tree_moradores.get_children(): self.tree_moradores.delete(item)
        self.cursor.execute("SELECT * FROM moradores ORDER BY nome ASC")
        for row in self.cursor.fetchall(): self.tree_moradores.insert("", tk.END, values=row)

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.conn.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaPortariaMultiThread(root, "LPR Condomínio - Alta Fluidez (Multi-Thread)")
    root.mainloop()