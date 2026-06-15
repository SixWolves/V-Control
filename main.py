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
import threading
import time

# ATENÇÃO: Ajuste o caminho abaixo se estiver usando Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SistemaPortariaHibrido:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("950x800")
        
        self.init_db()
        self.cap = cv2.VideoCapture(0)
        
        # --- VARIÁVEIS DE CONTROLE ---
        self.placa_exibida = ""       
        self.coords_exibidas = (0, 0, 0, 0) 
        self.timestamp_expira = 0.0   
        self.ultima_placa_gravada = "" 
        self.ocr_em_andamento = False 
        
        # --- EXPRESSÕES REGULARES (PADRÕES BRASILEIROS) ---
        self.padrao_antigo = re.compile(r'^[A-Z]{3}-[0-9]{4}$')             # Espera a leitura exata ABC-1234
        self.padrao_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')    # ABC1D23

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

        self.frame_status = tk.LabelFrame(self.tab_monitoramento, text=" Status do Sistema ", font=("Arial", 11, "bold"), padx=15, pady=10)
        self.frame_status.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_placa = tk.Label(self.frame_status, text="Buscando veículos (Antigo/Mercosul)...", font=("Arial", 14, "bold"), fg="gray")
        self.lbl_placa.grid(row=0, column=0, padx=20, sticky=tk.W)

        self.lbl_vinculo = tk.Label(self.frame_status, text="Vínculo: --", font=("Arial", 12), fg="gray")
        self.lbl_vinculo.grid(row=0, column=1, padx=20, sticky=tk.W)

    def update_video(self):
        """Thread Principal: Renderização fluida da interface gráfica"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            tempo_atual = time.time()

            # Se houver uma placa retida nos 5 segundos de exibição
            if self.placa_exibida and tempo_atual < self.timestamp_expira:
                x, y, w, h = self.coords_exibidas
                
                # Renderiza o enquadramento estável sobre o veículo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, self.placa_exibida, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                self.lbl_placa.config(text=f"Placa Identificada: {self.placa_exibida}", fg="blue")
                
                # Grava no banco apenas uma vez por ciclo de detecção
                if self.placa_exibida != self.ultima_placa_gravada:
                    morador = self.verificar_morador(self.placa_exibida)
                    vinculo = f"Morador: {morador[0]} ({morador[1]})" if morador else "Visitante"
                    self.lbl_vinculo.config(text=vinculo, fg="green" if morador else "orange")
                    
                    self.registrar_movimento(self.placa_exibida, vinculo)
                    self.ultima_placa_gravada = self.placa_exibida

            # Se o tempo expirou ou o visor está livre, busca novas placas
            else:
                if self.placa_exibida:
                    self.placa_exibida = ""
                    self.coords_exibidas = (0, 0, 0, 0)
                    self.ultima_placa_gravada = "" 

                self.lbl_placa.config(text="Buscando veículos (Antigo/Mercosul)...", fg="gray")
                self.lbl_vinculo.config(text="Vínculo: --", fg="gray")

                if not self.ocr_em_andamento:
                    self.ocr_em_andamento = True
                    frame_para_ocr = frame.copy()
                    threading.Thread(target=self.processar_ocr_background, args=(frame_para_ocr,), daemon=True).start()

            cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img)
            self.photo = ImageTk.PhotoImage(image=pil_img)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update_video)

    def processar_ocr_background(self, frame):
        """Thread Secundária: Filtros de imagem e processamento OCR sem travar a tela"""
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
                
                # Adicionado o hífen (-) no final da whitelist
                config_tesseract = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                texto = pytesseract.image_to_string(cropped_image, config=config_tesseract)
                
                # Modificado para manter letras, números e o hífen
                texto_limpo = "".join(e for e in texto if e.isalnum() or e == '-').upper()
                
                # Teste 1: Validação do Padrão Clássico (ABC-1234)
                if self.padrao_antigo.match(texto_limpo):
                    self.placa_exibida = texto_limpo # Já recebe do OCR com o hífen nativo
                    self.coords_exibidas = (x, y, w, h)
                    self.timestamp_expira = time.time() + 5.0
                    return
                # Teste 2: Validação do Padrão Mercosul (ABC1D23)
                elif self.padrao_mercosul.match(texto_limpo):
                    self.placa_exibida = texto_limpo # Mantém contínuo conforme o padrão original
                    self.coords_exibidas = (x, y, w, h)
                    self.timestamp_expira = time.time() + 5.0
                    return

        except Exception as e:
            print(f"Erro no processamento OCR: {e}")
        finally:
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

    # --- ABA: HISTÓRICO ---
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

    # --- ABA: CADASTRO DE MORADORES ---
    def configurar_aba_moradores(self):
        frame = tk.LabelFrame(self.tab_moradores, text=" Cadastro de Moradores ")
        frame.pack(fill=tk.X, padx=15, pady=10)
        self.ent_nome = tk.Entry(frame, width=20); self.ent_nome.pack(side=tk.LEFT, padx=5, pady=5)
        self.ent_unidade = tk.Entry(frame, width=10); self.ent_unidade.pack(side=tk.LEFT, padx=5, pady=5)
        self.ent_placa = tk.Entry(frame, width=15); self.ent_placa.pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(frame, text="Salvar", bg="#3498db", fg="white", command=self.salvar_morador).pack(side=tk.LEFT, padx=10)

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
                messagebox.showinfo("Sucesso", "Morador cadastrado com sucesso!")
            except: messagebox.showerror("Erro", "Verifique se os dados ou a placa já existem.")

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
    app = SistemaPortariaHibrido(root, "Monitoramento LPR - Dual Padrão (5s Retenção)")
    root.mainloop()