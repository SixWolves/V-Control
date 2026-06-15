from tkinter import ttk

# O nosso "Root" de variáveis CSS
CORES = {
    "bg_main": "#121212",      # Fundo principal escuro
    "bg_panel": "#1e1e1e",      # Fundo dos frames
    "fg_text": "#ffffff",       # Texto branco
    "fg_muted": "#b3b3b3",      # Texto cinza (secundário)
    "accent": "#3498db",        # Azul destaque
    "success": "#2ecc71",       # Verde (Entrada/Sucesso)
    "warning": "#f39c12",       # Laranja (Visitante)
    "danger": "#e74c3c",        # Vermelho (Avisos/Botões)
    "input_bg": "#2b2b2b",      # Fundo das caixas de texto
    "tree_header": "#2c3e50"    # Fundo do cabeçalho da tabela
}

def aplicar_tema():
    """Injeta a estilização nos componentes ttk (Abas e Tabelas)"""
    style = ttk.Style()
    style.theme_use('clam')
    
    # --- Estilo das Abas (Notebook) ---
    style.configure("TNotebook", background=CORES["bg_main"], borderwidth=0)
    style.configure("TNotebook.Tab", background=CORES["bg_panel"], foreground=CORES["fg_text"], padding=[15, 5], font=("Arial", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", CORES["accent"])])
    
    # --- Estilo das Tabelas (Treeview) ---
    style.configure("Treeview", background=CORES["bg_panel"], foreground=CORES["fg_text"], fieldbackground=CORES["bg_panel"], borderwidth=0)
    style.configure("Treeview.Heading", background=CORES["tree_header"], foreground=CORES["fg_text"], font=("Arial", 10, "bold"))
    style.map("Treeview", background=[('selected', CORES["accent"])])