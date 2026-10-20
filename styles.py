from tkinter import ttk

# Paleta de cores moderna (Dark Slate / Dashboard de Segurança)
CORES = {
    "bg_main": "#0f172a",        # Fundo principal escuro profundo (Slate 900)
    "bg_panel": "#1e293b",       # Fundo dos painéis e cards (Slate 800)
    "bg_header": "#0b1120",      # Fundo do cabeçalho superior
    "border": "#334155",         # Bordas sutis (Slate 700)
    "fg_text": "#f8fafc",        # Texto principal branco suave (Slate 50)
    "fg_muted": "#94a3b8",       # Texto secundário cinza-azulado (Slate 400)
    "accent": "#3b82f6",         # Azul destaque moderno (Blue 500)
    "accent_hover": "#2563eb",   # Azul hover (Blue 600)
    "success": "#10b981",        # Verde esmeralda (Emerald 500)
    "warning": "#f59e0b",        # Âmbar / Laranja (Amber 500)
    "danger": "#ef4444",         # Vermelho coral (Red 500)
    "danger_hover": "#dc2626",   # Vermelho hover (Red 600)
    "input_bg": "#0f172a",       # Fundo das caixas de texto
    "tree_header": "#0f172a",    # Fundo do cabeçalho da tabela
    "row_even": "#1e293b",       # Linha par da tabela
    "row_odd": "#172033"         # Linha ímpar da tabela (zebra)
}

# Tipografia moderna do Windows (Segoe UI + Consolas para placas)
FONTES = {
    "titulo_app": ("Segoe UI", 14, "bold"),
    "subtitulo_app": ("Segoe UI", 9),
    "secao": ("Segoe UI", 11, "bold"),
    "label": ("Segoe UI", 10),
    "label_bold": ("Segoe UI", 10, "bold"),
    "placa_destaque": ("Consolas", 18, "bold"),
    "status_vinculo": ("Segoe UI", 13, "bold"),
    "botao": ("Segoe UI", 10, "bold"),
    "tabela": ("Segoe UI", 10),
    "tabela_header": ("Segoe UI", 10, "bold")
}


def aplicar_tema():
    """Injeta a estilização moderna nos componentes ttk (Abas, Tabelas, Botões e Scrollbars)"""
    style = ttk.Style()
    style.theme_use("clam")

    # --- Estilo Geral de Frames TTK ---
    style.configure("TFrame", background=CORES["bg_main"])

    # --- Estilo das Abas (Notebook) ---
    style.configure(
        "TNotebook",
        background=CORES["bg_main"],
        borderwidth=0,
        tabmargins=[15, 10, 15, 0]
    )
    style.configure(
        "TNotebook.Tab",
        background=CORES["bg_panel"],
        foreground=CORES["fg_muted"],
        padding=[20, 8],
        font=FONTES["label_bold"],
        borderwidth=0,
        focuscolor=CORES["bg_main"]
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", CORES["accent"]),
            ("active", CORES["border"])
        ],
        foreground=[
            ("selected", "#ffffff"),
            ("active", CORES["fg_text"])
        ]
    )

    # --- Estilo das Tabelas (Treeview) ---
    style.configure(
        "Treeview",
        background=CORES["bg_panel"],
        foreground=CORES["fg_text"],
        fieldbackground=CORES["bg_panel"],
        borderwidth=0,
        rowheight=32,
        font=FONTES["tabela"]
    )
    style.configure(
        "Treeview.Heading",
        background=CORES["tree_header"],
        foreground=CORES["fg_text"],
        font=FONTES["tabela_header"],
        padding=[10, 8],
        borderwidth=0,
        relief="flat"
    )
    style.map(
        "Treeview.Heading",
        background=[("active", CORES["border"])]
    )
    style.map(
        "Treeview",
        background=[("selected", CORES["accent"])],
        foreground=[("selected", "#ffffff")]
    )

    # --- Estilo da Barra de Rolagem (Scrollbar) ---
    style.configure(
        "Vertical.TScrollbar",
        background=CORES["bg_panel"],
        troughcolor=CORES["bg_main"],
        bordercolor=CORES["bg_main"],
        arrowcolor=CORES["fg_muted"],
        relief="flat"
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", CORES["border"])]
    )

    # --- Estilo de Botões TTK (Primário e Perigo) ---
    style.configure(
        "Accent.TButton",
        background=CORES["accent"],
        foreground="#ffffff",
        font=FONTES["botao"],
        padding=[14, 6],
        borderwidth=0,
        focuscolor=CORES["accent"]
    )
    style.map(
        "Accent.TButton",
        background=[("active", CORES["accent_hover"])]
    )

    style.configure(
        "Danger.TButton",
        background=CORES["danger"],
        foreground="#ffffff",
        font=FONTES["botao"],
        padding=[14, 6],
        borderwidth=0,
        focuscolor=CORES["danger"]
    )
    style.map(
        "Danger.TButton",
        background=[("active", CORES["danger_hover"])]
    )
