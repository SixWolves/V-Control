import cv2
import pytesseract
import sqlite3
import datetime
import time

# 1. Configuração do Banco de Dados
conn = sqlite3.connect('controle_acesso.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS acessos
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   placa TEXT, 
                   tipo TEXT, 
                   data_hora TEXT)''')
conn.commit()

# 2. Inicialização das Câmeras
cam_entrada = cv2.VideoCapture(0) # Substituir pelo IP/ID correto da câmera de entrada
cam_saida = cv2.VideoCapture(1)   # Substituir pelo IP/ID correto da câmera de saída

def processar_ocr(frame):
    """Converte para cinza e tenta ler a placa."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Ajuste o PSM conforme o padrão de corte da placa
    texto = pytesseract.image_to_string(gray, config='--psm 8').strip()
    return texto if len(texto) >= 7 else None

def registrar_acesso(placa, tipo):
    """Salva no banco de dados e imprime no terminal."""
    agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO acessos (placa, tipo, data_hora) VALUES (?, ?, ?)", 
                   (placa, tipo, agora))
    conn.commit()
    print(f"[{agora}] {tipo} Registrada | Placa: {placa}")

# 3. Integração com Sensores Ultrassônicos (ESP32)
def sensor_detectou_entrada():
    # TODO: Inserir lógica de leitura Serial/MQTT do sensor ultrassônico
    return False 

def sensor_detectou_saida():
    # TODO: Inserir lógica de leitura Serial/MQTT do sensor ultrassônico
    return False

print("Sistema de Monitoramento LPR Iniciado...")

try:
    while True:
        # Captura contínua para exibir o vídeo em tempo real
        ret_in, frame_in = cam_entrada.read()
        ret_out, frame_out = cam_saida.read()

        if ret_in:
            cv2.imshow("Video Entrada", frame_in)
        if ret_out:
            cv2.imshow("Video Saida", frame_out)

        # Fluxo de Entrada (Sensor aciona o OCR)
        if sensor_detectou_entrada() and ret_in:
            placa = processar_ocr(frame_in)
            if placa:
                registrar_acesso(placa, "ENTRADA")
            time.sleep(3) 

        # Fluxo de Saída (Sensor aciona o OCR)
        if sensor_detectou_saida() and ret_out:
            placa = processar_ocr(frame_out)
            if placa:
                registrar_acesso(placa, "SAÍDA")
            time.sleep(3)

        # Atualiza a janela e aguarda a tecla 'q' para fechar o programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Erro: {e}")
finally:
    print("Encerrando sistema...")
    cam_entrada.release()
    cam_saida.release()
    cv2.destroyAllWindows() # Fecha as janelas de vídeo
    conn.close()