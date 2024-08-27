###########################################################################################################################################
##                                                                                                                                       ##
## Contagem de pessoas utilizando o modelo Yolo e o OpenCV                                                                               ##
##                                                                                                                                       ##
## Trabalho desenvolvido no 2° Quadrimestre de 2024 para a matéria Visão Computacional da Universidade Federal Do ABC                    ##
##                                                                                                                                       ##
## Professor Dr. Celso S Kurashima                                                                                                       ##
##                                                                                                                                       ##
## Integrantes: Diego Aoyagi de Souza - RA: 11066516, Gabriel Gomes de Oliveira - RA: 11108214, Gustavo Cardoso Bezerra- RA: 11201822553 ##
##                                                                                                                                       ##
###########################################################################################################################################

import cv2
import tkinter as tk
from collections import deque
from statistics import mode
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

# Inicializa a janela principal com tkinter
root = tk.Tk()
root.title("Contagem de Pessoas")

# Frame para a exibição do vídeo
video_frame = tk.Frame(root)
video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Label para exibir o vídeo
lmain = tk.Label(video_frame)
lmain.pack()

# Frame para as configurações na parte inferior
config_frame = tk.Frame(root)
config_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Dicionário de mapeamento de classes
class_names = {"Pessoas": 0, "Celular": 67}

# Variáveis para armazenar os valores de conf e maxlen
conf_value = tk.DoubleVar(value=0.30)
maxlen_value = tk.IntVar(value=15)
cls_value = tk.StringVar(value="Pessoas")  # Armazena o nome da classe selecionada

# Função para atualizar os valores
def update_conf(value):
    conf_value.set(float(value))

def update_maxlen(value):
    maxlen_value.set(int(value))

def update_cls(value):
    cls_value.set(value)

# Slider para ajustar o valor de conf
tk.Label(config_frame, text="Índice de Confiabilidade").pack()
tk.Scale(config_frame, from_=0.0, to_=1.0, resolution=0.01, orient=tk.HORIZONTAL, variable=conf_value, command=update_conf).pack()

# Slider para ajustar o valor de maxlen
tk.Label(config_frame, text="Contagem a cada x Frames").pack()
tk.Scale(config_frame, from_=3, to_=107, resolution=4, orient=tk.HORIZONTAL, variable=maxlen_value, command=update_maxlen).pack()

# Menu para selecionar a classe
tk.Label(config_frame, text="Classe").pack()
cls_menu = tk.OptionMenu(config_frame, cls_value, *class_names.keys(), command=update_cls)
cls_menu.pack()

# Carrega o modelo YOLO
model = YOLO('yolov8n.pt')

# Abre a captura de vídeo
cap = cv2.VideoCapture(1)

# Obtem a taxa de quadros real da webcam
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:  # Se não conseguir pegar a taxa de FPS, define um padrão
    fps = 30.0

# Obtem a largura e altura dos frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define o codec e cria o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Carrega a imagem da marca d'água
watermark = cv2.imread('logotipo-ufabc-abaixo.png', cv2.IMREAD_UNCHANGED)

# Redimensiona a marca d'água se necessário
(wH, wW) = watermark.shape[:2]
scale = 0.05  # escala da marca d'água
watermark = cv2.resize(watermark, (int(wW * scale), int(wH * scale)))

# Define a posição da marca d'água (inferior direito)
(wH, wW) = watermark.shape[:2]
posX = frame_width - wW - 10
posY = frame_height - wH - 10

# Inicializa o histórico da contagem de pessoas
person_count_history = deque(maxlen=maxlen_value.get())  # Mantem a contagem dos últimos frames

def show_frame():
    global person_count_history  # Adiciona essa linha para usar a variável global

    success, im0 = cap.read()
    if success:
        # Atualiza o tamanho da fila em tempo real
        person_count_history = deque(person_count_history, maxlen=maxlen_value.get())

        # Faz a predição na imagem capturada da webcam
        results = model.predict(source=im0)
        # Inicializa o contador de pessoas
        current_count = 0

        # Extrai as caixas delimitadoras, classes e confiabilidades
        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        # Adiciona fundo preto no texto de contagem de pessoas
        x,y,w,h = 0,0,255,45
        cv2.rectangle(im0, (x, x), (x + w, y + h), (0,0,0), -1)

        if boxes is not None:
            for box, cls, conf in zip(boxes, clss, confs):
                # Verifica se a classe é a selecionada e se a confiança é maior que o valor configurado
                if int(cls) == class_names[cls_value.get()] and conf > conf_value.get():
                    current_count += 1

                    # Desenha a caixa delimitadora e o rótulo na imagem
                    color = colors(int(cls), True)
                    cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                    label = f'{model.names[int(cls)]}-{conf:.2f}'
                    cv2.putText(im0, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Adiciona a contagem atual à história
        person_count_history.append(current_count)

        # Calcula a moda da contagem de pessoas
        if len(person_count_history) > 0:
            smoothed_count = mode(person_count_history)
        else:
            smoothed_count = current_count

        # Adiciona a marca d'água
        if watermark.shape[2] == 4:
            b, g, r, a = cv2.split(watermark)
            overlay = im0.copy()
            mask = cv2.merge((b, g, r))
            alpha_mask = a / 255.0
            for c in range(0, 3):
                overlay[posY:posY + wH, posX:posX + wW, c] = (alpha_mask * mask[:, :, c] + (1 - alpha_mask) * overlay[posY:posY + wH, posX:posX + wW, c])
        else:
            overlay[posY:posY + wH, posX:posX + wW] = watermark

        # Exibe a moda da contagem de pessoas na tela
        cv2.putText(overlay, f'Quantidade: {smoothed_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Converte a imagem de BGR para RGB e exibe no Tkinter
        cv2image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)  # Converte para imagem PIL
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        # Verifica se a tecla 'q' foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            root.quit()  # Encerra a janela Tkinter
            return

        lmain.after(10, show_frame)
    else:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

show_frame()
root.mainloop()

