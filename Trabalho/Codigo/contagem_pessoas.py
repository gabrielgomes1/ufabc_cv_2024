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
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

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

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("O frame está vazio ou o processo está completo com sucesso")
        break

    # Faz a predição na imagem capturada da webcam
    results = model.predict(source=im0)

    # Inicializa o contador de pessoas
    person_count = 0

    # Extrai as caixas delimitadoras, classes e confiabilidades
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()

    if boxes is not None:
        for box, cls, conf in zip(boxes, clss, confs):
            # Verifica se a classe é 0 (pessoa) e se a confiança é maior que 0,50
            if int(cls) == 0 and conf > 0.50:
                # Incrementa o contador de pessoas
                person_count += 1

                # Desenha a caixa delimitadora e o rótulo na imagem
                color = colors(int(cls), True)
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                label = f'{model.names[int(cls)]}-{conf:.2f}'
                cv2.putText(im0, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibe o número de pessoas detectadas na tela
    cv2.putText(im0, f'Pessoas: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Escreve o frame no arquivo de saída
    out.write(im0)

    # Exibe a imagem processada
    cv2.imshow("Output", im0)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera tudo após a conclusão
cap.release()
out.release()
cv2.destroyAllWindows()
