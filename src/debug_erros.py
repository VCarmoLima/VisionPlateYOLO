import os
import cv2
from ocr_engine import *

# Configuração
MODELO_PLACA = "models/placa_v113/weights/best.pt"
MODELO_CHARS = "models/caracteres_v11/weights/best.pt"
PASTA_PATIO = "datasets/patio_recortadas"
PASTA_ERROS = "erros_visuais"

def main():
    os.makedirs(PASTA_ERROS, exist_ok=True)
    motor = OCREngine(MODELO_PLACA, MODELO_CHARS)
    arquivos = os.listdir(PASTA_PATIO)[:20] # Vamos testar só as 20 primeiras pra ser rápido

    print(f"📸 Gerando imagens de diagnóstico em '{PASTA_ERROS}'...")

    for arquivo in arquivos:
        caminho = os.path.join(PASTA_PATIO, arquivo)
        img = cv2.imread(caminho)

        # Roda o OCR e pega a imagem desenhada (img_debug)
        texto, img_debug = motor.reconhecer_caracteres(img)

        # Salva para você ver
        nome_salvo = f"{texto}_{arquivo}" # Ex: OBFP403_BPQ0F43.jpg
        cv2.imwrite(os.path.join(PASTA_ERROS, nome_salvo), img_debug)
        print(f"Salvo: {nome_salvo}")

if __name__ == "__main__":
    main()