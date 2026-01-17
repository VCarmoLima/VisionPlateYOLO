from ultralytics import YOLO
import cv2
import numpy as np
import logging

# Silencia logs chatos do YOLO no terminal
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class OCREngine:
    def __init__(self, model_placa_path, model_chars_path):
        """
        Inicializa o motor de OCR carregando os dois modelos YOLOv11.
        """
        print(f"🔄 Carregando modelos YOLOv11 na GPU...")
        try:
            self.detector_placa = YOLO(model_placa_path)
            self.detector_chars = YOLO(model_chars_path)
            print("✅ Modelos carregados com sucesso!")
        except Exception as e:
            print(f"❌ Erro crítico ao carregar modelos: {e}")
            self.detector_placa = None
            self.detector_chars = None

    def detectar_placa(self, imagem):
        """
        Passo 1: Encontrar a placa na imagem da moto/carro.
        Retorna: A imagem recortada (crop) da placa.
        """
        # conf=0.25 é um bom equilíbrio padrão
        resultados = self.detector_placa(imagem, conf=0.25, verbose=False)

        melhor_crop = None
        maior_conf = 0

        for r in resultados:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > maior_conf:
                    # Coordenadas
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Padding (Margem de segurança para não cortar a beirada da letra)
                    h, w = imagem.shape[:2]
                    margin = 5
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)

                    melhor_crop = imagem[y1:y2, x1:x2]
                    maior_conf = conf

        return melhor_crop

    def reconhecer_caracteres(self, placa_crop):
        if placa_crop is None: return "", None

        # 1. Tratamento de Imagem
        gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)
        img_input = cv2.merge([gray, gray, gray])

        # 2. Inferência (Baixei conf para 0.1 para não perder nada)
        resultados = self.detector_chars(img_input, conf=0.1, verbose=False)

        letras_detectadas = []
        h_img, w_img, _ = placa_crop.shape

        for r in resultados:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                letra = self.detector_chars.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # --- FILTRO DE RUÍDO (SUAVE) ---
                altura_box = y2 - y1
                # Só descarta se for MINÚSCULO (menos de 5% da altura)
                if altura_box < (h_img * 0.05):
                    continue

                letras_detectadas.append({
                    'letra': letra,
                    'x1': x1,
                    'x2': x2, # Precisamos do x2 para calcular centro
                    'conf': conf,
                    'coords': (x1, y1, x2, y2)
                })

        # --- ORDENAÇÃO ROBUSTA ---
        letras_detectadas.sort(key=lambda x: x['x1'])

        # --- FILTRO DE SOBREPOSIÇÃO INTELIGENTE ---
        letras_finais = []
        for item in letras_detectadas:
            if not letras_finais:
                letras_finais.append(item)
                continue

            ultimo = letras_finais[-1]

            # Calcula o centro das caixas
            centro_atual = (item['x1'] + item['x2']) / 2
            centro_ultimo = (ultimo['x1'] + ultimo['x2']) / 2

            # Largura da letra anterior
            largura_ultimo = ultimo['x2'] - ultimo['x1']

            # Se o centro da nova letra estiver "dentro" da letra anterior (sobreposição forte)
            # Limite: Distância entre centros menor que 40% da largura da letra
            distancia = abs(centro_atual - centro_ultimo)

            if distancia < (largura_ultimo * 0.4):
                # É a mesma letra duplicada (ex: B e 8). Fica com a de maior confiança.
                if item['conf'] > ultimo['conf']:
                    letras_finais[-1] = item
            else:
                letras_finais.append(item)

        texto_final = "".join([item['letra'] for item in letras_finais])

        # Debug Visual
        img_debug = placa_crop.copy()
        for item in letras_finais:
            x1, y1, x2, y2 = item['coords']
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img_debug, item['letra'], (x1, y1-2), 0, 0.5, (0, 255, 0), 1)

        return texto_final, img_debug

    def processar_imagem(self, imagem_path):
        """
        Fluxo completo: Lê arquivo -> Acha Placa -> Lê Letras -> Retorna Resultado
        """
        img = cv2.imread(imagem_path)
        if img is None:
            return {"erro": "Imagem não encontrada"}

        # 1. Detectar
        placa_crop = self.detectar_placa(img)

        if placa_crop is None:
            return {"sucesso": False, "mensagem": "Nenhuma placa detectada"}

        # 2. Ler
        texto, img_debug = self.reconhecer_caracteres(placa_crop)

        return {
            "sucesso": True,
            "texto": texto,
            "crop_placa": placa_crop,
            "crop_debug": img_debug
        }