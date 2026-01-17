import os
import cv2
from ocr_engine import OCREngine
from utils import converter_antiga_para_mercosul, validar_placa

# --- CONFIGURAÇÃO ---
# Caminhos dos modelos (ajuste o nome da pasta se mudar no futuro)
MODELO_PLACA = "models/placa_v113/weights/best.pt"  # <--- Seu modelo 99% preciso
MODELO_CHARS = "models/caracteres_v11/weights/best.pt" # <--- O que está treinando agora

# Pasta com as imagens das motos para testar
PASTA_IMAGENS = "datasets/placa/test/images" # Vamos usar as imagens de teste do dataset
PASTA_RESULTADOS = "resultados_finais"

def main():
    # 1. Verifica se os modelos existem
    if not os.path.exists(MODELO_PLACA):
        print(f"❌ Modelo de Placa não encontrado: {MODELO_PLACA}")
        return

    # (Só vai funcionar quando o treino atual terminar)
    if not os.path.exists(MODELO_CHARS):
        print(f"⚠️ Modelo de Caracteres ainda não existe (Treino em andamento?).")
        print(f"   Esperando em: {MODELO_CHARS}")
        return

    # 2. Inicializa o motor
    motor = OCREngine(MODELO_PLACA, MODELO_CHARS)

    # 3. Cria pasta de resultados
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    # 4. Pega algumas imagens para testar
    arquivos = [f for f in os.listdir(PASTA_IMAGENS) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    # Vamos processar as 5 primeiras para teste rápido
    arquivos = arquivos[:5]

    print(f"🚀 Iniciando processamento de {len(arquivos)} imagens...\n")

    for arquivo in arquivos:
        caminho_img = os.path.join(PASTA_IMAGENS, arquivo)
        print(f"📸 Processando: {arquivo}")

        resultado = motor.processar_imagem(caminho_img)

# ... dentro do loop for ...
    if resultado["sucesso"]:
        texto_bruto = resultado["texto"]

        # 1. Padroniza tudo para Mercosul (Sua ideia genial)
        texto_padronizado = converter_antiga_para_mercosul(texto_bruto)

        # 2. Valida se o resultado final faz sentido
        eh_valida, tipo_placa = validar_placa(texto_padronizado)

        if eh_valida:
            print(f"   ✅ PLACA LIDA: {texto_padronizado} ({tipo_placa})")
            # Se quiser ver o original: print(f"      (Lido: {texto_bruto})")
        else:
            print(f"   ⚠️ Leitura Suspeita: {texto_bruto} (Não parece uma placa)")

    print(f"\n✨ Teste finalizado! Confira a pasta '{PASTA_RESULTADOS}'")

if __name__ == "__main__":
    main()