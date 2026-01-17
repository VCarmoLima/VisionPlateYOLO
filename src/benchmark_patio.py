import os
import cv2
import pandas as pd
import time
from ocr_engine import OCREngine
from utils import *

# --- CONFIGURAÇÃO ---
MODELO_PLACA = "models/placa_v113/weights/best.pt"
MODELO_CHARS = "models/caracteres_v11/weights/best.pt"
PASTA_PATIO = "datasets/patio_recortadas" # Onde você colocou as 107 imagens

def normalizar_texto(texto):
    """Limpa o texto para comparação justa (Tira traço, espaço e extensão .jpg)"""
    if not texto: return ""
    return texto.replace("-", "").replace(" ", "").upper().split(".")[0]

def main():
    # Verifica caminhos
    if not os.path.exists(PASTA_PATIO):
        print(f"❌ Erro: A pasta '{PASTA_PATIO}' não existe.")
        print("   -> Crie a pasta e coloque as 107 imagens lá dentro.")
        return

    if not os.path.exists(MODELO_CHARS):
        print(f"❌ Erro: Modelo de caracteres não encontrado em '{MODELO_CHARS}'")
        return

    print("🔄 Carregando Motor de IA...")
    # Carrega a engine (ignoramos o detector de placa se as imagens já forem crop)
    motor = OCREngine(MODELO_PLACA, MODELO_CHARS)

    arquivos = [f for f in os.listdir(PASTA_PATIO) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(arquivos) == 0:
        print("❌ Nenhuma imagem encontrada na pasta.")
        return

    print(f"🚀 Iniciando Auditoria em {len(arquivos)} imagens...\n")

    total = 0
    acertos = 0
    erros = []
    tempo_total = 0

    for arquivo in arquivos:
        total += 1
        caminho_img = os.path.join(PASTA_PATIO, arquivo)

        # 1. O Gabarito é o nome do arquivo (ex: 'ABC1234.jpg' -> 'ABC1234')
        gabarito = normalizar_texto(arquivo)

        # 2. Carrega imagem
        img = cv2.imread(caminho_img)
        if img is None:
            print(f"⚠️ Erro ao abrir: {arquivo}")
            continue

        # 3. Leitura da Placa (Cronometrada)
        inicio = time.time()

        # ATENÇÃO: Chamamos direto o reconhecedor de caracteres,
        # pois a imagem JÁ É O RECORTE da placa.
        texto_lido_bruto, _ = motor.reconhecer_caracteres(img)

        # APLICA A CORREÇÃO DE LOGICA
        texto_corrigido = corrigir_placa(texto_lido_bruto)
        texto_final = normalizar_texto(texto_corrigido)

        fim = time.time()
        tempo_total += (fim - inicio)

        # 4. Comparação
        if texto_final == gabarito:
            acertos += 1
            print(f"✅ [{acertos}/{total}] {arquivo} -> LIDO: {texto_final}")
        else:
            print(f"❌ [{acertos}/{total}] {arquivo} -> ERROU: {texto_final} (Esperado: {gabarito})")
            erros.append({
                "Arquivo": arquivo,
                "Esperado": gabarito,
                "Lido_IA": texto_final,
                "Match": False
            })

    # --- RESULTADO FINAL ---
    precisao = (acertos / total) * 100
    media_tempo = (tempo_total / total) * 1000 # em ms

    print("\n" + "="*40)
    print("📊 RESULTADO DA AUDITORIA (BENCHMARK)")
    print("="*40)
    print(f"📂 Total Auditado: {total}")
    print(f"✅ Acertos Exatos: {acertos}")
    print(f"❌ Erros:          {len(erros)}")
    print(f"🎯 PRECISÃO FINAL: {precisao:.2f}%")
    print(f"⚡ Tempo Médio:    {media_tempo:.1f} ms por placa")
    print("="*40)

    if erros:
        # Salva relatório de erros para análise
        df = pd.DataFrame(erros)
        df.to_csv("relatorio_erros.csv", index=False)
        print("📄 Detalhes dos erros salvos em 'relatorio_erros.csv'")
    else:
        print("🏆 PERFEITO! NENHUM ERRO ENCONTRADO.")

if __name__ == "__main__":
    main()