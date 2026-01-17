import re

def corrigir_placa(texto_bruto):
    """
    Tenta corrigir erros comuns de OCR baseados na posição dos caracteres.
    Ex: 'AX04J04' (O zero na letra) vira 'AXO4J04'.
    """
    if not texto_bruto: return ""

    # Remove tudo que não é alfanumérico
    texto = re.sub(r'[^a-zA-Z0-9]', '', texto_bruto).upper()

    # Se tiver menos ou mais que 7 caracteres, a heurística falha (ou tentamos cortar)
    if len(texto) != 7:
        return texto # Retorna o original pois não dá para inferir posição

    # Listas de Texto (mutável)
    chars = list(texto)

    # --- DICIONÁRIOS DE CORREÇÃO ---
    # Se a IA ler número onde deveria ser letra:
    nums_para_letras = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B'
    }

    # Se a IA ler letra onde deveria ser número:
    letras_para_nums = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'J': '1', 'L': '1',
        'Z': '2',
        'B': '8',
        'S': '5',
        'G': '6',
        'A': '4'
    }

    # --- REGRAS POSICIONAIS (Mercosul e Antiga) ---
    # Posições 0, 1, 2: SEMPRE LETRAS (LLL....)
    for i in [0, 1, 2]:
        if chars[i] in nums_para_letras:
            chars[i] = nums_para_letras[chars[i]]

    # Posição 3: SEMPRE NÚMERO (...N...)
    if chars[3] in letras_para_nums:
        chars[3] = letras_para_nums[chars[3]]

    # Posição 4: HÍBRIDA
    # Se for Mercosul (LLLNLNN), é Letra. Se for Antiga (LLLNNNN), é Número.
    # Aqui é difícil decidir sem saber o padrão.
    # Mas se a IA detectou letra, confiamos que é Mercosul. Se número, Antiga.
    # Podemos forçar correções óbvias, mas vamos deixar quieto por enquanto.

    # Posições 5, 6: SEMPRE NÚMEROS (.....NN)
    for i in [5, 6]:
        if chars[i] in letras_para_nums:
            chars[i] = letras_para_nums[chars[i]]

    return "".join(chars)

def converter_antiga_para_mercosul(placa):
    """Converte padrão antigo para Mercosul"""
    placa = corrigir_placa(placa) # Aplica correção antes

    if re.match(r'^[A-Z]{3}[0-9]{4}$', placa):
        mapa_conversao = {
            '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
            '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J'
        }
        parte1 = placa[0:4]
        numero_meio = placa[4]
        parte2 = placa[5:]
        nova_letra = mapa_conversao.get(numero_meio, numero_meio)
        return f"{parte1}{nova_letra}{parte2}"

    return placa

def validar_placa(placa):
    placa = placa.replace("-", "").replace(" ", "").upper()
    if re.match(r'^[A-Z]{3}[0-9][0-9A-Z][0-9]{2}$', placa):
        return True, "Válida"
    return False, "Inválida"