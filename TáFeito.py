import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==============================================================================
# 1. CONFIGURAÇÃO E DIVISÃO 50% DESENVOLVIMENTO / 50% TESTE
# ===============================================================================
PASTA_IMAGENS = Path(r"Folder das imagens")
FICHEIRO_DIVISAO = Path("divisao_dataset.json")
SEED = 100  # Garante reprodutibilidade científica para o artigo IEEE

def obter_divisao_dataset(pasta: Path, ficheiro_json: Path, seed: int = 42):
    if not pasta.exists():
        raise FileNotFoundError(f"A pasta não foi encontrada: {pasta}")

    # Se a divisão já foi feita anteriormente, carrega-a para não misturar dados
    if ficheiro_json.exists():
        with open(ficheiro_json, "r", encoding="utf-8") as f:
            dados = json.load(f)
        print(f"[INFO] Divisão carregada de '{ficheiro_json}' ({len(dados['dev'])} dev / {len(dados['test'])} test).")
        return dados["dev"], dados["test"]

    # Listar todos os ficheiros .tiff / .tif
    ficheiros = sorted([
        f.name for f in pasta.iterdir()
        if f.is_file() and f.suffix.lower() in {".tiff", ".tif"}
    ])

    if len(ficheiros) == 0:
        raise ValueError("Nenhum ficheiro .tiff encontrado na pasta especificada.")

    random.seed(seed)
    baralhados = ficheiros.copy()
    random.shuffle(baralhados)

    meio = len(baralhados) // 2
    dev_files = sorted(baralhados[:meio])
    test_files = sorted(baralhados[meio:])

    divisao = {
        "seed": seed,
        "total_ficheiros": len(ficheiros),
        "dev": dev_files,
        "test": test_files
    }

    with open(ficheiro_json, "w", encoding="utf-8") as f:
        json.dump(divisao, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Nova divisão 50/50 criada e guardada em '{ficheiro_json}':")
    print(f"       -> Desenvolvimento: {len(dev_files)} imagens")
    print(f"       -> Teste (INTACTO): {len(test_files)} imagens")
    return dev_files, test_files


# ==============================================================================
# 2. FUNÇÕES DE ENTROPIA, AGRUPAMENTO E PREDIÇÃO
# ==============================================================================
def calcular_entropia(dados: np.ndarray) -> float:
    """Calcula a entropia de 1ª ordem H(X) em bits/símbolo."""
    _, contagens = np.unique(dados, return_counts=True)
    probs = contagens / contagens.sum()
    return float(-np.sum(probs * np.log2(probs)))


def entropia_agrupamento_espacial(img: np.ndarray) -> float:
    """
    Agrupa pares de píxeis adjacentes na horizontal (X_i, X_{i+1}) por canal.
    Devolve H(X_1, X_2) / 2 (bits por símbolo) para comparação direta.
    """
    canais = [img] if img.ndim == 2 else [img[:, :, c] for c in range(img.shape[2])]
    entropias = []

    for ch in canais:
        h, w = ch.shape
        if w % 2 != 0:
            ch = ch[:, :-1]
        pares = ch.reshape(-1, 2).astype(np.uint16)
        # Empacotar dois uint8 num único uint16 (equivalente a pares únicos, 10x mais rápido)
        simbolos_agrupados = (pares[:, 0] << 8) | pares[:, 1]
        entropias.append(calcular_entropia(simbolos_agrupados) / 2.0)

    return float(np.mean(entropias))


def entropia_agrupamento_espectral(img: np.ndarray) -> float:
    """
    Agrupa o tripleto (R, G, B) de cada píxel num único símbolo conjunto.
    Devolve H(R, G, B) / 3 (bits por canal).
    """
    if img.ndim != 3 or img.shape[2] < 3:
        return calcular_entropia(img)

    rgb = img[:, :, :3].reshape(-1, 3).astype(np.uint32)
    # Empacotar 3 bytes (R, G, B) num único inteiro de 24 bits
    tripletos = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return calcular_entropia(tripletos) / 3.0


def preditor_1d_esquerda(canal: np.ndarray) -> np.ndarray:
    """Predição linear 1D: x_hat(i, j) = x(i, j-1)."""
    c = canal.astype(np.int32)
    return c[:, 1:] - c[:, :-1]


def preditor_2d_media(canal: np.ndarray) -> np.ndarray:
    """Predição linear 2D: x_hat(i, j) = floor((Esquerda + Cima) / 2)."""
    c = canal.astype(np.int32)
    esquerda = c[1:, :-1]
    cima = c[:-1, 1:]
    pred = (esquerda + cima) // 2
    return c[1:, 1:] - pred


def preditor_2d_med(canal: np.ndarray) -> np.ndarray:
    """
    Median Edge Detector (MED - base do padrão JPEG-LS).
    Adapta-se a arestas horizontais, verticais e zonas suaves.
    """
    c = canal.astype(np.int32)
    a = c[1:, :-1]   # Esquerda (West)
    b = c[:-1, 1:]   # Cima (North)
    d = c[:-1, :-1]  # Diagonal superior esquerda (North-West)

    max_ab = np.maximum(a, b)
    min_ab = np.minimum(a, b)

    pred = np.where(d >= max_ab, min_ab,
           np.where(d <= min_ab, max_ab, a + b - d))

    return c[1:, 1:] - pred


def transformada_cor_reversivel_rct(img: np.ndarray) -> tuple:
    """
    Reversible Color Transform (RCT - JPEG 2000 Lossless).
    Descorrelaciona os canais R, G, B sem perda de informação.
    """
    r = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    b = img[:, :, 2].astype(np.int32)

    cb = b - g
    cr = r - g
    y = g + ((cb + cr) >> 2)
    return y, cb, cr


# ==============================================================================
# 3. EXECUÇÃO DA ANÁLISE (EXCLUSIVAMENTE NOS DADOS DE DESENVOLVIMENTO)
# ==============================================================================
def analisar_conjunto_desenvolvimento(pasta: Path, dev_files: list):
    metricas = {
        "H_1a_Ordem": [],
        "H_Agrup_Espacial_2px": [],
        "H_Agrup_Espectral_RGB": [],
        "H_Pred_1D_Esquerda": [],
        "H_Pred_2D_Media": [],
        "H_Pred_2D_MED": [],
        "H_RCT_Apenas": [],
        "H_RCT_mais_MED": []
    }

    print(f"\nA analisar {len(dev_files)} imagens do conjunto de DESENVOLVIMENTO...\n")

    for nome in dev_files:
        caminho = pasta / nome
        img = np.array(Image.open(caminho))

        # 1. Entropia de 1ª ordem (sem processamento)
        h_base = calcular_entropia(img)
        metricas["H_1a_Ordem"].append(h_base)

        # 2. Possibilidades de Agrupamento
        metricas["H_Agrup_Espacial_2px"].append(entropia_agrupamento_espacial(img))
        metricas["H_Agrup_Espectral_RGB"].append(entropia_agrupamento_espectral(img))

        # 3. Possibilidades de Predição Espacial por canal
        canais = [img] if img.ndim == 2 else [img[:, :, c] for c in range(img.shape[2])]

        h_1d = np.mean([calcular_entropia(preditor_1d_esquerda(c)) for c in canais])
        h_2d_med = np.mean([calcular_entropia(preditor_2d_media(c)) for c in canais])
        h_med = np.mean([calcular_entropia(preditor_2d_med(c)) for c in canais])

        metricas["H_Pred_1D_Esquerda"].append(h_1d)
        metricas["H_Pred_2D_Media"].append(h_2d_med)
        metricas["H_Pred_2D_MED"].append(h_med)

        # 4. Combinação: Descorrelação Espectral (RCT) + Predição Espacial (MED)
        if img.ndim == 3 and img.shape[2] >= 3:
            y, cb, cr = transformada_cor_reversivel_rct(img)
            h_rct = np.mean([calcular_entropia(c) for c in (y, cb, cr)])
            h_rct_med = np.mean([calcular_entropia(preditor_2d_med(c)) for c in (y, cb, cr)])
        else:
            h_rct = h_base
            h_rct_med = h_med

        metricas["H_RCT_Apenas"].append(h_rct)
        metricas["H_RCT_mais_MED"].append(h_rct_med)

    return metricas


def apresentar_resultados_e_grafico(metricas: dict):
    print("=" * 75)
    print(f"{'MÉTODO / ABORDAGEM':<32} | {'ENTROPIA MÉDIA (bits/símbolo)':<28} | {'GANHO'}")
    print("=" * 75)

    base_media = np.mean(metricas["H_1a_Ordem"])

    nomes_legiveis = {
        "H_1a_Ordem": "1. Original (1ª Ordem)",
        "H_Agrup_Espacial_2px": "2. Agrupamento Espacial (2 px)",
        "H_Agrup_Espectral_RGB": "3. Agrupamento Espectral (RGB)",
        "H_RCT_Apenas": "4. Transformada Cor (RCT)",
        "H_Pred_1D_Esquerda": "5. Predição 1D (Esquerda)",
        "H_Pred_2D_Media": "6. Predição 2D (Média)",
        "H_Pred_2D_MED": "7. Predição 2D (MED / JPEG-LS)",
        "H_RCT_mais_MED": "8. Híbrido (RCT + MED)"  
    }

    medias = []
    desvios = []
    labels = []

    for chave, titulo in nomes_legiveis.items():
        vals = np.array(metricas[chave])
        m = np.mean(vals)
        s = np.std(vals)
        ganho_pct = ((base_media - m) / base_media) * 100.0

        medias.append(m)
        desvios.append(s)
        labels.append(titulo.split(". ")[1])

        print(f"{titulo:<32} | {m:>7.4f} ± {s:<7.4f} bits      | -{ganho_pct:>5.1f}%")

    print("=" * 75)

    # Gerar gráfico pronto para o artigo IEEE e PowerPoint
    plt.figure(figsize=(10, 5))
    cores = ['#4a5568', '#3182ce', '#3182ce', '#38a169', '#dd6b20', '#dd6b20', '#e53e3e', '#805ad5']
    barras = plt.bar(labels, medias, yerr=desvios, capsize=4, color=cores, edgecolor='black', alpha=0.9)

    plt.ylabel("Entropia Média (bits / símbolo)")
    plt.title("Análise de Entropia, Agrupamento e Predição — Dataset Aerials (Desenvolvimento)")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 8.8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for barra, m, s in zip(barras, medias, desvios):
        plt.text(barra.get_x() + barra.get_width() / 2, m + s + 0.12, f"{m:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("grafico_fase1_aerials.png", dpi=300)
    print("\n[INFO] Gráfico guardado em alta resolução: 'grafico_fase1_aerials.png'")
    plt.show()


if __name__ == "__main__":
    dev_files, test_files = obter_divisao_dataset(PASTA_IMAGENS, FICHEIRO_DIVISAO, SEED)
    resultados = analisar_conjunto_desenvolvimento(PASTA_IMAGENS, dev_files)
    apresentar_resultados_e_grafico(resultados)