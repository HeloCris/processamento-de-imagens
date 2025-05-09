# 5ª Trabalho Prático - Processamento Digital de Imagens
# Discentes: Helorrayne Cristine de Alcantara Rodrigues e Wanderson Almeida de Mello
# Equalização de histograma normalizado de uma imagem em tons de cinza

from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def salvar(imagem_obj, nome_saida):
    """
    Salva uma imagem PIL no diretório 'output' com o nome especificado.

    Parâmetros:
        imagem_obj (PIL.Image): Objeto da imagem a ser salva.
        nome_saida (str): Nome do arquivo de imagem de saída.
    """
    os.makedirs("output", exist_ok=True)
    caminho_saida = os.path.join("output", nome_saida)
    imagem_obj.save(caminho_saida)
    print(f"Imagem salva em: {caminho_saida}")

def salvar_histograma(hist, titulo, nome_arquivo):
    """
    Gera e salva um gráfico de histograma a partir de uma lista de frequências.

    Parâmetros:
        hist (list): Lista com as frequências dos níveis de cinza.
        titulo (str): Título do gráfico.
        nome_arquivo (str): Nome do arquivo de imagem do histograma a ser salvo.
    """
    os.makedirs("output", exist_ok=True)
    plt.figure()
    plt.title(titulo)
    plt.xlabel("Nível de Cinza")
    plt.ylabel("Frequência")
    plt.bar(range(256), hist, color='gray')
    plt.xlim([0, 255])
    plt.savefig(os.path.join("output", nome_arquivo))
    plt.close()
    print(f"Histograma salvo em: output/{nome_arquivo}")

def imagem_para_matriz(caminho_imagem):
    """
    Converte uma imagem em tons de cinza para matriz NumPy.

    Parâmetros:
        caminho_imagem (str): Caminho para o arquivo da imagem.

    Retorna:
        tuple: Matriz NumPy da imagem em escala de cinza e objeto da imagem original.
    """
    try:
        imagem = Image.open(caminho_imagem).convert("L")
        return np.array(imagem), imagem
    except Exception as erro:
        print("Erro ao carregar a imagem:", erro)
        return None, None

def equalizar_histograma(matriz):
    """
    Aplica a equalização de histograma normalizado a uma matriz de imagem.

    Parâmetros:
        matriz (np.ndarray): Matriz da imagem original.

    Retorna:
        tuple: Matriz equalizada, histograma original, histograma equalizado.
    """
    L = 256
    linhas, colunas = matriz.shape
    total = linhas * colunas

    # Cálculo do histograma original
    hist = [0] * L
    for i in range(linhas):
        for j in range(colunas):
            hist[matriz[i, j]] += 1

    # Normalização
    pr = [h / total for h in hist] 

    # CDF acumulada
    cdf = [0] * L
    acumulado = 0.0
    for k in range(L):
        acumulado += pr[k]
        cdf[k] = acumulado

    # Gera a tabela de transformação (LUT)
    lut = [int(round((L - 1) * cdf[k])) for k in range(L)]

    # Aplica a LUT para criar a imagem equalizada
    resultado = np.zeros_like(matriz, dtype=np.uint8)
    for i in range(linhas):
        for j in range(colunas):
            resultado[i, j] = lut[matriz[i, j]]

    # Calcula o novo histograma
    novo_hist = [0] * L
    for i in range(linhas):
        for j in range(colunas):
            novo_hist[resultado[i, j]] += 1

    return resultado, hist, novo_hist

def main():
    """
    Função principal que executa o processo de:
      - Leitura de imagem
      - Equalização do histograma
      - Geração e salva das imagens e histogramas
    """
    caminho_imagem = "imagem.png"
    matriz, imagem_original = imagem_para_matriz(caminho_imagem)
    if matriz is not None:
        equalizada, hist_original, hist_equalizado = equalizar_histograma(matriz)
        salvar(Image.fromarray(equalizada, mode="L"), "equalizada.jpg")
        salvar_histograma(hist_original, "Histograma Original", "histograma_original.png")
        salvar_histograma(hist_equalizado, "Histograma Equalizado", "histograma_equalizado.png")
        salvar(imagem_original, "original.jpg")

if __name__ == "__main__":
    main()
