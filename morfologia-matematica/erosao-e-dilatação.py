
#Helorrayne e Wanderson Mello
# Operações morfológicas
import cv2
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagem_tons_de_cinza(caminho_imagem):
    """Carrega uma imagem e a converte para tons de cinza."""
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {caminho_imagem}")
    return img

def erosao_tons_de_cinza(imagem, elemento_estruturante):
    """Aplica a operação de erosão em uma imagem em tons de cinza."""
    return cv2.erode(imagem, elemento_estruturante, iterations=1)

def dilatacao_tons_de_cinza(imagem, elemento_estruturante):
    """Aplica a operação de dilatação em uma imagem em tons de cinza."""
    return cv2.dilate(imagem, elemento_estruturante, iterations=1)

def gradiente_morfologico_tons_de_cinza(imagem, elemento_estruturante):
    """Calcula o gradiente morfológico de uma imagem em tons de cinza."""
    img_dilatada = cv2.dilate(imagem, elemento_estruturante, iterations=1)
    img_erodida = cv2.erode(imagem, elemento_estruturante, iterations=1)
    gradiente = cv2.subtract(img_dilatada, img_erodida)
    
    return gradiente

def exibir_imagens(imagens, titulos):
    """Exibe uma lista de imagens com seus respectivos títulos."""
    if len(imagens) != len(titulos):
        raise ValueError("O número de imagens deve ser igual ao número de títulos.")

    plt.figure(figsize=(15, 5)) # Ajustado para acomodar mais uma imagem
    for i, (imagem, titulo) in enumerate(zip(imagens, titulos)):
        plt.subplot(1, len(imagens), i + 1)
        plt.imshow(imagem, cmap='gray')
        plt.title(titulo)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Solicita ao usuário o caminho da imagem
    caminho_imagem_entrada = input("Digite o caminho para a imagem de entrada: ")

    try:
        # Carrega a imagem em tons de cinza
        img_original_cinza = carregar_imagem_tons_de_cinza(caminho_imagem_entrada)

        # Define o elemento estruturante (kernel)
       
        kernel = np.ones((5,5), np.uint8) # Kernel retangular 5x5

        # Aplica as operações morfológicas
        img_erodida = erosao_tons_de_cinza(img_original_cinza, kernel)
        img_dilatada = dilatacao_tons_de_cinza(img_original_cinza, kernel)
        img_gradiente = gradiente_morfologico_tons_de_cinza(img_original_cinza, kernel)

        # Exibe as imagens
        imagens_para_exibir = [img_original_cinza, img_erodida, img_dilatada, img_gradiente]
        titulos_para_exibir = ['Original Cinza', 'Erosão', 'Dilatação', 'Gradiente Morfológico']

        exibir_imagens(imagens_para_exibir, titulos_para_exibir)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")