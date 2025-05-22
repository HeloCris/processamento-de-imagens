# Filtro Gradiente
#  Helorrayne Cristine e Wanderson Almeida
# Processamento de Imagens

from PIL import Image
import numpy as np
import os

# Reutiliza a função apply_convolution definida anteriormente
# from your_previous_script import apply_convolution # Se estivesse em um arquivo separado

def apply_convolution(image_array, kernel):
    """
    Aplica uma convolução 2D a uma imagem.
    image_array: numpy array da imagem.
    kernel: numpy array do kernel de convolução.
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image_array.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Adiciona padding à imagem para lidar com as bordas
    # O padding com 'edge' replica os valores da borda
    padded_image = np.pad(image_array, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
    
    output_array = np.zeros_like(image_array, dtype=float) # Usar float para resultados intermediários

    # Aplica a convolução
    for y in range(image_height):
        for x in range(image_width):
            # Extrai a região da imagem correspondente ao kernel
            region = padded_image[y:y + kernel_height, x:x + kernel_width]
            # Realiza a operação de convolução (multiplicação elemento a elemento e soma)
            output_array[y, x] = np.sum(region * kernel)
            
    return output_array

def sobel_filter(image_path, negative_handling='clip_zero', output_dir='sobel_results'):
    """
    Aplica o filtro de Sobel em uma imagem.
    image_path: Caminho para a imagem de entrada.
    negative_handling: 'clip_zero' para atribuir 0 a negativos ou 'scale' para normalizar para [0, 255].
    output_dir: Diretório para salvar os resultados.
    """
    try:
        img = Image.open(image_path).convert('L')  # Converte para tons de cinza
    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada em {image_path}")
        return
    
    img_array = np.array(img)

    # Máscaras de Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    print(f"Aplicando filtro de Sobel com tratamento de negativos: '{negative_handling}'...")

    # Aplica convolução para Gx e Gy
    Gx = apply_convolution(img_array, sobel_x)
    Gy = apply_convolution(img_array, sobel_y)

    # Calcula a magnitude do gradiente
    gradient_magnitude = np.sqrt(Gx**2 + Gy**2)

    # Lidar com valores negativos (aplicado à magnitude do gradiente, que já é não-negativa)
    # No caso de Sobel, a magnitude já é positiva.
    # No entanto, se Gx ou Gy tivessem sido processados individualmente com 'clip_zero'
    # antes do cálculo da magnitude, o resultado final seria diferente.
    # Para Sobel, o tratamento 'clip_zero' na magnitude não faz sentido, pois ela é sqrt(x^2+y^2).
    # O 'scale' é o mais relevante aqui para mapear a magnitude para 0-255.
    
    if negative_handling == 'clip_zero':
        # Para a magnitude, clip_zero na verdade apenas garante que não exceda 255 (se houver).
        # A magnitude já é não-negativa.
        final_image_array = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        suffix = '_clip0' # Manter o sufixo para consistência, embora menos impactante aqui
    elif negative_handling == 'scale':
        # Escala para o intervalo [0, 255]
        min_val = gradient_magnitude.min()
        max_val = gradient_magnitude.max()
        if max_val == min_val: # Evita divisão por zero
            final_image_array = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        else:
            final_image_array = ((gradient_magnitude - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        suffix = '_scale'
    else:
        print("Opção de tratamento de negativos inválida. Use 'clip_zero' ou 'scale'.")
        return

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path).split('.')[0]
    output_filename = os.path.join(output_dir, f"{base_name}_sobel{suffix}.png")
    
    result_img = Image.fromarray(final_image_array)
    result_img.save(output_filename)
    print(f"Resultado salvo em: {output_filename}")

# --- Testando o Filtro Sobel ---
if __name__ == "__main__":
    # Garanta que a imagem de teste exista
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        img_size = (200, 200)
        img_data = np.zeros(img_size, dtype=np.uint8)
        img_data[50:150, 50:150] = 255
        test_img = Image.fromarray(img_data)
        test_img.save(test_image_path)
        print(f"Imagem de teste criada: {test_image_path}")

    print("\n--- Testes do Filtro Sobel ---")
    sobel_filter(test_image_path, negative_handling='clip_zero')
    sobel_filter(test_image_path, negative_handling='scale')