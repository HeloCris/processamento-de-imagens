# Filtro Laplaciano
# Helorrayne Cristine e Wanderson Almeida
# Processamento de Imagens

from PIL import Image
import numpy as np
import os

def apply_convolution(image_array, kernel):
    """
    Aplica a convolução de um kernel em uma imagem.
    image_array: array numpy da imagem (tons de cinza).
    kernel: array numpy da máscara (kernel).
    Retorna o array da imagem resultante após a convolução.
    """
    image_height, image_width = image_array.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calcula o padding necessário
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Adiciona padding zero à imagem
    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output_image = np.zeros_like(image_array, dtype=float) # Usamos float para permitir valores negativos

    for i in range(image_height):
        for j in range(image_width):
            # Extrai a região da imagem para convolução
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Realiza a convolução (produto Hadamard e soma)
            output_image[i, j] = np.sum(region * kernel)
            
    return output_image

def laplacian_filter(image_path, mask_type, negative_handling='clip_zero', output_dir='laplacian_results'):
    """
    Aplica o filtro Laplaciano em uma imagem.
    image_path: Caminho para a imagem de entrada.
    mask_type: Tipo de máscara Laplaciana (1, 2, 3, 4).
    negative_handling: 'clip_zero' para atribuir 0 a negativos ou 'scale' para normalizar para [0, 255].
    output_dir: Diretório para salvar os resultados.
    """
    try:
        img = Image.open(image_path).convert('L')  # Converte para tons de cinza
    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada em {image_path}")
        return
    
    img_array = np.array(img)

    laplacian_masks = {
        1: np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        2: np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
        3: np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        4: np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    }

    if mask_type not in laplacian_masks:
        print("Máscara Laplaciana inválida. Escolha 1, 2, 3 ou 4.")
        return

    kernel = laplacian_masks[mask_type]

    print(f"Aplicando filtro Laplaciano com Máscara {mask_type} e tratamento de negativos: '{negative_handling}'...")
    filtered_image_array = apply_convolution(img_array, kernel)

    # Lidar com valores negativos
    if negative_handling == 'clip_zero':
        # Atribui 0 a valores negativos
        final_image_array = np.maximum(filtered_image_array, 0)
        # Normaliza para 0-255 (se houver valores maiores que 255)
        final_image_array = np.clip(final_image_array, 0, 255).astype(np.uint8)
        suffix = '_clip0'
    elif negative_handling == 'scale':
        # Escala para o intervalo [0, 255]
        min_val = filtered_image_array.min()
        max_val = filtered_image_array.max()
        if max_val == min_val: # Evita divisão por zero
            final_image_array = np.zeros_like(filtered_image_array, dtype=np.uint8)
        else:
            final_image_array = ((filtered_image_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        suffix = '_scale'
    else:
        print("Opção de tratamento de negativos inválida. Use 'clip_zero' ou 'scale'.")
        return

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path).split('.')[0]
    output_filename = os.path.join(output_dir, f"{base_name}_laplacian_mask{mask_type}{suffix}.png")
    
    result_img = Image.fromarray(final_image_array)
    result_img.save(output_filename)
    print(f"Resultado salvo em: {output_filename}")

# --- Testando o Filtro Laplaciano ---
if __name__ == "__main__":
    # Crie uma imagem de teste ou use uma existente
    # Por exemplo, uma imagem preta com um quadrado branco no centro
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        img_size = (200, 200)
        img_data = np.zeros(img_size, dtype=np.uint8)
        # Adiciona um quadrado branco no centro
        img_data[50:150, 50:150] = 255
        test_img = Image.fromarray(img_data)
        test_img.save(test_image_path)
        print(f"Imagem de teste criada: {test_image_path}")

    # Testar com as 4 máscaras e as 2 opções de tratamento de negativos
    print("\n--- Testes do Filtro Laplaciano ---")
    for mask_id in range(1, 5):
        laplacian_filter(test_image_path, mask_id, negative_handling='clip_zero')
        laplacian_filter(test_image_path, mask_id, negative_handling='scale')