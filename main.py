import numpy as np
import matplotlib.pyplot as plt
import cv2

# Función para crear una matriz de Hopfield y entrenarla de manera simétrica


def entrenar_hopfield(imagen):
    num_pixeles = imagen.size
    hopfield_matrix = np.zeros((num_pixeles, num_pixeles))

    for i in range(num_pixeles):
        for j in range(num_pixeles):
            if i != j:
                hopfield_matrix[i, j] = imagen[i] * imagen[j]

    # Hacer que la matriz de pesos sea simétrica
    hopfield_matrix = 0.5 * (hopfield_matrix + hopfield_matrix.T)

    return hopfield_matrix

# Función para reconstruir la imagen dañada utilizando la red Hopfield


def reconstruir_imagen(imagen_danada, hopfield_matrix, max_iteraciones):
    imagen_reconstruida = imagen_danada.copy()
    for _ in range(max_iteraciones):
        for i in range(imagen_reconstruida.size):
            s = np.dot(hopfield_matrix[i], imagen_reconstruida)
            imagen_reconstruida[i] = 1 if s > 0 else -1

    return imagen_reconstruida


# Paso 1: Cargar la imagen original en color
imagen_original = cv2.imread('data/figure.png')

# Paso 2: Convertir la imagen original a escala de grises y redimensionar
imagen_original_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
imagen_original_gris = cv2.resize(imagen_original_gris, (100, 100))

# Paso 3: Preparar la imagen original y entrenar la red Hopfield
imagen_plana = (imagen_original_gris.flatten() > 128) * 2 - \
    1  # Convertir a patrones binarios (+1 y -1)
hopfield_matrix = entrenar_hopfield(imagen_plana)

# Paso 4: Cargar la imagen dañada en color
imagen_danada = cv2.imread('train/figure_test.png')

# Paso 5: Convertir la imagen dañada a escala de grises y redimensionar
imagen_danada_gris = cv2.cvtColor(imagen_danada, cv2.COLOR_BGR2GRAY)
imagen_danada_gris = cv2.resize(imagen_danada_gris, (100, 100))
imagen_danada_binaria = (imagen_danada_gris > 128) * 2 - 1

# Paso 6: Reconstrucción de la imagen dañada
max_iteraciones = 20
imagen_reconstruida = reconstruir_imagen(
    imagen_danada_binaria.flatten(), hopfield_matrix, max_iteraciones)

# Paso 7: Convertir la imagen reconstruida a escala de grises y redimensionar
imagen_reconstruida_gris = (imagen_reconstruida.reshape(100, 100) + 1) // 2

# Paso 8: Visualización de la imagen original, dañada y reconstruida
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Imagen Original")
plt.imshow(imagen_original_gris, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Imagen Dañada")
plt.imshow(imagen_danada_gris, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Imagen Reconstruida")
plt.imshow(imagen_reconstruida_gris, cmap='gray')

plt.show()
