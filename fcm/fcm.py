import numpy as np
import skfuzzy as fuzz
import cv2
import matplotlib.pyplot as plt

# Загрузим изображение и преобразуем его в оттенки серого
image = cv2.imread('sample_image_4.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (1024, 1024))  # Изменяем размер изображения для простоты обработки
img_flat = image.flatten()  # Развернем изображение в одномерный массив

# Настройки FCM
n_clusters = 2  # Количество кластеров
m = 2.0  # Степень "нечеткости" для FCM

# Нормализуем значения пикселей (обязательное условие для FCM)
img_norm = img_flat / 255.0
data = np.vstack([img_norm, img_norm])  # Создаем набор данных для кластеризации

# Применяем FCM к данным
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, n_clusters, m, error=0.005, maxiter=1000, init=None)

# Определим принадлежность пикселя к кластеру (используем максимальное значение принадлежности)
cluster_membership = np.argmax(u, axis=0)
segmented_img = cluster_membership.reshape(image.shape)

# Визуализируем результат
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(image)#, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Сегментированное изображение (FCM)')
plt.imshow(segmented_img, cmap='magma')
plt.axis('off')

plt.show()