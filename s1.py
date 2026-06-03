# -----------------------------------------------------------------------------
# PROCESADOR DE IMÁGENES DIGITALES
# -----------------------------------------------------------------------------

# --- Importación de Librerías ---
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
from PIL import Image
import cv2
import numpy as np
import os
import colorsys
import math
from scipy.stats import skew
from scipy import ndimage
import random
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def log2(x):
    if x <= 0:
        return 0
    return math.log2(x)

# --- 1. Funciones de Procesamiento Básicas ---
def separar_canales(imagen_pil):
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def separar_canales_rgb(imagen_pil):
    """Separa canales RGB de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    return imagen_pil.split()

def separar_canales_cmy(imagen_pil):
    """Separa canales CMY de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    r, g, b = imagen_pil.split()
    r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)
    
    c = 255 - r_arr
    m = 255 - g_arr
    y = 255 - b_arr
    
    return Image.fromarray(c), Image.fromarray(m), Image.fromarray(y)

def separar_canales_hsv(imagen_pil):
    """Separa canales HSV de una imagen."""
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    return Image.fromarray(h), Image.fromarray(s), Image.fromarray(v)

def dibujar_canales_rgb(fig, r, g, b):
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.imshow(r, cmap='Reds')
    ax1.set_title("Componente R")
    ax1.axis('off')
    ax2.imshow(g, cmap='Greens')
    ax2.set_title("Componente G")
    ax2.axis('off')
    ax3.imshow(b, cmap='Blues')
    ax3.set_title("Componente B")
    ax3.axis('off')
    fig.tight_layout()

def dibujar_canales_cmy(fig, c, m, y):
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    ax1.imshow(c, cmap='cool')
    ax1.set_title("Componente C")
    ax1.axis('off')
    ax2.imshow(m, cmap='magma')
    ax2.set_title("Componente M")
    ax2.axis('off')
    ax3.imshow(y, cmap='YlOrBr')
    ax3.set_title("Componente Y")
    ax3.axis('off')
    fig.tight_layout()

def dibujar_canales_hsv(fig, h, s, v):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv')
    ax1.set_title("Componente H (Hue)")
    ax1.axis('off')
    ax2.imshow(s, cmap='plasma')
    ax2.set_title("Componente S (Saturation)")
    ax2.axis('off')
    ax3.imshow(v, cmap='gray')
    ax3.set_title("Componente V (Value)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo YIQ ---
def separar_canales_yiq(imagen_pil):
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
    rgb_array = np.array(imagen_pil) / 255.0
    yiq_array = np.zeros_like(rgb_array)
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            yiq_array[i, j] = colorsys.rgb_to_yiq(*rgb_array[i, j])
    y = (yiq_array[:, :, 0] * 255).astype(np.uint8)
    i_norm = ((yiq_array[:, :, 1] - yiq_array[:, :, 1].min()) / (yiq_array[:, :, 1].max() - yiq_array[:, :, 1].min() + 1e-6) * 255).astype(np.uint8)
    q_norm = ((yiq_array[:, :, 2] - yiq_array[:, :, 2].min()) / (yiq_array[:, :, 2].max() - yiq_array[:, :, 2].min() + 1e-6) * 255).astype(np.uint8)
    return y, i_norm, q_norm

def dibujar_canales_yiq(fig, y, i, q):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(y, cmap='gray')
    ax1.set_title("Componente Y (Luminancia)")
    ax1.axis('off')
    ax2.imshow(i, cmap='RdBu')
    ax2.set_title("Componente I (Crominancia)")
    ax2.axis('off')
    ax3.imshow(q, cmap='PiYG')
    ax3.set_title("Componente Q (Crominancia)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para el Modelo HSI ---
def separar_canales_hsi(imagen_pil):
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.array(imagen_pil) / 255.0
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        i = (r + g + b) / 3.0
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = 1 - (3 / (r + g + b + 1e-6)) * min_rgb
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b) * (g - b))
        theta = np.arccos(np.clip(num / (den + 1e-6), -1, 1))
        h = np.copy(theta)
        h[b > g] = (2 * np.pi) - h[b > g]
        h = h / (2 * np.pi)
    h_out = (h * 255).astype(np.uint8)
    s_out = (s * 255).astype(np.uint8)
    i_out = (i * 255).astype(np.uint8)
    return h_out, s_out, i_out

def dibujar_canales_hsi(fig, h, s, i):
    fig.clear()
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    ax1.imshow(h, cmap='hsv')
    ax1.set_title("Componente H (Hue)")
    ax1.axis('off')
    ax2.imshow(s, cmap='plasma')
    ax2.set_title("Componente S (Saturation)")
    ax2.axis('off')
    ax3.imshow(i, cmap='gray')
    ax3.set_title("Componente I (Intensity)")
    ax3.axis('off')
    fig.tight_layout()

# --- Funciones para Escala de Grises y Binarización ---
def convertir_a_grises_en_ax(imagen_pil, ax):
    ax.clear()
    imagen_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris_cv = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2GRAY)
    ax.imshow(gris_cv, cmap='gray')
    ax.set_title("Imagen en Escala de Grises")
    ax.axis('off')
    return gris_cv

def binarizar_imagen_en_ax(imagen_gris_cv, ax, umbral=128):
    ax.clear()
    _, binaria = cv2.threshold(imagen_gris_cv, umbral, 255, cv2.THRESH_BINARY)
    ax.imshow(binaria, cmap='gray')
    ax.set_title(f"Imagen binarizada (umbral = {umbral})")
    ax.axis('off')
    return binaria

def calcular_histograma_rgb(imagen_pil):
    imagen_np = np.array(imagen_pil)
    if len(imagen_np.shape) == 3:
        hist_r = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([imagen_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([imagen_np], [2], None, [256], [0, 256])
        return hist_r, hist_g, hist_b
    else:
        hist = cv2.calcHist([imagen_np], [0], None, [256], [0, 256])
        return hist

def dibujar_histograma_imagen_original(fig, imagen_pil):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_pil)
    ax_img.set_title("Imagen Original")
    ax_img.axis('off')
    hist_r, hist_g, hist_b = calcular_histograma_rgb(imagen_pil)
    ax_hist.plot(hist_r, color='red', alpha=0.7, label='Canal R')
    ax_hist.plot(hist_g, color='green', alpha=0.7, label='Canal G')
    ax_hist.plot(hist_b, color='blue', alpha=0.7, label='Canal B')
    ax_hist.set_title("Histograma RGB")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()

# --- Funciones para el Dibujo de Histogramas ---
def dibujar_histograma_canales_rgb(fig, r, g, b):
    fig.clear()
    gs = fig.add_gridspec(2, 3, hspace=0.3)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_r.imshow(r, cmap='Reds')
    ax_r.set_title("Canal R")
    ax_r.axis('off')
    ax_g.imshow(g, cmap='Greens')
    ax_g.set_title("Canal G")
    ax_g.axis('off')
    ax_b.imshow(b, cmap='Blues')
    ax_b.set_title("Canal B")
    ax_b.axis('off')
    ax_hist_r = fig.add_subplot(gs[1, 0])
    ax_hist_g = fig.add_subplot(gs[1, 1])
    ax_hist_b = fig.add_subplot(gs[1, 2])
    hist_r = cv2.calcHist([np.array(r)], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([np.array(g)], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([np.array(b)], [0], None, [256], [0, 256])
    ax_hist_r.plot(hist_r, color='red')
    ax_hist_r.set_title("Histograma R")
    ax_hist_r.grid(True, alpha=0.3)
    ax_hist_g.plot(hist_g, color='green')
    ax_hist_g.set_title("Histograma G")
    ax_hist_g.grid(True, alpha=0.3)
    ax_hist_b.plot(hist_b, color='blue')
    ax_hist_b.set_title("Histograma B")
    ax_hist_b.grid(True, alpha=0.3)

def dibujar_histograma_escala_grises(fig, imagen_gris_cv):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_gris_cv, cmap='gray')
    ax_img.set_title("Imagen en Escala de Grises")
    ax_img.axis('off')
    hist = cv2.calcHist([imagen_gris_cv], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color='black')
    ax_hist.set_title("Histograma Escala de Grises")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()

def dibujar_histograma_binaria(fig, imagen_binaria_cv):
    fig.clear()
    ax_img, ax_hist = fig.subplots(1, 2)
    ax_img.imshow(imagen_binaria_cv, cmap='gray')
    ax_img.set_title("Imagen Binarizada")
    ax_img.axis('off')
    hist = cv2.calcHist([imagen_binaria_cv], [0], None, [256], [0, 256])
    ax_hist.bar([0, 255], [hist[0].item(), hist[255].item()], color='black', width=50)
    ax_hist.set_title("Histograma Binario")
    ax_hist.set_xlabel("Intensidad de Pixel")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_xlim(-25, 280)
    ax_hist.grid(True, alpha=0.3)
    fig.tight_layout()

def calcular_caracteristicas_estadisticas(imagen_pil):
    imagen_rgb = np.array(imagen_pil)
    if imagen_pil.mode != 'RGB':
        imagen_pil = imagen_pil.convert('RGB')
        imagen_rgb = np.array(imagen_pil)
    resultados = {}
    for i, canal in enumerate(['Red', 'Green', 'Blue']):
        datos = imagen_rgb[:, :, i].flatten()
        histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
        if histograma.sum() == 0:
            continue
        prob = histograma / histograma.sum()
        energia = np.sum(prob ** 2)
        entropia = -np.sum([p * log2(p) for p in prob if p > 0])
        asimetria = skew(datos)
        media = np.mean(datos)
        varianza = np.var(datos)
        resultados[canal] = {
            'Energía': energia,
            'Entropía': entropia,
            'Asimetría': asimetria,
            'Media': media,
            'Varianza': varianza
        }
    return resultados

def calcular_caracteristicas_escala_grises(imagen_gris_cv):
    datos = imagen_gris_cv.flatten()
    histograma, _ = np.histogram(datos, bins=256, range=(0, 256))
    if histograma.sum() == 0:
        return None
    prob = histograma / histograma.sum()
    energia = np.sum(prob ** 2)
    entropia = -np.sum([p * log2(p) for p in prob if p > 0])
    asimetria = skew(datos)
    media = np.mean(datos)
    varianza = np.var(datos)
    return {
        'Energía': energia,
        'Entropía': entropia,
        'Asimetría': asimetria,
        'Media': media,
        'Varianza': varianza
    }

# --- 2. FUNCIONES DE MORFOLOGÍA MATEMÁTICA ---

def crear_kernel(tamano, forma='rect'):
    """Crea un elemento estructurante (kernel)."""
    if forma == 'rect':
        return np.ones((tamano, tamano), np.uint8)
    elif forma == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
    elif forma == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamano, tamano))
    else:
        return np.ones((tamano, tamano), np.uint8)

# Operaciones básicas
def erosion_morfologica(imagen, kernel, iteraciones=1):
    return cv2.erode(imagen, kernel, iterations=iteraciones)

def dilatacion_morfologica(imagen, kernel, iteraciones=1):
    return cv2.dilate(imagen, kernel, iterations=iteraciones)

def apertura_tradicional(imagen, kernel, iteraciones=1):
    erosionada = cv2.erode(imagen, kernel, iterations=iteraciones)
    return cv2.dilate(erosionada, kernel, iterations=iteraciones)

def apertura_opencv(imagen, kernel, iteraciones=1):
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=iteraciones)

def cierre_tradicional(imagen, kernel, iteraciones=1):
    dilatada = cv2.dilate(imagen, kernel, iterations=iteraciones)
    return cv2.erode(dilatada, kernel, iterations=iteraciones)

def cierre_opencv(imagen, kernel, iteraciones=1):
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)

# Operaciones avanzadas para Morfología Binaria
def obtener_frontera(imagen, kernel):
    """Frontera: Original - Erosión."""
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(imagen, erosionada)

def adelgazamiento(imagen, kernel, iteraciones=1):
    """Adelgazamiento (Thinning) morfológico."""
    resultado = imagen.copy()
    for _ in range(iteraciones):
        erosionada = cv2.erode(resultado, kernel, iterations=1)
        temp = cv2.dilate(erosionada, kernel, iterations=1)
        temp = cv2.subtract(resultado, temp)
        resultado = cv2.bitwise_or(erosionada, temp)
    return resultado

def hit_or_miss(imagen, kernel1, kernel2):
    """Transformada Hit-or-Miss."""
    hit = cv2.erode(imagen, kernel1, iterations=1)
    miss = cv2.erode(cv2.bitwise_not(imagen), kernel2, iterations=1)
    return cv2.bitwise_and(hit, miss)

def esqueleto_morfologico(imagen, kernel):
    """Calcula el esqueleto morfológico de una imagen binaria."""
    esqueleto = np.zeros(imagen.shape, dtype=np.uint8)
    temp = imagen.copy()
    
    while True:
        erosionada = cv2.erode(temp, kernel, iterations=1)
        temp_abierta = cv2.morphologyEx(erosionada, cv2.MORPH_OPEN, kernel)
        subset = cv2.subtract(erosionada, temp_abierta)
        esqueleto = cv2.bitwise_or(esqueleto, subset)
        temp = erosionada.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    return esqueleto

# Operaciones para Morfología en Laticces (Escala de Grises)
def gradiente_morfologico_simetrico(imagen, kernel):
    """Gradiente simétrico: Dilatación - Erosión."""
    dilatada = cv2.dilate(imagen, kernel, iterations=1)
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(dilatada, erosionada)

def gradiente_por_erosion(imagen, kernel):
    """Gradiente por erosión: Original - Erosión."""
    erosionada = cv2.erode(imagen, kernel, iterations=1)
    return cv2.subtract(imagen, erosionada)

def gradiente_por_dilatacion(imagen, kernel):
    """Gradiente por dilatación: Dilatación - Original."""
    dilatada = cv2.dilate(imagen, kernel, iterations=1)
    return cv2.subtract(dilatada, imagen)

def top_hat(imagen, kernel):
    """Top Hat: Original - Apertura (detecta puntos brillantes)."""
    return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)

def bottom_hat(imagen, kernel):
    """Bottom Hat: Cierre - Original (detecta puntos oscuros)."""
    return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)

def filtro_suavizado_apertura(imagen, kernel):
    """Filtro de suavizado mediante apertura."""
    return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)

def filtro_suavizado_cierre(imagen, kernel):
    """Filtro de suavizado mediante cierre."""
    return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)

def filtro_suavizado_apertura_cierre(imagen, kernel):
    """Suavizado: Apertura seguida de Cierre."""
    apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

def filtro_suavizado_cierre_apertura(imagen, kernel):
    """Suavizado: Cierre seguida de Apertura."""
    cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(cierre, cv2.MORPH_OPEN, kernel)


# --- NUEVAS FUNCIONES DE FILTRADO AVANZADO (Manuales) ---

def filtro_contraharmonico(img, k_size, Q):
    """
    Filtro de Media Contraharmónica.
    Q > 0: Elimina ruido pimienta (puntos negros).
    Q < 0: Elimina ruido sal (puntos blancos).
    """
    img = img.astype(np.float32)
    epsilon = 1e-5  # Para evitar división por cero

    # Calcular numerador y denominador con convolución
    # Fórmula: Sum(I^(Q+1)) / Sum(I^Q)
    numerador = cv2.filter2D(np.power(img, Q + 1), -1, np.ones((k_size, k_size)))
    denominador = cv2.filter2D(np.power(img, Q), -1, np.ones((k_size, k_size)))

    resultado = numerador / (denominador + epsilon)
    resultado = np.clip(resultado, 0, 255)  # Asegurar rango 0-255
    return resultado.astype(np.uint8)


def filtro_mediana_adaptativo(img, s_max=7):
    """
    Filtro de Mediana Adaptativo (RAMP).
    Versión corregida: Convierte a int para evitar overflow en restas negativas.
    """
    filas, cols = img.shape
    img_out = img.copy()

    pad_max = s_max // 2
    img_padded = cv2.copyMakeBorder(img, pad_max, pad_max, pad_max, pad_max, cv2.BORDER_REPLICATE)

    for i in range(pad_max, filas + pad_max):
        for j in range(pad_max, cols + pad_max):
            s_curr = 3
            while s_curr <= s_max:
                pad = s_curr // 2
                ventana = img_padded[i - pad:i + pad + 1, j - pad:j + pad + 1]

                # --- CAMBIO IMPORTANTE: Convertir a int() ---
                # Esto permite que las restas den negativo sin error de overflow
                z_min = int(np.min(ventana))
                z_max = int(np.max(ventana))
                z_med = int(np.median(ventana))
                z_xy = int(img_padded[i, j])
                # --------------------------------------------

                # Nivel A: ¿La mediana es ruido?
                a1 = z_med - z_min
                a2 = z_med - z_max

                if a1 > 0 and a2 < 0:
                    # Nivel B: ¿El pixel central es ruido?
                    b1 = z_xy - z_min
                    b2 = z_xy - z_max
                    if b1 > 0 and b2 < 0:
                        img_out[i - pad_max, j - pad_max] = z_xy  # No es ruido
                    else:
                        img_out[i - pad_max, j - pad_max] = z_med  # Es ruido
                    break  # Salir del while
                else:
                    s_curr += 2  # Aumentar ventana
                    if s_curr > s_max:
                        img_out[i - pad_max, j - pad_max] = z_med  # Fallback

    return img_out

def filtro_mediana_ponderada(img, k_size=3, peso_central=5):
    """
    Mediana Ponderada: El pixel central se repite 'peso_central' veces
    antes de calcular la mediana, dándole más importancia.
    """
    from scipy.ndimage import generic_filter

    def funcion_mediana_ponderada(buffer):
        # El buffer es la ventana aplanada
        centro_idx = len(buffer) // 2
        pixel_central = buffer[centro_idx]

        # Crear lista con vecinos + el central repetido
        vecinos = list(buffer)
        # Añadir el central peso-1 veces extra (porque ya está una vez)
        vecinos.extend([pixel_central] * (peso_central - 1))

        return np.median(vecinos)

    return generic_filter(img, funcion_mediana_ponderada, size=k_size)

# ---OPERACIONES ARITMÉTICAS Y LÓGICAS ---

def sumar_escalar(imagen, escalar):
    """Suma un escalar a la imagen."""
    matriz_escalar = np.full(imagen.shape, int(escalar), dtype=np.uint8)
    return cv2.add(imagen, matriz_escalar)

def restar_escalar(imagen, escalar):
    """Resta un escalar de la imagen."""
    matriz_escalar = np.full(imagen.shape, int(escalar), dtype=np.uint8)
    return cv2.subtract(imagen, matriz_escalar)

def multiplicar_escalar(imagen, escalar):
    """Multiplica la imagen por un escalar."""
    return cv2.multiply(imagen, float(escalar))

def sumar_imagenes(img1, img2):
    """Suma dos imágenes (ajustando tamaños)."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.add(img1_resized, img2_resized)

def restar_imagenes(img1, img2):
    """Resta dos imágenes (ajustando tamaños)."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.subtract(img1_resized, img2_resized)

def operacion_and(img1, img2):
    """AND lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_and(img1_resized, img2_resized)

def operacion_or(img1, img2):
    """OR lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_or(img1_resized, img2_resized)

def operacion_xor(img1, img2):
    """XOR lógico entre dos imágenes."""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (w, h))
    img2_resized = cv2.resize(img2, (w, h))
    return cv2.bitwise_xor(img1_resized, img2_resized)

def operacion_not(imagen):
    """NOT lógico (inversión)."""
    return cv2.bitwise_not(imagen)

def operacion_relacional(imagen, umbral, operador):
    """Aplica operación relacional (>, <, ==)."""
    if operador == '>':
        mask = imagen > umbral
    elif operador == '<':
        mask = imagen < umbral
    elif operador == '==':
        mask = imagen == umbral
    else:
        return imagen
    return np.uint8(mask) * 255

def agregar_ruido_sal_pimienta(imagen, porcentaje, modo='mixto'):
    """Agrega ruido sal y pimienta."""
    img_ruidosa = imagen.copy()
    alto, ancho = img_ruidosa.shape
    num_pixeles = int((porcentaje / 100) * alto * ancho)

    for _ in range(num_pixeles):
        y = random.randint(0, alto - 1)
        x = random.randint(0, ancho - 1)
        
        if modo == 'mixto':
            if random.random() < 0.5:
                img_ruidosa[y, x] = 255
            else:
                img_ruidosa[y, x] = 0
        elif modo == 'sal':
            img_ruidosa[y, x] = 255
        elif modo == 'pimienta':
            img_ruidosa[y, x] = 0
                
    return img_ruidosa

def agregar_ruido_gaussiano(imagen, media=0, sigma=25):
    """Agrega ruido gaussiano."""
    img_ruidosa = imagen.copy().astype(np.float64)
    ruido = np.random.normal(media, sigma, imagen.shape)
    img_ruidosa += ruido
    img_ruidosa = np.clip(img_ruidosa, 0, 255)
    return img_ruidosa.astype(np.uint8)

def etiquetar_componentes(imagen, vecindad=8):
    """Etiqueta componentes conexas."""
    if vecindad == 4:
        estructura = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    else:
        estructura = np.ones((3, 3), dtype=int)
    
    etiquetas, num_objetos = ndimage.label(imagen, structure=estructura)
    return etiquetas, num_objetos


# --- MÓDULO DE CLASIFICACIÓN ---

def extraer_caracteristicas_forma(imagen_binaria):
    """Extrae características de una forma binaria."""
    # 1. Área
    area = cv2.countNonZero(imagen_binaria)
    
    # 2. Perímetro (usando frontera)
    kernel = np.ones((3,3), np.uint8)
    frontera = obtener_frontera(imagen_binaria, kernel)
    perimetro = cv2.countNonZero(frontera)
    
    # 3. Compacidad
    if area > 0:
        compacidad = (perimetro ** 2) / (4 * np.pi * area)
    else:
        compacidad = 0
    
    # 4. Relación de aspecto
    contours, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        relacion_aspecto = float(w) / h if h > 0 else 0
    else:
        relacion_aspecto = 0
    
    # 5. Momentos de Hu (invariantes)
    momentos = cv2.moments(imagen_binaria)
    hu_momentos = cv2.HuMoments(momentos).flatten()
    
    # 6. Características estadísticas
    stats = calcular_caracteristicas_escala_grises(imagen_binaria)
    
    return {
        'area': area,
        'perimetro': perimetro,
        'compacidad': compacidad,
        'relacion_aspecto': relacion_aspecto,
        'hu_momentos': hu_momentos,
        'energia': stats['Energía'] if stats else 0,
        'entropia': stats['Entropía'] if stats else 0
    }

def crear_vector_caracteristicas(caracteristicas):
    """Crea vector numérico de características."""
    return np.array([
        caracteristicas['compacidad'],
        caracteristicas['relacion_aspecto'],
        caracteristicas['energia'],
        caracteristicas['entropia'],
        caracteristicas['hu_momentos'][0],
        caracteristicas['hu_momentos'][1]
    ])

def distancia_euclidiana(v1, v2):
    """Calcula distancia euclidiana entre dos vectores."""
    return np.linalg.norm(v1 - v2)

def distancia_manhattan(v1, v2):
    """Calcula distancia de Manhattan."""
    return np.sum(np.abs(v1 - v2))

def clasificar_por_distancia(vector_prueba, prototipos, metrica='euclidiana'):
    """Clasifica una imagen comparando con prototipos."""
    distancias = {}
    
    for clase, prototipo in prototipos.items():
        if metrica == 'euclidiana':
            dist = distancia_euclidiana(vector_prueba, prototipo)
        elif metrica == 'manhattan':
            dist = distancia_manhattan(vector_prueba, prototipo)
        distancias[clase] = dist
    
    # Retorna la clase con menor distancia
    clase_predicha = min(distancias, key=distancias.get)
    return clase_predicha, distancias


def generar_dataset_figuras():
    """Genera dataset de figuras geométricas."""
    # Círculos
    for i in range(10):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), random.randint(40, 80), 255, -1)
        cv2.imwrite(f'dataset/circulo_{i}.png', img)
    
    # Cuadrados
    for i in range(10):
        img = np.zeros((200, 200), dtype=np.uint8)
        size = random.randint(60, 100)
        x1 = 100 - size//2
        y1 = 100 - size//2
        cv2.rectangle(img, (x1, y1), (x1+size, y1+size), 255, -1)
        cv2.imwrite(f'dataset/cuadrado_{i}.png', img)
    
    # Triángulos
    for i in range(10):
        img = np.zeros((200, 200), dtype=np.uint8)
        pts = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
        cv2.fillPoly(img, [pts], 255)
        cv2.imwrite(f'dataset/triangulo_{i}.png', img)


# --- FILTRADO Y BORDES ---

def filtro_moda(imagen, kernel_size=3):
    """Implementación del filtro de Moda (basado en el PDF)."""
    import scipy.stats as st  # Alias local por seguridad
    salida = np.copy(imagen)
    h, w = imagen.shape
    pad = kernel_size // 2
    imagen_padded = np.pad(imagen, pad, mode='constant', constant_values=0)

    # Nota: Este proceso es lento pixel por pixel, pero es lo que pide la teoría
    for i in range(h):
        for j in range(w):
            window = imagen_padded[i:i + kernel_size, j:j + kernel_size].flatten()
            moda_result = st.mode(window, axis=None, keepdims=False)
            # Manejo de versiones de scipy
            if isinstance(moda_result.mode, np.ndarray):
                salida[i, j] = moda_result.mode[0]  # Versiones nuevas
            else:
                salida[i, j] = moda_result.mode  # Versiones viejas
    return salida


def filtro_maximo(imagen, kernel_size=3):
    """Filtro Máximo (equivalente a dilatación en escala de grises)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(imagen, kernel)


def filtro_minimo(imagen, kernel_size=3):
    """Filtro Mínimo (equivalente a erosión en escala de grises)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(imagen, kernel)


def filtro_promedio_pesado(imagen):
    """Filtro promediador con peso central (Kernel del PDF)."""
    kernel = np.array([[1, 1, 1],
                       [1, 5, 1],
                       [1, 1, 1]]) / 13
    return cv2.filter2D(imagen, -1, kernel)


def aplicar_detector_bordes(imagen, tipo):
    """Aplica diferentes detectores de bordes según el tipo."""
    img_float = imagen.astype(np.float64)

    if tipo == 'sobel':
        sx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        magnitud = cv2.magnitude(sx, sy)
        return cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'prewitt':
        kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        ix = cv2.filter2D(img_float, -1, kx)
        iy = cv2.filter2D(img_float, -1, ky)
        magnitud = cv2.magnitude(ix, iy)  # Aproximación de magnitud
        return cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'roberts':
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        ix = cv2.filter2D(img_float, -1, kx)
        iy = cv2.filter2D(img_float, -1, ky)
        magnitud = cv2.magnitude(ix, iy)
        return cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'laplaciano':
        lap = cv2.Laplacian(img_float, cv2.CV_64F)
        lap = np.absolute(lap)
        return cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'laplaciano8':  # Máscara de 8 vecinos del PDF
        k = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        lap = cv2.filter2D(img_float, -1, k)
        lap = np.absolute(lap)
        return cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'kirsch':
        # Definición de los 8 kernels de Kirsch (brújula)
        k1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
        k2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
        k3 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
        k4 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
        k5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
        k6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
        k7 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        k8 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
        kernels = [k1, k2, k3, k4, k5, k6, k7, k8]
        # Aplicar todos y tomar el máximo
        responses = [cv2.filter2D(img_float, -1, k) for k in kernels]
        magnitud = np.max(responses, axis=0)
        return cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    elif tipo == 'canny':
        # Canny requiere uint8 de entrada y devuelve binario
        return cv2.Canny(imagen, 100, 200)

    return imagen


# --- SEGMENTACIÓN Y BRILLO ---

def umbral_otsu_implementacion(imagen):
    # Otsu devuelve el umbral calculado y la imagen
    umbral, binarizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada, umbral


def umbral_media(imagen):
    umbral = np.mean(imagen)
    _, binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return binarizada, umbral


def umbral_kapur(imagen):
    # Implementación basada en el PDF [cite: 247-277]
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()
    total_pixeles = imagen.size
    probabilidades = hist / total_pixeles

    max_entropia = -1.0
    umbral_optimo = 0

    # Tablas de entropía acumulada para optimizar
    # (Implementación simplificada del concepto para velocidad)
    for t in range(1, 255):
        w0 = np.sum(probabilidades[:t])  # Probabilidad clase 0
        w1 = np.sum(probabilidades[t:])  # Probabilidad clase 1

        if w0 == 0 or w1 == 0: continue

        # Normalizar probabilidades para cada clase y calcular entropía
        # Se suma 1e-10 para evitar log(0)
        p0 = probabilidades[:t] / w0
        p1 = probabilidades[t:] / w1

        ent0 = -np.sum(p0 * np.log(p0 + 1e-10))
        ent1 = -np.sum(p1 * np.log(p1 + 1e-10))

        entropia_total = ent0 + ent1

        if entropia_total > max_entropia:
            max_entropia = entropia_total
            umbral_optimo = t

    _, binarizada = cv2.threshold(imagen, umbral_optimo, 255, cv2.THRESH_BINARY)
    return binarizada, umbral_optimo


def umbral_minimo_histograma(imagen):
    from scipy.signal import find_peaks
    hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten()

    # Suavizar un poco el histograma para evitar picos de ruido
    hist_smooth = ndimage.gaussian_filter1d(hist, sigma=2)

    # Encontrar picos con una distancia mínima
    picos, _ = find_peaks(hist_smooth, distance=20)

    if len(picos) >= 2:
        # Buscar el mínimo entre los dos primeros picos principales
        inicio, fin = picos[0], picos[1]
        # argmin devuelve índice relativo, sumamos 'inicio' para absoluto
        umbral = np.argmin(hist_smooth[inicio:fin]) + inicio
    else:
        # Fallback si no encuentra dos picos: usar la media
        umbral = int(np.mean(imagen))

    _, binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return binarizada, umbral


def umbral_banda(imagen, t1, t2):
    # [cite: 353-355]
    salida = np.zeros_like(imagen)
    # Píxeles dentro del rango a blanco, resto a negro
    mask = (imagen >= t1) & (imagen <= t2)
    salida[mask] = 255
    return salida


# --- FUNCIONES DE AJUSTE DE BRILLO (ECUALIZACIONES) ---
def ecualizacion_uniforme(imagen):
    return cv2.equalizeHist(imagen)


def ecualizacion_exponencial(imagen):
    img_norm = imagen / 255.0
    # Formula PDF: 255 * (1 - exp(-img/255)) ? Revisa formula exacta del PDF,
    # a veces alpha varía, usaremos la lógica estándar inversa
    alpha = 5.0  # Parámetro ajustable
    res = 255 * (1 - np.exp(-img_norm * alpha))
    # Normalizar para asegurar rango 0-255
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    return res.astype(np.uint8)


def ecualizacion_rayleigh(imagen):
    # Formula PDF: 255 * sqrt(imagen / 255) (Simplificación común)
    # Una implementación más estricta usa distribución acumulada
    img_norm = imagen / 255.0
    alpha = 0.4  # Coeficiente de distribución
    # P(r) = r/a^2 * exp(-r^2 / 2a^2) -> Mapeo basado en histograma
    # Usaremos la aproximación del PDF [cite: 393]
    res = 255 * np.sqrt(img_norm)  # Esta es la formula simple que da el PDF
    return res.astype(np.uint8)


def ecualizacion_hipercubica(imagen):
    # [cite: 398-401]
    img_norm = imagen / 255.0
    res = 255 * (img_norm ** 3)  # El PDF dice hipercúbica (potencia 3 o 4)
    return res.astype(np.uint8)


def ecualizacion_log_hiperbolica(imagen):
    # [cite: 406]
    # log(1 + imagen) / log(1 + 255)
    img_float = imagen.astype(np.float32)
    res = 255 * (np.log1p(img_float) / np.log1p(255.0))
    return res.astype(np.uint8)

def funcion_potencia(imagen, exponente=2):
    """Función potencia: s = c * r^γ"""
    img_norm = imagen / 255.0
    res = np.power(img_norm, exponente)
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    return res.astype(np.uint8)

def desplazamiento_histograma(imagen, valor):
    """Desplaza el histograma sumando/restando un valor constante"""
    if valor >= 0:
        return cv2.add(imagen, valor)
    else:
        return cv2.subtract(imagen, abs(valor))

def contraccion_histograma(imagen, factor=0.5):
    """Reduce el rango dinámico (reduce contraste)"""
    media = np.mean(imagen)
    resultado = ((imagen - media) * factor) + media
    return np.clip(resultado, 0, 255).astype(np.uint8)


def multiumbralizado(imagen, umbrales=[85, 170]):
    """
    Segmentación con múltiples umbrales.
    Genera N+1 regiones usando N umbrales.

    Ejemplo con umbrales=[85, 170]:
    - [0-84]     → Negro (0)
    - [85-169]   → Gris (127)
    - [170-255]  → Blanco (255)
    """
    resultado = np.zeros_like(imagen)
    umbrales_sorted = sorted(umbrales)  # Asegurar orden ascendente

    # Número de niveles = número de umbrales + 1
    num_niveles = len(umbrales_sorted) + 1

    # Asignar valores a cada región
    for i in range(num_niveles):
        if i == 0:
            # Primera región: [0, primer_umbral)
            mascara = imagen < umbrales_sorted[0]
            resultado[mascara] = 0
        elif i == num_niveles - 1:
            # Última región: [último_umbral, 255]
            mascara = imagen >= umbrales_sorted[-1]
            resultado[mascara] = 255
        else:
            # Regiones intermedias
            mascara = (imagen >= umbrales_sorted[i - 1]) & (imagen < umbrales_sorted[i])
            # Distribuir valores entre 0 y 255
            valor = int(255 * i / (num_niveles - 1))
            resultado[mascara] = valor

    return resultado

def correccion_gamma(imagen, gamma=1.0):
    # [cite: 416-421]
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(imagen, table)


def expansion_histograma(imagen):
    # Contrast Stretching: (img - min) * 255 / (max - min)
    min_val = np.min(imagen)
    max_val = np.max(imagen)
    if max_val - min_val == 0: return imagen
    res = (imagen - min_val) * (255.0 / (max_val - min_val))
    return res.astype(np.uint8)

# --- 3. Clase de la Aplicación GUI ---
class MatplotlibApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes - Morfología Avanzada")
        self.root.geometry("1200x800")

        self.original_image = None
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_cmy_channels = None
        self.current_hsv_channels = None
        self.current_yiq_channels = None
        self.current_hsi_channels = None
        self.current_state = "original"
        self.morphology_result = None
        self.prototipos_clases = {}
        
        # Sistema de historial de operaciones
        self.imagen_trabajo_actual = None
        self.historial_operaciones = []
        self.historial_imagenes = []
        
        # Frame principal dividido
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo con SCROLLBAR
        left_container = ttk.Frame(main_container, width=280)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_container.pack_propagate(False)
        
        bottom_bar_left = ttk.Frame(left_container)
        bottom_bar_left.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Separator(bottom_bar_left).pack(fill=tk.X, pady=(0, 5))
        
        revert_frame = ttk.Frame(bottom_bar_left)
        revert_frame.pack(fill=tk.X, padx=5)
        
        ttk.Button(revert_frame, text="🔄 Revertir", 
                   command=self.revertir_ultima_operacion).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=2)
        ttk.Button(revert_frame, text="🔙 Original", 
                   command=self.revertir_a_original).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1, pady=2)
        
        ttk.Label(bottom_bar_left, text="GUARDAR", 
                  font=('Arial', 10, 'bold')).pack(pady=(8, 5))
        ttk.Button(bottom_bar_left, text="💾 Guardar Resultado", 
                   command=self.guardar_imagen_actual).pack(fill=tk.X, padx=5, pady=2)

        # Canvas para permitir scroll
        canvas_left = tk.Canvas(left_container, width=260, highlightthickness=0)
        scrollbar_left = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=canvas_left.yview)
        
        # Frame scrollable dentro del canvas
        left_panel = ttk.Frame(canvas_left)
        
        # Configurar scroll
        left_panel.bind(
            "<Configure>",
            lambda e: canvas_left.configure(scrollregion=canvas_left.bbox("all"))
        )
        
        canvas_left.create_window((0, 0), window=left_panel, anchor="nw")
        canvas_left.configure(yscrollcommand=scrollbar_left.set)
        
        # Empaquetar canvas y scrollbar
        canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Habilitar scroll con rueda del mouse
        def _on_mousewheel(event):
            canvas_left.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_left.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Panel derecho: Visualización
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- CONTROLES EN PANEL IZQUIERDO (con scroll) ---

        # Sección: Operaciones Básicas
        ttk.Label(left_panel, text="OPERACIONES BÁSICAS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))

        ttk.Button(left_panel, text="📁 Cargar Imagen", command=self.cargar_imagen).pack(fill=tk.X, padx=5, pady=2)

        # Subsección: Modelos de Color
        ttk.Label(left_panel, text="Modelos de Color:", font=('Arial', 9, 'bold')).pack(anchor='w', padx=5, pady=(8, 2))
        ttk.Button(left_panel, text="🎨 RGB", command=self.aplicar_separacion_rgb).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🖌️ CMY", command=self.aplicar_separacion_cmy).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🌈 HSV", command=self.aplicar_separacion_hsv).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📺 YIQ", command=self.aplicar_separacion_yiq).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🎭 HSI", command=self.aplicar_separacion_hsi).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(left_panel, text="🔍 Comparar Todos",
                   command=self.comparar_modelos_color).pack(fill=tk.X, padx=5, pady=2)
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Button(left_panel, text="⚫ Escala de Grises", command=self.aplicar_escala_de_grises).pack(fill=tk.X, padx=5,
                                                                                                      pady=2)
        ttk.Button(left_panel, text="◾ Binarizar", command=self.aplicar_binarizacion).pack(fill=tk.X, padx=5, pady=2)


        # Sección: Morfología
        ttk.Label(left_panel, text="MORFOLOGÍA", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="🔷 Morfología Básica", command=self.abrir_menu_morfologia_basica).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔶 Morfología Binaria", command=self.abrir_menu_morfologia_binaria).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔸 Morfología Laticces", command=self.abrir_menu_morfologia_laticces).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: Operaciones Avanzadas
        ttk.Label(left_panel, text="OPERACIONES AVANZADAS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="➕ Operaciones Aritméticas", command=self.abrir_menu_aritmeticas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔀 Operaciones Lógicas", command=self.abrir_menu_logicas).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="⚖️ Operaciones Relacionales", command=self.abrir_menu_relacionales).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🎲 Agregar Ruido", command=self.abrir_menu_ruido).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🏷️ Etiquetado Componentes", command=self.abrir_menu_etiquetado).pack(fill=tk.X, padx=5, pady=2)


        # --- SECCIÓN: FILTROS ---
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="FILTROS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))

        ttk.Button(left_panel, text="💧 Filtros de Suavizado", command=self.abrir_menu_suavizado).pack(fill=tk.X, padx=5,pady=2)
        ttk.Button(left_panel, text="📐 Detección de Bordes", command=self.abrir_menu_bordes).pack(fill=tk.X, padx=5,pady=2)

        # Sección: Análisis
        ttk.Label(left_panel, text="ANÁLISIS", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
        
        ttk.Button(left_panel, text="📊 Pseudocolor", command=self.abrir_opciones_pseudocolor).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📈 Histograma", command=self.mostrar_histograma).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📋 Características", command=self.mostrar_caracteristicas).pack(fill=tk.X, padx=5, pady=2)
        # Busca la sección donde tienes los botones de ANÁLISIS y agrega esto:
        ttk.Button(left_panel, text="📏 Medir Objetos (Área/Perímetro)",
                   command=self.analizar_objetos_individuales).pack(fill=tk.X, padx=5, pady=2)
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Sección: SEGMENTACION

        ttk.Label(left_panel, text="SEGMENTACION", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        ttk.Button(left_panel, text="🎛️ Segmentación Avanzada", command=self.abrir_menu_segmentacion_p5).pack(fill=tk.X,padx=5,pady=2)
        ttk.Button(left_panel, text="💡 Ajuste de Brillo", command=self.abrir_menu_brillo_p5).pack(fill=tk.X, padx=5, pady=2)
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        
        # Frame con scrollbar para historial
        historial_frame = ttk.Frame(left_panel)
        historial_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.historial_text = tk.Text(historial_frame, height=8, width=30, wrap=tk.WORD, font=('Courier', 8))
        historial_scroll = ttk.Scrollbar(historial_frame, orient=tk.VERTICAL, command=self.historial_text.yview)
        self.historial_text.configure(yscrollcommand=historial_scroll.set)
        
        self.historial_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        historial_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.historial_text.config(state=tk.DISABLED)
        
        # --- CANVAS DE VISUALIZACIÓN ---
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _reset_states(self):
        """Resetea los estados de procesamiento."""
        self.grayscale_image_cv = None
        self.binary_image_cv = None
        self.current_rgb_channels = None
        self.current_cmy_channels = None
        self.current_hsv_channels = None
        self.current_yiq_channels = None
        self.current_hsi_channels = None
        self.morphology_result = None
        self.imagen_trabajo_actual = None
        self.historial_operaciones = []
        self.actualizar_texto_historial()
        self.historial_imagenes = []

    def agregar_a_historial(self, operacion):
        """Agrega una operación al historial."""
        self.historial_operaciones.append(operacion)
        if self.imagen_trabajo_actual is not None:
            self.historial_imagenes.append(self.imagen_trabajo_actual.copy())
        self.actualizar_texto_historial()
    
    def actualizar_texto_historial(self):
        """Actualiza el texto del historial en la interfaz."""
        self.historial_text.config(state=tk.NORMAL)
        self.historial_text.delete(1.0, tk.END)
        
        if not self.historial_operaciones:
            self.historial_text.insert(tk.END, "Sin operaciones aplicadas")
        else:
            for i, op in enumerate(self.historial_operaciones, 1):
                self.historial_text.insert(tk.END, f"{i}. {op}\n")
        
        self.historial_text.config(state=tk.DISABLED)
        self.historial_text.see(tk.END)
    
    def revertir_ultima_operacion(self):
        """Revierte la última operación aplicada, paso por paso."""
        if not self.historial_operaciones:
            messagebox.showinfo("Info", "No hay operaciones para revertir.")
            return
        
        # Remover última operación
        operacion_eliminada = self.historial_operaciones.pop()
    
        # Eliminar última imagen guardada
        if self.historial_imagenes:
            self.historial_imagenes.pop()
        
        # Restaurar estado anterior
        if not self.historial_operaciones:
            # Volver a imagen base (grises o binaria)
            if self.binary_image_cv is not None:
                self.imagen_trabajo_actual = self.binary_image_cv.copy()
            elif self.grayscale_image_cv is not None:
                self.imagen_trabajo_actual = self.grayscale_image_cv.copy()
            estado = "imagen base"
        else:
            # Restaurar penúltima imagen
            if self.historial_imagenes:
                self.imagen_trabajo_actual = self.historial_imagenes[-1].copy()
            estado = self.historial_operaciones[-1]
        
        # Actualizar visualización
        self.morphology_result = self.imagen_trabajo_actual.copy()
        self.actualizar_texto_historial()
        self._mostrar_imagen_actual()
        
        messagebox.showinfo("Revertido", 
                        f"✓ Operación revertida: {operacion_eliminada}\n"
                        f"→ Ahora estás en: {estado}")

    def _preparar_lienzo_unico(self):
        self.fig.clear()
        return self.fig.add_subplot(111)

    def _mostrar_imagen_original(self):
        ax = self._preparar_lienzo_unico()
        ax.imshow(self.original_image)
        ax.set_title("Imagen Original")
        ax.axis('off')
        self._reset_states()
        self.current_state = "original"
        self.canvas.draw()
    
    def _mostrar_imagen_actual(self):
        """Muestra la imagen de trabajo actual."""
        if self.imagen_trabajo_actual is not None:
            ax = self._preparar_lienzo_unico()
            ax.imshow(self.imagen_trabajo_actual, cmap='gray')
            ax.set_title(f"Imagen Procesada ({len(self.historial_operaciones)} operaciones)")
            ax.axis('off')
            self.canvas.draw()

    def cargar_imagen(self):
        ruta_imagen = filedialog.askopenfilename(
            title="Selecciona un archivo de imagen", 
            filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not ruta_imagen:
            return
        
        try:
            self.original_image = Image.open(ruta_imagen).convert('RGB')
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(self.original_image)
            ax.set_title("Imagen Original")
            ax.axis('off')
            self._reset_states()
            self.current_state = "original"
            self.canvas.draw()
            messagebox.showinfo("Éxito", "Imagen cargada correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen: {e}")

    def aplicar_separacion_rgb(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        r, g, b = separar_canales_rgb(self.original_image)
        self.current_rgb_channels = (r, g, b)
        dibujar_canales_rgb(self.fig, r, g, b)
        self.current_state = "rgb_channels"
        self.canvas.draw()

    def aplicar_separacion_cmy(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        c, m, y = separar_canales_cmy(self.original_image)
        self.current_cmy_channels = (c, m, y)
        dibujar_canales_cmy(self.fig, c, m, y)
        self.current_state = "cmy_channels"
        self.canvas.draw()

    def aplicar_separacion_hsv(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        h, s, v = separar_canales_hsv(self.original_image)
        self.current_hsv_channels = (h, s, v)
        dibujar_canales_hsv(self.fig, h, s, v)
        self.current_state = "hsv_channels"
        self.canvas.draw()

    def aplicar_separacion_yiq(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        y, i, q = separar_canales_yiq(self.original_image)
        self.current_yiq_channels = (y, i, q)
        dibujar_canales_yiq(self.fig, y, i, q)
        self.current_state = "yiq_channels"
        self.canvas.draw()
        
    def aplicar_separacion_hsi(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        h, s, i = separar_canales_hsi(self.original_image)
        self.current_hsi_channels = (h, s, i)
        dibujar_canales_hsi(self.fig, h, s, i)
        self.current_state = "hsi_channels"
        self.canvas.draw()

    def comparar_modelos_color(self):
        """Muestra todos los modelos de color en una sola vista."""
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return

        try:
            # Extraer todos los canales
            r, g, b = separar_canales_rgb(self.original_image)
            c, m, y = separar_canales_cmy(self.original_image)
            h_hsv, s_hsv, v = separar_canales_hsv(self.original_image)
            yiq_y, yiq_i, yiq_q = separar_canales_yiq(self.original_image)
            h_hsi, s_hsi, i_hsi = separar_canales_hsi(self.original_image)

            # Crear figura con subplots
            self.fig.clear()
            fig = self.fig
            fig.suptitle('Comparación de Modelos de Color', fontsize=14, fontweight='bold')

            # 5 filas (RGB, CMY, HSV, YIQ, HSI) x 3 columnas (3 canales cada uno)
            gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)

            # RGB
            ax_rgb = [fig.add_subplot(gs[0, i]) for i in range(3)]
            ax_rgb[0].imshow(r, cmap='Reds');
            ax_rgb[0].set_title('R');
            ax_rgb[0].axis('off')
            ax_rgb[1].imshow(g, cmap='Greens');
            ax_rgb[1].set_title('G');
            ax_rgb[1].axis('off')
            ax_rgb[2].imshow(b, cmap='Blues');
            ax_rgb[2].set_title('B');
            ax_rgb[2].axis('off')

            # CMY
            ax_cmy = [fig.add_subplot(gs[1, i]) for i in range(3)]
            ax_cmy[0].imshow(c, cmap='cool');
            ax_cmy[0].set_title('C');
            ax_cmy[0].axis('off')
            ax_cmy[1].imshow(m, cmap='magma');
            ax_cmy[1].set_title('M');
            ax_cmy[1].axis('off')
            ax_cmy[2].imshow(y, cmap='YlOrBr');
            ax_cmy[2].set_title('Y');
            ax_cmy[2].axis('off')

            # HSV
            ax_hsv = [fig.add_subplot(gs[2, i]) for i in range(3)]
            ax_hsv[0].imshow(h_hsv, cmap='hsv');
            ax_hsv[0].set_title('H');
            ax_hsv[0].axis('off')
            ax_hsv[1].imshow(s_hsv, cmap='plasma');
            ax_hsv[1].set_title('S');
            ax_hsv[1].axis('off')
            ax_hsv[2].imshow(v, cmap='gray');
            ax_hsv[2].set_title('V');
            ax_hsv[2].axis('off')

            # YIQ
            ax_yiq = [fig.add_subplot(gs[3, i]) for i in range(3)]
            ax_yiq[0].imshow(yiq_y, cmap='gray');
            ax_yiq[0].set_title('Y');
            ax_yiq[0].axis('off')
            ax_yiq[1].imshow(yiq_i, cmap='RdBu');
            ax_yiq[1].set_title('I');
            ax_yiq[1].axis('off')
            ax_yiq[2].imshow(yiq_q, cmap='PiYG');
            ax_yiq[2].set_title('Q');
            ax_yiq[2].axis('off')

            # HSI
            ax_hsi = [fig.add_subplot(gs[4, i]) for i in range(3)]
            ax_hsi[0].imshow(h_hsi, cmap='hsv');
            ax_hsi[0].set_title('H');
            ax_hsi[0].axis('off')
            ax_hsi[1].imshow(s_hsi, cmap='plasma');
            ax_hsi[1].set_title('S');
            ax_hsi[1].axis('off')
            ax_hsi[2].imshow(i_hsi, cmap='gray');
            ax_hsi[2].set_title('I');
            ax_hsi[2].axis('off')

            # Labels de fila
            labels = ['RGB', 'CMY', 'HSV', 'YIQ', 'HSI']
            for i, label in enumerate(labels):
                ax_label = fig.add_subplot(gs[i, 3])
                ax_label.text(0.5, 0.5, label, fontsize=12, fontweight='bold',
                              ha='center', va='center', rotation=0)
                ax_label.axis('off')

            self.canvas.draw()
            self.current_state = "color_comparison"

        except Exception as e:
            messagebox.showerror("Error", f"Error al comparar modelos: {e}")
    def aplicar_escala_de_grises(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        self._reset_states()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.grayscale_image_cv = convertir_a_grises_en_ax(self.original_image, ax)
        self.imagen_trabajo_actual = self.grayscale_image_cv.copy()
        self.current_state = "grayscale"
        self.historial_operaciones = []
        self.actualizar_texto_historial()
        self.canvas.draw()

    def aplicar_binarizacion(self):
        if self.imagen_trabajo_actual is None and self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Primero convierte la imagen a escala de grises.")
            return
        self.abrir_dialogo_binarizacion()

    # --- MENÚS DE MORFOLOGÍA ---
    
    def abrir_menu_morfologia_basica(self):
        """Menú para operaciones morfológicas básicas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.\nPresiona 'Escala de Grises' primero.")
            return
        
        # Establecer imagen de trabajo
        if self.imagen_trabajo_actual is None:
            if self.binary_image_cv is not None:
                self.imagen_trabajo_actual = self.binary_image_cv.copy()
            else:
                self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología Básica")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Básicas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Erosión", 
                   command=lambda: self.aplicar_operacion_secuencial('erosion', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Dilatación", 
                   command=lambda: self.aplicar_operacion_secuencial('dilatacion', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('apertura', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('cierre', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    # --- NUEVOS MENÚS: OPERACIONES AVANZADAS ---
    
    def abrir_menu_aritmeticas(self):
        """Menú para operaciones aritméticas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Aritméticas")
        dialog.geometry("350x280")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Aritméticas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Con escalar
        ttk.Label(main_frame, text="Con Escalar:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        ttk.Button(main_frame, text="Sumar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('suma_escalar', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Restar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('resta_escalar', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Multiplicar Escalar", 
                   command=lambda: self.aplicar_operacion_secuencial('mult_escalar', dialog)).pack(fill=tk.X, pady=2)
        
        # Con otra imagen
        ttk.Label(main_frame, text="Con otra Imagen:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Sumar Imágenes", 
                   command=lambda: self.aplicar_operacion_con_imagen('suma_img', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Restar Imágenes", 
                   command=lambda: self.aplicar_operacion_con_imagen('resta_img', dialog)).pack(fill=tk.X, pady=2)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_logicas(self):
        """Menú para operaciones lógicas."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Lógicas")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Lógicas", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="AND (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('and_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="OR (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('or_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="XOR (con otra imagen)", 
                   command=lambda: self.aplicar_operacion_con_imagen('xor_img', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="NOT (inversión)", 
                   command=lambda: self.aplicar_operacion_secuencial('not_img', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_relacionales(self):
        """Menú para operaciones relacionales."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Operaciones Relacionales")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Operaciones Relacionales", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Mayor que (>)", 
                   command=lambda: self.aplicar_operacion_secuencial('mayor', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Menor que (<)", 
                   command=lambda: self.aplicar_operacion_secuencial('menor', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Igual a (==)", 
                   command=lambda: self.aplicar_operacion_secuencial('igual', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_ruido(self):
        """Menú para agregar ruido."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Agregar Ruido")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Agregar Ruido", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Ruido Sal y Pimienta", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_sp', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Sal", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_sal', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Pimienta", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_pimienta', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Ruido Gaussiano", 
                   command=lambda: self.aplicar_operacion_secuencial('ruido_gauss', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_etiquetado(self):
        """Menú para etiquetado de componentes."""
        if self.binary_image_cv is None and self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "Requiere imagen binaria.\nAplica primero: Escala de Grises → Binarizar")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.binary_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Etiquetado de Componentes")
        dialog.geometry("350x220") 
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Etiquetado de Componentes", font=('Arial', 12, 'bold')).pack(pady=10)
        
        invertir_var = tk.BooleanVar(value=False)
        chk_invertir = ttk.Checkbutton(main_frame, 
                                        text="Invertir imagen (Objetos negros a blancos)", 
                                        variable=invertir_var)
        chk_invertir.pack(fill=tk.X, pady=5)

        ttk.Button(main_frame, text="Vecindad 4", 
                   command=lambda: self.aplicar_etiquetado_directo(4, invertir_var.get(), dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Vecindad 8", 
                   command=lambda: self.aplicar_etiquetado_directo(8, invertir_var.get(), dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()
    
    def aplicar_etiquetado_directo(self, vecindad, invertir, dialog):
        """Aplica etiquetado y muestra resultado."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_para_label = self.imagen_trabajo_actual.copy()
            descripcion_inversion = ""
            if invertir:
                imagen_para_label = cv2.bitwise_not(imagen_para_label)
                descripcion_inversion = " (Invertido)"

            etiquetas, num_objetos = etiquetar_componentes(imagen_para_label, vecindad)
            
            # Crear imagen coloreada
            if num_objetos == 0:
                img_color = np.zeros((etiquetas.shape[0], etiquetas.shape[1], 3), dtype=np.uint8)
            else:
                etiquetas_norm = np.uint8(255 * etiquetas / num_objetos)
                img_color = cv2.applyColorMap(etiquetas_norm, cv2.COLORMAP_JET)
                img_color[etiquetas == 0] = [0, 0, 0]
            
            # Convertir a escala de grises para mantener compatibilidad
            img_resultado = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Actualizar imagen de trabajo
            self.imagen_trabajo_actual = img_resultado.copy()
            self.morphology_result = img_resultado
            self.current_state = "morphology"
            
            descripcion = f"Etiquetado V{vecindad}{descripcion_inversion} ({num_objetos} objetos)"
            self.agregar_a_historial(descripcion)
            
            # Mostrar con imagen coloreada
            self.fig.clear()
            axs = self.fig.subplots(1, 2)
            
            axs[0].imshow(imagen_para_label, cmap='gray')
            axs[0].set_title(f'Antes{descripcion_inversion}')
            axs[0].axis('off')
            
            # Etiquetado en color
            img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
            axs[1].imshow(img_color_rgb)
            axs[1].set_title(f'Etiquetado V{vecindad}: {num_objetos} objetos')
            axs[1].axis('off')
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            messagebox.showinfo("Resultado", f"Se encontraron {num_objetos} objetos con vecindad {vecindad}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en etiquetado: {e}")
    
    def aplicar_operacion_con_imagen(self, operacion, dialog):
        """Aplica operación que requiere cargar otra imagen."""
        dialog.destroy()
        
        # Cargar segunda imagen
        ruta = filedialog.askopenfilename(
            title="Seleccionar segunda imagen",
            filetypes=[("Archivos de Imagen", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if not ruta:
            return
        
        try:
            img_color = cv2.imread(ruta)
            img2 = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            # Ejecutar operación
            if operacion == 'suma_img':
                resultado = sumar_imagenes(imagen_entrada, img2)
                descripcion = "Suma con imagen"
            elif operacion == 'resta_img':
                resultado = restar_imagenes(imagen_entrada, img2)
                descripcion = "Resta con imagen"
            elif operacion == 'and_img':
                resultado = operacion_and(imagen_entrada, img2)
                descripcion = "AND con imagen"
            elif operacion == 'or_img':
                resultado = operacion_or(imagen_entrada, img2)
                descripcion = "OR con imagen"
            elif operacion == 'xor_img':
                resultado = operacion_xor(imagen_entrada, img2)
                descripcion = "XOR con imagen"
            else:
                messagebox.showerror("Error", "Operación no reconocida.")
                return
            
            # Actualizar
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def abrir_menu_morfologia_binaria(self):
        """Menú para operaciones morfológicas binarias avanzadas."""
        if self.binary_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen BINARIA.\nPresiona 'Binarizar' primero.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.binary_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología Binaria Avanzada")
        dialog.geometry("350x280")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Morfología Binaria", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Button(main_frame, text="Frontera", 
                   command=lambda: self.aplicar_operacion_secuencial('frontera', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Adelgazamiento (Thinning)", 
                   command=lambda: self.aplicar_operacion_secuencial('adelgazamiento', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Hit-or-Miss", 
                   command=lambda: self.aplicar_operacion_secuencial('hitmiss', dialog)).pack(fill=tk.X, pady=3)
        ttk.Button(main_frame, text="Esqueleto Morfológico", 
                   command=lambda: self.aplicar_operacion_secuencial('esqueleto', dialog)).pack(fill=tk.X, pady=3)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def abrir_menu_morfologia_laticces(self):
        """Menú para operaciones morfológicas en laticces (escala de grises)."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.\nPresiona 'Escala de Grises' primero.")
            return
        
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Morfología en Laticces")
        dialog.geometry("400x420")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Morfología en Laticces", font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Gradientes
        ttk.Label(main_frame, text="Gradientes:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(5, 2))
        ttk.Button(main_frame, text="Gradiente Simétrico", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_simetrico', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Gradiente por Erosión", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_erosion', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Gradiente por Dilatación", 
                   command=lambda: self.aplicar_operacion_secuencial('gradiente_dilatacion', dialog)).pack(fill=tk.X, pady=2)
        
        # Transformadas
        ttk.Label(main_frame, text="Transformadas:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Top Hat (Brillantes)", 
                   command=lambda: self.aplicar_operacion_secuencial('tophat', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Bottom Hat (Oscuros)", 
                   command=lambda: self.aplicar_operacion_secuencial('bottomhat', dialog)).pack(fill=tk.X, pady=2)
        
        # Filtros
        ttk.Label(main_frame, text="Filtros de Suavizado:", font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 2))
        ttk.Button(main_frame, text="Suavizado por Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_apertura', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado por Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_cierre', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado Apertura+Cierre", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_ac', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(main_frame, text="Suavizado Cierre+Apertura", 
                   command=lambda: self.aplicar_operacion_secuencial('suavizado_ca', dialog)).pack(fill=tk.X, pady=2)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack()

    def aplicar_operacion_secuencial(self, operacion, dialog):
        """Aplica operación morfológica de forma secuencial sobre la imagen de trabajo."""
        dialog.destroy()
        self.abrir_dialogo_configuracion_morfologia(operacion)

    def abrir_dialogo_configuracion_morfologia(self, operacion):
        """Diálogo para configurar parámetros del kernel y operaciones."""
        
        # Operaciones que necesitan parámetros especiales
        if operacion in ['suma_escalar', 'resta_escalar', 'mult_escalar']:
            self.dialogo_operacion_escalar(operacion)
            return
        elif operacion in ['mayor', 'menor', 'igual']:
            self.dialogo_operacion_relacional(operacion)
            return
        elif operacion in ['ruido_sp', 'ruido_sal', 'ruido_pimienta']:
            self.dialogo_ruido_sal_pimienta(operacion)
            return
        elif operacion == 'ruido_gauss':
            self.dialogo_ruido_gaussiano()
            return
        elif operacion == 'not_img':
            # NOT no necesita configuración
            self.ejecutar_operacion_simple(operacion)
            return
        
        # Diálogo estándar para morfología
        config_dialog = tk.Toplevel(self.root)
        config_dialog.title(f"Configuración - {operacion.capitalize()}")
        config_dialog.geometry("420x350")
        config_dialog.transient(self.root)
        config_dialog.grab_set()
        config_dialog.resizable(False, False)
        
        main_frame = ttk.Frame(config_dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"Configurar {operacion.capitalize()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        # Tamaño del kernel
        ttk.Label(main_frame, text="Tamaño del Kernel:").pack(anchor='w', pady=(5, 0))
        tamano_var = tk.IntVar(value=5)
        tamano_frame = ttk.Frame(main_frame)
        tamano_frame.pack(fill=tk.X, pady=5)
        for tam in [3, 5, 7, 9, 11]:
            ttk.Radiobutton(tamano_frame, text=f"{tam}x{tam}", 
                            variable=tamano_var, value=tam).pack(side=tk.LEFT, padx=5)

        # Forma del kernel
        ttk.Label(main_frame, text="Forma del Kernel:").pack(anchor='w', pady=(10, 0))
        forma_var = tk.StringVar(value='rect')
        forma_frame = ttk.Frame(main_frame)
        forma_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(forma_frame, text="Rectangular", 
                        variable=forma_var, value='rect').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(forma_frame, text="Elíptico", 
                        variable=forma_var, value='ellipse').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(forma_frame, text="Cruz", 
                        variable=forma_var, value='cross').pack(side=tk.LEFT, padx=5)

        # Iteraciones
        ttk.Label(main_frame, text="Iteraciones:").pack(anchor='w', pady=(10, 0))
        iteraciones_var = tk.IntVar(value=1)
        iteraciones_spinbox = ttk.Spinbox(main_frame, from_=1, to=10, 
                                          textvariable=iteraciones_var, width=10)
        iteraciones_spinbox.pack(anchor='w', pady=5)

        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_morfologia_secuencial(operacion, 
                                                           tamano_var.get(), 
                                                           forma_var.get(), 
                                                           iteraciones_var.get(), 
                                                           config_dialog)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Cancelar", 
                   command=config_dialog.destroy).pack(side=tk.LEFT, padx=5)

    # --- DIÁLOGOS ESPECIALES PARA NUEVAS OPERACIONES ---
    
    def dialogo_operacion_escalar(self, operacion):
        """Diálogo para operaciones con escalar."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Configuración - {operacion}")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"{operacion.replace('_', ' ').title()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Valor escalar:").pack(anchor='w', pady=(5, 0))
        escalar_var = tk.DoubleVar(value=50)
        escalar_entry = ttk.Entry(main_frame, textvariable=escalar_var, width=15)
        escalar_entry.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_operacion_escalar(operacion, escalar_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_operacion_relacional(self, operacion):
        """Diálogo para operaciones relacionales."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Configuración - {operacion}")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ops = {'mayor': '>', 'menor': '<', 'igual': '=='}
        ttk.Label(main_frame, text=f"Operación: {ops[operacion]}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Umbral (0-255):").pack(anchor='w', pady=(5, 0))
        umbral_var = tk.IntVar(value=128)
        umbral_spinbox = ttk.Spinbox(main_frame, from_=0, to=255, 
                                      textvariable=umbral_var, width=15)
        umbral_spinbox.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_operacion_relacional(operacion, umbral_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_ruido_sal_pimienta(self, operacion):
        """Diálogo para ruido sal y pimienta."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Ruido")
        dialog.geometry("320x240")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=f"Ruido {operacion.split('_')[1].title()}", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Porcentaje (0-100):").pack(anchor='w', pady=(5, 0))
        porcentaje_var = tk.IntVar(value=10)
        porcentaje_scale = tk.Scale(main_frame, from_=0, to=100, orient="horizontal", 
                                     variable=porcentaje_var, length=200)
        porcentaje_scale.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_ruido_sp(operacion, porcentaje_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def dialogo_ruido_gaussiano(self):
        """Diálogo para ruido gaussiano."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Ruido Gaussiano")
        dialog.geometry("320x240")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Ruido Gaussiano", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Media:").pack(anchor='w', pady=(5, 0))
        media_var = tk.DoubleVar(value=0)
        media_entry = ttk.Entry(main_frame, textvariable=media_var, width=15)
        media_entry.pack(anchor='w', pady=5)

        ttk.Label(main_frame, text="Desviación estándar (sigma):").pack(anchor='w', pady=(10, 0))
        sigma_var = tk.DoubleVar(value=25)
        sigma_entry = ttk.Entry(main_frame, textvariable=sigma_var, width=15)
        sigma_entry.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_ruido_gaussiano(media_var.get(), sigma_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)


        """Diálogo para seleccionar umbral de binarización."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configuración - Binarizar")
        dialog.geometry("320x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Ajustar Umbral", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Label(main_frame, text="Umbral (0-255):").pack(anchor='w', pady=(5, 0))
        
        umbral_var = tk.IntVar(value=128)
        
        umbral_spinbox = ttk.Spinbox(main_frame, from_=0, to=255, 
                                      textvariable=umbral_var, width=15)
        umbral_spinbox.pack(anchor='w', pady=5)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 0))
        
        ttk.Button(button_frame, text="Aplicar", 
                   command=lambda: self.ejecutar_binarizacion(umbral_var.get(), dialog)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancelar", 
                   command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def abrir_dialogo_binarizacion(self):
        """Diálogo para seleccionar umbral de binarización con vista previa en tiempo real."""
        
        # Determinar imagen a binarizar
        imagen_a_binarizar = None
        if self.imagen_trabajo_actual is not None:
            imagen_a_binarizar = self.imagen_trabajo_actual
        elif self.grayscale_image_cv is not None:
            imagen_a_binarizar = self.grayscale_image_cv
        else:
            messagebox.showerror("Error", "No hay imagen en escala de grises para binarizar.")
            return
        
        # Crear ventana de diálogo
        dialog = tk.Toplevel(self.root)
        dialog.title("Binarización Interactiva")
        dialog.geometry("850x550")  # Tamaño fijo más pequeño
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)  # Permitir redimensionar si quiere
        
        # **NUEVO: Frame principal con Canvas y Scrollbar**
        main_container = ttk.Frame(dialog)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para scroll
        canvas_scroll = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=canvas_scroll.yview)
        
        # Frame scrollable
        scrollable_frame = ttk.Frame(canvas_scroll)
        
        # Configurar scroll
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        # Empaquetar canvas y scrollbar
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # **HABILITAR SCROLL CON RUEDA DEL MOUSE**
        def _on_mousewheel(event):
            canvas_scroll.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas_scroll.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Frame de contenido (ahora dentro del scrollable_frame)
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(main_frame, text="Ajustar Umbral de Binarización", 
                font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        # Frame para la visualización
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Crear figura de matplotlib para vista previa
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig_preview = Figure(figsize=(8, 3.5), dpi=90)  # Más pequeño
        canvas_preview = FigureCanvasTkAgg(fig_preview, master=preview_frame)
        canvas_preview.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para controles
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Variable para el umbral
        umbral_var = tk.IntVar(value=128)
        
        # Label que muestra el valor actual
        valor_label = ttk.Label(control_frame, text=f"Umbral: {umbral_var.get()}", 
                                font=('Arial', 11, 'bold'))
        valor_label.pack(pady=(0, 5))
        
        # Función para actualizar la vista previa
        def actualizar_preview(valor):
            umbral = int(float(valor))
            valor_label.config(text=f"Umbral: {umbral}")
            
            # Aplicar binarización
            _, img_binaria = cv2.threshold(imagen_a_binarizar, umbral, 255, cv2.THRESH_BINARY)
            
            # Actualizar visualización
            fig_preview.clear()
            ax1, ax2 = fig_preview.subplots(1, 2)
            
            # Imagen original en grises
            ax1.imshow(imagen_a_binarizar, cmap='gray')
            ax1.set_title('Original', fontsize=9)
            ax1.axis('off')
            
            # Imagen binarizada
            ax2.imshow(img_binaria, cmap='gray')
            ax2.set_title(f'Binarizada (Umbral={umbral})', fontsize=9)
            ax2.axis('off')
            
            fig_preview.tight_layout()
            canvas_preview.draw()
        
        # Slider para el umbral
        slider = tk.Scale(
            control_frame, 
            from_=0, 
            to=255, 
            orient="horizontal",
            variable=umbral_var,
            command=actualizar_preview,
            length=500,
            width=18,
            sliderlength=25,
            font=('Arial', 9)
        )
        slider.pack(fill=tk.X, padx=20, pady=5)
        
        # Labels de referencia
        ref_frame = ttk.Frame(control_frame)
        ref_frame.pack(fill=tk.X, padx=20)
        
        ttk.Label(ref_frame, text="0 (Oscuro)", font=('Arial', 8)).pack(side=tk.LEFT)
        ttk.Label(ref_frame, text="255 (Claro)", font=('Arial', 8)).pack(side=tk.RIGHT)
        
        # Botones de valores predefinidos
        ttk.Label(control_frame, text="Valores rápidos:", 
                font=('Arial', 9, 'bold')).pack(pady=(10, 5))
        
        preset_frame = ttk.Frame(control_frame)
        preset_frame.pack()
        
        def set_preset(valor):
            umbral_var.set(valor)
            actualizar_preview(valor)
        
        for valor in [50, 100, 128, 150, 200]:
            ttk.Button(preset_frame, text=str(valor), width=6,
                    command=lambda v=valor: set_preset(v)).pack(side=tk.LEFT, padx=2)
        
        # Frame de botones finales
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(15, 5))
        
        # Variable para guardar el resultado
        resultado_umbral = [None]
        
        def aplicar_y_cerrar():
            resultado_umbral[0] = umbral_var.get()
            # Desactivar el scroll del mouse antes de cerrar
            canvas_scroll.unbind_all("<MouseWheel>")
            dialog.destroy()
        
        def cancelar():
            resultado_umbral[0] = None
            # Desactivar el scroll del mouse antes de cerrar
            canvas_scroll.unbind_all("<MouseWheel>")
            dialog.destroy()
        
        ttk.Button(button_frame, text="✓ Aplicar", 
                command=aplicar_y_cerrar, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="✗ Cancelar", 
                command=cancelar, width=12).pack(side=tk.LEFT, padx=5)
        
        # Agregar indicador de scroll (opcional pero útil)
        hint_label = ttk.Label(main_frame, 
                            text="💡 Usa la rueda del mouse para desplazarte",
                            font=('Arial', 8, 'italic'),
                            foreground='gray')
        hint_label.pack(pady=(5, 0))
        
        # Mostrar vista previa inicial
        actualizar_preview(128)
        
        # **IMPORTANTE: Hacer scroll al inicio**
        canvas_scroll.yview_moveto(0)
        
        # Esperar a que se cierre el diálogo
        dialog.wait_window()
        
        # Si se aplicó, ejecutar binarización
        if resultado_umbral[0] is not None:
            self.ejecutar_binarizacion(resultado_umbral[0], None)
 



    def ejecutar_binarizacion(self, umbral, dialog):
        """Aplica la binarización con el umbral dado."""
        if dialog is not None:  # AGREGAR ESTA LÍNEA
            dialog.destroy()
        
        imagen_a_binarizar = None
        if self.imagen_trabajo_actual is not None:
            imagen_a_binarizar = self.imagen_trabajo_actual
        elif self.grayscale_image_cv is not None:
            imagen_a_binarizar = self.grayscale_image_cv
        else:
            messagebox.showerror("Error", "No hay imagen en escala de grises para binarizar.")
            return

        if imagen_a_binarizar is None:
            messagebox.showerror("Error", "Se perdió la imagen en escala de grises.")
            return
        
        try:
            ax = self._preparar_lienzo_unico()
            
            self.binary_image_cv = binarizar_imagen_en_ax(imagen_a_binarizar, ax, umbral)
            
            self.imagen_trabajo_actual = self.binary_image_cv.copy()
            self.current_state = "binary"
            
            self.agregar_a_historial(f"Binarizar (Umbral={umbral})")
            
            self.actualizar_texto_historial()
            self.canvas.draw()
        
        except Exception as e:
            messagebox.showerror("Error", f"Error al binarizar: {e}")
    
    # --- EJECUTORES DE OPERACIONES ESPECIALES ---
    
    def ejecutar_operacion_simple(self, operacion):
        """Ejecuta operación que no necesita parámetros."""
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            if operacion == 'not_img':
                resultado = operacion_not(imagen_entrada)
                descripcion = "NOT (inversión)"
            else:
                return
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_operacion_escalar(self, operacion, escalar, dialog):
        """Ejecuta operación aritmética con escalar."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            if operacion == 'suma_escalar':
                resultado = sumar_escalar(imagen_entrada, escalar)
                descripcion = f"Suma escalar {escalar}"
            elif operacion == 'resta_escalar':
                resultado = restar_escalar(imagen_entrada, escalar)
                descripcion = f"Resta escalar {escalar}"
            elif operacion == 'mult_escalar':
                resultado = multiplicar_escalar(imagen_entrada, escalar)
                descripcion = f"Multiplicar {escalar}"
            else:
                return
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_operacion_relacional(self, operacion, umbral, dialog):
        """Ejecuta operación relacional."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            ops = {'mayor': '>', 'menor': '<', 'igual': '=='}
            operador = ops[operacion]
            
            resultado = operacion_relacional(imagen_entrada, umbral, operador)
            descripcion = f"Relacional {operador} {umbral}"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_ruido_sp(self, operacion, porcentaje, dialog):
        """Ejecuta ruido sal y pimienta."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            modos = {'ruido_sp': 'mixto', 'ruido_sal': 'sal', 'ruido_pimienta': 'pimienta'}
            modo = modos[operacion]
            
            resultado = agregar_ruido_sal_pimienta(imagen_entrada, porcentaje, modo)
            descripcion = f"Ruido {modo} {porcentaje}%"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_ruido_gaussiano(self, media, sigma, dialog):
        """Ejecuta ruido gaussiano."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo.")
            return
        
        try:
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            resultado = agregar_ruido_gaussiano(imagen_entrada, media, sigma)
            descripcion = f"Ruido Gaussiano σ={sigma}"
            
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            self.agregar_a_historial(descripcion)
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def ejecutar_morfologia_secuencial(self, operacion, tamano, forma, iteraciones, dialog):
        """Ejecuta operación morfológica sobre la imagen de trabajo actual."""
        dialog.destroy()
        
        if self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "No hay imagen de trabajo disponible.")
            return

        try:
            kernel = crear_kernel(tamano, forma)
            imagen_entrada = self.imagen_trabajo_actual.copy()
            
            # Ejecutar operación
            if operacion == 'erosion':
                resultado = erosion_morfologica(imagen_entrada, kernel, iteraciones)
                descripcion = f"Erosión {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'dilatacion':
                resultado = dilatacion_morfologica(imagen_entrada, kernel, iteraciones)
                descripcion = f"Dilatación {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'apertura':
                resultado = apertura_opencv(imagen_entrada, kernel, iteraciones)
                descripcion = f"Apertura {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'cierre':
                resultado = cierre_opencv(imagen_entrada, kernel, iteraciones)
                descripcion = f"Cierre {tamano}x{tamano} {forma} (iter:{iteraciones})"
            
            # Morfología Binaria
            elif operacion == 'frontera':
                resultado = obtener_frontera(imagen_entrada, kernel)
                descripcion = f"Frontera {tamano}x{tamano} {forma}"
                
            elif operacion == 'adelgazamiento':
                resultado = adelgazamiento(imagen_entrada, kernel, iteraciones)
                descripcion = f"Adelgazamiento {tamano}x{tamano} {forma} (iter:{iteraciones})"
                
            elif operacion == 'hitmiss':
                kernel2 = crear_kernel(tamano, forma)
                resultado = hit_or_miss(imagen_entrada, kernel, kernel2)
                descripcion = f"Hit-or-Miss {tamano}x{tamano} {forma}"
                
            elif operacion == 'esqueleto':
                resultado = esqueleto_morfologico(imagen_entrada, kernel)
                descripcion = f"Esqueleto {tamano}x{tamano} {forma}"
            
            # Morfología en Laticces
            elif operacion == 'gradiente_simetrico':
                resultado = gradiente_morfologico_simetrico(imagen_entrada, kernel)
                descripcion = f"Grad. Simétrico {tamano}x{tamano} {forma}"
                
            elif operacion == 'gradiente_erosion':
                resultado = gradiente_por_erosion(imagen_entrada, kernel)
                descripcion = f"Grad. Erosión {tamano}x{tamano} {forma}"
                
            elif operacion == 'gradiente_dilatacion':
                resultado = gradiente_por_dilatacion(imagen_entrada, kernel)
                descripcion = f"Grad. Dilatación {tamano}x{tamano} {forma}"
                
            elif operacion == 'tophat':
                resultado = top_hat(imagen_entrada, kernel)
                descripcion = f"Top Hat {tamano}x{tamano} {forma}"
                
            elif operacion == 'bottomhat':
                resultado = bottom_hat(imagen_entrada, kernel)
                descripcion = f"Bottom Hat {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_apertura':
                resultado = filtro_suavizado_apertura(imagen_entrada, kernel)
                descripcion = f"Suavizado Apertura {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_cierre':
                resultado = filtro_suavizado_cierre(imagen_entrada, kernel)
                descripcion = f"Suavizado Cierre {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_ac':
                resultado = filtro_suavizado_apertura_cierre(imagen_entrada, kernel)
                descripcion = f"Suavizado A+C {tamano}x{tamano} {forma}"
                
            elif operacion == 'suavizado_ca':
                resultado = filtro_suavizado_cierre_apertura(imagen_entrada, kernel)
                descripcion = f"Suavizado C+A {tamano}x{tamano} {forma}"
            
            else:
                messagebox.showerror("Error", f"Operación '{operacion}' no reconocida.")
                return
            
            # Actualizar imagen de trabajo
            self.imagen_trabajo_actual = resultado.copy()
            self.morphology_result = resultado
            self.current_state = "morphology"
            
            # Agregar al historial
            self.agregar_a_historial(descripcion)
            
            # Mostrar resultado
            self._mostrar_comparacion_antes_despues(imagen_entrada, resultado, descripcion)
            
        except Exception as e:
            messagebox.showerror("Error en Morfología", f"Ocurrió un error: {e}")

    def _mostrar_comparacion_antes_despues(self, antes, despues, titulo):
        """Muestra comparación lado a lado: antes y después."""
        self.fig.clear()
        axs = self.fig.subplots(1, 2)
        
        axs[0].imshow(antes, cmap='gray')
        axs[0].set_title('Antes')
        axs[0].axis('off')
        
        axs[1].imshow(despues, cmap='gray')
        axs[1].set_title(f'Después: {titulo}')
        axs[1].axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

    # --- FUNCIONES DE PSEUDOCOLOR ---
    def abrir_opciones_pseudocolor(self):
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Requiere imagen en escala de grises.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Opciones de Pseudocolor")
        dialog.geometry("350x220") 
        dialog.transient(self.root) 
        dialog.grab_set() 
        
        main_frame = ttk.Frame(dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Seleccione mapas de color:", 
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        ttk.Button(main_frame, text="JET, HOT, OCEAN", 
                   command=lambda: self.aplicar_pseudocolor_cv2(['JET', 'HOT', 'OCEAN'], dialog)).pack(fill=tk.X, pady=5)
        ttk.Button(main_frame, text="BONE, PINK, AUTUMN", 
                   command=lambda: self.aplicar_pseudocolor_cv2(['BONE', 'PINK', 'AUTUMN'], dialog)).pack(fill=tk.X, pady=5)
        ttk.Button(main_frame, text="Personalizado", 
                   command=lambda: self.aplicar_pseudocolor_pastel(dialog)).pack(fill=tk.X, pady=5)
        
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(main_frame, text="Cancelar", command=dialog.destroy).pack(side=tk.RIGHT)

    def aplicar_pseudocolor_cv2(self, map_names, dialog):
        if self.grayscale_image_cv is None: 
            messagebox.showerror("Error", "Se perdió la imagen.")
            dialog.destroy()
            return

        CV2_COLORMAPS = {
            'JET': cv2.COLORMAP_JET, 'HOT': cv2.COLORMAP_HOT, 'OCEAN': cv2.COLORMAP_OCEAN,
            'BONE': cv2.COLORMAP_BONE, 'PINK': cv2.COLORMAP_PINK, 'AUTUMN': cv2.COLORMAP_AUTUMN,
        }
        
        try:
            imagen_gris = self.grayscale_image_cv
            
            img_map1 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[0]])
            img_map2 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[1]])
            img_map3 = cv2.applyColorMap(imagen_gris, CV2_COLORMAPS[map_names[2]])
            
            img_map1_rgb = cv2.cvtColor(img_map1, cv2.COLOR_BGR2RGB)
            img_map2_rgb = cv2.cvtColor(img_map2, cv2.COLOR_BGR2RGB)
            img_map3_rgb = cv2.cvtColor(img_map3, cv2.COLOR_BGR2RGB)
            
            self.fig.clear()
            axs = self.fig.subplots(2, 2) 
            
            axs[0, 0].imshow(imagen_gris, cmap='gray')
            axs[0, 0].set_title('Escala de Grises')
            axs[0, 1].imshow(img_map1_rgb)
            axs[0, 1].set_title(f'{map_names[0]}')
            axs[1, 0].imshow(img_map2_rgb)
            axs[1, 0].set_title(f'{map_names[1]}')
            axs[1, 1].imshow(img_map3_rgb)
            axs[1, 1].set_title(f'{map_names[2]}')
            
            for ax in axs.flat:
                ax.axis('off')
            
            self.fig.tight_layout()
            self.current_state = "pseudocolor"
            self.canvas.draw()
            dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            dialog.destroy()

    def aplicar_pseudocolor_pastel(self, dialog):
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Se perdió la imagen.")
            dialog.destroy()
            return

        try:
            imagen_gris = self.grayscale_image_cv
            colores_pastel = [(1.0, 0.8, 0.9), (0.8, 1.0, 0.8), (0.8, 0.9, 1.0), (1.0, 1.0, 0.8), (0.9, 0.8, 1.0)]
            mapa_pastel = LinearSegmentedColormap.from_list("PastelMap", colores_pastel, N=256)
                                                            
            self.fig.clear()
            axs = self.fig.subplots(1, 2)
            
            axs[0].imshow(imagen_gris, cmap='gray')
            axs[0].set_title('Escala de grises')
            axs[0].axis('off')
            
            axs[1].imshow(imagen_gris, cmap=mapa_pastel)
            axs[1].set_title('Pseudocolor Pastel')
            axs[1].axis('off')
            
            self.fig.tight_layout()
            self.current_state = "pseudocolor"
            self.canvas.draw()
            dialog.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")
            dialog.destroy()

    # --- FUNCIÓN DE HISTOGRAMA ---
    def mostrar_histograma(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return

        if self.current_state == "original":
            dibujar_histograma_imagen_original(self.fig, self.original_image)
        elif self.current_state == "rgb_channels":
            if self.current_rgb_channels is None:
                messagebox.showerror("Error", "No hay canales RGB.")
                return
            r, g, b = self.current_rgb_channels
            dibujar_histograma_canales_rgb(self.fig, r, g, b)
        elif self.current_state in ["grayscale", "pseudocolor", "morphology"]:
            # Usar imagen de trabajo si existe
            img_mostrar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
            if img_mostrar is None:
                messagebox.showerror("Error", "No hay imagen disponible.")
                return
            dibujar_histograma_escala_grises(self.fig, img_mostrar)
        elif self.current_state == "binary":
            if self.binary_image_cv is None:
                messagebox.showerror("Error", "No hay imagen binaria.")
                return
            dibujar_histograma_binaria(self.fig, self.binary_image_cv)
        
        self.canvas.draw()
    
    # --- CARACTERÍSTICAS ESTADÍSTICAS ---
    def mostrar_caracteristicas(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero debes cargar una imagen.")
            return
        
        try:
            caracteristicas = None
            tipo_imagen = ""
            
            if self.current_state == "original":
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Imagen Original (RGB)"
            elif self.current_state in ["grayscale", "pseudocolor", "morphology"]:
                img_analizar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
                if img_analizar is not None:
                    caracteristicas_grises = calcular_caracteristicas_escala_grises(img_analizar)
                    if caracteristicas_grises:
                        caracteristicas = {"Escala de Grises": caracteristicas_grises}
                    tipo_imagen = "Imagen Procesada" if self.imagen_trabajo_actual is not None else "Imagen en Escala de Grises"
            elif self.current_state == "binary" and self.binary_image_cv is not None:
                tipo_imagen = "Imagen Binarizada"; stats = calcular_caracteristicas_escala_grises(self.binary_image_cv)
                if stats: caracteristicas = {"Binaria": stats}
            elif self.current_state == "rgb_channels":
                caracteristicas = calcular_caracteristicas_estadisticas(self.original_image)
                tipo_imagen = "Canales RGB"
            
            if not caracteristicas:
                messagebox.showerror("Error", "No se pudieron calcular características.")
                return
            
            ventana_stats = tk.Toplevel(self.root)
            ventana_stats.title("Características Estadísticas")
            ventana_stats.geometry("650x450")
            
            main_frame = ttk.Frame(ventana_stats)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            ttk.Label(main_frame, text="Características Estadísticas", 
                      font=('Arial', 14, 'bold')).pack(pady=(0, 5))
            ttk.Label(main_frame, text=f"Tipo: {tipo_imagen}", 
                      font=('Arial', 11, 'italic')).pack(pady=(0, 10))
            
            text_frame = ttk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            texto_scroll = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
            scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=texto_scroll.yview)
            texto_scroll.configure(yscrollcommand=scrollbar.set)
            
            resultado_texto = ""
            for canal, stats in caracteristicas.items():
                resultado_texto += f"\n{'='*35}\n"
                resultado_texto += f"{canal.upper()}\n"
                resultado_texto += f"{'='*35}\n"
                resultado_texto += f"Energía:    {stats['Energía']:.6f}\n"
                resultado_texto += f"Entropía:   {stats['Entropía']:.6f}\n"
                resultado_texto += f"Asimetría:  {stats['Asimetría']:.6f}\n"
                resultado_texto += f"Media:      {stats['Media']:.2f}\n"
                resultado_texto += f"Varianza:   {stats['Varianza']:.2f}\n"
            
            texto_scroll.insert(tk.END, resultado_texto)
            texto_scroll.config(state=tk.DISABLED)
            
            texto_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=10)
            
            ttk.Button(button_frame, text="Exportar", 
                       command=lambda: self.exportar_caracteristicas(caracteristicas, tipo_imagen)).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Cerrar", 
                       command=ventana_stats.destroy).pack(side=tk.LEFT, padx=5)
                       
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
    
    def exportar_caracteristicas(self, caracteristicas, tipo_imagen):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Exportar características..."
        )
        
        if ruta_archivo:
            try:
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write("CARACTERÍSTICAS ESTADÍSTICAS\n")
                    f.write("=" * 45 + "\n")
                    f.write(f"Tipo: {tipo_imagen}\n\n")
                    
                    for canal, stats in caracteristicas.items():
                        f.write(f"\n{canal.upper()}:\n")
                        f.write(f"  Energía:    {stats['Energía']:.6f}\n")
                        f.write(f"  Entropía:   {stats['Entropía']:.6f}\n")
                        f.write(f"  Asimetría:  {stats['Asimetría']:.6f}\n")
                        f.write(f"  Media:      {stats['Media']:.2f}\n")
                        f.write(f"  Varianza:   {stats['Varianza']:.2f}\n")
                
                messagebox.showinfo("Éxito", f"Exportado: {os.path.basename(ruta_archivo)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")
    
    # --- FUNCIONES DE GUARDADO ---
    
    def revertir_a_original(self):
        if self.original_image is None:
            messagebox.showerror("Error", "No hay imagen original.")
            return
        self._mostrar_imagen_original()
        messagebox.showinfo("Revertido", "Revertido a imagen original.")

    def guardar_imagen_actual(self):
        if self.original_image is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        try:
            if self.current_state == "morphology" and self.morphology_result is not None:
                self._guardar_resultado_morfologia()
            elif self.current_state == "original":
                self._guardar_original()
            elif self.current_state == "rgb_channels":
                self._guardar_canales_individuales()
            elif self.current_state == "grayscale":
                self._guardar_escala_grises()
            elif self.current_state == "binary":
                self._guardar_imagen_binaria()
            elif self.current_state == "pseudocolor":
                self._guardar_figura_actual("Guardar vista Pseudocolor...")
                
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"Error: {e}")

    def _guardar_resultado_morfologia(self):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar resultado morfológico..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(self.morphology_result)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_figura_actual(self, title="Guardar vista actual..."):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title=title
        )
        if ruta_archivo:
            try:
                self.fig.savefig(ruta_archivo, dpi=150)
                messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {e}")

    def _guardar_original(self):
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen original..."
        )
        if ruta_archivo:
            self.original_image.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_canales_individuales(self):
        if self.current_rgb_channels is None:
            messagebox.showerror("Error", "No hay canales RGB.")
            return
            
        ruta_base = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar canales RGB..."
        )
        if not ruta_base:
            return
            
        directorio, nombre_completo = os.path.split(ruta_base)
        nombre_base, extension = os.path.splitext(nombre_completo)

        r, g, b = self.current_rgb_channels
        
        imagen_roja = Image.merge('RGB', (r, Image.new('L', r.size, 0), Image.new('L', r.size, 0)))
        imagen_verde = Image.merge('RGB', (Image.new('L', g.size, 0), g, Image.new('L', g.size, 0)))
        imagen_azul = Image.merge('RGB', (Image.new('L', b.size, 0), Image.new('L', b.size, 0), b))
        
        imagen_roja.save(os.path.join(directorio, f"{nombre_base}_R.png"))
        imagen_verde.save(os.path.join(directorio, f"{nombre_base}_G.png"))
        imagen_azul.save(os.path.join(directorio, f"{nombre_base}_B.png"))
        
        messagebox.showinfo("Éxito", f"Canales guardados en:\n{directorio}")

    def _guardar_escala_grises(self):
        # Si hay imagen de trabajo procesada, guardar esa
        img_guardar = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv
        
        if img_guardar is None:
            messagebox.showerror("Error", "No hay imagen en escala de grises.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Guardar imagen en escala de grises..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(img_guardar)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

    def _guardar_imagen_binaria(self):
        if self.binary_image_cv is None:
            messagebox.showerror("Error", "No hay imagen binaria.")
            return
            
        ruta_archivo = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Guardar imagen binarizada..."
        )
        if ruta_archivo:
            imagen_pil = Image.fromarray(self.binary_image_cv)
            imagen_pil.save(ruta_archivo)
            messagebox.showinfo("Éxito", f"Guardado: {os.path.basename(ruta_archivo)}")

# --- MÉTODOS DE CLASIFICACIÓN ---

    def entrenar_modelo(self):
        """Genera el dataset, extrae características y calcula prototipos."""
        try:
            # 1. Crear carpeta si no existe
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
            
            # 2. Generar imágenes sintéticas (usando tu función)
            generar_dataset_figuras()
            
            # 3. Entrenar (Calcular vectores promedio por figura)
            caracteristicas_acumuladas = {'circulo': [], 'cuadrado': [], 'triangulo': []}
            
            # Leer archivos generados
            archivos = [f for f in os.listdir('dataset') if f.endswith('.png')]
            
            if not archivos:
                messagebox.showerror("Error", "No se generaron imágenes en /dataset")
                return

            for archivo in archivos:
                # Determinar clase por el nombre del archivo
                clase = archivo.split('_')[0] 
                if clase not in caracteristicas_acumuladas: continue
                
                # Cargar y procesar
                ruta = os.path.join('dataset', archivo)
                img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                _, binaria = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                
                # Extraer vector
                feats = extraer_caracteristicas_forma(binaria)
                vector = crear_vector_caracteristicas(feats)
                caracteristicas_acumuladas[clase].append(vector)
            
            # Calcular promedios (Centroides)
            self.prototipos_clases = {}
            for clase, vectores in caracteristicas_acumuladas.items():
                if vectores:
                    promedio = np.mean(vectores, axis=0)
                    self.prototipos_clases[clase] = promedio
            
            messagebox.showinfo("Entrenamiento Completado", 
                              f"Modelo entrenado con éxito.\n"
                              f"Clases aprendidas: {list(self.prototipos_clases.keys())}")
            
        except Exception as e:
            messagebox.showerror("Error en Entrenamiento", f"Detalle: {e}")

    def clasificar_figura_actual(self):
        """Clasifica la imagen que está en pantalla."""
        if not self.prototipos_clases:
            messagebox.showwarning("Atención", "Primero debes presionar 'Generar/Entrenar Dataset'")
            return
            
        # Usar la imagen binaria actual o binarizar la de trabajo
        img_analisis = None
        
        if self.binary_image_cv is not None:
            img_analisis = self.binary_image_cv
        elif self.imagen_trabajo_actual is not None:
            # Intentar binarizar al vuelo si es gris
            _, img_analisis = cv2.threshold(self.imagen_trabajo_actual, 127, 255, cv2.THRESH_BINARY)
        else:
            messagebox.showerror("Error", "Carga una imagen y binarízala primero.")
            return

        try:
            # 1. Extraer características de la imagen actual
            feats = extraer_caracteristicas_forma(img_analisis)
            vector_prueba = crear_vector_caracteristicas(feats)
            
            # 2. Clasificar
            clase_predicha, distancias = clasificar_por_distancia(vector_prueba, self.prototipos_clases)
            
            # 3. Mostrar resultados visualmente
            resultado_texto = f"PREDICCIÓN: {clase_predicha.upper()}\n\n"
            resultado_texto += "Distancias a prototipos:\n"
            for k, v in distancias.items():
                resultado_texto += f"- {k}: {v:.4f}\n"
            
            resultado_texto += "\nCaracterísticas detectadas:\n"
            resultado_texto += f"- Compacidad: {feats['compacidad']:.4f}\n"
            resultado_texto += f"- Rel. Aspecto: {feats['relacion_aspecto']:.4f}"

            messagebox.showinfo("Resultado de Clasificación", resultado_texto)
            self.agregar_a_historial(f"Clasificación: {clase_predicha.upper()}")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo clasificar. ¿La imagen está binarizada correctamente?\nError: {e}")

    # --- MÉTODOS FILTRADO ---

    def abrir_menu_suavizado(self):
        """Menú para filtros Pasa-Bajas (Suavizado) con Clasificación."""
        if self.grayscale_image_cv is None and self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "Requiere imagen (preferiblemente en escala de grises).")
            return

        # Usar la imagen de trabajo actual si existe, si no la gris base
        if self.imagen_trabajo_actual is None:
            self.imagen_trabajo_actual = self.grayscale_image_cv.copy()

        dialog = tk.Toplevel(self.root)
        dialog.title("Filtros de Suavizado (Pasa-Bajas)")
        dialog.geometry("300x480")  # Aumenté la altura para que quepan los títulos

        ttk.Label(dialog, text="MENU SUAVIZADO", font=('Arial', 12, 'bold')).pack(pady=5)

        # --- SECCIÓN 1: FILTROS LINEALES ---
        ttk.Label(dialog, text="-- Lineales Paso Bajas (Convolución) --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="blue").pack(pady=(10, 2))

        ttk.Button(dialog, text="Promedio (Blur 5x5)",
                   command=lambda: self.ejecutar_suavizado('promedio', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Promedio Pesado",
                   command=lambda: self.ejecutar_suavizado('pesado', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Gaussiano (5x5)",
                   command=lambda: self.ejecutar_suavizado('gaussiano', dialog)).pack(fill=tk.X, pady=2)

        # --- SECCIÓN 2: FILTROS NO LINEALES ---
        ttk.Separator(dialog).pack(fill=tk.X, pady=5)
        ttk.Label(dialog, text="-- No Lineales (Estadísticos) --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="red").pack(pady=(5, 2))

        ttk.Button(dialog, text="Bilateral (Preserva Bordes)",
                   command=lambda: self.ejecutar_suavizado('bilateral', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Mediana (Ideal Sal y Pimienta)",
                   command=lambda: self.ejecutar_suavizado('mediana', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Moda",
                   command=lambda: self.ejecutar_suavizado('moda', dialog)).pack(fill=tk.X, pady=2)

        ttk.Frame(dialog, height=5).pack()  # Espaciador pequeño

        ttk.Button(dialog, text="Máximo (Quita Pimienta)",
                   command=lambda: self.ejecutar_suavizado('maximo', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Mínimo (Quita Sal)",
                   command=lambda: self.ejecutar_suavizado('minimo', dialog)).pack(fill=tk.X, pady=2)

        ttk.Separator(dialog).pack(fill=tk.X, pady=5)
        ttk.Label(dialog, text="-- Avanzados --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="green").pack(pady=(5, 2))

        ttk.Button(dialog, text="Adaptativo de Mediana (Preciso)",
                   command=lambda: self.ejecutar_suavizado('adaptativo', dialog)).pack(fill=tk.X, pady=2)

        ttk.Button(dialog, text="Mediana Ponderada (Detalles)",
                   command=lambda: self.ejecutar_suavizado('ponderada', dialog)).pack(fill=tk.X, pady=2)

        ttk.Button(dialog, text="Contraharmónico (Q=1.5, Quita Pimienta)",
                   command=lambda: self.ejecutar_suavizado('contra_pimienta', dialog)).pack(fill=tk.X, pady=2)

        ttk.Button(dialog, text="Contraharmónico (Q=-1.5, Quita Sal)",
                   command=lambda: self.ejecutar_suavizado('contra_sal', dialog)).pack(fill=tk.X, pady=2)

    def abrir_menu_bordes(self):
        """Menú para detección de bordes con Clasificación."""
        if self.grayscale_image_cv is None:
            messagebox.showerror("Error", "Primero cargue una imagen y conviértala a escala de grises.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Detección de Bordes (Pasa-Altas)")
        dialog.geometry("300x500")  # Altura ajustada

        ttk.Label(dialog, text="MENU BORDES", font=('Arial', 12, 'bold')).pack(pady=5)

        # --- SECCIÓN 1: LINEALES PASO ALTAS ---
        ttk.Label(dialog, text="-- Lineales Paso Altas (1er Orden) --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="blue").pack(pady=(10, 2))

        ttk.Button(dialog, text="Sobel (X e Y)",
                   command=lambda: self.ejecutar_bordes('sobel', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Prewitt",
                   command=lambda: self.ejecutar_bordes('prewitt', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Roberts",
                   command=lambda: self.ejecutar_bordes('roberts', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Kirsch (Brújula/Direccional)",
                   command=lambda: self.ejecutar_bordes('kirsch', dialog)).pack(fill=tk.X, pady=2)

        # --- SECCIÓN 2: LAPLACIANO ---
        ttk.Separator(dialog).pack(fill=tk.X, pady=5)
        ttk.Label(dialog, text="-- Laplaciano (2do Orden) --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="purple").pack(pady=(5, 2))

        ttk.Button(dialog, text="Laplaciano Estándar",
                   command=lambda: self.ejecutar_bordes('laplaciano', dialog)).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Laplaciano 8 Vecinos (Más fuerte)",
                   command=lambda: self.ejecutar_bordes('laplaciano8', dialog)).pack(fill=tk.X, pady=2)

        # --- SECCIÓN 3: AVANZADOS ---
        ttk.Separator(dialog).pack(fill=tk.X, pady=5)
        ttk.Label(dialog, text="-- Algoritmos Avanzados --", font=('Arial', 10, 'bold', 'italic'),
                  foreground="red").pack(pady=(5, 2))

        ttk.Button(dialog, text="Canny (Óptimo)",
                   command=lambda: self.ejecutar_bordes('canny', dialog)).pack(fill=tk.X, pady=2)
    def ejecutar_suavizado(self, tipo, dialog):
        dialog.destroy()
        if self.imagen_trabajo_actual is None: return

        try:
            img = self.imagen_trabajo_actual.copy()
            resultado = None
            desc = ""

            if tipo == 'promedio':
                resultado = cv2.blur(img, (5, 5))
                desc = "Filtro Promedio 5x5"
            elif tipo == 'pesado':
                resultado = filtro_promedio_pesado(img)
                desc = "Filtro Promedio Pesado"
            elif tipo == 'gaussiano':
                resultado = cv2.GaussianBlur(img, (5, 5), 0)
                desc = "Filtro Gaussiano"
            elif tipo == 'bilateral':
                # Filtro Bilateral: Suaviza texturas pero mantiene bordes nítidos
                # Parámetros del PDF: d=9, sigmaColor=75, sigmaSpace=75
                resultado = cv2.bilateralFilter(img, 9, 75, 75)
                desc = "Filtro Bilateral (Bordes Preservados)"
            elif tipo == 'mediana':
                resultado = cv2.medianBlur(img, 5)
                desc = "Filtro Mediana k=5"
            elif tipo == 'moda':
                messagebox.showinfo("Procesando", "Calculando Moda... esto puede tardar unos segundos.")
                self.root.update()  # Actualizar GUI
                resultado = filtro_moda(img, 3)
                desc = "Filtro Moda"
            elif tipo == 'maximo':
                resultado = filtro_maximo(img, 3)
                desc = "Filtro Máximo"
            elif tipo == 'minimo':
                resultado = filtro_minimo(img, 3)
                desc = "Filtro Mínimo"
            elif tipo == 'adaptativo':
                # Puede tardar unos segundos
                messagebox.showinfo("Procesando", "El filtro Adaptativo puede tardar unos segundos. Espere...")
                resultado = filtro_mediana_adaptativo(img, s_max=7)
                desc = "Filtro Mediana Adaptativo (S_max=7)"

            elif tipo == 'ponderada':
                # Requiere scipy
                resultado = filtro_mediana_ponderada(img, k_size=3, peso_central=5)
                desc = "Filtro Mediana Ponderada (W=5)"

            elif tipo == 'contra_pimienta':
                # Q positivo elimina Pimienta (negro)
                resultado = filtro_contraharmonico(img, k_size=3, Q=1.5)
                desc = "Filtro Contraharmónico (Q=1.5)"

            elif tipo == 'contra_sal':
                # Q negativo elimina Sal (blanco)
                resultado = filtro_contraharmonico(img, k_size=3, Q=-1.5)
                desc = "Filtro Contraharmónico (Q=-1.5)"

            self.imagen_trabajo_actual = resultado
            self.agregar_a_historial(desc)
            self._mostrar_comparacion_antes_despues(img, resultado, desc)

        except Exception as e:
            messagebox.showerror("Error", f"Fallo al aplicar filtro: {e}")

    def ejecutar_bordes(self, tipo, dialog):
        dialog.destroy()
        if self.imagen_trabajo_actual is None: return

        try:
            img = self.imagen_trabajo_actual.copy()
            resultado = aplicar_detector_bordes(img, tipo)
            desc = f"Bordes: {tipo.capitalize()}"

            self.imagen_trabajo_actual = resultado
            self.agregar_a_historial(desc)
            self._mostrar_comparacion_antes_despues(img, resultado, desc)

        except Exception as e:
            messagebox.showerror("Error", f"Fallo en bordes: {e}")

    def abrir_menu_segmentacion_p5(self):
        if self.grayscale_image_cv is None and self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "Requiere imagen en grises.")
            return

        img = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv

        dialog = tk.Toplevel(self.root)
        dialog.title("Segmentación Avanzada (Práctica 5)")
        dialog.geometry("300x330")  # ⬅️ Aumenté la altura para el nuevo botón

        def aplicar_seg(metodo):
            dialog.destroy()
            try:
                if metodo == 'otsu':
                    res, umbral = umbral_otsu_implementacion(img)
                    txt = f"Otsu (T={int(umbral)})"
                elif metodo == 'kapur':
                    messagebox.showinfo("Procesando", "Calculando entropía (puede tardar)...")
                    self.root.update()
                    res, umbral = umbral_kapur(img)
                    txt = f"Kapur (T={umbral})"
                elif metodo == 'minimo':
                    res, umbral = umbral_minimo_histograma(img)
                    txt = f"Mínimo Hist. (T={umbral})"
                elif metodo == 'media':
                    res, umbral = umbral_media(img)
                    txt = f"Media (T={int(umbral)})"
                elif metodo == 'banda':
                    t1 = simpledialog.askinteger("Umbral 1", "Valor T1:", initialvalue=80, minvalue=0, maxvalue=255)
                    t2 = simpledialog.askinteger("Umbral 2", "Valor T2:", initialvalue=150, minvalue=0, maxvalue=255)
                    if t1 is None or t2 is None: return
                    res = umbral_banda(img, t1, t2)
                    txt = f"Banda [{t1}-{t2}]"

                # ⬇️ NUEVO MÉTODO
                elif metodo == 'multi':
                    # Pedir umbrales al usuario
                    umbrales_str = simpledialog.askstring(
                        "Multi-umbralización",
                        "Ingrese umbrales separados por comas (ej: 85,170):",
                        initialvalue="85,170"
                    )
                    if umbrales_str is None: return

                    # Convertir string a lista de números
                    try:
                        umbrales = [int(u.strip()) for u in umbrales_str.split(',')]
                        umbrales = sorted(umbrales)  # Ordenar de menor a mayor
                    except:
                        messagebox.showerror("Error", "Formato inválido. Use: 85,170,200")
                        return

                    res = multiumbralizado(img, umbrales)
                    txt = f"Multi-umbral {umbrales}"

                self.imagen_trabajo_actual = res
                self.binary_image_cv = res  # Actualizamos binaria global también
                self.agregar_a_historial(txt)
                self._mostrar_comparacion_antes_despues(img, res, txt)
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(dialog, text="Método Otsu (Automático)", command=lambda: aplicar_seg('otsu')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Método Kapur (Entropía)", command=lambda: aplicar_seg('kapur')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Mínimo del Histograma", command=lambda: aplicar_seg('minimo')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Umbral por Media", command=lambda: aplicar_seg('media')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Umbral Banda", command=lambda: aplicar_seg('banda')).pack(fill=tk.X, pady=2)

        # ⬇️ NUEVO BOTÓN
        ttk.Button(dialog, text="Multi-umbralización", command=lambda: aplicar_seg('multi')).pack(fill=tk.X, pady=2)
    def abrir_menu_brillo_p5(self):
        if self.grayscale_image_cv is None and self.imagen_trabajo_actual is None:
            messagebox.showerror("Error", "Requiere imagen en grises.")
            return

        img = self.imagen_trabajo_actual if self.imagen_trabajo_actual is not None else self.grayscale_image_cv

        dialog = tk.Toplevel(self.root)
        dialog.title("Ajuste de Brillo (Práctica 5)")
        dialog.geometry("300x450")  # ⬅️ Aumenté la altura para que quepan todos los botones

        def aplicar_bri(metodo):
            dialog.destroy()
            try:
                res = None
                txt = ""
                if metodo == 'uniforme':
                    res = ecualizacion_uniforme(img)
                    txt = "Ecualización Uniforme"
                elif metodo == 'exponencial':
                    res = ecualizacion_exponencial(img)
                    txt = "Ecualización Exponencial"
                elif metodo == 'rayleigh':
                    res = ecualizacion_rayleigh(img)
                    txt = "Ecualización Rayleigh"
                elif metodo == 'hipercubica':
                    res = ecualizacion_hipercubica(img)
                    txt = "Ecualización Hipercúbica"
                elif metodo == 'log_hiper':
                    res = ecualizacion_log_hiperbolica(img)
                    txt = "Ecualización Log-Hiperbólica"
                elif metodo == 'gamma':
                    g = simpledialog.askfloat("Gamma", "Valor Gamma:", initialvalue=1.5)
                    if g is None: return
                    res = correccion_gamma(img, g)
                    txt = f"Corrección Gamma ({g})"
                elif metodo == 'expansion':
                    res = expansion_histograma(img)
                    txt = "Expansión Histograma"

                # ⬇️ NUEVOS MÉTODOS
                elif metodo == 'potencia':
                    exp = simpledialog.askfloat("Función Potencia", "Exponente (ej: 2.0):", initialvalue=2.0)
                    if exp is None: return
                    res = funcion_potencia(img, exp)
                    txt = f"Función Potencia (exp={exp})"

                elif metodo == 'desplazamiento':
                    valor = simpledialog.askinteger("Desplazamiento", "Valor (-255 a 255):", initialvalue=50)
                    if valor is None: return
                    res = desplazamiento_histograma(img, valor)
                    txt = f"Desplazamiento ({valor:+d})"

                elif metodo == 'contraccion':
                    factor = simpledialog.askfloat("Contracción", "Factor (0.0-1.0):", initialvalue=0.5)
                    if factor is None: return
                    res = contraccion_histograma(img, factor)
                    txt = f"Contracción (factor={factor})"

                self.imagen_trabajo_actual = res
                self.agregar_a_historial(txt)
                self._mostrar_comparacion_antes_despues(img, res, txt)

            except Exception as e:
                messagebox.showerror("Error", str(e))

        # Botones existentes
        ttk.Button(dialog, text="Ecualización Uniforme", command=lambda: aplicar_bri('uniforme')).pack(fill=tk.X,
                                                                                                       pady=2)
        ttk.Button(dialog, text="Ecualización Exponencial", command=lambda: aplicar_bri('exponencial')).pack(fill=tk.X,
                                                                                                             pady=2)
        ttk.Button(dialog, text="Ecualización Rayleigh", command=lambda: aplicar_bri('rayleigh')).pack(fill=tk.X,
                                                                                                       pady=2)
        ttk.Button(dialog, text="Ecualización Hipercúbica", command=lambda: aplicar_bri('hipercubica')).pack(fill=tk.X,
                                                                                                             pady=2)
        ttk.Button(dialog, text="Ecualización Log-Hiper", command=lambda: aplicar_bri('log_hiper')).pack(fill=tk.X,
                                                                                                         pady=2)

        ttk.Separator(dialog).pack(fill=tk.X, pady=5)

        ttk.Button(dialog, text="Corrección Gamma", command=lambda: aplicar_bri('gamma')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Expansión (Contrast Stretching)", command=lambda: aplicar_bri('expansion')).pack(
            fill=tk.X, pady=2)

        # ⬇️ NUEVOS BOTONES
        ttk.Button(dialog, text="Función Potencia", command=lambda: aplicar_bri('potencia')).pack(fill=tk.X, pady=2)
        ttk.Button(dialog, text="Desplazamiento Histograma", command=lambda: aplicar_bri('desplazamiento')).pack(
            fill=tk.X, pady=2)
        ttk.Button(dialog, text="Contracción Histograma", command=lambda: aplicar_bri('contraccion')).pack(fill=tk.X,
                                                                                                           pady=2)

    def analizar_objetos_individuales(self):
        """
        Identifica objetos, corrige automáticamente el fondo y muestra tabla interactiva de características.
        """
        # 1. Obtener imagen binaria
        if self.binary_image_cv is None:
            if self.imagen_trabajo_actual is not None:
                _, binaria = cv2.threshold(self.imagen_trabajo_actual, 127, 255, cv2.THRESH_BINARY)
            else:
                messagebox.showerror("Error", "Primero carga una imagen.")
                return
        else:
            binaria = self.binary_image_cv.copy()

        # --- CORRECCIÓN AUTOMÁTICA DE FONDO ---
        pixeles_blancos = cv2.countNonZero(binaria)
        pixeles_totales = binaria.size

        fondo_invertido = False
        if pixeles_blancos > (pixeles_totales / 2):
            binaria = cv2.bitwise_not(binaria)
            fondo_invertido = True

        # 2. Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            messagebox.showinfo("Info", "No se encontraron objetos.")
            return

        # 3. Ventana de Resultados
        ventana_res = tk.Toplevel(self.root)
        ventana_res.title(f"Análisis: {len(contornos)} Objetos Detectados")
        ventana_res.geometry("900x600")

        # Frame principal con dos paneles
        main_frame = ttk.Frame(ventana_res)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Panel izquierdo: Tabla
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Panel derecho: Imagen
        right_frame = ttk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        # Mensaje informativo
        aviso = "Nota: Se detectó fondo blanco y se invirtió para el análisis.\n" if fondo_invertido else ""
        lbl_info = tk.Label(left_frame,
                            text=f"{aviso}Se encontraron {len(contornos)} figuras.\n🖱️ Haz clic en una fila para resaltar el objeto",
                            fg="blue", font=('Arial', 9))
        lbl_info.pack(pady=5)

        # Frame para Treeview (tabla interactiva)
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        # Treeview (tabla)
        tree = ttk.Treeview(tree_frame,
                            columns=("ID", "Área", "Perímetro", "Circularidad", "Clasificación"),
                            show="headings",
                            yscrollcommand=vsb.set,
                            xscrollcommand=hsb.set,
                            selectmode="browse")

        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Configurar columnas
        tree.heading("ID", text="ID")
        tree.heading("Área", text="Área")
        tree.heading("Perímetro", text="Perímetro")
        tree.heading("Circularidad", text="Circularidad")
        tree.heading("Clasificación", text="Clasificación")

        tree.column("ID", width=50, anchor="center")
        tree.column("Área", width=80, anchor="center")
        tree.column("Perímetro", width=80, anchor="center")
        tree.column("Circularidad", width=100, anchor="center")
        tree.column("Clasificación", width=120, anchor="center")

        # Empaquetar
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Canvas para imagen interactiva
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig_preview = Figure(figsize=(5, 5), dpi=90)
        ax_preview = fig_preview.add_subplot(111)
        canvas_preview = FigureCanvasTkAgg(fig_preview, master=right_frame)
        canvas_preview.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Preparar imagen base
        img_base = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)

        # Almacenar datos de objetos
        objetos_datos = []

        # 4. Procesar objetos
        count_validos = 0
        for i, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)

            if area < 50:
                continue

            count_validos += 1
            perimetro = cv2.arcLength(cnt, True)

            if perimetro > 0:
                circularidad = (4 * np.pi * area) / (perimetro ** 2)
            else:
                circularidad = 0

            # Clasificación
            clasificacion = "Desconocido"
            if circularidad > 0.88:
                clasificacion = "CÍRCULO"
            elif 0.72 <= circularidad <= 0.88:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.9 <= aspect_ratio <= 1.1:
                    clasificacion = "CUADRADO"
                else:
                    clasificacion = "RECTÁNGULO"
            elif circularidad < 0.65:
                clasificacion = "TRIÁNGULO"

            # Guardar datos
            x, y, w, h = cv2.boundingRect(cnt)
            objetos_datos.append({
                'id': i,
                'contorno': cnt,
                'bbox': (x, y, w, h),
                'area': int(area),
                'perimetro': int(perimetro),
                'circularidad': circularidad,
                'clasificacion': clasificacion
            })

            # Insertar en tabla
            tree.insert("", "end", values=(
                i,
                int(area),
                int(perimetro),
                f"{circularidad:.4f}",
                clasificacion
            ))

        # 5. Función para actualizar visualización
        def actualizar_visualizacion(objeto_id=None):
            """Actualiza la imagen mostrando todos los objetos y resaltando uno específico"""
            img_display = img_base.copy()

            # Dibujar todos los objetos con contorno fino
            for obj in objetos_datos:
                x, y, w, h = obj['bbox']
                color = (100, 100, 100)  # Gris para objetos no seleccionados
                grosor = 1

                cv2.rectangle(img_display, (x, y), (x + w, y + h), color, grosor)

            # Si hay un objeto seleccionado, resaltarlo
            if objeto_id is not None:
                # Buscar el objeto
                obj_seleccionado = None
                for obj in objetos_datos:
                    if obj['id'] == objeto_id:
                        obj_seleccionado = obj
                        break

                if obj_seleccionado:
                    x, y, w, h = obj_seleccionado['bbox']

                    # Dibujar rectángulo grueso amarillo
                    cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 255), 4)

                    # Dibujar el contorno relleno semi-transparente
                    overlay = img_display.copy()
                    cv2.drawContours(overlay, [obj_seleccionado['contorno']], -1, (255, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, img_display, 0.7, 0, img_display)

                    # Agregar etiqueta con ID y clasificación
                    label = f"ID:{obj_seleccionado['id']} - {obj_seleccionado['clasificacion']}"
                    font_scale = 0.7
                    thickness = 2

                    # Fondo para el texto
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                   thickness)
                    cv2.rectangle(img_display,
                                  (x, y - text_height - 10),
                                  (x + text_width + 10, y),
                                  (0, 255, 255), -1)

                    # Texto
                    cv2.putText(img_display, label, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                    # Información adicional
                    info_text = f"Área: {obj_seleccionado['area']} | Perímetro: {obj_seleccionado['perimetro']}"
                    cv2.putText(img_display, info_text, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

            # Actualizar imagen en canvas
            ax_preview.clear()
            img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            ax_preview.imshow(img_display_rgb)
            ax_preview.axis('off')

            if objeto_id is not None:
                ax_preview.set_title(f"Objeto Seleccionado: ID {objeto_id}", fontweight='bold', fontsize=11)
            else:
                ax_preview.set_title(f"Todos los objetos ({count_validos} totales)", fontsize=11)

            fig_preview.tight_layout()
            canvas_preview.draw()

        # 6. Evento de selección en la tabla
        def on_tree_select(event):
            """Cuando se selecciona una fila en la tabla"""
            selection = tree.selection()
            if selection:
                item = tree.item(selection[0])
                objeto_id = item['values'][0]  # El ID está en la primera columna
                actualizar_visualizacion(objeto_id)

        tree.bind('<<TreeviewSelect>>', on_tree_select)

        # 7. Botón para deseleccionar
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=5)

        def deseleccionar():
            tree.selection_remove(tree.selection())
            actualizar_visualizacion(None)

        ttk.Button(btn_frame, text="🔄 Ver Todos", command=deseleccionar).pack(side=tk.LEFT, padx=5)

        # Botón para exportar resultados
        def exportar_resultados():
            ruta = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
            )
            if ruta:
                try:
                    with open(ruta, 'w', encoding='utf-8') as f:
                        f.write("ID,Área,Perímetro,Circularidad,Clasificación\n")
                        for obj in objetos_datos:
                            f.write(
                                f"{obj['id']},{obj['area']},{obj['perimetro']},{obj['circularidad']:.4f},{obj['clasificacion']}\n")
                    messagebox.showinfo("Éxito", f"Datos exportados a:\n{os.path.basename(ruta)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error al exportar: {e}")

        ttk.Button(btn_frame, text="💾 Exportar CSV", command=exportar_resultados).pack(side=tk.LEFT, padx=5)

        # Mostrar vista inicial (todos los objetos)
        actualizar_visualizacion(None)

        # Actualizar canvas principal también
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        img_resultados = img_base.copy()
        for obj in objetos_datos:
            x, y, w, h = obj['bbox']
            cv2.rectangle(img_resultados, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_resultados, str(obj['id']), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        ax.imshow(img_resultados)
        ax.set_title(f"Análisis Completado: {count_validos} figuras medidas")
        ax.axis('off')
        self.canvas.draw()

    # --- MÉTODOS PARA EL PROYECTO FINAL DE CAFÉ ---

    def entrenar_modelo_cafe(self):
        """
        Entrena el modelo CON SELECCIÓN DE MÉTODO DE NORMALIZACIÓN
        e integra el entrenamiento del SVM.
        """
        global X_train_cafe, y_train_cafe
        global scaler_min_cafe, scaler_max_cafe
        global scaler_mean_cafe, scaler_std_cafe
        global metodo_normalizacion

        # --- DIÁLOGO PARA SELECCIONAR MÉTODO ---
        dialog_metodo = tk.Toplevel(self.root)
        dialog_metodo.title("Seleccionar Método de Normalización")
        dialog_metodo.geometry("450x280")
        dialog_metodo.transient(self.root)
        dialog_metodo.grab_set()

        main_frame = ttk.Frame(dialog_metodo, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text="Elige el método de normalización:",
                  font=('Arial', 11, 'bold')).pack(pady=(0, 15))

        metodo_var = tk.StringVar(value='minmax')

        # Opción Min-Max
        rb1_frame = ttk.Frame(main_frame)
        rb1_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(rb1_frame, text="Min-Max Scaling",
                        variable=metodo_var, value='minmax').pack(side=tk.LEFT)
        ttk.Label(rb1_frame, text="(Escala a [0,1])",
                  font=('Arial', 9, 'italic'), foreground='blue').pack(side=tk.LEFT, padx=5)

        info1 = ttk.Label(main_frame,
                          text="✓ Mejor para datos con distribución uniforme\n"
                               "✓ Preserva relaciones entre valores\n"
                               "✗ Sensible a outliers extremos",
                          font=('Arial', 8), foreground='gray')
        info1.pack(anchor='w', padx=30, pady=(0, 10))

        # Opción Standard
        rb2_frame = ttk.Frame(main_frame)
        rb2_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(rb2_frame, text="Standard Scaling (Z-Score)",
                        variable=metodo_var, value='standard').pack(side=tk.LEFT)
        ttk.Label(rb2_frame, text="(Media=0, Std=1)",
                  font=('Arial', 9, 'italic'), foreground='green').pack(side=tk.LEFT, padx=5)

        info2 = ttk.Label(main_frame,
                          text="✓ Robusto ante outliers\n"
                               "✓ Mejor para datos con distribución gaussiana\n"
                               "✗ Puede generar valores negativos",
                          font=('Arial', 8), foreground='gray')
        info2.pack(anchor='w', padx=30, pady=(0, 15))

        resultado = [None]

        def continuar():
            resultado[0] = metodo_var.get()
            dialog_metodo.destroy()

        ttk.Button(main_frame, text="Continuar", command=continuar).pack()

        dialog_metodo.wait_window()

        if resultado[0] is None:
            return

        metodo_normalizacion = resultado[0]

        # --- SELECCIONAR CARPETA ---
        messagebox.showinfo("Entrenamiento",
                            f"Método seleccionado: {metodo_normalizacion.upper()}\n\n"
                            "Selecciona la carpeta 'train'.")
        dataset_path = filedialog.askdirectory(title="Seleccionar carpeta TRAIN")
        if not dataset_path: return

        X_train_cafe = []
        y_train_cafe = []
        clases = ['dark', 'green', 'light']

        # Ventana de progreso
        ventana_progreso = tk.Toplevel(self.root)
        ventana_progreso.title(f"Entrenando con {metodo_normalizacion.upper()}...")
        lbl_prog = ttk.Label(ventana_progreso, text="Iniciando...")
        lbl_prog.pack(pady=20, padx=20)

        total_imagenes = 0

        try:
            # PASO 1: Extraer características SIN normalizar
            for clase in clases:
                path_clase = os.path.join(dataset_path, clase)
                if not os.path.exists(path_clase): continue

                imagenes = os.listdir(path_clase)

                for img_name in imagenes:
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                    img_path = os.path.join(path_clase, img_name)

                    # Leer imagen (manejo de tildes/rutas complejas)
                    with open(img_path, "rb") as f:
                        bytes_img = bytearray(f.read())
                        np_arr = np.asarray(bytes_img, dtype=np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if img is not None:
                        feats = extraer_features_cafe(img)
                        X_train_cafe.append(feats)
                        y_train_cafe.append(clase)
                        total_imagenes += 1

                        # Debug visual (cada 5 imágenes)
                        if total_imagenes % 5 == 0:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            gray = cv2.GaussianBlur(gray, (3, 3), 0)
                            _, mask_viz = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            if cv2.countNonZero(mask_viz) > mask_viz.size / 2:
                                mask_viz = cv2.bitwise_not(mask_viz)

                            mask_viz_bgr = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
                            h, w = img.shape[:2]
                            scale = 200 / h
                            dim = (int(w * scale), 200)
                            img_small = cv2.resize(img, dim)
                            mask_small = cv2.resize(mask_viz_bgr, dim)
                            combined = np.hstack((img_small, mask_small))
                            cv2.putText(combined, f"Clase: {clase}", (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.imshow("DEBUG ENTRENAMIENTO", combined)
                            cv2.waitKey(1)

                        lbl_prog.config(text=f"Procesando: {clase} ({total_imagenes})")
                        ventana_progreso.update()

            cv2.destroyAllWindows()

            if total_imagenes == 0:
                ventana_progreso.destroy()
                messagebox.showerror("Error", "No se encontraron imágenes.")
                return

            # PASO 2: Calcular parámetros del scaler
            X_train_cafe = np.array(X_train_cafe)

            if metodo_normalizacion == 'minmax':
                scaler_min_cafe = np.min(X_train_cafe, axis=0)
                scaler_max_cafe = np.max(X_train_cafe, axis=0)
                scaler_mean_cafe = None
                scaler_std_cafe = None
            elif metodo_normalizacion == 'standard':
                scaler_mean_cafe = np.mean(X_train_cafe, axis=0)
                scaler_std_cafe = np.std(X_train_cafe, axis=0)
                scaler_min_cafe = None
                scaler_max_cafe = None

            # PASO 3: Normalizar todos los vectores
            lbl_prog.config(text="Normalizando datos...")
            ventana_progreso.update()

            X_train_cafe_normalized = []
            for vector in X_train_cafe:
                vector_norm = normalizar_vector(vector)
                X_train_cafe_normalized.append(vector_norm)

            X_train_cafe = X_train_cafe_normalized

            # --- AQUÍ EMPIEZA LO NUEVO: ENTRENAMIENTO SVM ---

            lbl_prog.config(text="Entrenando SVM (Kernel RBF)...")
            ventana_progreso.update()

            # Llamada al método interno que entrena el SVM con los datos ya normalizados
            svm_ok = self.entrenar_svm_interno()

            # Mensaje condicional según si funcionó o no
            svm_msg = "\n🤖 Modelo SVM: Entrenado OK" if svm_ok else "\n🤖 Modelo SVM: Falló"

            ventana_progreso.destroy()

            # Mensaje final con todos los detalles
            messagebox.showinfo("Éxito",
                                f"✅ Modelo entrenado con {total_imagenes} granos.\n"
                                f"📊 Método Normalización: {metodo_normalizacion.upper()}\n"
                                f"{svm_msg}")

        except Exception as e:
            cv2.destroyAllWindows()
            try:
                ventana_progreso.destroy()
            except:
                pass
            messagebox.showerror("Error", f"Fallo en entrenamiento: {e}")

    def tabla_resultados_cafe(self):
        """
        Muestra una tabla detallada con las características y clasificación
        de cada grano de café detectado en la imagen actual.
        """
        # 1. Validaciones
        if not X_train_cafe:
            messagebox.showwarning("Atención", "El modelo no está entrenado.")
            return
        if self.original_image is None:
            messagebox.showerror("Error", "Carga una imagen primero.")
            return

        # 2. Preparar imagen y detección
        img_pil = self.original_image
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_vis = img_bgr.copy()  # Imagen base para visualización

        # -----------------------------------------------------------
        # NUEVA BINARIZACIÓN ANTI-SOMBRAS (Usando Canal de Saturación)
        # -----------------------------------------------------------

        # 1. Convertir a HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # 2. Usar el Canal S (Saturación) en lugar de Grises
        # La sombra es gris (S=bajo), el grano tiene color (S=alto)
        # Aplicamos un desenfoque suave al canal S
        s_blurred = cv2.GaussianBlur(S, (3, 3), 0)

        # 3. Binarización Otsu sobre la Saturación
        # NOTA: Guardamos en 'binaria' para que coincida con el resto del código
        _, binaria = cv2.threshold(s_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Operaciones Morfológicas (Limpieza)
        kernel_clean = np.ones((3, 3), np.uint8)

        # "Cierre" para rellenar huecos dentro del grano (brillos de aceite)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_clean, iterations=2)

        # "Apertura" para quitar ruido externo
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_clean, iterations=1)

        # NOTA: En el canal de Saturación, el fondo (papel blanco) tiene S=0 (Negro),
        # y el grano tiene S>0 (Blanco). Por lo tanto, NO solemos necesitar invertir.
        # Pero mantenemos la seguridad por si acaso hay un borde extraño:
        if cv2.countNonZero(binaria) > binaria.size / 2:
            binaria = cv2.bitwise_not(binaria)

        # -----------------------------------------------------------

        # Separación de objetos pegados (Erosión)
        # Ahora sí funciona porque 'binaria' ya está definida arriba
        kernel_erode = np.ones((3, 3), np.uint8)
        binaria = cv2.erode(binaria, kernel_erode, iterations=1)

        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            messagebox.showinfo("Resultado", "No se detectaron granos claros en la imagen.")
            return

        # 3. Crear Ventana de Resultados
        ventana_tabla = tk.Toplevel(self.root)
        ventana_tabla.title(f"Reporte de Clasificación: {len(contornos)} Granos")
        ventana_tabla.geometry("1000x600")

        # Layout Principal
        main_frame = ttk.Frame(ventana_tabla)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Panel Izquierdo: Tabla ---
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        lbl_instruccion = tk.Label(left_frame,
                                   text="👇 Selecciona una fila para ver el grano",
                                   fg="blue", font=('Arial', 10, 'bold'))
        lbl_instruccion.pack(pady=5)

        # Configuración del Treeview
        cols = ("ID", "Clase", "Confianza", "Área", "Perímetro", "Circularidad")
        tree = ttk.Treeview(left_frame, columns=cols, show="headings", selectmode="browse")

        # Encabezados
        tree.heading("ID", text="ID")
        tree.heading("Clase", text="Clasificación")
        tree.heading("Confianza", text="Votos (%)")
        tree.heading("Área", text="Área (px)")
        tree.heading("Perímetro", text="Perímetro (px)")
        tree.heading("Circularidad", text="Circularidad")

        # Anchos de columna
        tree.column("ID", width=40, anchor="center")
        tree.column("Clase", width=100, anchor="center")
        tree.column("Confianza", width=80, anchor="center")
        tree.column("Área", width=80, anchor="center")
        tree.column("Perímetro", width=80, anchor="center")
        tree.column("Circularidad", width=80, anchor="center")

        # Scrollbar
        vsb = ttk.Scrollbar(left_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Panel Derecho: Vista Previa ---
        right_frame = ttk.Frame(main_frame, width=450)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        right_frame.pack_propagate(False)  # Respetar ancho fijo

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig_preview = Figure(figsize=(5, 5), dpi=90)
        ax_preview = fig_preview.add_subplot(111)
        canvas_preview = FigureCanvasTkAgg(fig_preview, master=right_frame)
        canvas_preview.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Procesar Datos y Llenar Tabla
        datos_granos = []  # Lista para guardar la info de cada grano

        for i, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)
            if area < 200: continue  # Ignorar ruido

            # Geometría
            perimetro = cv2.arcLength(cnt, True)
            circularidad = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
            x, y, w, h = cv2.boundingRect(cnt)

            # Clasificación
            # Recorte seguro (ROI)
            pad = 5
            y1, y2 = max(0, y - pad), min(img_bgr.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(img_bgr.shape[1], x + w + pad)

            roi = img_bgr[y1:y2, x1:x2]
            # IMPORTANTE: Recortar también la máscara correcta
            mask_roi = binaria[y1:y2, x1:x2]

            if roi.size == 0: continue

            try:
                # Usamos tu función de extracción y clasificación
                feats = extraer_features_cafe(roi, mask_roi)

                # Verificamos qué método seleccionó el usuario en el RadioButton
                if self.clasificador_activo.get() == "SVM":
                    clase_predicha, votos = self.clasificar_svm_cafe(feats)
                else:
                    # Por defecto usa KNN
                    clase_predicha, votos = clasificar_knn_cafe(feats, k=5)

                # Calcular % de confianza
                total_votos = sum(votos.values())
                confianza = (votos.get(clase_predicha, 0) / total_votos) * 100
            except:
                clase_predicha = "Error"
                confianza = 0.0

            # Guardar en memoria
            datos_grano = {
                'id': i,
                'bbox': (x, y, w, h),
                'contorno': cnt,
                'clase': clase_predicha,
                'area': area,
                'perimetro': perimetro,
                'circularidad': circularidad
            }
            datos_granos.append(datos_grano)

            # Insertar en Treeview
            tree.insert("", "end", values=(
                i,
                clase_predicha.upper(),
                f"{confianza:.1f}%",
                int(area),
                int(perimetro),
                f"{circularidad:.3f}"
            ))

        # 5. Funciones de Interacción
        def actualizar_vista(id_seleccionado=None):
            img_show = img_vis.copy()

            # Dibujar todos los granos (tenues)
            for dato in datos_granos:
                bx, by, bw, bh = dato['bbox']
                color = (200, 200, 200)  # Gris
                cv2.rectangle(img_show, (bx, by), (bx + bw, by + bh), color, 1)

            # Resaltar seleccionado
            titulo = "Vista General"
            if id_seleccionado is not None:
                # Buscar datos
                seleccionado = next((d for d in datos_granos if d['id'] == id_seleccionado), None)
                if seleccionado:
                    sx, sy, sw, sh = seleccionado['bbox']

                    # Color según clase
                    colores = {'dark': (0, 0, 0), 'medium': (0, 140, 255), 'light': (0, 255, 255), 'green': (0, 255, 0)}
                    c = colores.get(seleccionado['clase'], (0, 0, 255))

                    # Rectángulo grueso
                    cv2.rectangle(img_show, (sx, sy), (sx + sw, sy + sh), c, 3)

                    # Rellenar contorno (transparente)
                    overlay = img_show.copy()
                    cv2.drawContours(overlay, [seleccionado['contorno']], -1, c, -1)
                    cv2.addWeighted(overlay, 0.4, img_show, 0.6, 0, img_show)

                    titulo = f"Grano ID {id_seleccionado}: {seleccionado['clase'].upper()}"

            # Mostrar en matplotlib
            ax_preview.clear()
            ax_preview.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
            ax_preview.set_title(titulo, fontsize=10, fontweight='bold')
            ax_preview.axis('off')
            canvas_preview.draw()

        def al_seleccionar_fila(event):
            seleccion = tree.selection()
            if seleccion:
                item = tree.item(seleccion[0])
                id_sel = item['values'][0]
                actualizar_vista(id_sel)

        # Vincular evento
        tree.bind("<<TreeviewSelect>>", al_seleccionar_fila)

        # Botón para exportar CSV
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(pady=10, fill=tk.X)

        def exportar_csv():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                with open(path, 'w') as f:
                    f.write("ID,Clase,Area,Perimetro,Circularidad\n")
                    for d in datos_granos:
                        f.write(f"{d['id']},{d['clase']},{d['area']},{d['perimetro']},{d['circularidad']:.4f}\n")
                messagebox.showinfo("Éxito", "Datos exportados correctamente.")

        ttk.Button(btn_frame, text="💾 Exportar Datos a CSV", command=exportar_csv).pack(fill=tk.X)

        # Vista inicial
        actualizar_vista()

    def clasificar_multiples_granos(self):
        """
        Versión DEPURADA Y ACTUALIZADA (HSV):
        Usa el canal de Saturación para detectar granos ignorando sombras.
        """
        if not X_train_cafe or self.original_image is None:
            messagebox.showerror("Error", "Entrena el modelo y carga una imagen primero.")
            return

        try:
            # 1. Preparar imagen
            img_pil = self.original_image
            img_original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_vis = img_original.copy()

            # -----------------------------------------------------------
            # NUEVA BINARIZACIÓN ANTI-SOMBRAS (Usando Canal de Saturación)
            # -----------------------------------------------------------

            # A. Convertir a HSV y extraer Saturación
            hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(hsv)

            # B. Desenfoque suave en S (3x3 es suficiente ahora)
            s_blurred = cv2.GaussianBlur(S, (3, 3), 0)

            # C. Otsu sobre Saturación
            # La sombra (gris) se vuelve negro, el grano (color) se vuelve blanco
            _, binaria = cv2.threshold(s_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # D. Limpieza Morfológica
            kernel_clean = np.ones((3, 3), np.uint8)

            # "Cierre" (Close) para tapar agujeros de aceite dentro del grano
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel_clean, iterations=2)

            # "Apertura" (Open) para quitar ruido del fondo
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_clean, iterations=1)

            # E. Inversión de seguridad (raro en S, pero necesario si el fondo tiene color y el objeto no)
            if cv2.countNonZero(binaria) > binaria.size / 2:
                binaria = cv2.bitwise_not(binaria)
                print("DEBUG: Invertí la máscara por seguridad.")

            # F. Separación de objetos pegados (Erosión final)
            # Usamos 3x3 para no comer demasiado grano ahora que no hay sombra
            binaria = cv2.erode(binaria, kernel_clean, iterations=1)

            # -----------------------------------------------------------

            # --- MOSTRAR LO QUE VE LA MÁQUINA (DEBUG) ---
            # Esto te permitirá confirmar que las sombras desaparecieron
            cv2.imshow("DEBUG: Mascara HSV (Saturacion)", binaria)
            cv2.waitKey(1)  # Puse 1 para que no bloquee, si quieres pausar pon 0
            # ---------------------------------------------

            # 3. Encontrar contornos
            contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            granos_detectados = 0

            for cnt in contornos:
                area = cv2.contourArea(cnt)

                # Filtro de ruido (ajustable según tu resolución)
                if area < 200: continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Recortar (con validación de bordes para no crashear)
                pad = 5
                y1, y2 = max(0, y - pad), min(img_original.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(img_original.shape[1], x + w + pad)

                roi_grano = img_original[y1:y2, x1:x2]

                # IMPORTANTE: Usar la máscara binaria NUEVA para el recorte
                roi_mask = binaria[y1:y2, x1:x2]

                if roi_grano.size == 0: continue

                try:
                    # --- EXTRACCIÓN Y CLASIFICACIÓN ---
                    # Ahora 'extraer_features_cafe' también debe estar actualizada a HSV
                    features = extraer_features_cafe(roi_grano, roi_mask)

                    if self.clasificador_activo.get() == "SVM":
                        clase_predicha, votos = self.clasificar_svm_cafe(features)
                        # print(f"SVM: {clase_predicha}")
                    else:
                        clase_predicha, votos = clasificar_knn_cafe(features, k=5)
                        # print(f"KNN: {clase_predicha}")
                    # ------------------------------------------------

                    # Dibujar
                    colores = {'dark': (0, 0, 0), 'medium': (0, 140, 255), 'light': (0, 255, 255), 'green': (0, 255, 0)}
                    c = colores.get(clase_predicha, (255, 0, 255))  # Rosa si falla el color

                    cv2.rectangle(img_vis, (x, y), (x + w, y + h), c, 2)
                    cv2.putText(img_vis, clase_predicha.upper(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                    granos_detectados += 1
                except Exception as e:
                    print(f"Error clasificando un grano: {e}")
                    continue

            # Mostrar resultado final en la interfaz Tkinter
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Detectados: {granos_detectados} (Modo: {self.clasificador_activo.get()})")
            ax.axis('off')
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error en clasificación múltiple: {e}")

    def ver_inspeccion_numerica(self):
        """
        Muestra los VALORES CRUDOS (Vectores) para comparar numéricamente las clases.
        Te ayuda a saber si el modelo distingue cosas reales o si los datos son basura.
        """
        if not X_train_cafe:
            messagebox.showwarning("Atención", "Entrena el modelo primero.")
            return

        ventana_insp = tk.Toplevel(self.root)
        ventana_insp.title("Inspección de Datos Numéricos (Rayos X)")
        ventana_insp.geometry("1100x600")

        # Pestañas
        tab_control = ttk.Notebook(ventana_insp)
        tab_resumen = ttk.Frame(tab_control)
        tab_detalle = ttk.Frame(tab_control)
        tab_control.add(tab_resumen, text='📊 Promedios (La "Firma" de la Clase)')
        tab_control.add(tab_detalle, text='🔢 Datos Crudos (Vectores)')
        tab_control.pack(expand=1, fill="both")

        # --- PESTAÑA 1: PROMEDIOS POR CLASE ---
        lbl_info = tk.Label(tab_resumen,
                            text="Si estos números son DIFERENTES entre clases, el modelo está funcionando bien.\n"
                                 "Si son IGUALES, el modelo está ciego.",
                            bg="#f0f0f0", pady=10, font=('Arial', 10))
        lbl_info.pack(fill=tk.X)

        # Columnas clave (Color RGB, Textura, Forma)
        cols = ["Clase", "Rojo (Norm)", "Verde (Norm)", "Azul (Norm)", "Textura (Contraste)", "Textura (Energía)",
                "Forma (Hu1)"]
        tree_avg = ttk.Treeview(tab_resumen, columns=cols, show="headings")

        for col in cols:
            tree_avg.heading(col, text=col)
            tree_avg.column(col, width=130, anchor="center")

        tree_avg.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Calcular promedios usando numpy
        X = np.array(X_train_cafe)
        y = np.array(y_train_cafe)
        clases_unicas = np.unique(y)

        for clase in clases_unicas:
            indices = np.where(y == clase)
            vectores_clase = X[indices]
            prom = np.mean(vectores_clase, axis=0)

            # Según tu función extraer_features_cafe:
            # 0=R, 1=G, 2=B, 3=Hu1 ... 10=Contraste ... 13=Energía
            vals = (
                clase.upper(),
                f"{prom[0]:.4f}",  # R
                f"{prom[1]:.4f}",  # G
                f"{prom[2]:.4f}",  # B
                f"{prom[10]:.4f}",  # Contraste
                f"{prom[13]:.4f}",  # Energía
                f"{prom[3]:.4f}"  # Hu1
            )
            tree_avg.insert("", "end", values=vals)

        # --- PESTAÑA 2: DETALLE DE VECTORES ---
        # Cabeceras cortas para que quepan
        headers = ["R", "G", "B"] + [f"Hu{i}" for i in range(1, 8)] + ["Con", "Dis", "Hom", "Ene", "Cor"]
        cols_det = ["ID", "Clase"] + [f"F{i}" for i in range(15)]

        tree_det = ttk.Treeview(tab_detalle, columns=cols_det, show="headings")
        tree_det.heading("ID", text="ID");
        tree_det.column("ID", width=40)
        tree_det.heading("Clase", text="Clase");
        tree_det.column("Clase", width=80)

        for i, h in enumerate(headers):
            col_id = f"F{i}"
            tree_det.heading(col_id, text=h)
            tree_det.column(col_id, width=55, anchor="center")

        sb = ttk.Scrollbar(tab_detalle, orient="vertical", command=tree_det.yview)
        tree_det.configure(yscrollcommand=sb.set)
        tree_det.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Llenar con las primeras 200 muestras
        for i in range(min(len(X), 200)):
            # Formatear números a 3 decimales
            datos_fmt = [f"{v:.3f}" for v in X[i]]
            tree_det.insert("", "end", values=[i, y[i]] + datos_fmt)

        # Botón Exportar
        def exportar():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                with open(path, 'w') as f:
                    f.write("Clase," + ",".join(headers) + "\n")
                    for i in range(len(X)):
                        row = [y[i]] + [str(v) for v in X[i]]
                        f.write(",".join(row) + "\n")
                messagebox.showinfo("Listo", "CSV Exportado.")

        ttk.Button(tab_detalle, text="💾 Descargar CSV (Para analizar en Excel)", command=exportar).pack(fill=tk.X)

    def ver_scaler_detallado(self):
        """
        Muestra los parámetros del scaler y estadísticas de normalización.
        """
        global scaler_min_cafe, scaler_max_cafe
        global scaler_mean_cafe, scaler_std_cafe
        global metodo_normalizacion

        if metodo_normalizacion == 'minmax' and scaler_min_cafe is None:
            messagebox.showwarning("Atención", "Entrena el modelo primero.")
            return
        if metodo_normalizacion == 'standard' and scaler_mean_cafe is None:
            messagebox.showwarning("Atención", "Entrena el modelo primero.")
            return

        ventana = tk.Toplevel(self.root)
        ventana.title(f"Parámetros del Scaler - {metodo_normalizacion.upper()}")
        ventana.geometry("700x500")

        # Crear notebook con tabs
        tab_control = ttk.Notebook(ventana)
        tab_params = ttk.Frame(tab_control)
        tab_graficas = ttk.Frame(tab_control)

        tab_control.add(tab_params, text='📊 Parámetros')
        tab_control.add(tab_graficas, text='📈 Visualización')
        tab_control.pack(expand=1, fill="both")

        # --- TAB 1: PARÁMETROS ---
        text = tk.Text(tab_params, wrap=tk.WORD, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(tab_params, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        headers = ["R", "G", "B"] + [f"Hu{i}" for i in range(1, 8)] + \
                  ["Contraste", "Disim", "Homog", "Energía", "Correl"]

        text.insert(tk.END, f"MÉTODO ACTIVO: {metodo_normalizacion.upper()}\n")
        text.insert(tk.END, "=" * 70 + "\n\n")

        if metodo_normalizacion == 'minmax':
            text.insert(tk.END, "CARACTERÍSTICA | MIN | MAX | RANGO\n")
            text.insert(tk.END, "-" * 70 + "\n")

            for i, h in enumerate(headers):
                rango = scaler_max_cafe[i] - scaler_min_cafe[i]
                text.insert(tk.END, f"{h:12} | {scaler_min_cafe[i]:8.4f} | {scaler_max_cafe[i]:8.4f} | {rango:8.4f}\n")

        elif metodo_normalizacion == 'standard':
            text.insert(tk.END, "CARACTERÍSTICA | MEDIA | DESV.STD | COEF.VAR\n")
            text.insert(tk.END, "-" * 70 + "\n")

            for i, h in enumerate(headers):
                cv = (scaler_std_cafe[i] / (scaler_mean_cafe[i] + 1e-10)) * 100
                text.insert(tk.END, f"{h:12} | {scaler_mean_cafe[i]:8.4f} | {scaler_std_cafe[i]:8.4f} | {cv:8.2f}%\n")

        text.config(state=tk.DISABLED)

        # --- TAB 2: GRÁFICAS ---
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        fig = Figure(figsize=(9, 6), dpi=90)

        if metodo_normalizacion == 'minmax':
            ax = fig.add_subplot(111)
            rangos = scaler_max_cafe - scaler_min_cafe
            ax.bar(range(len(headers)), rangos, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(headers)))
            ax.set_xticklabels(headers, rotation=45, ha='right')
            ax.set_ylabel('Rango (Max - Min)')
            ax.set_title('Rango de Valores por Característica (Min-Max)')
            ax.grid(axis='y', alpha=0.3)

        elif metodo_normalizacion == 'standard':
            ax1 = fig.add_subplot(211)
            ax1.bar(range(len(headers)), scaler_mean_cafe, color='green', alpha=0.6)
            ax1.set_xticks(range(len(headers)))
            ax1.set_xticklabels(headers, rotation=45, ha='right')
            ax1.set_ylabel('Media')
            ax1.set_title('Media por Característica')
            ax1.grid(axis='y', alpha=0.3)

            ax2 = fig.add_subplot(212)
            ax2.bar(range(len(headers)), scaler_std_cafe, color='orange', alpha=0.6)
            ax2.set_xticks(range(len(headers)))
            ax2.set_xticklabels(headers, rotation=45, ha='right')
            ax2.set_ylabel('Desviación Estándar')
            ax2.set_title('Variabilidad por Característica')
            ax2.grid(axis='y', alpha=0.3)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=tab_graficas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def ver_analisis_entrenamiento(self):
        """
        Versión FINAL SIN LÍMITES (CORREGIDA).
        Lee TODAS las imágenes y permite elegir entre SVM o KNN para la matriz.
        """
        if not X_train_cafe:
            messagebox.showwarning("Atención", "Entrena el modelo primero.")
            return

        # 1. Preguntar fuente de datos
        usar_ram = messagebox.askyesno("Fuente de Datos",
                                       "¿Quieres analizar los datos que YA tienes en memoria (Entrenamiento)?\n\n"
                                       "✅ SÍ: Usar datos de RAM (Instantáneo).\n"
                                       "❌ NO: Cargar carpeta TEST (Tarda más, pero es la validación real).")

        X_t = None
        y_t = None
        nombres_t = None
        conteo = {}

        if usar_ram:
            # --- OPCIÓN A: USAR MEMORIA RAM ---
            try:
                X_t = np.array(X_train_cafe)
                y_t = np.array(y_train_cafe)
                nombres_t = np.array([f"Grano_Train_{i}" for i in range(len(y_t))])

                clases_unicas = np.unique(y_t)
                conteo = {c: np.sum(y_t == c) for c in clases_unicas}
                messagebox.showinfo("Éxito", f"Datos cargados desde RAM: {len(X_t)} muestras.")
            except Exception as e:
                messagebox.showerror("Error", f"Error leyendo RAM: {e}")
                return

        else:
            # --- OPCIÓN B: CARGAR DESDE CARPETA (SIN LÍMITE) ---
            test_path = filedialog.askdirectory(title="Seleccionar carpeta (TEST o TRAIN)")
            if not test_path: return

            ventana_loading = tk.Toplevel(self.root)
            ventana_loading.title("Cargando datos COMPLETOS...")
            # Barra de progreso indeterminada
            pbar = ttk.Progressbar(ventana_loading, mode='indeterminate')
            pbar.pack(pady=10, padx=20, fill=tk.X)
            pbar.start(10)

            lbl_status = ttk.Label(ventana_loading,
                                   text="Leyendo todas las imágenes...\nEsto puede tardar unos minutos.")
            lbl_status.pack(pady=10)

            self.root.update()

            X_list = []
            y_list = []
            nombres_list = []
            clases = ['dark', 'green', 'light', 'medium']
            conteo = {c: 0 for c in clases}

            try:
                for clase in clases:
                    path_clase = os.path.join(test_path, clase)
                    if not os.path.exists(path_clase): continue

                    imagenes = os.listdir(path_clase)

                    # Actualizar texto para que el usuario no se desespere
                    lbl_status.config(text=f"Procesando carpeta: {clase} ({len(imagenes)} fotos)...")
                    self.root.update()

                    for img_name in imagenes:
                        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                        full_path = os.path.join(path_clase, img_name)
                        with open(full_path, "rb") as f:
                            bytes_img = bytearray(f.read())
                            np_arr = np.asarray(bytes_img, dtype=np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if img is not None:
                            feats = extraer_features_cafe(img)
                            X_list.append(feats)
                            y_list.append(clase)
                            nombres_list.append(img_name)
                            conteo[clase] += 1

                X_t = np.array(X_list)
                y_t = np.array(y_list)
                nombres_t = np.array(nombres_list)

            except Exception as e:
                ventana_loading.destroy()
                messagebox.showerror("Error", f"Error leyendo carpeta: {e}")
                return

            ventana_loading.destroy()

        if X_t is None or len(X_t) == 0:
            messagebox.showerror("Error", "No hay datos para graficar.")
            return

        # --- GRAFICAR ---
        try:
            # Barajar datos
            indices_random = np.arange(len(X_t))
            np.random.shuffle(indices_random)
            X_t = X_t[indices_random]
            y_t = y_t[indices_random]
            nombres_t = nombres_t[indices_random]

            ventana_analisis = tk.Toplevel(self.root)
            origen = "RAM" if usar_ram else "CARPETA COMPLETA"
            ventana_analisis.title(f"Análisis Completo - {len(X_t)} Muestras - Fuente: {origen}")
            ventana_analisis.geometry("1100x700")

            tab_control = ttk.Notebook(ventana_analisis)
            tab1 = ttk.Frame(tab_control)
            tab2 = ttk.Frame(tab_control)
            tab3 = ttk.Frame(tab_control)

            tab_control.add(tab1, text='🌌 3D Interactivo')
            tab_control.add(tab2, text='🗺️ 2D Plano')
            tab_control.add(tab3, text='🎯 Matriz de Confusión')
            tab_control.pack(expand=1, fill="both")

            # PCA
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_t)

            colors_map = {'dark': 'black', 'medium': 'orange', 'light': '#CCCC00', 'green': 'green'}
            colors = [colors_map.get(c, 'blue') for c in y_t]
            clases_presentes = np.unique(y_t)

            # PESTAÑA 1: 3D
            fig1 = plt.Figure(figsize=(8, 6), dpi=100)
            ax1 = fig1.add_subplot(111, projection='3d')
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=colors, s=20, alpha=0.6, edgecolors='none',
                        picker=5)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_map.get(c, 'blue'), label=c,
                       markersize=10) for c in clases_presentes]
            ax1.legend(handles=legend_elements)
            ax1.set_title(f'Distribución 3D ({len(X_t)} puntos)')

            def on_pick(event):
                try:
                    ind = event.ind[0]
                    ax1.set_title(f"Punto: {nombres_t[ind]} ({y_t[ind]})", color='red')
                    canvas1.draw()
                except:
                    pass

            canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas1.mpl_connect('pick_event', on_pick)

            # PESTAÑA 2: 2D
            fig2 = plt.Figure(figsize=(8, 6), dpi=100)
            ax2 = fig2.add_subplot(111)
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=20, alpha=0.5, edgecolors='none')
            ax2.legend(handles=legend_elements)
            ax2.set_title(f"Proyección 2D")
            ax2.grid(True, alpha=0.3)

            canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
            canvas2.draw()
            canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # ---------------------------------------------------------
            # PESTAÑA 3: MATRIZ DE CONFUSIÓN (MODIFICADO AQUÍ)
            # ---------------------------------------------------------
            y_pred = []

            # Preguntamos si el SVM está disponible
            usar_svm = False
            if modelo_svm_cafe is not None:
                # Preguntamos al usuario qué modelo quiere visualizar
                usar_svm = messagebox.askyesno("Elegir Modelo para Matriz",
                                               "¿Quieres generar la matriz usando el modelo SVM?\n\n"
                                               "✅ SÍ = Usar SVM\n"
                                               "❌ NO = Usar KNN (Por defecto)")

            # Título dinámico para la gráfica
            nombre_modelo = "SVM" if usar_svm else "KNN"

            for vector in X_t:
                if usar_svm:
                    # Usamos SVM (llamada interna con self)
                    pred, _ = self.clasificar_svm_cafe(vector)
                else:
                    # Usamos KNN (función global)
                    pred, _ = clasificar_knn_cafe(vector, k=5)

                y_pred.append(pred)
            # ---------------------------------------------------------

            labels_reales = [l for l in ['dark', 'green', 'light', 'medium'] if l in clases_presentes]
            cm = confusion_matrix(y_t, y_pred, labels=labels_reales)

            fig3 = plt.Figure(figsize=(8, 6), dpi=100)
            ax3 = fig3.add_subplot(111)
            cax = ax3.matshow(cm, cmap='Blues')
            fig3.colorbar(cax)

            ax3.set_xticks(np.arange(len(labels_reales)))
            ax3.set_yticks(np.arange(len(labels_reales)))
            ax3.set_xticklabels(labels_reales, rotation=45)
            ax3.set_yticklabels(labels_reales)

            for i in range(len(labels_reales)):
                for j in range(len(labels_reales)):
                    ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                             color='white' if cm[i, j] > cm.max() / 2 else 'black')

            precision = (np.trace(cm) / np.sum(cm)) * 100
            # Título actualizado con el nombre del modelo
            ax3.set_title(f'Matriz de Confusión: {nombre_modelo} (Precisión: {precision:.1f}%)')
            ax3.set_xlabel('Predicción')
            ax3.set_ylabel('Real')

            canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
            canvas3.draw()
            canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar: {e}")
    # --- MÉTODOS PARA SVM ---

    def entrenar_svm_interno(self):
        """
        Entrena el modelo SVM usando los datos globales cargados en X_train_cafe.
        Justificación: Clasificación mediante vectores de características y SVM con kernel gaussiano.
        """
        global X_train_cafe, y_train_cafe, modelo_svm_cafe

        if not X_train_cafe:
            return False

        try:
            # Creamos el SVM con kernel Radial Basis Function (Gaussiano)
            # probability=True es necesario para obtener porcentajes de confianza
            modelo_svm_cafe = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

            modelo_svm_cafe.fit(X_train_cafe, y_train_cafe)
            return True
        except Exception as e:
            print(f"Error entrenando SVM: {e}")
            return False

    def clasificar_svm_cafe(self, vector_features):
        """
        Clasifica un vector usando el modelo SVM entrenado.
        Retorna: clase_predicha, diccionario_probabilidades
        """

        if modelo_svm_cafe is None:
            return "Modelo no entrenado", {}

        # 1. Normalizar el vector (CRÍTICO PARA SVM)
        # El SVM es muy sensible a escalas, usamos la misma normalización que en el training
        vector_norm = normalizar_vector(vector_features)

        # SVM de sklearn espera una matriz 2D (1 muestra, N features)
        vector_2d = vector_norm.reshape(1, -1)

        # 2. Predecir Clase
        prediccion = modelo_svm_cafe.predict(vector_2d)[0]

        # 3. Obtener Probabilidades (Confianza)
        try:
            probs = modelo_svm_cafe.predict_proba(vector_2d)[0]
            clases = modelo_svm_cafe.classes_

            # Formatear como diccionario para que sea compatible con tu código existente
            votos = {clase: prob for clase, prob in zip(clases, probs)}

        except:
            # Fallback si probability=False
            votos = {prediccion: 1.0}

        return prediccion, votos

# --- 4. Bloque Principal de Ejecución ---
if __name__ == "__main__":
    root = tk.Tk()
    app = MatplotlibApp(root)
    root.mainloop()