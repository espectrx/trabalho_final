import os
from tkinter import filedialog
from tkinter import Tk
import cv2  # manipula imagens_roupas
import numpy as np
import matplotlib.pyplot as plt

def vibrance_contraste_suave(imagem):
    # CLAHE muito leve no canal L (clareza sem exagero)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))
    img_bgr_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)

    # Conversão para HSV para aplicar vibrance "manual"
    hsv = cv2.cvtColor(img_bgr_clahe, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)

    # Vibrance: aumenta mais onde a saturação é baixa
    vibrance_mask = s < 150  # onde a saturação é média ou baixa
    s[vibrance_mask] *= 1.25  # aumento seletivo
    s = np.clip(s, 0, 255)

    hsv_vibrant = cv2.merge([h, s, v])
    result_bgr = cv2.cvtColor(hsv_vibrant.astype("uint8"), cv2.COLOR_HSV2BGR)

    return result_bgr

# Cria a janela para escolher o arquivo
Tk().withdraw()  # Oculta a janela principal do tkinter
caminho = filedialog.askopenfilename(title="Selecione a imagem",
                                     filetypes=[("Arquivos de imagem", "*.jpg;*.jpeg;*.png")])


if not os.path.exists(caminho):
    print(f"Erro: imagem '{caminho}' não encontrada na pasta {os.getcwd()}")
else:
    imagem = cv2.imread(caminho)
    if imagem is None:
        print("Erro ao carregar a imagem")
    else:
        # Carregamento da imagem
        img = cv2.imread("/mnt/data/45c92eeb-79d9-4194-90d0-83d1a410258b.png")
        resultado = vibrance_contraste_suave(imagem)

        # Exibe o resultado
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Com Cores Realçadas")
        plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()