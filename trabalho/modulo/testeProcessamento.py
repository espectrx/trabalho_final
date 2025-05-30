import os
from tkinter import filedialog
from tkinter import Tk
import cv2
from processamento import extrair_dados_da_imagem, visualizar_resultados
from recomendacao import recomendar_roupas

if __name__ == "__main__":
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
            medidas, resultado = extrair_dados_da_imagem(imagem)

            print("Medidas extraídas:")
            for k, v in medidas.items():
                print(f'{k}:{v}')

            if 'tom_de_pele' in medidas:
                tom_de_cabelo = medidas.get('tom_de_cabelo', None)
                visualizar_resultados(imagem,
                                      resultado,
                                      medidas['tom_de_pele'],
                                      medidas['pouco_cabelo'],
                                      tom_de_cabelo,
                                      medidas['tom_de_olho'])
            recomendar_roupas(medidas)