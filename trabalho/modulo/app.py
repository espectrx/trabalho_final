import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64

# Importa suas funções
from processamento import extrair_dados_da_imagem
from recomendacao import recomendar_roupas

# Configuração da página
st.set_page_config(
    page_title="Análise de Coloração Pessoal",
    page_icon="🎨",
    layout="wide"
)


# Função para converter imagem PIL para OpenCV
def pil_to_opencv(pil_image):
    open_cv_image = np.array(pil_image)
    # Converte RGB para BGR (OpenCV usa BGR)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


# Função para converter OpenCV para base64 (para exibir no Streamlit)
def opencv_to_base64(cv_img):
    _, buffer = cv2.imencode('.png', cv_img)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"


# Função para criar visualizações que substituem cv2.imshow()
def criar_visualizacoes(imagem, medidas, resultado):
    visualizacoes = {}

    # 1. Imagem original
    visualizacoes['original'] = imagem.copy()

    # 2. Análise corporal (equivalente ao seu visualizar_resultados)
    if resultado and resultado.pose_landmarks:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        imagem_landmarks = imagem.copy()
        mp_drawing.draw_landmarks(
            imagem_landmarks,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        visualizacoes['landmarks'] = imagem_landmarks

    # 3. Painel de cores
    painel_cores = criar_painel_cores(medidas)
    visualizacoes['cores'] = painel_cores

    return visualizacoes


def criar_painel_cores(medidas):
    # Cria um painel similar ao seu código
    painel = np.full((400, 600, 3), 240, dtype=np.uint8)

    y_pos = 50

    # Tom de pele
    if 'tom_de_pele' in medidas:
        cv2.putText(painel, "Tom de Pele:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_pele = tuple(map(int, medidas['tom_de_pele']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_pele, -1)
        cv2.putText(painel, f"BGR: {list(cor_pele)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Tom de cabelo
    if 'tom_de_cabelo' in medidas and not medidas.get('pouco_cabelo', True):
        cv2.putText(painel, "Tom de Cabelo:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_cabelo = tuple(map(int, medidas['tom_de_cabelo']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_cabelo, -1)
        cv2.putText(painel, f"BGR: {list(cor_cabelo)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Tom dos olhos
    if 'tom_de_olho' in medidas:
        cv2.putText(painel, "Tom dos Olhos:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_olho = tuple(map(int, medidas['tom_de_olho']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_olho, -1)
        cv2.putText(painel, f"BGR: {list(cor_olho)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Classificação
    if 'Classificação' in medidas:
        cv2.putText(painel, f"Contraste: {medidas['Classificação']}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        y_pos += 40

    if 'Subtom' in medidas:
        cv2.putText(painel, f"Subtom: {medidas['Subtom']}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    return painel


# Interface principal
def main():
    st.title("🎨 Análise de Coloração Pessoal")
    st.markdown("Upload uma foto para análise completa das suas características!")

    # Upload de imagem
    uploaded_file = st.file_uploader(
        "Escolha uma imagem",
        type=['png', 'jpg', 'jpeg'],
        help="Faça upload de uma foto com boa iluminação"
    )

    if uploaded_file is not None:
        # Mostra a imagem enviada
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📸 Imagem Enviada")
            st.image(image, caption="Sua foto", use_column_width=True)

        with col2:
            if st.button("🔍 Analisar Imagem", type="primary"):
                with st.spinner("Analisando sua coloração pessoal..."):
                    # Converte PIL para OpenCV
                    cv_image = pil_to_opencv(image)

                    # Chama sua função de análise
                    try:
                        medidas, resultado = extrair_dados_da_imagem(cv_image)

                        # Cria as visualizações
                        visualizacoes = criar_visualizacoes(cv_image, medidas, resultado)

                        # Armazena no session_state
                        st.session_state.medidas = medidas
                        st.session_state.visualizacoes = visualizacoes

                        st.success("✅ Análise concluída!")

                    except Exception as e:
                        st.error(f"Erro na análise: {str(e)}")

    # Exibe resultados se existirem
    if 'medidas' in st.session_state:
        st.divider()

        # Seção 1: Dicionário de medidas
        st.subheader("📊 Resultados da Análise")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🧍 Medidas Corporais")
            medidas_corporais = {
                k: v for k, v in st.session_state.medidas.items()
                if k in ['altura_total', 'largura_ombros', 'largura_quadril', 'proporção', 'tipo_corpo']
            }
            st.json(medidas_corporais)

        with col2:
            st.markdown("### 🎨 Análise de Cores")
            analise_cores = {
                k: v for k, v in st.session_state.medidas.items()
                if k in ['Classificação', 'Subtom', 'Tom de pele (escala 0-10)',
                         'Tom de cabelo (escala 0-10)', 'Tom dos olhos (escala 0-10)']
            }
            st.json(analise_cores)

        # Seção 2: Visualizações (substitui cv2.imshow)
        st.divider()
        st.subheader("🖼️ Visualizações da Análise")

        tabs = st.tabs(["📸 Original", "🔍 Landmarks", "🎨 Painel de Cores"])

        with tabs[0]:
            if 'original' in st.session_state.visualizacoes:
                st.image(st.session_state.visualizacoes['original'],
                         caption="Imagem Original", channels="BGR")

        with tabs[1]:
            if 'landmarks' in st.session_state.visualizacoes:
                st.image(st.session_state.visualizacoes['landmarks'],
                         caption="Detecção de Landmarks Corporais", channels="BGR")

        with tabs[2]:
            if 'cores' in st.session_state.visualizacoes:
                st.image(st.session_state.visualizacoes['cores'],
                         caption="Análise de Tons", channels="BGR")

        # Seção 3: Recomendações de roupas
        st.divider()
        st.subheader("👗 Recomendações de Roupas")

        if st.button("🛍️ Gerar Recomendações"):
            with st.spinner("Buscando roupas ideais para você..."):
                try:
                    # Chama sua função de recomendação
                    # Nota: você precisará adaptar a função para retornar dados em vez de usar cv2.imshow
                    st.info("Função de recomendação integrada! (adapte recomendar_roupas para retornar dados)")

                except Exception as e:
                    st.error(f"Erro nas recomendações: {str(e)}")

        # Dicionário completo (expandível)
        with st.expander("📋 Ver Dicionário Completo"):
            st.json(st.session_state.medidas)


if __name__ == "__main__":
    main()