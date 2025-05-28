import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import base64
import os
import traceback

# Try to import your custom functions - with error handling
try:
    from processamento import extrair_dados_da_imagem

    PROCESSAMENTO_AVAILABLE = True
except ImportError:
    st.error("⚠️ Módulo 'processamento' não encontrado. Algumas funcionalidades estarão limitadas.")
    PROCESSAMENTO_AVAILABLE = False

try:
    from recomendacao import recomendar_roupas

    RECOMENDACAO_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Módulo 'recomendacao' não encontrado. Usando versão simplificada.")
    RECOMENDACAO_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Análise de Coloração Pessoal",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .color-box {
        display: inline-block;
        width: 60px;
        height: 60px;
        margin: 5px;
        border: 2px solid #ddd;
        border-radius: 8px;
        text-align: center;
        vertical-align: top;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def pil_to_opencv(pil_image):
    """Convert PIL image to OpenCV format"""
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR (OpenCV uses BGR)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def opencv_to_base64(cv_img):
    """Convert OpenCV image to base64 for display in Streamlit"""
    _, buffer = cv2.imencode('.png', cv_img)
    img_base64 = base64.b64encode(buffer).decode()
    return f"data:image/png;base64,{img_base64}"


def opencv_to_pil(cv_img):
    """Convert OpenCV image to PIL using io buffer"""
    # Convert BGR to RGB
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Encode to PNG in memory
    _, buffer = cv2.imencode('.png', cv_img_rgb)
    # Use io.BytesIO to create a bytes buffer
    img_buffer = io.BytesIO(buffer)
    # Convert to PIL Image
    pil_img = Image.open(img_buffer)
    return pil_img


def create_downloadable_image(cv_img, filename="analysis_result.png"):
    """Create a downloadable image using io buffer"""
    # Convert to PIL first
    pil_img = opencv_to_pil(cv_img)

    # Create buffer for download
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)

    return buffer.getvalue(), filename


def criar_painel_cores(medidas):
    """Create a color panel showing extracted colors"""
    painel = np.full((400, 600, 3), 240, dtype=np.uint8)
    y_pos = 50

    # Skin tone
    if 'tom_de_pele' in medidas:
        cv2.putText(painel, "Tom de Pele:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_pele = tuple(map(int, medidas['tom_de_pele']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_pele, -1)
        cv2.putText(painel, f"BGR: {list(cor_pele)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Hair tone
    if 'tom_de_cabelo' in medidas and not medidas.get('pouco_cabelo', True):
        cv2.putText(painel, "Tom de Cabelo:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_cabelo = tuple(map(int, medidas['tom_de_cabelo']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_cabelo, -1)
        cv2.putText(painel, f"BGR: {list(cor_cabelo)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Eye tone
    if 'tom_de_olho' in medidas:
        cv2.putText(painel, "Tom dos Olhos:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cor_olho = tuple(map(int, medidas['tom_de_olho']))
        cv2.rectangle(painel, (200, y_pos - 20), (300, y_pos + 20), cor_olho, -1)
        cv2.putText(painel, f"BGR: {list(cor_olho)}", (320, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 80

    # Classification
    if 'Classificação' in medidas:
        cv2.putText(painel, f"Contraste: {medidas['Classificação'].capitalize()}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_pos += 40

    if 'Subtom' in medidas:
        cv2.putText(painel, f"Subtom: {medidas['Subtom'].capitalize()}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return painel


def criar_visualizacoes(imagem, medidas, resultado=None):
    """Create visualizations replacing cv2.imshow()"""
    visualizacoes = {}

    # 1. Original image
    visualizacoes['original'] = imagem.copy()

    # 2. Body analysis (equivalent to your visualizar_resultados)
    if resultado and hasattr(resultado, 'pose_landmarks') and resultado.pose_landmarks:
        try:
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
        except ImportError:
            st.warning("MediaPipe não disponível para visualização de landmarks")

    # 3. Color panel
    painel_cores = criar_painel_cores(medidas)
    visualizacoes['cores'] = painel_cores

    return visualizacoes


def gerar_recomendacoes_web(dicionario):
    """Web version of clothing recommendation function"""
    try:
        # Try multiple paths for the CSV file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '../../trabalho/data/catalogo_roupas.csv'),
            os.path.join(os.getcwd(), 'data', 'catalogo_roupas.csv'),
            os.path.join(os.getcwd(), 'catalogo_roupas.csv'),
            'catalogo_roupas.csv'
        ]

        caminho_csv = None
        for path in possible_paths:
            if os.path.exists(path):
                caminho_csv = path
                break

        if not caminho_csv:
            st.error(
                "❌ Arquivo CSV do catálogo não encontrado. Certifique-se de que 'catalogo_roupas.csv' está no diretório correto.")
            return []

        catalogo = pd.read_csv(caminho_csv)
        catalogo.columns = catalogo.columns.str.strip().str.lower()

        # Create copy of catalog
        roupas_filtradas = catalogo.copy()

        # Recommendation rules
        classificacao = dicionario.get('Classificação', '').lower()
        subtom = dicionario.get('Subtom', '').lower()

        if "baixo contraste escuro" in classificacao or "baixo contraste claro" in classificacao:
            if 'contraste' in roupas_filtradas.columns:
                roupas_filtradas = roupas_filtradas[
                    roupas_filtradas['contraste'].str.contains("contraste baixo", case=False, na=False)]
        else:
            if 'estação' in roupas_filtradas.columns:
                if "quente" in subtom:
                    roupas_filtradas = roupas_filtradas[
                        roupas_filtradas['estação'].str.contains("quente|verão", case=False, na=False)]
                elif "frio" in subtom:
                    roupas_filtradas = roupas_filtradas[
                        roupas_filtradas['estação'].str.contains("frio|inverno", case=False, na=False)]
                elif "neutro" in subtom:
                    roupas_filtradas = roupas_filtradas[
                        roupas_filtradas['estação'].str.contains("puro|pura", case=False, na=False)]

        # Convert string "[146 28 63]" to list [146, 28, 63]
        if 'cor bgr' in roupas_filtradas.columns:
            roupas_filtradas["cor bgr"] = roupas_filtradas["cor bgr"].apply(
                lambda x: list(map(int, str(x).strip("[]").split())) if pd.notna(x) else [0, 0, 0]
            )

        # Filter by type (shirts)
        # if 'tipo' in roupas_filtradas.columns:
        #     roupas_filtradas = roupas_filtradas[roupas_filtradas["tipo"] == "camisa"]

        # Extract colors
        cores_bgr = []
        for _, row in roupas_filtradas.iterrows():
            if 'cor bgr' in row and isinstance(row['cor bgr'], list) and len(row['cor bgr']) == 3:
                cores_bgr.append(row['cor bgr'])

        return cores_bgr

    except Exception as e:
        st.error(f"Erro ao processar recomendações: {str(e)}")
        return []


def create_color_palette_report(cores_bgr, medidas):
    """Create a downloadable text report of the color analysis using io"""
    report_buffer = io.StringIO()

    # Write header
    report_buffer.write("RELATÓRIO DE ANÁLISE DE COLORAÇÃO PESSOAL\n")
    report_buffer.write("=" * 50 + "\n\n")

    # Write personal analysis
    report_buffer.write("ANÁLISE PESSOAL:\n")
    report_buffer.write("-" * 20 + "\n")

    for key, value in medidas.items():
        if key in ['Classificação', 'Subtom', 'Tom de pele (escala 0-10)',
                   'Tom de cabelo (escala 0-10)', 'Tom dos olhos (escala 0-10)']:
            report_buffer.write(f"{key}: {value}\n")

    report_buffer.write("\n")

    # Write recommended colors
    report_buffer.write("CORES RECOMENDADAS (RGB):\n")
    report_buffer.write("-" * 30 + "\n")

    for i, cor_bgr in enumerate(cores_bgr, 1):
        cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])  # Convert BGR to RGB
        cor_hex = f"#{cor_rgb[0]:02x}{cor_rgb[1]:02x}{cor_rgb[2]:02x}"
        report_buffer.write(f"Cor {i:2d}: RGB{cor_rgb} - HEX: {cor_hex}\n")

    report_buffer.write(f"\nTotal de cores recomendadas: {len(cores_bgr)}\n")
    report_buffer.write("\nRelatório gerado automaticamente pelo sistema de análise de coloração pessoal.\n")

    # Get the content and close the buffer
    content = report_buffer.getvalue()
    report_buffer.close()

    return content


def display_color_grid(cores_bgr):
    """Display colors in a grid format"""
    if not cores_bgr:
        st.warning("Nenhuma cor encontrada para exibir.")
        return

    st.markdown("### 🎨 Cores Recomendadas para Você")

    # Create color grid
    cols_per_row = 5
    rows = len(cores_bgr) // cols_per_row + (1 if len(cores_bgr) % cols_per_row else 0)

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            color_idx = row * cols_per_row + col_idx
            if color_idx < len(cores_bgr):
                with cols[col_idx]:
                    cor_bgr = cores_bgr[color_idx]
                    # Convert BGR to RGB for correct display
                    cor_rgb = (cor_bgr[2], cor_bgr[1], cor_bgr[0])
                    cor_hex = f"#{cor_rgb[0]:02x}{cor_rgb[1]:02x}{cor_rgb[2]:02x}"

                    # Create colored square
                    st.markdown(f"""
                    <div style="
                        width: 80px; 
                        height: 80px; 
                        background-color: {cor_hex}; 
                        border: 2px solid #ddd;
                        border-radius: 8px;
                        margin: 5px auto;
                    "></div>
                    <p style="text-align: center; font-size: 10px; margin: 0;">
                        RGB: {cor_rgb}
                    </p>
                    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Análise de Coloração Pessoal</h1>', unsafe_allow_html=True)
    st.markdown("**Upload uma foto para análise completa das suas características de cor e estilo!**")

    # Sidebar with instructions
    with st.sidebar:
        st.header("📋 Instruções")
        st.markdown("""
        **Para melhores resultados:**
        - Coloque uma foto de corpo inteiro
        - Use uma foto com boa iluminação natural
        - Certifique-se de que o rosto está bem visível
        - Evite filtros ou edições na imagem
        - Use fundo neutro se possível
        """)

        st.header("ℹ️ Sobre a Análise")
        st.markdown("""
        Esta ferramenta analisa:
        - Tom de pele, cabelo e olhos
        - Contraste facial
        - Subtom (quente/frio/neutro/oliva)
        - Recomendações de cores para roupas
        """)

    # File upload
    uploaded_file = st.file_uploader(
        "Escolha uma imagem",
        type=['png', 'jpg', 'jpeg'],
        help="Faça upload de uma foto com boa iluminação"
    )

    if uploaded_file is not None:
        # Show uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📸 Imagem Enviada")
            st.image(image, caption="Sua foto", use_container_width=True)

        with col2:
            st.subheader("📊 Resultados da Análise")

            if not PROCESSAMENTO_AVAILABLE:
                st.error(
                    "Módulo de processamento não disponível. Verifique se o arquivo 'processamento.py' está presente.")
                st.stop()  # Use stop() em vez de return para interromper a execução

            # Executa a análise automaticamente
            with st.spinner("Analisando sua coloração pessoal..."):
                try:
                    # Convert PIL to OpenCV
                    cv_image = pil_to_opencv(image)

                    # Call your analysis function
                    medidas, resultado = extrair_dados_da_imagem(cv_image)

                    # Create visualizations
                    visualizacoes = criar_visualizacoes(cv_image, medidas, resultado)

                    # Store in session_state
                    st.session_state.medidas = medidas
                    st.session_state.visualizacoes = visualizacoes
                    st.session_state.analysis_complete = True


                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")
                    st.code(traceback.format_exc())


            # Display results if they exist - MOVED INSIDE COL2
            if st.session_state.get('analysis_complete', False) and 'medidas' in st.session_state:

                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.markdown("### 🧍 Medidas Corporais")
                    medidas_corporais = {
                        k: v for k, v in st.session_state.medidas.items()
                        if k in ['altura_total', 'largura_ombros', 'largura_quadril', 'proporção', 'tipo_corpo']
                    }
                    if medidas_corporais:
                        for key, value in medidas_corporais.items():
                            st.metric(key.replace('_', ' ').title(), value)
                    else:
                        st.info("Medidas corporais não detectadas")

                with col_res2:
                    st.markdown("### 🎨 Análise de Cores")
                    analise_cores = {
                        k: v for k, v in st.session_state.medidas.items()
                        if k in ['Classificação', 'Subtom', 'Tom de pele (escala 0-10)',
                                 'Tom de cabelo (escala 0-10)', 'Tom dos olhos (escala 0-10)']
                    }
                    if analise_cores:
                        for key, value in analise_cores.items():
                            st.metric(key, value)
                    else:
                        st.info("Análise de cores não disponível")

        # Section 2: Visualizations
        st.divider()
        st.subheader("🖼️ Visualizações da Análise")

        if 'visualizacoes' in st.session_state:
            tabs = st.tabs(["📸 Original", "🔍 Landmarks", "🎨 Painel de Cores"])

            with tabs[0]:
                if 'original' in st.session_state.visualizacoes:
                    st.image(st.session_state.visualizacoes['original'],
                             caption="Imagem Original", channels="BGR")

            with tabs[1]:
                if 'landmarks' in st.session_state.visualizacoes:
                    st.image(st.session_state.visualizacoes['landmarks'],
                             caption="Detecção de Landmarks Corporais", channels="BGR")

                    # Add download button for landmarks image
                    try:
                        img_data, filename = create_downloadable_image(
                            st.session_state.visualizacoes['landmarks'],
                            "landmarks_analysis.png"
                        )
                        st.download_button(
                            label="📥 Baixar Análise de Landmarks",
                            data=img_data,
                            file_name=filename,
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Erro ao preparar download: {e}")
                else:
                    st.info("Landmarks não detectados ou MediaPipe não disponível")

            with tabs[2]:
                if 'cores' in st.session_state.visualizacoes:
                    st.image(st.session_state.visualizacoes['cores'],
                             caption="Análise de Tons", channels="BGR")

                    # Add download button for color analysis
                    try:
                        img_data, filename = create_downloadable_image(
                            st.session_state.visualizacoes['cores'],
                            "color_analysis.png"
                        )
                        st.download_button(
                            label="📥 Baixar Painel de Cores",
                            data=img_data,
                            file_name=filename,
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Erro ao preparar download: {e}")

        # Section 3: Clothing recommendations
        st.divider()
        st.subheader("👗 Recomendações de Roupas")

        if st.button("🛍️ Gerar Recomendações", type="secondary", use_container_width=True):
            with st.spinner("Buscando roupas ideais para você..."):
                try:
                    cores_recomendadas = gerar_recomendacoes_web(st.session_state.medidas)

                    if cores_recomendadas:
                        st.success("✅ Recomendações geradas!")
                        display_color_grid(cores_recomendadas)

                        # Create downloadable color palette
                        try:
                            palette_data = create_color_palette_report(cores_recomendadas, st.session_state.medidas)
                            st.download_button(
                                label="📥 Baixar Relatório de Cores",
                                data=palette_data,
                                file_name="color_palette_report.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Erro ao criar relatório: {e}")
                    else:
                        st.warning(
                            "⚠️ Nenhuma roupa recomendada encontrada. Verifique se o arquivo CSV do catálogo está disponível.")

                except Exception as e:
                    st.error(f"Erro nas recomendações: {str(e)}")
                    st.code(traceback.format_exc())

        # Complete dictionary (expandable)
        with st.expander("📋 Ver Dicionário Completo de Análise"):
            st.json(st.session_state.medidas)

if __name__ == "__main__":
    main()
