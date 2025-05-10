import cv2  # manipula imagens_roupas
import mediapipe as mp  # detecta as partes do corpo
import numpy as np
import matplotlib.pyplot as plt

medidas = {}
def extrair_dados_da_imagem(imagem):
    # recebe uma imagem (formato BGR do OpenCV) e retorna um dicionário com medidas extraídas dela
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh

    # Converter para RGB uma única vez
    img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # Processar pose e face juntos (melhor desempenho)
    with mp_pose.Pose(static_image_mode=True) as pose, mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
    ) as face_mesh:

        resultado = pose.process(img_rgb)
        resultado_face = face_mesh.process(img_rgb)

    # CORPO -------------------------------------------------------------------------------------
    if resultado.pose_landmarks:
        try:
            # ALTURA TOTAL (CABEÇA AO TORNOZELO)
            topo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]  # ou .LEFT_EYE
            tornozelo = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            altura_total = round(abs(tornozelo.y - topo.y), 2)
            medidas['altura_total'] = altura_total

            # DISTANCIA_OMBROS
            l_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            dx = (l_shoulder.x - r_shoulder.x)
            dy = (l_shoulder.y - r_shoulder.y)
            distancia = round(np.sqrt(dx ** 2 + dy ** 2), 2)
            medidas['largura_ombros'] = distancia

            # PROPORÇÃO TRONCO E PERNA
            quadril = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            altura_tronco = abs(l_shoulder.y - quadril.y)
            altura_pernas = abs(quadril.y - tornozelo.y)
            proporcao_tronco_pernas = round(altura_tronco / altura_pernas, 2)
            medidas['proporção'] = proporcao_tronco_pernas

            # LARGURA QUADRIL
            l_hip = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            dx_hip = (l_hip.x - r_hip.x)
            dy_hip = (l_hip.y - r_hip.y)
            largura_quadril = round(np.sqrt(dx_hip ** 2 + dy_hip ** 2), 2)
            medidas['largura_quadril'] = largura_quadril

            ombros = medidas.get('largura_ombros')
            quadril = medidas.get('largura_quadril')
            proporcao = medidas.get('proporção')
            diferenca = abs(ombros - quadril)

            if diferenca < 0.03:
                if proporcao < 0.9:
                    tipo_corpo = "Ampulheta"
                else:
                    tipo_corpo = "Retângulo"
            elif ombros > quadril:
                tipo_corpo = "Triângulo Invertido"
            elif quadril > ombros:
                tipo_corpo = "Pêra (Triângulo)"
            else:
                tipo_corpo = "Desconhecido"

            medidas['tipo_corpo'] = tipo_corpo

        except:
            print("Não foi possível calcular todas as medidas corporais.")
    else:
        print("Landmarks corporais não detectados.")

    cv2.imshow("Imagem de Entrada", imagem)

    # ROSTO -----------------------------------------------------------------
    h, w, _ = imagem.shape
    # Usa o ponto do nariz como centro da área de amostragem
    nariz = None
    if resultado.pose_landmarks:
        nariz = resultado.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        cx, cy = int(nariz.x * w), int(nariz.y * h)
    elif resultado_face.multi_face_landmarks:
        face_landmarks = resultado_face.multi_face_landmarks[0]
        nariz_face = face_landmarks.landmark[4]  # ponto da ponta do nariz
        cx, cy = int(nariz_face.x * w), int(nariz_face.y * h)
    else:
        print("Não foi possível localizar o nariz.")  # ponto da ponta do nariz no FaceMesh

    # Define o tamanho da área ao redor do nariz
    offset = 15
    x1, y1 = max(cx - offset, 0), max(cy - offset, 0)
    x2, y2 = min(cx + offset, w), min(cy + offset, h)
    rosto = imagem[y1:y2, x1:x2]

    if rosto.size > 0:
        tom_medio = np.mean(rosto, axis=(0, 1))  # BGR
    else:
        print('Não foi possível extrair o tom de pele')

    # Definir ROI (zoom 3x ao redor do nariz)
    zoom_factor = 3  # Ajuste conforme necessário
    roi_size = 150  # Tamanho da região ampliada (pixels)
    x1 = max(cx - roi_size // 2, 0)
    y1 = max(cy - roi_size // 2, 0)
    x2 = min(cx + roi_size // 2, w)
    y2 = min(cy + roi_size // 2, h)

    roi = imagem[y1:y2, x1:x2]

    # Ampliar a ROI (zoom no rosto)
    if roi.size > 0:
        roi_ampliada = cv2.resize(roi, (roi_size * zoom_factor, roi_size * zoom_factor))

        # Aplicar Face Mesh na ROI ampliada
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as face_mesh:
            roi_rgb = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2RGB)
            resultado_face = face_mesh.process(roi_rgb)

            if resultado_face.multi_face_landmarks:
                face_landmarks = resultado_face.multi_face_landmarks[0]

                # Usar ponto 4 (ponta do nariz) e dimensões da ROI
                ponto_nariz = face_landmarks.landmark[4]
                h_roi, w_roi, _ = roi_ampliada.shape
                x_nose, y_nose = int(ponto_nariz.x * w_roi), int(ponto_nariz.y * h_roi)

                offset = 8  # Reduzido para área mais precisa
                x1 = max(x_nose - offset, 0)
                y1 = max(y_nose - offset, 0)
                x2 = min(x_nose + offset, w_roi)  # Corrigido para w_roi
                y2 = min(y_nose + offset, h_roi)  # Corrigido para h_roi

                coordenadas_roi = (x1, y1, x2, y2)
                regiao_pele = roi_ampliada[y1:y2, x1:x2]

                if regiao_pele.size > 0:
                    # Aplicar filtro HSV
                    regiao_hsv = cv2.cvtColor(regiao_pele, cv2.COLOR_BGR2HSV)
                    mask_pele = cv2.inRange(regiao_hsv, (0, 30, 60), (25, 150, 255))
                    regiao_filtrada = cv2.bitwise_and(regiao_pele, regiao_pele, mask=mask_pele)

                    # Calcular média apenas nos pixels de pele
                    tom_pele = cv2.mean(regiao_filtrada, mask=mask_pele)[:3]
                    medidas['tom_de_pele'] = np.array(tom_pele).astype(int)

                    # Debug visual
                    debug_img = roi_ampliada.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.imshow("Nariz Analisado", debug_img)
            else:
                print("Landmarks faciais não detectados.")
    # CABELO ---------------------------------------------------------------------------------
    if face_landmarks:
        try:
            h_roi, w_roi, _ = roi_ampliada.shape

            # Encontrar ponto mais baixo do rosto (queixo)
            ponto_queixo = max(
                [(int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[152:155]],
                key=lambda p: p[1]
            )

            # Converter para HSV
            hsv = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2HSV)

            # Faixa para tons loiros (amarelos claros a dourados)
            loiro_min = np.array([15, 40, 160])  # H, S, V
            loiro_max = np.array([45, 180, 255])

            mask_loiro = cv2.inRange(hsv, loiro_min, loiro_max)

            # Máscara para destacar áreas escuras (potencial cabelo)
            gray = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2GRAY)
            _, mask_escura = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

            # Máscara do rosto baseada nos landmarks
            mask_rosto = np.zeros_like(gray)
            pontos_rosto = np.array([
                (int(lm.x * w_roi), int(lm.y * h_roi)) for lm in face_landmarks.landmark[:468]
            ])
            cv2.fillConvexPoly(mask_rosto, pontos_rosto, 255)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_rosto = cv2.dilate(mask_rosto, kernel)

            # Subtrair o rosto da máscara escura
            mask_cabelo_total = cv2.bitwise_or(mask_loiro, mask_escura)
            mask_cabelo = cv2.subtract(mask_cabelo_total, mask_rosto)

            # Restringir a região superior (acima do queixo)
            mask_regiao = np.zeros_like(gray)
            cv2.rectangle(mask_regiao, (0, 0), (w_roi, ponto_queixo[1] - 10), 255, -1)
            mask_cabelo = cv2.bitwise_and(mask_cabelo, mask_regiao)

            # Encontrar contornos
            contornos, _ = cv2.findContours(mask_cabelo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                maior_contorno = max(contornos, key=cv2.contourArea)
                mascara_final = np.zeros_like(gray)
                cv2.drawContours(mascara_final, [maior_contorno], -1, 255, -1)

                media_hsv = cv2.mean(hsv, mask=mascara_final)
                if media_hsv[1] < 25 or media_hsv[2] > 245:
                    # Saturação baixa ou brilho muito alto -> provável fundo
                    print("Descartado: cor muito clara (fundo ou pele)")
                    medidas['pouco_cabelo'] = True
                else:
                    media_bgr = cv2.mean(roi_ampliada, mask=mascara_final)[:3]
                    medidas['tom_de_cabelo'] = np.array(media_bgr).astype(int)

                # Extrair pixels do cabelo
                pixels_cabelo = cv2.bitwise_and(roi_ampliada, roi_ampliada, mask=mascara_final)
                media_cabelo = cv2.mean(pixels_cabelo, mask=mascara_final)[:3]
                medidas['tom_de_cabelo'] = np.array(media_cabelo).astype(int)
                medidas['pouco_cabelo'] = False

                # DEBUG VISUAL
                cv2.drawContours(debug_img, [maior_contorno], -1, (0, 255, 0), 2)
                cv2.line(debug_img, (0, ponto_queixo[1] - 10), (w_roi, ponto_queixo[1] - 10), (0, 0, 255), 2)
                #cv2.imshow("Região Analisada - Cabelo", debug_img)
                area_cabelo = cv2.countNonZero(mascara_final)
                limite_area_minima = 500  # Ajuste conforme seus testes (valor empírico)

                if area_cabelo < limite_area_minima:
                    medidas['pouco_cabelo'] = True
                    medidas['tom_de_cabelo'] = None
                else:
                    # Extrair pixels do cabelo
                    pixels_cabelo = cv2.bitwise_and(roi_ampliada, roi_ampliada, mask=mascara_final)
                    media_cabelo = cv2.mean(pixels_cabelo, mask=mascara_final)[:3]
                    medidas['tom_de_cabelo'] = np.array(media_cabelo).astype(int)
                    medidas['pouco_cabelo'] = False

            else:
                medidas['pouco_cabelo'] = True

        except Exception as e:
            print(f"Erro na análise de cabelo: {str(e)}")
            medidas['pouco_cabelo'] = True
    # OLHO --------------------------------------------------------------------------------------
    # Eye landmarks (adjust if needed)
    olho_esquerdo = [33, 133]  # Indices for left eye region
    olho_direito = [362, 263]  # Indices for right eye region

    for face_landmarks in resultado_face.multi_face_landmarks:
        h, w, _ = roi_ampliada.shape

        # --- Left Eye Color Extraction ---
        left_eye_coords = np.array([(int(face_landmarks.landmark[i].x * w),
                                     int(face_landmarks.landmark[i].y * h)) for i in olho_esquerdo])

        # Define a region around the eye landmarks
        min_x = int(min(left_eye_coords[:, 0]))
        max_x = int(max(left_eye_coords[:, 0]))
        min_y = int(min(left_eye_coords[:, 1]))
        max_y = int(max(left_eye_coords[:, 1]))

        # Extract eye region (add padding if desired)
        eye_region = roi_ampliada[min_y - 5: max_y + 5, min_x - 5: max_x + 5]
        #cv2.imshow('Olho analisado', eye_region.copy())

        # Calculate average color
        average_color = np.mean(eye_region, axis=(0, 1))
        medidas['tom_de_olho'] = np.round(average_color).astype(int)

        cv2.rectangle(debug_img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
        cv2.imshow("Rosto analisado", debug_img)


    # CONTRASTE =====================================================================================
    # Adicionando após obter os tons de pele e cabelo:
    def bgr_to_gray_scale_0_10(bgr):
        gray = int(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
        escala = np.clip(round(gray / 255 * 10), 0, 10)
        return escala

    # Obter escala de cinza dos tons extraídos
    if 'tom_de_pele' in medidas and 'tom_de_cabelo' in medidas and medidas['tom_de_cabelo'] is not None:
        escala_pele = bgr_to_gray_scale_0_10(medidas['tom_de_pele'])
        escala_cabelo = bgr_to_gray_scale_0_10(medidas['tom_de_cabelo'])
        escala_olhos = bgr_to_gray_scale_0_10(medidas['tom_de_olho'])
    else:
        escala_pele = bgr_to_gray_scale_0_10(medidas['tom_de_pele'])
        escala_cabelo = escala_pele
        escala_olhos = bgr_to_gray_scale_0_10(medidas['tom_de_olho'])

    # Encontrar tons extremos
    tons = [escala_pele, escala_cabelo, escala_olhos]
    tom_min = min(tons)
    tom_max = max(tons)
    intervalo = tom_max - tom_min

    # Classificação de contraste
    if intervalo <= 3:
        if escala_pele <= 6:
            contraste = "baixo contraste escuro"
        else:
            contraste = "baixo contraste claro"
    elif intervalo <= 5:
        contraste = "contraste médio"
    else:
        contraste = "alto contraste"

    medidas["Tom de pele (escala 0-10)"] = escala_pele
    medidas["Tom de cabelo (escala 0-10)"] = escala_cabelo
    medidas["Tom dos olhos (escala 0-10)"] = escala_olhos
    medidas["Intervalo de contraste"] = intervalo
    medidas["Classificação"] = contraste

    def vibrance_contraste_suave(roi_ampliada):
        # CLAHE muito leve no canal L (clareza sem exagero)
        lab = cv2.cvtColor(roi_ampliada, cv2.COLOR_BGR2LAB)
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

    # Carregamento da imagem
    img = cv2.imread("/mnt/data/45c92eeb-79d9-4194-90d0-83d1a410258b.png")
    imagem_realcada = vibrance_contraste_suave(roi_ampliada)

    # Aplicar Face Mesh na ROI ampliada
    coordenadas_roi = (x1, y1, x2, y2)
    regiao_pele = imagem_realcada[y1:y2, x1:x2]

    if regiao_pele.size > 0:
        # Aplicar filtro HSV
        regiao_hsv = cv2.cvtColor(regiao_pele, cv2.COLOR_BGR2HSV)
        mask_pele = cv2.inRange(regiao_hsv, (0, 30, 60), (25, 150, 255))
        regiao_filtrada = cv2.bitwise_and(regiao_pele, regiao_pele, mask=mask_pele)

        # Calcular média apenas nos pixels de pele
        tom_pele = cv2.mean(regiao_filtrada, mask=mask_pele)[:3]
        medidas['cor_saturada'] = np.array(tom_pele).astype(int)

        # Debug visual
        debug_img = imagem_realcada.copy()
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Rosto novo", debug_img)

    # COMPARANDO RESULTADOS ==============================================================
    # --- Dicionários de Subtons Base (em BGR) ---
    subtons_bgr = {
        "baixo contraste escuro": {
            "frio": [81, 113, 219],
            "neutro": [80, 117, 214],
            "quente": [66, 112, 207],
            "oliva": [66, 113, 185]
        },
        "baixo contraste claro": {
            "frio": [175, 188, 233],
            "neutro": [180, 196, 231],
            "quente": [170, 198, 230],
            "oliva": [180, 205, 235]
        },
        "medio contraste": {
            "frio": [138, 169, 255],
            "neutro": [135, 169, 254],
            "quente": [114, 158, 246],
            "oliva": [120, 169, 240]
        }
    }

    # --- Conversão BGR para LAB ---
    def bgr_para_lab(bgr):
        pix = np.uint8([[bgr]])  # shape (1,1,3)
        lab = cv2.cvtColor(pix, cv2.COLOR_BGR2LAB)
        return lab[0, 0]

    # --- Distância Euclidiana entre dois tons em LAB ---
    def distancia_lab(lab1, lab2):
        return np.linalg.norm(np.array(lab1, float) - np.array(lab2, float))

    # --- Classificação do subtom baseado em cor BGR de entrada ---
    def classificar_subtom(bgr_input):
        # Seleciona o conjunto de subtons correto
        if medidas["Classificação"] == "baixo contraste escuro":
            subtons_select = subtons_bgr["baixo contraste escuro"]
        if medidas["Classificação"] == "baixo contraste claro":
            subtons_select = subtons_bgr["baixo contraste claro"]
        else:
            subtons_select = subtons_bgr["medio contraste"]

        # Converte cada subtom base para LAB
        subtons_lab = {
            nome: bgr_para_lab(bgr) for nome, bgr in subtons_select.items()
        }

        # Converte o tom de entrada para LAB
        lab_input = bgr_para_lab(bgr_input)

        # Calcula distâncias e encontra o mínimo
        distancias = {
            nome: distancia_lab(lab_input, lab_base)
            for nome, lab_base in subtons_lab.items()
        }
        subtom_proximo = min(distancias, key=distancias.get)

        return subtom_proximo, distancias

    # Exemplo de uso
    subtom, dist = classificar_subtom(medidas['cor_saturada'])
    medidas['Subtom'] = subtom
    for k, v in dist.items():
        v = int(v)
        dist[k] = v
    medidas['Distâncias'] = dist

    return medidas, resultado


def visualizar_resultados(imagem, resultado, tom_de_pele=None, pouco_cabelo=None, tom_de_cabelo=None, tom_de_olho=None):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Primeiro processamos a imagem ORIGINAL (sem bordas)
    imagem_landmarks = imagem.copy()

    # Landmarks CORPORAIS (exatamente como na sua versão original)
    if resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagem_landmarks,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    # Criar painel de resultados
    painel_resultados = np.full((imagem.shape[0], 300, 3), 240, dtype=np.uint8)

    # Cabeçalho
    cv2.putText(painel_resultados, "RESULTADOS", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Variável auxiliar para controle da posição vertical
    y_atual = 60  # Começa após o cabeçalho (evita sobreposição)

    # ---- TOM DE PELE ----
    cv2.putText(painel_resultados, "Tom de pele:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(painel_resultados, (20, y_atual + 20), (120, y_atual + 100),
                  tuple([int(c) for c in tom_de_pele]), -1)
    cv2.putText(painel_resultados, f"RGB: {list(tom_de_pele)}", (20, y_atual + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    y_atual += 140  # Espaço suficiente para o próximo bloco

    # ---- TOM DE CABELO ----
    cv2.putText(painel_resultados, "Tom de cabelo:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    if not pouco_cabelo and tom_de_cabelo is not None:
        cv2.rectangle(painel_resultados, (20, y_atual + 30), (120, y_atual + 120),
                      tuple([int(c) for c in tom_de_cabelo]), -1)
        cv2.putText(painel_resultados, f"RGB: {list(tom_de_cabelo)}", (20, y_atual + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        cv2.putText(painel_resultados, "Nao detectado", (20, y_atual + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    y_atual += 160  # Mais espaço caso tenha texto extra ("Nao detectado")

    # ---- TOM DE OLHO ----
    cv2.putText(painel_resultados, "Tom de olho:", (20, y_atual + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(painel_resultados, (20, y_atual + 20), (120, y_atual + 100),
                  tuple([int(c) for c in tom_de_olho]), -1)
    cv2.putText(painel_resultados, f"RGB: {list(tom_de_olho)}", (20, y_atual + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Combinar as imagens HORIZONTALMENTE (não usando copyMakeBorder)
    imagem_final = np.hstack((imagem_landmarks, painel_resultados))

    # Exibição
    cv2.namedWindow("Analise Corporal", cv2.WINDOW_NORMAL)
    cv2.imshow("Analise Corporal", imagem_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
