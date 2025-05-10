import os
import pandas as pd
from processamento import extrair_dados_da_imagem


def recomendar_roupas(medidas):
    base_dir = os.path.dirname(__file__)
    caminho_csv = os.path.abspath(os.path.join(base_dir, '../../trabalho/data/catalogo_roupas.csv'))

    print("\nLendo CSV em:", caminho_csv)
    catalogo = pd.read_csv(caminho_csv)

    # Mostra o conteúdo lido
    catalogo.columns = catalogo.columns.str.strip().str.lower()
    print(catalogo.head())

    # Exemplo de dados do usuário (simulados)
    roupas_filtradas = catalogo.copy()

    # Regras de recomendação
    if medidas.get('Classificação', 0) == "baixo contraste escuro":
        roupas_filtradas = roupas_filtradas[roupas_filtradas['tipo'].str.contains("blusa", case=False)]
    elif medidas.get('Classificação', 0) == "baixo contraste claro":
        roupas_filtradas = roupas_filtradas[roupas_filtradas['tipo'].str.contains("blusa", case=False)]
    elif medidas.get('Classificação', 0) == "baixo contraste claro":
        roupas_filtradas = roupas_filtradas[roupas_filtradas['tipo'].str.contains("blusa", case=False)]
    else:
        pass
    print(roupas_filtradas.head(3))  # Top 3 sugestões

