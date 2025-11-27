"""
Tarefa 5: Visualização com KeplerMapper

Este módulo implementa a visualização do complexo simplicial usando
KeplerMapper, uma biblioteca para Topological Data Analysis (TDA).

KeplerMapper cria uma representação visual da estrutura topológica
dos dados, revelando clusters e conexões entre municípios baseados
na similaridade de suas curvas epidêmicas.
"""

import numpy as np
import pandas as pd
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series
from tarefa2_normalizacao import normalize_all_series_by_total, POPULACAO_CENSO_2010
from tarefa3_distancias import compute_both_distance_matrices


def prepare_data_for_mapper(normalized_series: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """
    Prepara os dados normalizados para uso com KeplerMapper.
    
    Parameters
    ----------
    normalized_series : dict
        Dicionário {município: série_normalizada}.
    
    Returns
    -------
    tuple
        (data_matrix, municipality_names)
    """
    municipalities = list(normalized_series.keys())
    data_matrix = np.array([normalized_series[mun] for mun in municipalities])
    
    return data_matrix, municipalities


def create_mapper_visualization(
    data_matrix: np.ndarray,
    municipality_names: List[str],
    lens_type: str = 'pca',
    n_cubes: int = 10,
    overlap: float = 0.5,
    eps: float = 1.5,
    min_samples: int = 1,
    output_path: str = None,
    title: str = "KeplerMapper - Dengue RJ"
) -> str:
    """
    Cria visualização interativa do complexo simplicial usando KeplerMapper.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Matriz de dados (municípios x semanas).
    municipality_names : list
        Lista de nomes dos municípios.
    lens_type : str
        Tipo de lente: 'pca', 'tsne', 'mds', 'l2norm', 'mean'.
    n_cubes : int
        Número de cubos (resolução da cobertura).
    overlap : float
        Sobreposição entre cubos (0-1).
    eps : float
        Parâmetro epsilon do DBSCAN.
    min_samples : int
        Número mínimo de amostras do DBSCAN.
    output_path : str, optional
        Caminho para salvar o HTML.
    title : str
        Título da visualização.
    
    Returns
    -------
    str
        HTML da visualização.
    """
    # Inicializar KeplerMapper
    mapper = km.KeplerMapper(verbose=1)
    
    # Escalar dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    # Criar lente (projeção)
    print(f"\n--- Criando lente: {lens_type} ---")
    if lens_type == 'pca':
        lens = mapper.fit_transform(data_scaled, projection=PCA(n_components=2))
    elif lens_type == 'tsne':
        lens = mapper.fit_transform(data_scaled, projection=TSNE(n_components=2, random_state=42, perplexity=min(30, len(data_matrix)-1)))
    elif lens_type == 'mds':
        lens = mapper.fit_transform(data_scaled, projection=MDS(n_components=2, random_state=42))
    elif lens_type == 'l2norm':
        lens = mapper.fit_transform(data_scaled, projection="l2norm")
    elif lens_type == 'mean':
        lens = mapper.fit_transform(data_scaled, projection="mean")
    elif lens_type == 'sum':
        lens = mapper.fit_transform(data_scaled, projection="sum")
    else:
        # Default: PCA
        lens = mapper.fit_transform(data_scaled, projection=PCA(n_components=2))
    
    print(f"  Lente shape: {lens.shape}")
    
    # Configurar clusterer com parâmetros mais flexíveis
    cluster_algo = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Criar o grafo do Mapper
    print(f"\n--- Construindo grafo Mapper ---")
    print(f"  n_cubes: {n_cubes}, overlap: {overlap}, eps: {eps}, min_samples: {min_samples}")
    
    graph = mapper.map(
        lens,
        data_scaled,
        clusterer=cluster_algo,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    )
    
    # Estatísticas do grafo
    n_nodes = len(graph['nodes'])
    n_edges = len(graph['links']) if 'links' in graph else 0
    print(f"\n  Nós no grafo: {n_nodes}")
    print(f"  Arestas no grafo: {n_edges}")
    
    # Criar tooltip com nomes dos municípios
    tooltips = np.array(municipality_names)
    
    # Calcular cores baseadas em características dos dados
    # Usar média de casos como cor
    color_values = data_matrix.mean(axis=1)
    
    # Criar visualização HTML
    print(f"\n--- Gerando visualização HTML ---")
    
    html = mapper.visualize(
        graph,
        path_html=output_path if output_path else "mapper_output.html",
        title=title,
        custom_tooltips=tooltips,
        color_values=color_values,
        color_function_name="Média de Casos (normalizada)",
        node_color_function=["mean", "std", "median", "max"]
    )
    
    if output_path:
        print(f"✓ Visualização salva em: {output_path}")
    
    return html


def create_mapper_from_distance_matrix(
    distance_matrix: np.ndarray,
    municipality_names: List[str],
    n_cubes: int = 10,
    overlap: float = 0.5,
    eps: float = 1.0,
    min_samples: int = 1,
    output_path: str = None,
    title: str = "KeplerMapper - Matriz de Distância"
) -> str:
    """
    Cria visualização usando a matriz de distância diretamente.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância entre municípios.
    municipality_names : list
        Lista de nomes dos municípios.
    n_cubes : int
        Número de cubos.
    overlap : float
        Sobreposição.
    eps : float
        Parâmetro epsilon do DBSCAN.
    min_samples : int
        Número mínimo de amostras do DBSCAN.
    output_path : str, optional
        Caminho para salvar.
    title : str
        Título.
    
    Returns
    -------
    str
        HTML da visualização.
    """
    # Inicializar KeplerMapper
    mapper = km.KeplerMapper(verbose=1)
    
    # Usar MDS para projetar a matriz de distância em 2D
    print("\n--- Projetando matriz de distância com MDS ---")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    data_projected = mds.fit_transform(distance_matrix)
    
    # Escalar
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_projected)
    
    # Criar lente (usar as próprias coordenadas MDS)
    lens = data_scaled
    
    print(f"  Projeção shape: {lens.shape}")
    
    # Clusterer com parâmetros ajustados
    cluster_algo = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Criar grafo
    print(f"\n--- Construindo grafo Mapper ---")
    print(f"  n_cubes: {n_cubes}, overlap: {overlap}, eps: {eps}")
    
    graph = mapper.map(
        lens,
        data_scaled,
        clusterer=cluster_algo,
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    )
    
    n_nodes = len(graph['nodes'])
    print(f"  Nós no grafo: {n_nodes}")
    
    # Tooltips
    tooltips = np.array(municipality_names)
    
    # Cor: distância média para outros municípios
    color_values = distance_matrix.mean(axis=1)
    
    # Gerar HTML
    print(f"\n--- Gerando visualização HTML ---")
    
    html = mapper.visualize(
        graph,
        path_html=output_path if output_path else "mapper_distance.html",
        title=title,
        custom_tooltips=tooltips,
        color_values=color_values,
        color_function_name="Distância Média",
    )
    
    if output_path:
        print(f"✓ Visualização salva em: {output_path}")
    
    return html


def create_multiple_mapper_views(
    data_matrix: np.ndarray,
    municipality_names: List[str],
    distance_matrix: np.ndarray,
    output_dir: str,
    year: int = 2013
):
    """
    Cria múltiplas visualizações KeplerMapper com diferentes configurações.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Matriz de dados.
    municipality_names : list
        Nomes dos municípios.
    distance_matrix : np.ndarray
        Matriz de distância.
    output_dir : str
        Diretório de saída.
    year : int
        Ano da análise.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GERANDO MÚLTIPLAS VISUALIZAÇÕES KEPLER MAPPER")
    print("=" * 70)
    
    # 1. Visualização com PCA
    print("\n[1/4] Visualização com lente PCA...")
    create_mapper_visualization(
        data_matrix, municipality_names,
        lens_type='pca',
        n_cubes=8,
        overlap=0.6,
        eps=2.0,
        min_samples=1,
        output_path=str(output_path / f"kmapper_pca_{year}.html"),
        title=f"KeplerMapper - Dengue RJ {year} (PCA)"
    )
    
    # 2. Visualização com t-SNE
    print("\n[2/4] Visualização com lente t-SNE...")
    create_mapper_visualization(
        data_matrix, municipality_names,
        lens_type='tsne',
        n_cubes=8,
        overlap=0.6,
        eps=2.0,
        min_samples=1,
        output_path=str(output_path / f"kmapper_tsne_{year}.html"),
        title=f"KeplerMapper - Dengue RJ {year} (t-SNE)"
    )
    
    # 3. Visualização com norma L2
    print("\n[3/4] Visualização com lente L2 norm...")
    create_mapper_visualization(
        data_matrix, municipality_names,
        lens_type='l2norm',
        n_cubes=10,
        overlap=0.5,
        eps=2.5,
        min_samples=1,
        output_path=str(output_path / f"kmapper_l2norm_{year}.html"),
        title=f"KeplerMapper - Dengue RJ {year} (L2 Norm)"
    )
    
    # 4. Visualização baseada na matriz de distância
    print("\n[4/4] Visualização com matriz de distância...")
    create_mapper_from_distance_matrix(
        distance_matrix, municipality_names,
        n_cubes=8,
        overlap=0.6,
        eps=1.5,
        min_samples=1,
        output_path=str(output_path / f"kmapper_distancia_{year}.html"),
        title=f"KeplerMapper - Dengue RJ {year} (Distância L1)"
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZAÇÕES GERADAS COM SUCESSO!")
    print("=" * 70)
    print(f"\nArquivos salvos em: {output_path}")
    print("  • kmapper_pca_{year}.html - Projeção PCA")
    print("  • kmapper_tsne_{year}.html - Projeção t-SNE")
    print("  • kmapper_l2norm_{year}.html - Projeção L2 Norm")
    print("  • kmapper_distancia_{year}.html - Baseado em distância L1")
    print("\nAbra os arquivos HTML no navegador para visualização interativa!")


def analyze_mapper_clusters(
    graph: dict,
    municipality_names: List[str],
    data_matrix: np.ndarray
) -> pd.DataFrame:
    """
    Analisa os clusters identificados pelo Mapper.
    
    Parameters
    ----------
    graph : dict
        Grafo do Mapper.
    municipality_names : list
        Nomes dos municípios.
    data_matrix : np.ndarray
        Matriz de dados.
    
    Returns
    -------
    pd.DataFrame
        Análise dos clusters.
    """
    results = []
    
    for node_id, indices in graph['nodes'].items():
        municipalities_in_node = [municipality_names[i] for i in indices]
        node_data = data_matrix[indices]
        
        results.append({
            'node_id': node_id,
            'n_municipalities': len(indices),
            'municipalities': ', '.join(municipalities_in_node[:5]),
            'mean_cases': node_data.mean(),
            'std_cases': node_data.std(),
            'peak_week': np.argmax(node_data.mean(axis=0)) + 1
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 5: VISUALIZAÇÃO COM KEPLER MAPPER")
    print("=" * 70)
    
    # Carregar dados reais
    filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    df = load_and_merge_dengue_data(filepath)
    
    # Usar ano de 2013
    year = 2013
    print(f"\n--- Análise para o ano {year} ---")
    
    # Extrair e normalizar séries temporais
    time_series = get_yearly_time_series(df, year, normalize_to_52=True)
    time_series_filtered = {mun: series for mun, series in time_series.items() 
                           if mun in POPULACAO_CENSO_2010 and np.sum(series) > 0}
    
    normalized_series = normalize_all_series_by_total(time_series_filtered)
    
    print(f"Municípios analisados: {len(normalized_series)}")
    
    # Preparar dados
    data_matrix, municipalities = prepare_data_for_mapper(normalized_series)
    print(f"Matriz de dados: {data_matrix.shape}")
    
    # Calcular matriz de distância
    MD1, MD2, _ = compute_both_distance_matrices(normalized_series)
    
    # Diretório de saída
    output_dir = Path(__file__).parent.parent / "output"
    
    # Criar múltiplas visualizações
    create_multiple_mapper_views(
        data_matrix, municipalities, MD1, 
        output_dir=str(output_dir),
        year=year
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSÕES:")
    print("=" * 70)
    print("""
    O KeplerMapper revela a ESTRUTURA TOPOLÓGICA dos dados de dengue:
    
    1. NODOS representam grupos de municípios com curvas similares
    2. CONEXÕES indicam sobreposição entre grupos
    3. A FORMA do grafo revela padrões globais da epidemia
    
    Diferentes lentes (PCA, t-SNE, L2) destacam diferentes aspectos:
    - PCA: Variância principal dos dados
    - t-SNE: Estrutura local (vizinhança)
    - L2 Norm: Intensidade geral da epidemia
    
    Abra os arquivos HTML no navegador para explorar interativamente!
    - Passe o mouse sobre os nodos para ver os municípios
    - Observe as cores que indicam características dos clusters
    """)
