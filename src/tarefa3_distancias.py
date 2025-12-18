"""
Tarefa 3: Cálculo de Distâncias entre Séries Temporais

Este módulo implementa:
1. Distância L1 (Manhattan) entre pares de séries temporais
2. Distância L2 (Euclidiana) entre pares de séries temporais
3. Construção de matrizes de distância MD1 (L1) e MD2 (L2)

As distâncias são calculadas entre séries normalizadas (área unitária)
para identificar municípios com dinâmicas epidêmicas sincronizadas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cityblock, euclidean
from typing import Dict, Tuple, List
from pathlib import Path
import seaborn as sns

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series
from src.tarefa2_normalizacao import normalize_all_series_by_total, POPULACAO_CENSO_2010


def l1_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calcula a distância L1 (Manhattan) entre duas séries temporais.
    
    L1 = Σ|x_i - y_i|
    
    A distância L1 é menos sensível a outliers do que a L2.
    
    Parameters
    ----------
    series1 : np.ndarray
        Primeira série temporal.
    series2 : np.ndarray
        Segunda série temporal.
    
    Returns
    -------
    float
        Distância L1 entre as séries.
    
    Raises
    ------
    ValueError
        Se as séries têm tamanhos diferentes.
    """
    if len(series1) != len(series2):
        raise ValueError(f"Séries devem ter o mesmo tamanho: {len(series1)} vs {len(series2)}")
    
    return cityblock(series1, series2)


def l2_distance(series1: np.ndarray, series2: np.ndarray) -> float:
    """
    Calcula a distância L2 (Euclidiana) entre duas séries temporais.
    
    L2 = √(Σ(x_i - y_i)²)
    
    A distância L2 dá mais peso a grandes diferenças.
    
    Parameters
    ----------
    series1 : np.ndarray
        Primeira série temporal.
    series2 : np.ndarray
        Segunda série temporal.
    
    Returns
    -------
    float
        Distância L2 entre as séries.
    
    Raises
    ------
    ValueError
        Se as séries têm tamanhos diferentes.
    """
    if len(series1) != len(series2):
        raise ValueError(f"Séries devem ter o mesmo tamanho: {len(series1)} vs {len(series2)}")
    
    return euclidean(series1, series2)


def compute_distance_matrix(time_series: Dict[str, np.ndarray], 
                           metric: str = 'l1') -> Tuple[np.ndarray, List[str]]:
    """
    Calcula a matriz de distância entre todas as séries temporais.
    
    Parameters
    ----------
    time_series : dict
        Dicionário {município: série_temporal}.
    metric : str
        Métrica de distância: 'l1' (Manhattan) ou 'l2' (Euclidiana).
    
    Returns
    -------
    tuple
        (matriz_distancia, lista_municipios)
    """
    municipalities = list(time_series.keys())
    n = len(municipalities)
    
    # Criar matriz de dados (cada linha é uma série temporal)
    data_matrix = np.array([time_series[mun] for mun in municipalities])
    
    # Calcular matriz de distância
    if metric.lower() == 'l1':
        distances = pdist(data_matrix, metric='cityblock')
    elif metric.lower() == 'l2':
        distances = pdist(data_matrix, metric='euclidean')
    else:
        raise ValueError(f"Métrica desconhecida: {metric}. Use 'l1' ou 'l2'.")
    
    # Converter para matriz quadrada
    distance_matrix = squareform(distances)
    
    return distance_matrix, municipalities


def compute_both_distance_matrices(time_series: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Calcula ambas as matrizes de distância MD1 (L1) e MD2 (L2).
    
    Parameters
    ----------
    time_series : dict
        Dicionário {município: série_temporal}.
    
    Returns
    -------
    tuple
        (MD1, MD2, lista_municipios)
    """
    MD1, municipalities = compute_distance_matrix(time_series, metric='l1')
    MD2, _ = compute_distance_matrix(time_series, metric='l2')
    
    return MD1, MD2, municipalities


def distance_matrix_to_dataframe(distance_matrix: np.ndarray, 
                                municipalities: List[str]) -> pd.DataFrame:
    """
    Converte matriz de distância para DataFrame com rótulos.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com matriz de distância rotulada.
    """
    return pd.DataFrame(distance_matrix, 
                       index=municipalities, 
                       columns=municipalities)


def plot_distance_matrix(distance_matrix: np.ndarray, 
                        municipalities: List[str],
                        metric_name: str = 'L1',
                        output_path: str = None,
                        figsize: Tuple[int, int] = (12, 10)):
    """
    Plota heatmap da matriz de distância.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    metric_name : str
        Nome da métrica para o título.
    output_path : str, optional
        Caminho para salvar o gráfico.
    figsize : tuple
        Tamanho da figura.
    """
    plt.figure(figsize=figsize)
    
    # Criar heatmap
    sns.heatmap(distance_matrix, 
                xticklabels=municipalities,
                yticklabels=municipalities,
                cmap='viridis',
                annot=len(municipalities) <= 10,  # Mostrar valores se poucos municípios
                fmt='.3f' if len(municipalities) <= 10 else '',
                square=True)
    
    plt.title(f'Matriz de Distância {metric_name}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def find_most_similar_pairs(distance_matrix: np.ndarray, 
                           municipalities: List[str],
                           top_n: int = 10) -> pd.DataFrame:
    """
    Encontra os pares de municípios mais similares (menor distância).
    
    Municípios com curvas epidêmicas similares podem indicar
    sincronização na dinâmica da doença.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    top_n : int
        Número de pares a retornar.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com os pares mais similares.
    """
    n = len(municipalities)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'municipio_1': municipalities[i],
                'municipio_2': municipalities[j],
                'distancia': distance_matrix[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('distancia').head(top_n)
    
    return pairs_df


def find_most_dissimilar_pairs(distance_matrix: np.ndarray, 
                              municipalities: List[str],
                              top_n: int = 10) -> pd.DataFrame:
    """
    Encontra os pares de municípios mais dissimilares (maior distância).
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    top_n : int
        Número de pares a retornar.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com os pares mais dissimilares.
    """
    n = len(municipalities)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                'municipio_1': municipalities[i],
                'municipio_2': municipalities[j],
                'distancia': distance_matrix[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('distancia', ascending=False).head(top_n)
    
    return pairs_df


def compute_synchronization_ranking(distance_matrix: np.ndarray,
                                   municipalities: List[str]) -> pd.DataFrame:
    """
    Calcula ranking de sincronização baseado na distância média.
    
    Municípios com menor distância média para os outros são mais
    representativos do padrão geral.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    
    Returns
    -------
    pd.DataFrame
        Ranking de sincronização.
    """
    mean_distances = []
    
    for i, mun in enumerate(municipalities):
        # Média das distâncias para todos os outros municípios
        distances = distance_matrix[i, :]
        # Excluir distância para si mesmo (0)
        mean_dist = np.mean(distances[distances > 0])
        mean_distances.append({
            'municipio': mun,
            'distancia_media': mean_dist
        })
    
    ranking = pd.DataFrame(mean_distances)
    ranking = ranking.sort_values('distancia_media')
    ranking['ranking'] = range(1, len(ranking) + 1)
    
    return ranking


def save_distance_matrices(MD1: np.ndarray, MD2: np.ndarray,
                          municipalities: List[str],
                          output_dir: str = 'output'):
    """
    Salva as matrizes de distância em arquivos CSV.
    
    Parameters
    ----------
    MD1 : np.ndarray
        Matriz de distância L1.
    MD2 : np.ndarray
        Matriz de distância L2.
    municipalities : list
        Lista de nomes dos municípios.
    output_dir : str
        Diretório de saída.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Converter para DataFrames
    df_MD1 = distance_matrix_to_dataframe(MD1, municipalities)
    df_MD2 = distance_matrix_to_dataframe(MD2, municipalities)
    
    # Salvar
    df_MD1.to_csv(output_path / 'matriz_distancia_L1.csv')
    df_MD2.to_csv(output_path / 'matriz_distancia_L2.csv')
    
    print(f"Matrizes salvas em: {output_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 3: CÁLCULO DE DISTÂNCIAS L1 E L2 - DADOS REAIS")
    print("=" * 70)
    
    # Carregar dados reais
    filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    df = load_and_merge_dengue_data(filepath)
    
    # Usar ano de 2013 (ano com muitos casos)
    year = 2013
    print(f"\n--- Análise para o ano {year} ---")
    
    # Extrair séries temporais
    time_series = get_yearly_time_series(df, year, normalize_to_52=True)
    
    # Filtrar municípios com dados de população e com casos > 0
    time_series_filtered = {}
    for mun, series in time_series.items():
        if mun in POPULACAO_CENSO_2010 and np.sum(series) > 0:
            time_series_filtered[mun] = series
    
    print(f"Municípios analisados: {len(time_series_filtered)}")
    
    # Normalizar séries para área unitária
    print("\n--- Normalizando séries para área unitária ---")
    normalized_series = normalize_all_series_by_total(time_series_filtered)
    
    # Calcular matrizes de distância
    print("\n--- Calculando Matrizes de Distância ---")
    MD1, MD2, municipalities = compute_both_distance_matrices(normalized_series)
    
    print(f"\nDimensões das matrizes: {MD1.shape}")
    
    # Estatísticas das distâncias
    print(f"\nEstatísticas da Matriz L1:")
    print(f"  Mínimo (excl. diagonal): {MD1[MD1 > 0].min():.4f}")
    print(f"  Máximo: {MD1.max():.4f}")
    print(f"  Média: {MD1[MD1 > 0].mean():.4f}")
    
    print(f"\nEstatísticas da Matriz L2:")
    print(f"  Mínimo (excl. diagonal): {MD2[MD2 > 0].min():.4f}")
    print(f"  Máximo: {MD2.max():.4f}")
    print(f"  Média: {MD2[MD2 > 0].mean():.4f}")
    
    # Pares mais similares (mais sincronizados)
    print("\n--- TOP 15 PARES MAIS SINCRONIZADOS (L1) ---")
    similar_L1 = find_most_similar_pairs(MD1, municipalities, top_n=15)
    print(similar_L1.to_string(index=False))
    
    print("\n--- TOP 15 PARES MAIS SINCRONIZADOS (L2) ---")
    similar_L2 = find_most_similar_pairs(MD2, municipalities, top_n=15)
    print(similar_L2.to_string(index=False))
    
    # Pares mais dessincronizados
    print("\n--- TOP 10 PARES MENOS SINCRONIZADOS (L1) ---")
    dissimilar_L1 = find_most_dissimilar_pairs(MD1, municipalities, top_n=10)
    print(dissimilar_L1.to_string(index=False))
    
    # Ranking de sincronização
    print("\n--- RANKING DE SINCRONIZAÇÃO (TOP 15) ---")
    print("(Municípios com menor distância média são mais representativos do padrão geral)")
    ranking = compute_synchronization_ranking(MD1, municipalities)
    print(ranking.head(15).to_string(index=False))
    
    # Salvar matrizes
    output_dir = Path(__file__).parent.parent / "output"
    save_distance_matrices(MD1, MD2, municipalities, output_dir)
    
    # Plotar heatmap das matrizes (para subset de municípios)
    print("\n--- Plotando heatmaps ---")
    # Usar top 20 municípios por sincronização para visualização
    top_sync = ranking.head(20)['municipio'].tolist()
    
    # Índices dos top municípios
    top_indices = [municipalities.index(m) for m in top_sync if m in municipalities]
    
    # Submatriz
    MD1_sub = MD1[np.ix_(top_indices, top_indices)]
    MD2_sub = MD2[np.ix_(top_indices, top_indices)]
    top_municipalities = [municipalities[i] for i in top_indices]
    
    plot_distance_matrix(MD1_sub, top_municipalities, 
                        metric_name=f'L1 (Manhattan) - Top 20 Municípios - {year}')
    plot_distance_matrix(MD2_sub, top_municipalities, 
                        metric_name=f'L2 (Euclidiana) - Top 20 Municípios - {year}')
    
    print("\n" + "=" * 70)
    print("CONCLUSÕES:")
    print("=" * 70)
    print("""
1. Municípios com MENOR distância entre suas curvas normalizadas
   são mais SINCRONIZADOS na dinâmica da epidemia.

2. A distância L1 (Manhattan) é mais robusta a outliers.

3. A distância L2 (Euclidiana) penaliza mais diferenças grandes.

4. As matrizes MD1 e MD2 serão usadas na Tarefa 4 para criar
   complexos simpliciais com diferentes limiares.
    """)
