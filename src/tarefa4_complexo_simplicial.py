"""
Tarefa 4: Criação de Complexos Simpliciais

Este módulo implementa:
1. Criação de complexos simpliciais a partir de matrizes de distância
2. Limiar variável para filtrar conexões
3. Identificação de simplexos de diferentes dimensões (0, 1, 2, ...)
4. Visualização do complexo simplicial como grafo

Um complexo simplicial é uma estrutura topológica que generaliza grafos:
- 0-simplexo: vértice (um ponto)
- 1-simplexo: aresta (conecta 2 vértices)
- 2-simplexo: triângulo (conecta 3 vértices)
- k-simplexo: conecta k+1 vértices

Aplicação: Identificar grupos de municípios com dinâmica epidêmica similar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
from pathlib import Path
from itertools import combinations
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series
from tarefa2_normalizacao import normalize_all_series_by_total, POPULACAO_CENSO_2010
from tarefa3_distancias import compute_both_distance_matrices


class SimplicialComplex:
    """
    Classe para representar e manipular um complexo simplicial.
    """
    
    def __init__(self, vertices: List[str]):
        """
        Inicializa o complexo simplicial com um conjunto de vértices.
        
        Parameters
        ----------
        vertices : list
            Lista de nomes dos vértices (ex: municípios).
        """
        self.vertices = vertices
        self.vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        self.simplices = {0: set(range(len(vertices)))}  # 0-simplexos são os vértices
        
    def add_simplex(self, simplex: tuple):
        """
        Adiciona um simplexo ao complexo.
        
        Parameters
        ----------
        simplex : tuple
            Tupla de índices dos vértices que formam o simplexo.
        """
        dim = len(simplex) - 1
        if dim not in self.simplices:
            self.simplices[dim] = set()
        self.simplices[dim].add(tuple(sorted(simplex)))
        
    def get_simplices(self, dimension: int) -> Set[tuple]:
        """
        Retorna todos os simplexos de uma determinada dimensão.
        
        Parameters
        ----------
        dimension : int
            Dimensão dos simplexos (0 para vértices, 1 para arestas, etc.)
        
        Returns
        -------
        set
            Conjunto de simplexos da dimensão especificada.
        """
        return self.simplices.get(dimension, set())
    
    def count_simplices(self) -> Dict[int, int]:
        """
        Conta o número de simplexos por dimensão.
        
        Returns
        -------
        dict
            Dicionário {dimensão: contagem}.
        """
        return {dim: len(simplices) for dim, simplices in self.simplices.items()}
    
    def get_vertex_names(self, simplex: tuple) -> List[str]:
        """
        Retorna os nomes dos vértices de um simplexo.
        
        Parameters
        ----------
        simplex : tuple
            Tupla de índices dos vértices.
        
        Returns
        -------
        list
            Lista de nomes dos vértices.
        """
        return [self.vertices[i] for i in simplex]


def create_simplicial_complex_from_distance_matrix(
    distance_matrix: np.ndarray,
    vertices: List[str],
    threshold: float,
    max_dimension: int = 2
) -> SimplicialComplex:
    """
    Cria um complexo simplicial a partir de uma matriz de distância.
    
    Dois vértices são conectados se sua distância é menor que o limiar.
    Simplexos de dimensão maior são criados quando todos os pares de
    vértices estão conectados (formam uma clique).
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância entre vértices.
    vertices : list
        Lista de nomes dos vértices.
    threshold : float
        Limiar de distância para conexão.
    max_dimension : int
        Dimensão máxima dos simplexos a procurar.
    
    Returns
    -------
    SimplicialComplex
        Complexo simplicial construído.
    """
    n = len(vertices)
    complex_obj = SimplicialComplex(vertices)
    
    # Criar matriz de adjacência (1 se distância < limiar)
    adjacency = distance_matrix < threshold
    np.fill_diagonal(adjacency, False)  # Remover auto-loops
    
    # Adicionar 1-simplexos (arestas)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                complex_obj.add_simplex((i, j))
    
    # Adicionar simplexos de dimensão maior
    for dim in range(2, max_dimension + 1):
        # Para cada combinação de dim+1 vértices
        for combo in combinations(range(n), dim + 1):
            # Verificar se todos os pares estão conectados (formam clique)
            is_clique = True
            for pair in combinations(combo, 2):
                if not adjacency[pair[0], pair[1]]:
                    is_clique = False
                    break
            
            if is_clique:
                complex_obj.add_simplex(combo)
    
    return complex_obj


def analyze_thresholds(
    distance_matrix: np.ndarray,
    vertices: List[str],
    thresholds: List[float] = None,
    metric_name: str = 'L1'
) -> pd.DataFrame:
    """
    Analisa complexos simpliciais para diferentes valores de limiar.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância entre vértices.
    vertices : list
        Lista de nomes dos vértices.
    thresholds : list, optional
        Lista de limiares a testar. Se None, usa percentis.
    metric_name : str
        Nome da métrica para identificação.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com estatísticas para cada limiar.
    """
    if thresholds is None:
        # Usar percentis da distribuição de distâncias
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        thresholds = np.percentile(distances, [10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    results = []
    
    for threshold in thresholds:
        complex_obj = create_simplicial_complex_from_distance_matrix(
            distance_matrix, vertices, threshold, max_dimension=3
        )
        
        counts = complex_obj.count_simplices()
        
        results.append({
            'metrica': metric_name,
            'limiar': threshold,
            'vertices': counts.get(0, 0),
            'arestas': counts.get(1, 0),
            'triangulos': counts.get(2, 0),
            'tetraedros': counts.get(3, 0)
        })
    
    return pd.DataFrame(results)


def plot_simplicial_complex_as_graph(
    complex_obj: SimplicialComplex,
    title: str = "Complexo Simplicial",
    output_path: str = None,
    highlight_triangles: bool = True
):
    """
    Visualiza o complexo simplicial como um grafo.
    
    Parameters
    ----------
    complex_obj : SimplicialComplex
        Complexo simplicial a visualizar.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    highlight_triangles : bool
        Se True, destaca os triângulos (2-simplexos).
    """
    # Criar grafo
    G = nx.Graph()
    
    # Adicionar vértices
    G.add_nodes_from(range(len(complex_obj.vertices)))
    
    # Adicionar arestas (1-simplexos)
    for edge in complex_obj.get_simplices(1):
        G.add_edge(edge[0], edge[1])
    
    # Layout
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Desenhar triângulos (se houver)
    if highlight_triangles:
        triangles = complex_obj.get_simplices(2)
        for triangle in triangles:
            triangle = list(triangle)
            triangle_coords = [pos[v] for v in triangle]
            triangle_coords.append(triangle_coords[0])  # Fechar o triângulo
            xs, ys = zip(*triangle_coords)
            plt.fill(xs, ys, alpha=0.2, color='yellow', edgecolor='orange', linewidth=2)
    
    # Desenhar arestas
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1)
    
    # Desenhar vértices
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', 
                          edgecolors='navy', linewidths=1.5)
    
    # Rótulos dos vértices (usar nomes abreviados se muitos)
    if len(complex_obj.vertices) <= 20:
        labels = {i: complex_obj.vertices[i][:15] for i in range(len(complex_obj.vertices))}
    else:
        labels = {i: str(i) for i in range(len(complex_obj.vertices))}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Estatísticas
    counts = complex_obj.count_simplices()
    stats_text = f"Vértices: {counts.get(0, 0)} | Arestas: {counts.get(1, 0)} | Triângulos: {counts.get(2, 0)}"
    
    plt.title(f"{title}\n{stats_text}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def identify_clusters(
    distance_matrix: np.ndarray,
    vertices: List[str],
    threshold: float
) -> List[List[str]]:
    """
    Identifica clusters de vértices conectados no complexo simplicial.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância entre vértices.
    vertices : list
        Lista de nomes dos vértices.
    threshold : float
        Limiar de distância para conexão.
    
    Returns
    -------
    list
        Lista de clusters, onde cada cluster é uma lista de nomes de vértices.
    """
    n = len(vertices)
    
    # Criar matriz de adjacência
    adjacency = distance_matrix < threshold
    np.fill_diagonal(adjacency, False)
    
    # Criar grafo
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                G.add_edge(i, j)
    
    # Encontrar componentes conectados
    components = list(nx.connected_components(G))
    
    # Converter índices para nomes
    clusters = [[vertices[i] for i in component] for component in components]
    
    # Ordenar por tamanho (maior primeiro)
    clusters.sort(key=len, reverse=True)
    
    return clusters


def find_optimal_threshold(
    distance_matrix: np.ndarray,
    vertices: List[str],
    target_clusters: int = 5
) -> float:
    """
    Encontra um limiar que resulta aproximadamente no número desejado de clusters.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância entre vértices.
    vertices : list
        Lista de nomes dos vértices.
    target_clusters : int
        Número desejado de clusters.
    
    Returns
    -------
    float
        Limiar ótimo.
    """
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    
    # Testar diferentes limiares
    thresholds = np.linspace(distances.min(), distances.max(), 100)
    
    best_threshold = thresholds[0]
    best_diff = float('inf')
    
    for threshold in thresholds:
        clusters = identify_clusters(distance_matrix, vertices, threshold)
        diff = abs(len(clusters) - target_clusters)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    
    return best_threshold


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 4: COMPLEXOS SIMPLICIAIS A PARTIR DE MATRIZES DE DISTÂNCIA")
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
    
    # Calcular matrizes de distância
    MD1, MD2, municipalities = compute_both_distance_matrices(normalized_series)
    
    print(f"Número de municípios: {len(municipalities)}")
    
    # Analisar diferentes limiares
    print("\n--- Análise de Limiares (Matriz L1) ---")
    threshold_analysis = analyze_thresholds(MD1, municipalities, metric_name='L1')
    print(threshold_analysis.to_string(index=False))
    
    print("\n--- Análise de Limiares (Matriz L2) ---")
    threshold_analysis_L2 = analyze_thresholds(MD2, municipalities, metric_name='L2')
    print(threshold_analysis_L2.to_string(index=False))
    
    # Criar complexo com limiar específico
    # Usar percentil 30 como exemplo
    distances_L1 = MD1[np.triu_indices_from(MD1, k=1)]
    threshold_30 = np.percentile(distances_L1, 30)
    
    print(f"\n--- Complexo Simplicial com Limiar = {threshold_30:.4f} (percentil 30) ---")
    complex_30 = create_simplicial_complex_from_distance_matrix(
        MD1, municipalities, threshold_30, max_dimension=3
    )
    
    counts = complex_30.count_simplices()
    print(f"Estrutura do complexo:")
    print(f"  0-simplexos (vértices): {counts.get(0, 0)}")
    print(f"  1-simplexos (arestas): {counts.get(1, 0)}")
    print(f"  2-simplexos (triângulos): {counts.get(2, 0)}")
    print(f"  3-simplexos (tetraedros): {counts.get(3, 0)}")
    
    # Identificar clusters
    print("\n--- Clusters de Municípios Sincronizados ---")
    clusters = identify_clusters(MD1, municipalities, threshold_30)
    
    print(f"Número de clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters[:5], 1):
        print(f"\nCluster {i} ({len(cluster)} municípios):")
        for mun in cluster[:10]:  # Mostrar até 10
            print(f"  - {mun}")
        if len(cluster) > 10:
            print(f"  ... e mais {len(cluster) - 10} municípios")
    
    # Encontrar limiar ótimo para ~5 clusters
    print("\n--- Busca de Limiar Ótimo ---")
    optimal_threshold = find_optimal_threshold(MD1, municipalities, target_clusters=5)
    print(f"Limiar para ~5 clusters: {optimal_threshold:.4f}")
    
    clusters_optimal = identify_clusters(MD1, municipalities, optimal_threshold)
    print(f"Clusters obtidos: {len(clusters_optimal)}")
    
    # Plotar complexo (com subset de municípios para visualização)
    print("\n--- Visualização do Complexo ---")
    
    # Usar apenas os 30 municípios mais centrais (menor distância média)
    mean_distances = MD1.mean(axis=1)
    top_indices = np.argsort(mean_distances)[:30]
    
    top_municipalities = [municipalities[i] for i in top_indices]
    top_MD1 = MD1[np.ix_(top_indices, top_indices)]
    
    # Criar complexo para subset
    threshold_subset = np.percentile(top_MD1[np.triu_indices_from(top_MD1, k=1)], 40)
    complex_subset = create_simplicial_complex_from_distance_matrix(
        top_MD1, top_municipalities, threshold_subset, max_dimension=2
    )
    
    plot_simplicial_complex_as_graph(
        complex_subset, 
        title=f"Complexo Simplicial - Top 30 Municípios (limiar={threshold_subset:.3f})"
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSÕES:")
    print("=" * 70)
    print("""
1. O complexo simplicial revela a ESTRUTURA TOPOLÓGICA das relações
   entre municípios baseada na similaridade de suas curvas epidêmicas.

2. Limiares MENORES resultam em menos conexões (mais clusters isolados).
   Limiares MAIORES conectam mais municípios (menos clusters).

3. TRIÂNGULOS (2-simplexos) indicam trios de municípios onde todos
   os três têm dinâmicas epidêmicas similares.

4. Os CLUSTERS identificados representam grupos de municípios com
   padrões de dengue sincronizados, possivelmente indicando:
   - Proximidade geográfica
   - Fluxo populacional similar
   - Condições climáticas semelhantes

5. Este método pode ajudar a definir regiões de vigilância epidemiológica
   coordenada.
    """)
