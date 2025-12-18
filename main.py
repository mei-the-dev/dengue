"""
Script Principal: Execução Completa das Análises de Dengue

Este script executa todas as análises e gera todas as visualizações,
salvando os gráficos na pasta output/.

Uso:
    python main.py
"""

import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from epiweeks import Year

# Importar módulos do projeto
from tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series
from tarefa1_semanas_epidemiologicas import (
    get_weeks_per_year, identify_years_with_53_weeks, define_epidemic_period
)
from tarefa2_normalizacao import (
    POPULACAO_CENSO_2010, normalize_by_population, normalize_by_total_infected,
    normalize_all_series_by_total
)
from tarefa3_distancias import (
    compute_both_distance_matrices, find_most_similar_pairs, 
    find_most_dissimilar_pairs, compute_synchronization_ranking,
    save_distance_matrices
)
from tarefa4_complexo_simplicial import (
    create_simplicial_complex_from_distance_matrix, analyze_thresholds,
    identify_clusters, find_optimal_threshold
)
from visualizacoes import (
    ensure_output_dir,
    plot_weeks_per_year, plot_epidemic_curves_by_year, plot_year_transitions,
    plot_top_municipalities_by_year, plot_epidemic_period,
    plot_incidence_rates, plot_normalized_curves, plot_normalization_comparison,
    plot_distance_heatmap, plot_distance_distribution, plot_similar_pairs_curves,
    plot_synchronization_ranking,
    plot_threshold_analysis, plot_simplicial_complex_graph, plot_clusters,
    plot_summary_dashboard
)


def print_header(text: str):
    """Imprime cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_section(text: str):
    """Imprime seção formatada."""
    print(f"\n--- {text} ---")


def main():
    """Executa todas as análises e gera visualizações."""
    
    print_header("ANÁLISE EPIDEMIOLÓGICA DE DENGUE - RIO DE JANEIRO (2010-2016)")
    print("Executando todas as análises e gerando visualizações...")
    
    # Configurar diretórios
    project_dir = Path(__file__).parent
    data_file = project_dir / "Dengue_Brasil_2010-2016_RJ.xlsx"
    output_dir = ensure_output_dir(project_dir / "output")
    
    # Verificar se arquivo de dados existe
    if not data_file.exists():
        print(f"\n❌ ERRO: Arquivo de dados não encontrado: {data_file}")
        print("Por favor, coloque o arquivo 'Dengue_Brasil_2010-2016_RJ.xlsx' na pasta do projeto.")
        return
    
    # =========================================================================
    # ETAPA 0: Validação Inicial dos Dados
    # =========================================================================
    print_section("Validação inicial dos dados brutos")
    # Carregar dados para validação rápida
    df_raw = pd.read_excel(data_file)
    print(f"  Registros brutos: {len(df_raw)}")
    print(f"  Colunas: {df_raw.columns.tolist()}")
    print(f"  Valores ausentes por coluna:\n{df_raw.isnull().sum()}")
    # Checar valores negativos ou inconsistentes
    if (df_raw['casos'] < 0).any():
        print("  ⚠️ Atenção: Existem valores negativos de casos!")
    # ...pode-se adicionar mais validações conforme necessário...
    del df_raw
    
    # =========================================================================
    # CARREGAR DADOS
    # =========================================================================
    print_section("Carregando dados e padronizando estrutura")
    df = load_and_merge_dengue_data(data_file)
    print(f"✓ Dados carregados: {len(df)} registros")
    print(f"  Municípios: {df['municipio'].nunique()}")
    print(f"  Anos: {sorted(df['ano'].unique())}")
    # Remover registros com dados ausentes críticos
    df = df.dropna(subset=['municipio', 'ano', 'semana_epi', 'casos'])
    # Garantir tipos corretos
    df['casos'] = df['casos'].astype(int)
    # Ano principal para análises detalhadas
    YEAR = 2013  # Ano com muitos casos
    
    # =========================================================================
    # TAREFA 1: SEMANAS EPIDEMIOLÓGICAS
    # =========================================================================
    print_header("TAREFA 1: ANÁLISE DE SEMANAS EPIDEMIOLÓGICAS")
    
    # 1.1 Semanas por ano
    print_section("Semanas epidemiológicas por ano")
    weeks_per_year = get_weeks_per_year(2010, 2016)
    for year, weeks in weeks_per_year.items():
        marker = " *" if weeks == 53 else ""
        print(f"  {year}: {weeks} semanas{marker}")
    
    years_53 = identify_years_with_53_weeks(2010, 2016)
    print(f"\nAnos com 53 semanas: {years_53}")
    
    # Visualização: Semanas por ano
    plot_weeks_per_year(weeks_per_year, 
                        output_path=output_dir / "tarefa1_semanas_por_ano.png")
    
    # 1.2 Curvas epidêmicas por ano
    print_section("Curvas epidêmicas por ano")
    plot_epidemic_curves_by_year(df, 
                                 output_path=output_dir / "tarefa1_curvas_por_ano.png")
    
    # 1.3 Transições entre anos
    print_section("Transições entre anos")
    plot_year_transitions(df, 
                          output_path=output_dir / "tarefa1_transicoes_anos.png")
    
    # 1.4 Municípios mais afetados
    print_section(f"Municípios mais afetados em {YEAR}")
    plot_top_municipalities_by_year(df, top_n=15, year=YEAR,
                                    output_path=output_dir / f"tarefa1_top_municipios_{YEAR}.png")
    
    # 1.5 Período epidêmico para Rio de Janeiro
    print_section("Período epidêmico - Exemplo: Rio de Janeiro")
    rj_data = df[(df['municipio'] == 'Rio de Janeiro') & (df['ano'] == YEAR)]
    rj_series = rj_data.sort_values('semana_epi')['casos'].values
    
    epidemic_period = define_epidemic_period(rj_series, threshold_pct=0.1)
    if epidemic_period[0] is not None:
        print(f"  Período epidêmico: Semana {epidemic_period[0]+1} a {epidemic_period[1]+1}")
    
    plot_epidemic_period(rj_series, municipality='Rio de Janeiro', year=YEAR,
                        epidemic_period=epidemic_period,
                        output_path=output_dir / f"tarefa1_periodo_epidemico_rj_{YEAR}.png")
    
    # =========================================================================
    # TAREFA 2: NORMALIZAÇÃO
    # =========================================================================
    print_header("TAREFA 2: NORMALIZAÇÃO DE DADOS")
    
    # 2.1 Calcular taxas de incidência
    print_section(f"Taxas de incidência - {YEAR}")
    year_data = df[df['ano'] == YEAR]
    totals = year_data.groupby('municipio')['casos'].sum().reset_index()
    totals.columns = ['municipio', 'total_casos']
    totals['populacao'] = totals['municipio'].map(POPULACAO_CENSO_2010)
    totals['taxa_incidencia'] = (totals['total_casos'] / totals['populacao']) * 100000
    totals = totals.dropna()
    totals = totals.sort_values('taxa_incidencia', ascending=False)
    
    print(f"  Top 5 por taxa de incidência:")
    for _, row in totals.head(5).iterrows():
        print(f"    {row['municipio']}: {row['taxa_incidencia']:.1f} por 100.000 hab.")
    
    # Visualização: Taxas de incidência
    plot_incidence_rates(totals, top_n=20, year=YEAR,
                        output_path=output_dir / f"tarefa2_taxa_incidencia_{YEAR}.png")
    
    # 2.2 Séries normalizadas
    print_section("Normalizando séries temporais")
    time_series = get_yearly_time_series(df, YEAR, normalize_to_52=True)
    # Padronizar tamanho das séries (52 semanas)
    for mun, series in time_series.items():
        if len(series) != 52:
            print(f"  ⚠️ Série de {mun} tem {len(series)} semanas. Preenchendo com zeros.")
            time_series[mun] = np.pad(series, (0, 52 - len(series)), 'constant')
    # Filtrar municípios com dados de população e séries não nulas
    time_series_filtered = {mun: series for mun, series in time_series.items() 
                           if mun in POPULACAO_CENSO_2010 and np.sum(series) > 0}
    print(f"  Municípios com dados completos: {len(time_series_filtered)}")
    # Normalizar por área unitária
    normalized_series = normalize_all_series_by_total(time_series_filtered)
    # Visualização: Curvas normalizadas (top 10)
    top_mun_list = totals.head(10)['municipio'].tolist()
    top_normalized = {mun: normalized_series[mun] for mun in top_mun_list 
                      if mun in normalized_series}
    plot_normalized_curves(top_normalized, 
                          title=f"Curvas Normalizadas (Área Unitária) - Top 10 Municípios - {YEAR}",
                          output_path=output_dir / f"tarefa2_curvas_normalizadas_{YEAR}.png")
    
    # 2.3 Comparação de normalizações (exemplo)
    print_section("Comparação de normalizações - Rio de Janeiro")
    mun_example = 'Rio de Janeiro'
    if mun_example in time_series_filtered and mun_example in POPULACAO_CENSO_2010:
        original = time_series_filtered[mun_example]
        pop_norm = normalize_by_population(original, POPULACAO_CENSO_2010[mun_example])
        unit_norm = normalize_by_total_infected(original)
        
        plot_normalization_comparison(original, pop_norm, unit_norm,
                                     municipality=mun_example, year=YEAR,
                                     output_path=output_dir / f"tarefa2_comparacao_normalizacao_{YEAR}.png")
    
    # =========================================================================
    # TAREFA 3: DISTÂNCIAS
    # =========================================================================
    print_header("TAREFA 3: CÁLCULO DE DISTÂNCIAS L1 E L2")
    
    # 3.1 Calcular matrizes de distância
    print_section("Calculando matrizes de distância")
    if len(normalized_series) < 2:
        print("❌ Não há municípios suficientes para calcular distâncias.")
        return
    MD1, MD2, municipalities = compute_both_distance_matrices(normalized_series)
    print(f"  Dimensões das matrizes: {MD1.shape}")
    print(f"  Distância L1 - Mín: {MD1[MD1 > 0].min():.4f}, Máx: {MD1.max():.4f}, Média: {MD1[MD1 > 0].mean():.4f}")
    print(f"  Distância L2 - Mín: {MD2[MD2 > 0].min():.4f}, Máx: {MD2.max():.4f}, Média: {MD2[MD2 > 0].mean():.4f}")
    # Salvar matrizes em CSV
    save_distance_matrices(MD1, MD2, municipalities, output_dir)
    
    # 3.2 Visualização: Distribuição das distâncias
    plot_distance_distribution(MD1, MD2,
                              output_path=output_dir / f"tarefa3_distribuicao_distancias_{YEAR}.png")
    
    # 3.3 Pares mais similares
    print_section("Pares mais sincronizados")
    similar_L1 = find_most_similar_pairs(MD1, municipalities, top_n=15)
    print("  Top 5 pares mais similares (L1):")
    for _, row in similar_L1.head(5).iterrows():
        print(f"    {row['municipio_1'][:20]} ↔ {row['municipio_2'][:20]}: {row['distancia']:.4f}")
    
    # Visualização: Curvas dos pares similares
    plot_similar_pairs_curves(normalized_series, similar_L1, top_n=6,
                             output_path=output_dir / f"tarefa3_pares_similares_{YEAR}.png")
    
    # 3.4 Ranking de sincronização
    print_section("Ranking de sincronização")
    ranking = compute_synchronization_ranking(MD1, municipalities)
    print("  Top 5 municípios mais representativos:")
    for _, row in ranking.head(5).iterrows():
        print(f"    {row['ranking']}. {row['municipio']}: dist. média = {row['distancia_media']:.4f}")
    
    plot_synchronization_ranking(ranking, top_n=20,
                                output_path=output_dir / f"tarefa3_ranking_sincronizacao_{YEAR}.png")
    
    # 3.5 Heatmaps (subset para visualização)
    print_section("Gerando heatmaps")
    top_sync = ranking.head(25)['municipio'].tolist()
    top_indices = [municipalities.index(m) for m in top_sync if m in municipalities]
    
    MD1_sub = MD1[np.ix_(top_indices, top_indices)]
    MD2_sub = MD2[np.ix_(top_indices, top_indices)]
    top_mun_names = [municipalities[i] for i in top_indices]
    
    plot_distance_heatmap(MD1_sub, top_mun_names, metric_name='L1 (Manhattan)',
                         output_path=output_dir / f"tarefa3_heatmap_L1_{YEAR}.png")
    plot_distance_heatmap(MD2_sub, top_mun_names, metric_name='L2 (Euclidiana)',
                         output_path=output_dir / f"tarefa3_heatmap_L2_{YEAR}.png")
    
    # =========================================================================
    # TAREFA 4: COMPLEXO SIMPLICIAL
    # =========================================================================
    print_header("TAREFA 4: COMPLEXOS SIMPLICIAIS")
    
    # 4.1 Análise de limiares
    print_section("Análise de limiares")
    threshold_analysis = analyze_thresholds(MD1, municipalities, metric_name='L1')
    print(threshold_analysis.to_string(index=False))
    
    plot_threshold_analysis(threshold_analysis, metric_name='L1',
                           output_path=output_dir / f"tarefa4_analise_limiares_{YEAR}.png")
    
    # 4.2 Criar complexo com limiar ótimo
    print_section("Criando complexo simplicial")
    distances_L1 = MD1[np.triu_indices_from(MD1, k=1)]
    threshold_30 = np.percentile(distances_L1, 30)
    
    print(f"  Limiar (percentil 30): {threshold_30:.4f}")
    
    complex_obj = create_simplicial_complex_from_distance_matrix(
        MD1, municipalities, threshold_30, max_dimension=3
    )
    
    counts = complex_obj.count_simplices()
    print(f"  Estrutura do complexo:")
    print(f"    0-simplexos (vértices): {counts.get(0, 0)}")
    print(f"    1-simplexos (arestas): {counts.get(1, 0)}")
    print(f"    2-simplexos (triângulos): {counts.get(2, 0)}")
    print(f"    3-simplexos (tetraedros): {counts.get(3, 0)}")
    
    # 4.3 Identificar clusters
    print_section("Identificando clusters")
    clusters = identify_clusters(MD1, municipalities, threshold_30)
    print(f"  Número de clusters: {len(clusters)}")
    
    for i, cluster in enumerate(clusters[:5], 1):
        examples = ', '.join(cluster[:3])
        suffix = f" e mais {len(cluster)-3}" if len(cluster) > 3 else ""
        print(f"    Cluster {i} ({len(cluster)} mun.): {examples}{suffix}")
    
    # Visualização: Clusters
    plot_clusters(clusters, title=f"Clusters de Municípios (limiar={threshold_30:.4f})",
                 output_path=output_dir / f"tarefa4_clusters_{YEAR}.png")
    
    # 4.4 Visualização do complexo (subset)
    print_section("Visualizando complexo simplicial")
    
    # Usar top 30 municípios mais centrais
    mean_distances = MD1.mean(axis=1)
    top_indices = np.argsort(mean_distances)[:30]
    
    top_mun_complex = [municipalities[i] for i in top_indices]
    top_MD1 = MD1[np.ix_(top_indices, top_indices)]
    
    threshold_subset = np.percentile(top_MD1[np.triu_indices_from(top_MD1, k=1)], 40)
    complex_subset = create_simplicial_complex_from_distance_matrix(
        top_MD1, top_mun_complex, threshold_subset, max_dimension=2
    )
    
    plot_simplicial_complex_graph(complex_subset.simplices, top_mun_complex,
                                  title=f"Complexo Simplicial - Top 30 Municípios (limiar={threshold_subset:.3f})",
                                  output_path=output_dir / f"tarefa4_complexo_grafo_{YEAR}.png")
    
    # 4.5 Encontrar limiar ótimo para ~5 clusters
    print_section("Busca de limiar ótimo")
    optimal_threshold = find_optimal_threshold(MD1, municipalities, target_clusters=5)
    print(f"  Limiar para ~5 clusters: {optimal_threshold:.4f}")
    
    clusters_optimal = identify_clusters(MD1, municipalities, optimal_threshold)
    print(f"  Clusters obtidos: {len(clusters_optimal)}")
    
    plot_clusters(clusters_optimal, 
                 title=f"Clusters Ótimos (limiar={optimal_threshold:.4f}, ~5 clusters)",
                 output_path=output_dir / f"tarefa4_clusters_otimos_{YEAR}.png")
    
    # =========================================================================
    # DASHBOARD RESUMO
    # =========================================================================
    print_header("GERANDO DASHBOARD RESUMO")
    
    plot_summary_dashboard(df, year=YEAR, incidence_df=totals,
                          output_path=output_dir / "dashboard_resumo.png")
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print_header("ANÁLISE CONCLUÍDA")
    
    # Listar arquivos gerados
    output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.csv"))
    
    print(f"\n📁 Arquivos gerados em: {output_dir}")
    print("-" * 60)
    
    for f in sorted(output_files):
        size_kb = f.stat().st_size / 1024
        print(f"  • {f.name} ({size_kb:.1f} KB)")
    
    print("-" * 60)
    print(f"Total: {len(output_files)} arquivos")
    
    print("\n" + "=" * 80)
    print("  CONCLUSÕES DA ANÁLISE")
    print("=" * 80)
    print("""
    1. SEMANAS EPIDEMIOLÓGICAS:
       • A maioria dos anos tem 52 semanas; 2014 tem 53 semanas
       • 2016 tem dados incompletos (apenas 32 semanas)
       • Recomenda-se normalizar para 52 semanas para comparações

    2. NORMALIZAÇÃO:
       • A normalização por população revela municípios proporcionalmente
         mais afetados, independente do tamanho
       • A normalização por área unitária permite comparar formatos das
         curvas epidêmicas

    3. DISTÂNCIAS:
       • Municípios com menor distância L1/L2 são mais sincronizados
       • Pares similares indicam dinâmicas epidêmicas correlacionadas
       • O ranking de sincronização identifica municípios representativos

    4. COMPLEXOS SIMPLICIAIS:
       • Revelam estrutura topológica das relações entre municípios
       • Clusters identificados podem guiar vigilância epidemiológica
       • Limiares menores = mais clusters; maiores = mais conexões

    Para mais detalhes, consulte os gráficos gerados na pasta output/.
    """)


if __name__ == "__main__":
    main()
