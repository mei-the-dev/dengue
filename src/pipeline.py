"""
Dengue Analysis Pipeline
Centraliza o fluxo de dados, análise e visualização para uso em scripts e notebooks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tarefa0_carregar_dados import load_and_merge_dengue_data
from src.tarefa1_semanas_epidemiologicas import get_weeks_per_year, identify_years_with_53_weeks
from src.tarefa2_normalizacao import POPULACAO_CENSO_2010, normalize_by_population, normalize_by_total_infected, normalize_all_series_by_total
from src.tarefa3_distancias import compute_both_distance_matrices, find_most_similar_pairs, find_most_dissimilar_pairs
from src.tarefa4_complexo_simplicial import analyze_thresholds, find_optimal_threshold
from src.tarefa5_kmapper import prepare_data_for_mapper, create_mapper_visualization
from src.visualizacoes import *


def load_all_data(filepath=None):
    """Carrega e retorna o DataFrame unificado de dengue."""
    return load_and_merge_dengue_data(filepath)


def run_all_analyses(df, year=2013):
    """Executa as principais análises e retorna um dicionário de resultados."""
    # Filtrar ano de interesse
    df_year = df[df['ano'] == year].copy()
    weeks_per_year = get_weeks_per_year()
    # Normalização por população: taxa de incidência por município
    municipios = df_year['municipio'].unique()
    taxas_incidencia = []
    for mun in municipios:
        serie = df_year[df_year['municipio'] == mun]['casos'].values
        pop = POPULACAO_CENSO_2010.get(mun, None)
        if pop is not None:
            taxa = normalize_by_population(serie, pop)
            for semana, valor in zip(df_year[df_year['municipio'] == mun]['semana_epi'].values, taxa):
                taxas_incidencia.append({'municipio': mun, 'semana_epi': semana, 'taxa_incidencia': valor})
    norm_pop = pd.DataFrame(taxas_incidencia)
    # Garante que a coluna 'casos' é numérica
    df_year['casos'] = pd.to_numeric(df_year['casos'], errors='coerce')
    # Cria dicionário {municipio: array de casos} para normalização
    series_dict = {}
    for mun in df_year['municipio'].unique():
        arr = df_year[df_year['municipio'] == mun]['casos'].values
        arr = pd.to_numeric(arr, errors='coerce')
        arr = np.array(arr, dtype=np.float64)
        series_dict[mun] = arr
    norm_unit = normalize_all_series_by_total(series_dict)
    md1, md2, mun_list = compute_both_distance_matrices(norm_unit)
    similar_pairs = find_most_similar_pairs(md1, mun_list)
    dissimilar_pairs = find_most_dissimilar_pairs(md1, mun_list)
    thresholds = analyze_thresholds(md1, mun_list)
    optimal_thr = find_optimal_threshold(thresholds)
    return {
        'df': df,
        'df_year': df_year,
        'weeks_per_year': weeks_per_year,
        'norm_pop': norm_pop,
        'norm_unit': norm_unit,
        'md1': md1,
        'md2': md2,
        'mun_list': mun_list,
        'similar_pairs': similar_pairs,
        'dissimilar_pairs': dissimilar_pairs,
        'thresholds': thresholds,
        'optimal_thr': optimal_thr
    }


def generate_all_figures(results, output_dir='output'):
    """Gera e salva todas as figuras principais do projeto."""
    ensure_output_dir(output_dir)
    plot_weeks_per_year(results['weeks_per_year'], f'{output_dir}/weeks_per_year.pdf')
    plot_epidemic_curves_by_year(results['df'], f'{output_dir}/curvas_por_ano.pdf')
    plot_incidence_rates(results['norm_pop'], f'{output_dir}/incidencia.pdf')
    plot_normalized_curves(results['norm_unit'], f'{output_dir}/curvas_normalizadas.pdf')
    plot_distance_heatmap(results['md1'], results['mun_list'], f'{output_dir}/matriz_distancia_L1.pdf', title='Matriz de Distância L1')
    plot_distance_heatmap(results['md2'], results['mun_list'], f'{output_dir}/matriz_distancia_L2.pdf', title='Matriz de Distância L2')
    # Adicione outras visualizações conforme necessário


def get_summary_tables(results):
    """Retorna tabelas resumo úteis para notebook ou apresentação."""
    return {
        'top_similar_pairs': results['similar_pairs'],
        'top_dissimilar_pairs': results['dissimilar_pairs'],
        'weeks_per_year': results['weeks_per_year']
    }
