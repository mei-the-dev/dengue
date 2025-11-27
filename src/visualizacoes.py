"""
Módulo de Visualizações para Análise de Dengue

Este módulo centraliza todas as funções de visualização do projeto,
gerando e salvando gráficos para todas as análises.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import networkx as nx

# Configurações globais de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# Cores para os gráficos
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'info': '#17becf',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f'
}


def ensure_output_dir(output_dir: str = 'output') -> Path:
    """Garante que o diretório de saída existe."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


# =============================================================================
# TAREFA 1: Semanas Epidemiológicas
# =============================================================================

def plot_weeks_per_year(weeks_per_year: dict, output_path: str = None):
    """
    Plota gráfico de barras com o número de semanas por ano.
    
    Parameters
    ----------
    weeks_per_year : dict
        Dicionário {ano: número_de_semanas}.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = list(weeks_per_year.keys())
    weeks = list(weeks_per_year.values())
    
    colors = [COLORS['danger'] if w == 53 else COLORS['primary'] for w in weeks]
    
    bars = ax.bar(years, weeks, color=colors, edgecolor='black', linewidth=1)
    
    # Linha de referência em 52
    ax.axhline(y=52, color=COLORS['gray'], linestyle='--', linewidth=2, label='52 semanas')
    
    ax.set_xlabel('Ano')
    ax.set_ylabel('Número de Semanas Epidemiológicas')
    ax.set_title('Semanas Epidemiológicas por Ano (2010-2016)')
    ax.set_xticks(years)
    ax.set_ylim(0, 55)
    
    # Anotações nas barras
    for bar, week in zip(bars, weeks):
        ax.annotate(f'{week}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['primary'], label='52 semanas'),
        mpatches.Patch(facecolor=COLORS['danger'], label='53 semanas'),
        Line2D([0], [0], color=COLORS['gray'], linestyle='--', linewidth=2, label='Referência (52)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_epidemic_curves_by_year(df: pd.DataFrame, output_path: str = None):
    """
    Plota curvas epidêmicas agregadas por ano.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'ano', 'semana_epi', 'casos'.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    years = sorted(df['ano'].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i/len(years)) for i in range(len(years))]
    
    for i, year in enumerate(years):
        year_data = df[df['ano'] == year].groupby('semana_epi')['casos'].sum()
        ax.plot(year_data.index, year_data.values, 
               color=colors[i], linewidth=2, label=str(year), alpha=0.8)
    
    ax.set_xlabel('Semana Epidemiológica')
    ax.set_ylabel('Total de Casos')
    ax.set_title('Curvas Epidêmicas de Dengue por Ano - Rio de Janeiro')
    ax.legend(title='Ano', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_year_transitions(df: pd.DataFrame, output_path: str = None):
    """
    Plota transições entre última semana de um ano e primeira do seguinte.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados de dengue.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    transitions = []
    years = sorted(df['ano'].unique())
    
    for i in range(len(years) - 1):
        year = years[i]
        next_year = years[i + 1]
        
        last_week_cases = df[(df['ano'] == year) & 
                            (df['semana_epi'] == df[df['ano'] == year]['semana_epi'].max())]['casos'].sum()
        first_week_cases = df[(df['ano'] == next_year) & (df['semana_epi'] == 1)]['casos'].sum()
        
        transitions.append({
            'period': f'{year}→{next_year}',
            'last': last_week_cases,
            'first': first_week_cases
        })
    
    x = range(len(transitions))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], [t['last'] for t in transitions], 
                   width, label='Última semana', color=COLORS['primary'])
    bars2 = ax.bar([i + width/2 for i in x], [t['first'] for t in transitions], 
                   width, label='Primeira semana seguinte', color=COLORS['secondary'])
    
    ax.set_xlabel('Transição de Ano')
    ax.set_ylabel('Número de Casos')
    ax.set_title('Transição de Casos entre Anos')
    ax.set_xticks(x)
    ax.set_xticklabels([t['period'] for t in transitions])
    ax.legend()
    
    # Anotações
    for bar in bars1:
        ax.annotate(f'{int(bar.get_height()):,}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, rotation=45)
    for bar in bars2:
        ax.annotate(f'{int(bar.get_height()):,}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_top_municipalities_by_year(df: pd.DataFrame, top_n: int = 10, 
                                    year: int = 2013, output_path: str = None):
    """
    Plota os municípios mais afetados em um ano específico.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados de dengue.
    top_n : int
        Número de municípios a mostrar.
    year : int
        Ano a analisar.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    year_data = df[df['ano'] == year]
    top_mun = year_data.groupby('municipio')['casos'].sum().nlargest(top_n)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_mun)))[::-1]
    
    bars = ax.barh(range(len(top_mun)), top_mun.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_mun)))
    ax.set_yticklabels(top_mun.index)
    ax.invert_yaxis()
    
    ax.set_xlabel('Total de Casos')
    ax.set_ylabel('Município')
    ax.set_title(f'Top {top_n} Municípios Mais Afetados - {year}')
    
    # Anotações
    for bar in bars:
        ax.annotate(f'{int(bar.get_width()):,}', 
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_epidemic_period(series: np.ndarray, municipality: str = None,
                        year: int = None, epidemic_period: tuple = None,
                        output_path: str = None):
    """
    Plota curva epidêmica com período epidêmico destacado.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal de casos.
    municipality : str, optional
        Nome do município.
    year : int, optional
        Ano.
    epidemic_period : tuple, optional
        (semana_inicio, semana_fim).
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    weeks = np.arange(1, len(series) + 1)
    
    ax.plot(weeks, series, color=COLORS['primary'], linewidth=2, label='Casos')
    ax.fill_between(weeks, series, alpha=0.3, color=COLORS['primary'])
    
    if epidemic_period and epidemic_period[0] is not None:
        start, end = epidemic_period
        ax.axvline(x=start + 1, color=COLORS['success'], linestyle='--', 
                  linewidth=2, label=f'Início PE (sem. {start + 1})')
        ax.axvline(x=end + 1, color=COLORS['danger'], linestyle='--', 
                  linewidth=2, label=f'Fim PE (sem. {end + 1})')
        ax.axvspan(start + 1, end + 1, alpha=0.15, color=COLORS['warning'])
    
    title = 'Curva Epidêmica de Dengue'
    if municipality:
        title += f' - {municipality}'
    if year:
        title += f' ({year})'
    
    ax.set_xlabel('Semana Epidemiológica')
    ax.set_ylabel('Número de Casos')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


# =============================================================================
# TAREFA 2: Normalização
# =============================================================================

def plot_incidence_rates(incidence_df: pd.DataFrame, top_n: int = 20, 
                         year: int = None, output_path: str = None):
    """
    Plota gráfico de barras com taxas de incidência.
    
    Parameters
    ----------
    incidence_df : pd.DataFrame
        DataFrame com colunas 'municipio' e 'taxa_incidencia'.
    top_n : int
        Número de municípios a mostrar.
    year : int, optional
        Ano da análise.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    top_data = incidence_df.head(top_n)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top_data)))[::-1]
    
    bars = ax.barh(range(len(top_data)), top_data['taxa_incidencia'], 
                   color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['municipio'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Taxa de Incidência (por 100.000 habitantes)')
    ax.set_ylabel('Município')
    
    title = f'Top {top_n} Municípios por Taxa de Incidência de Dengue'
    if year:
        title += f' - {year}'
    ax.set_title(title)
    
    # Anotações
    for bar in bars:
        ax.annotate(f'{bar.get_width():.1f}', 
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_normalized_curves(normalized_series: Dict[str, np.ndarray],
                          municipalities: list = None,
                          title: str = "Curvas Normalizadas (Área Unitária)",
                          output_path: str = None):
    """
    Plota curvas normalizadas para comparação visual.
    
    Parameters
    ----------
    normalized_series : dict
        Dicionário {município: série_normalizada}.
    municipalities : list, optional
        Lista de municípios a plotar.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    if municipalities is None:
        municipalities = list(normalized_series.keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cmap = plt.cm.tab20
    
    for i, municipality in enumerate(municipalities):
        if municipality in normalized_series:
            series = normalized_series[municipality]
            weeks = np.arange(1, len(series) + 1)
            ax.plot(weeks, series, color=cmap(i % 20), 
                   label=municipality, linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Semana Epidemiológica')
    ax.set_ylabel('Proporção de Casos')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_normalization_comparison(original_series: np.ndarray, 
                                  pop_normalized: np.ndarray,
                                  unit_normalized: np.ndarray,
                                  municipality: str,
                                  year: int = None,
                                  output_path: str = None):
    """
    Compara as diferentes normalizações para um município.
    
    Parameters
    ----------
    original_series : np.ndarray
        Série original de casos.
    pop_normalized : np.ndarray
        Série normalizada pela população.
    unit_normalized : np.ndarray
        Série normalizada para área unitária.
    municipality : str
        Nome do município.
    year : int, optional
        Ano.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    weeks = np.arange(1, len(original_series) + 1)
    
    # Original
    axes[0].fill_between(weeks, original_series, alpha=0.3, color=COLORS['primary'])
    axes[0].plot(weeks, original_series, color=COLORS['primary'], linewidth=2)
    axes[0].set_xlabel('Semana Epidemiológica')
    axes[0].set_ylabel('Número de Casos')
    axes[0].set_title('Dados Originais')
    axes[0].grid(True, alpha=0.3)
    
    # Normalizado por população
    axes[1].fill_between(weeks, pop_normalized, alpha=0.3, color=COLORS['secondary'])
    axes[1].plot(weeks, pop_normalized, color=COLORS['secondary'], linewidth=2)
    axes[1].set_xlabel('Semana Epidemiológica')
    axes[1].set_ylabel('Taxa (por 100.000 hab.)')
    axes[1].set_title('Normalização por População')
    axes[1].grid(True, alpha=0.3)
    
    # Normalizado área unitária
    axes[2].fill_between(weeks, unit_normalized, alpha=0.3, color=COLORS['success'])
    axes[2].plot(weeks, unit_normalized, color=COLORS['success'], linewidth=2)
    axes[2].set_xlabel('Semana Epidemiológica')
    axes[2].set_ylabel('Proporção')
    axes[2].set_title('Normalização Área Unitária')
    axes[2].grid(True, alpha=0.3)
    
    title = f'Comparação de Normalizações - {municipality}'
    if year:
        title += f' ({year})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


# =============================================================================
# TAREFA 3: Distâncias
# =============================================================================

def plot_distance_heatmap(distance_matrix: np.ndarray, 
                          municipalities: List[str],
                          metric_name: str = 'L1',
                          output_path: str = None,
                          figsize: Tuple[int, int] = (14, 12)):
    """
    Plota heatmap da matriz de distância.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matriz de distância.
    municipalities : list
        Lista de nomes dos municípios.
    metric_name : str
        Nome da métrica.
    output_path : str, optional
        Caminho para salvar o gráfico.
    figsize : tuple
        Tamanho da figura.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determinar se deve mostrar valores
    show_values = len(municipalities) <= 15
    
    im = sns.heatmap(distance_matrix, 
                     xticklabels=municipalities,
                     yticklabels=municipalities,
                     cmap='viridis',
                     annot=show_values,
                     fmt='.3f' if show_values else '',
                     square=True,
                     cbar_kws={'label': f'Distância {metric_name}'},
                     ax=ax)
    
    ax.set_title(f'Matriz de Distância {metric_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_distance_distribution(MD1: np.ndarray, MD2: np.ndarray, 
                               output_path: str = None):
    """
    Plota distribuição das distâncias L1 e L2.
    
    Parameters
    ----------
    MD1 : np.ndarray
        Matriz de distância L1.
    MD2 : np.ndarray
        Matriz de distância L2.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extrair distâncias (apenas triângulo superior)
    distances_L1 = MD1[np.triu_indices_from(MD1, k=1)]
    distances_L2 = MD2[np.triu_indices_from(MD2, k=1)]
    
    # Histograma L1
    axes[0].hist(distances_L1, bins=50, color=COLORS['primary'], 
                edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(distances_L1), color=COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Média: {np.mean(distances_L1):.4f}')
    axes[0].axvline(np.median(distances_L1), color=COLORS['success'], 
                   linestyle=':', linewidth=2, label=f'Mediana: {np.median(distances_L1):.4f}')
    axes[0].set_xlabel('Distância L1 (Manhattan)')
    axes[0].set_ylabel('Frequência')
    axes[0].set_title('Distribuição das Distâncias L1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histograma L2
    axes[1].hist(distances_L2, bins=50, color=COLORS['secondary'], 
                edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(distances_L2), color=COLORS['danger'], 
                   linestyle='--', linewidth=2, label=f'Média: {np.mean(distances_L2):.4f}')
    axes[1].axvline(np.median(distances_L2), color=COLORS['success'], 
                   linestyle=':', linewidth=2, label=f'Mediana: {np.median(distances_L2):.4f}')
    axes[1].set_xlabel('Distância L2 (Euclidiana)')
    axes[1].set_ylabel('Frequência')
    axes[1].set_title('Distribuição das Distâncias L2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_similar_pairs_curves(normalized_series: Dict[str, np.ndarray],
                              similar_pairs: pd.DataFrame,
                              top_n: int = 5,
                              output_path: str = None):
    """
    Plota curvas dos pares mais similares.
    
    Parameters
    ----------
    normalized_series : dict
        Séries normalizadas.
    similar_pairs : pd.DataFrame
        DataFrame com pares similares.
    top_n : int
        Número de pares a plotar.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(similar_pairs.head(top_n).iterrows()):
        if i >= 6:
            break
            
        mun1, mun2 = row['municipio_1'], row['municipio_2']
        dist = row['distancia']
        
        if mun1 in normalized_series and mun2 in normalized_series:
            series1 = normalized_series[mun1]
            series2 = normalized_series[mun2]
            weeks = np.arange(1, len(series1) + 1)
            
            axes[i].plot(weeks, series1, color=COLORS['primary'], 
                        linewidth=2, label=mun1[:20])
            axes[i].plot(weeks, series2, color=COLORS['secondary'], 
                        linewidth=2, label=mun2[:20], linestyle='--')
            axes[i].set_xlabel('Semana Epidemiológica')
            axes[i].set_ylabel('Proporção')
            axes[i].set_title(f'Dist = {dist:.4f}')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Ocultar subplots vazios
    for i in range(min(top_n, 6), 6):
        axes[i].axis('off')
    
    fig.suptitle('Pares de Municípios Mais Sincronizados', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_synchronization_ranking(ranking: pd.DataFrame, top_n: int = 20,
                                 output_path: str = None):
    """
    Plota ranking de sincronização.
    
    Parameters
    ----------
    ranking : pd.DataFrame
        DataFrame com ranking de sincronização.
    top_n : int
        Número de municípios a mostrar.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_data = ranking.head(top_n)
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(top_data)))[::-1]
    
    bars = ax.barh(range(len(top_data)), top_data['distancia_media'], 
                   color=colors, edgecolor='black')
    ax.set_yticks(range(len(top_data)))
    ax.set_yticklabels(top_data['municipio'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Distância Média')
    ax.set_ylabel('Município')
    ax.set_title(f'Top {top_n} Municípios Mais Representativos\n(Menor Distância Média = Mais Sincronizado)')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


# =============================================================================
# TAREFA 4: Complexo Simplicial
# =============================================================================

def plot_threshold_analysis(threshold_df: pd.DataFrame, metric_name: str = 'L1',
                           output_path: str = None):
    """
    Plota análise de limiares para o complexo simplicial.
    
    Parameters
    ----------
    threshold_df : pd.DataFrame
        DataFrame com análise de limiares.
    metric_name : str
        Nome da métrica.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Número de simplexos por dimensão
    axes[0].plot(threshold_df['limiar'], threshold_df['arestas'], 
                'o-', color=COLORS['primary'], label='Arestas (1-simplexos)', linewidth=2)
    axes[0].plot(threshold_df['limiar'], threshold_df['triangulos'], 
                's-', color=COLORS['secondary'], label='Triângulos (2-simplexos)', linewidth=2)
    axes[0].plot(threshold_df['limiar'], threshold_df['tetraedros'], 
                '^-', color=COLORS['success'], label='Tetraedros (3-simplexos)', linewidth=2)
    
    axes[0].set_xlabel('Limiar de Distância')
    axes[0].set_ylabel('Número de Simplexos')
    axes[0].set_title(f'Evolução dos Simplexos com Limiar ({metric_name})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Logaritmo
    axes[1].semilogy(threshold_df['limiar'], threshold_df['arestas'] + 1, 
                    'o-', color=COLORS['primary'], label='Arestas', linewidth=2)
    axes[1].semilogy(threshold_df['limiar'], threshold_df['triangulos'] + 1, 
                    's-', color=COLORS['secondary'], label='Triângulos', linewidth=2)
    
    axes[1].set_xlabel('Limiar de Distância')
    axes[1].set_ylabel('Número de Simplexos (log)')
    axes[1].set_title('Escala Logarítmica')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_simplicial_complex_graph(simplices: dict, vertices: List[str],
                                  title: str = "Complexo Simplicial",
                                  output_path: str = None):
    """
    Visualiza o complexo simplicial como um grafo.
    
    Parameters
    ----------
    simplices : dict
        Dicionário {dimensão: conjunto de simplexos}.
    vertices : list
        Lista de nomes dos vértices.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Criar grafo
    G = nx.Graph()
    G.add_nodes_from(range(len(vertices)))
    
    # Adicionar arestas
    edges = simplices.get(1, set())
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Desenhar triângulos
    triangles = simplices.get(2, set())
    for triangle in triangles:
        triangle = list(triangle)
        triangle_coords = [pos[v] for v in triangle]
        triangle_coords.append(triangle_coords[0])
        xs, ys = zip(*triangle_coords)
        ax.fill(xs, ys, alpha=0.2, color='yellow', edgecolor='orange', linewidth=2)
    
    # Desenhar arestas
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1, ax=ax)
    
    # Desenhar vértices
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue', 
                          edgecolors='navy', linewidths=1.5, ax=ax)
    
    # Rótulos
    if len(vertices) <= 25:
        labels = {i: vertices[i][:12] for i in range(len(vertices))}
    else:
        labels = {i: str(i) for i in range(len(vertices))}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Estatísticas
    n_edges = len(edges)
    n_triangles = len(triangles)
    stats_text = f"Vértices: {len(vertices)} | Arestas: {n_edges} | Triângulos: {n_triangles}"
    
    ax.set_title(f"{title}\n{stats_text}", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


def plot_clusters(clusters: List[List[str]], title: str = "Clusters de Municípios",
                  output_path: str = None):
    """
    Plota visualização dos clusters.
    
    Parameters
    ----------
    clusters : list
        Lista de clusters.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Dados para o gráfico
    cluster_sizes = [len(c) for c in clusters[:10]]  # Top 10 clusters
    cluster_labels = [f'Cluster {i+1}\n({len(c)} mun.)' for i, c in enumerate(clusters[:10])]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
    
    bars = ax.bar(range(len(cluster_sizes)), cluster_sizes, color=colors, edgecolor='black')
    ax.set_xticks(range(len(cluster_sizes)))
    ax.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax.set_ylabel('Número de Municípios')
    ax.set_title(title)
    
    # Anotações com exemplos de municípios
    for i, (bar, cluster) in enumerate(zip(bars, clusters[:10])):
        examples = cluster[:3]
        text = '\n'.join([m[:15] for m in examples])
        if len(cluster) > 3:
            text += f'\n...'
        
        ax.annotate(text, 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=7, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {output_path}")
    
    plt.close()


# =============================================================================
# RESUMO GERAL
# =============================================================================

def plot_summary_dashboard(df: pd.DataFrame, year: int = 2013,
                          incidence_df: pd.DataFrame = None,
                          output_path: str = None):
    """
    Cria dashboard resumo com múltiplos gráficos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo com dados de dengue.
    year : int
        Ano de análise.
    incidence_df : pd.DataFrame, optional
        DataFrame com taxas de incidência.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Subplot 1: Curvas epidêmicas por ano
    ax1 = fig.add_subplot(2, 2, 1)
    years = sorted(df['ano'].unique())
    cmap = plt.cm.viridis
    for i, y in enumerate(years):
        year_data = df[df['ano'] == y].groupby('semana_epi')['casos'].sum()
        ax1.plot(year_data.index, year_data.values, 
                color=cmap(i/len(years)), linewidth=2, label=str(y))
    ax1.set_xlabel('Semana Epidemiológica')
    ax1.set_ylabel('Total de Casos')
    ax1.set_title('Curvas Epidêmicas por Ano')
    ax1.legend(title='Ano')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Total de casos por ano
    ax2 = fig.add_subplot(2, 2, 2)
    yearly_totals = df.groupby('ano')['casos'].sum()
    colors = [COLORS['danger'] if t == yearly_totals.max() else COLORS['primary'] 
              for t in yearly_totals]
    bars = ax2.bar(yearly_totals.index, yearly_totals.values, color=colors, edgecolor='black')
    ax2.set_xlabel('Ano')
    ax2.set_ylabel('Total de Casos')
    ax2.set_title('Total de Casos por Ano')
    for bar in bars:
        ax2.annotate(f'{int(bar.get_height()):,}', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Top 10 municípios do ano selecionado
    ax3 = fig.add_subplot(2, 2, 3)
    year_data = df[df['ano'] == year]
    top_mun = year_data.groupby('municipio')['casos'].sum().nlargest(10)
    colors3 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_mun)))[::-1]
    ax3.barh(range(len(top_mun)), top_mun.values, color=colors3, edgecolor='black')
    ax3.set_yticks(range(len(top_mun)))
    ax3.set_yticklabels(top_mun.index)
    ax3.invert_yaxis()
    ax3.set_xlabel('Total de Casos')
    ax3.set_title(f'Top 10 Municípios - {year}')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Taxa de incidência (se disponível)
    ax4 = fig.add_subplot(2, 2, 4)
    if incidence_df is not None and len(incidence_df) > 0:
        top_inc = incidence_df.head(10)
        colors4 = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_inc)))[::-1]
        ax4.barh(range(len(top_inc)), top_inc['taxa_incidencia'], 
                color=colors4, edgecolor='black')
        ax4.set_yticks(range(len(top_inc)))
        ax4.set_yticklabels(top_inc['municipio'])
        ax4.invert_yaxis()
        ax4.set_xlabel('Taxa por 100.000 hab.')
        ax4.set_title(f'Top 10 por Taxa de Incidência - {year}')
    else:
        ax4.text(0.5, 0.5, 'Dados de incidência\nnão disponíveis', 
                ha='center', va='center', fontsize=14)
        ax4.axis('off')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Dashboard: Análise Epidemiológica de Dengue - Rio de Janeiro',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard salvo: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("Módulo de Visualizações carregado.")
    print("Use as funções deste módulo para gerar gráficos das análises.")
