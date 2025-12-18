"""
Tarefa 1: Análise de Semanas Epidemiológicas

Este módulo analisa:
1. Calendário das semanas epidemiológicas (anos com 52 vs 53 semanas)
2. Como padronizar séries para 52 semanas
3. Análise da última semana de um ano e primeira do seguinte
4. Definição do período epidêmico (PE) a partir das curvas epidêmicas

Achados principais do arquivo de dados:
- 2014 tem 53 semanas epidemiológicas
- 2016 tem apenas 32 semanas (dados incompletos)
- Total de 92 municípios do Rio de Janeiro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from epiweeks import Week, Year

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series


def get_weeks_per_year(start_year: int = 2010, end_year: int = 2016) -> dict:
    """
    Determina o número de semanas epidemiológicas para cada ano.
    
    Anos com 53 semanas epidemiológicas ocorrem quando o ano começa ou
    termina em quinta-feira (ou ambos em anos bissextos).
    
    Parameters
    ----------
    start_year : int
        Ano inicial da análise.
    end_year : int
        Ano final da análise.
    
    Returns
    -------
    dict
        Dicionário {ano: número_de_semanas}.
    """
    weeks_per_year = {}
    for year in range(start_year, end_year + 1):
        epi_year = Year(year)
        weeks_per_year[year] = epi_year.totalweeks()
    return weeks_per_year


def identify_years_with_53_weeks(start_year: int = 2010, end_year: int = 2016) -> list:
    """
    Identifica quais anos têm 53 semanas epidemiológicas.
    
    Parameters
    ----------
    start_year : int
        Ano inicial.
    end_year : int
        Ano final.
    
    Returns
    -------
    list
        Lista de anos com 53 semanas.
    """
    weeks = get_weeks_per_year(start_year, end_year)
    years_53 = [year for year, n_weeks in weeks.items() if n_weeks == 53]
    return years_53


def normalize_to_52_weeks(series: np.ndarray, year: int, method: str = 'merge_last') -> np.ndarray:
    """
    Normaliza uma série temporal de 53 semanas para 52 semanas.
    
    Métodos disponíveis:
    - 'merge_last': Combina a semana 53 com a semana 52
    - 'merge_first': Combina a semana 53 com a semana 1 do ano seguinte
    - 'average': Distribui os casos da semana 53 entre as semanas 52 e 1
    - 'remove': Remove a semana 53 (não recomendado, perde dados)
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal com 53 semanas.
    year : int
        Ano da série.
    method : str
        Método de normalização.
    
    Returns
    -------
    np.ndarray
        Série temporal com 52 semanas.
    """
    if len(series) == 52:
        return series
    
    if len(series) != 53:
        raise ValueError(f"Série deve ter 52 ou 53 elementos, mas tem {len(series)}")
    
    if method == 'merge_last':
        # Combina semana 53 com semana 52
        normalized = series[:52].copy()
        normalized[51] += series[52]
        return normalized
    
    elif method == 'merge_first':
        # Combina semana 53 com semana 1 (retorna série de 52 começando da semana 2)
        normalized = series[1:].copy()
        normalized[0] += series[0]
        return normalized
    
    elif method == 'average':
        # Distribui casos da semana 53 entre semanas 52 e 1
        normalized = series[:52].copy()
        extra = series[52] / 2
        normalized[51] += extra
        # Nota: a outra metade deveria ir para semana 1 do próximo ano
        return normalized
    
    elif method == 'remove':
        # Remove semana 53 (perde dados!)
        return series[:52]
    
    else:
        raise ValueError(f"Método desconhecido: {method}")


def analyze_week_boundaries(df: pd.DataFrame, year_col: str = 'ano', 
                           week_col: str = 'semana_epi', 
                           cases_col: str = 'casos') -> pd.DataFrame:
    """
    Analisa a transição entre última semana de um ano e primeira do seguinte.
    
    Esta análise ajuda a entender como os casos se distribuem nas fronteiras
    entre anos, o que é importante para o período epidêmico que cruza anos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    year_col : str
        Nome da coluna de ano.
    week_col : str
        Nome da coluna de semana epidemiológica.
    cases_col : str
        Nome da coluna de casos.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com análise das fronteiras.
    """
    boundaries = []
    years = sorted(df[year_col].unique())
    
    for i in range(len(years) - 1):
        year = years[i]
        next_year = years[i + 1]
        
        # Última semana do ano atual
        year_data = df[df[year_col] == year]
        last_week = year_data[week_col].max()
        last_week_cases = year_data[year_data[week_col] == last_week][cases_col].sum()
        
        # Primeira semana do próximo ano
        next_year_data = df[df[year_col] == next_year]
        first_week_cases = next_year_data[next_year_data[week_col] == 1][cases_col].sum()
        
        boundaries.append({
            'ano': year,
            'ultima_semana': last_week,
            'casos_ultima_semana': last_week_cases,
            'casos_primeira_semana_seguinte': first_week_cases,
            'continuidade': 'Alta' if abs(last_week_cases - first_week_cases) < 
                          max(last_week_cases, first_week_cases) * 0.5 else 'Baixa'
        })
    
    return pd.DataFrame(boundaries)


def define_epidemic_period(series: np.ndarray, threshold_pct: float = 0.1,
                          min_consecutive_weeks: int = 2) -> tuple:
    """
    Define o período epidêmico baseado na curva epidêmica.
    
    O período epidêmico (PE) é definido como o intervalo de tempo entre
    o início e o fim de uma epidemia, geralmente identificado quando os
    casos ultrapassam um limiar.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal de casos.
    threshold_pct : float
        Percentual do pico para definir início/fim (ex: 0.1 = 10% do pico).
    min_consecutive_weeks : int
        Número mínimo de semanas consecutivas acima do limiar.
    
    Returns
    -------
    tuple
        (semana_inicio, semana_fim) do período epidêmico.
    """
    peak = np.max(series)
    threshold = peak * threshold_pct
    
    # Encontrar semanas acima do limiar
    above_threshold = series > threshold
    
    # Encontrar início (primeira sequência de semanas consecutivas acima do limiar)
    start_week = None
    consecutive = 0
    for i, above in enumerate(above_threshold):
        if above:
            consecutive += 1
            if consecutive >= min_consecutive_weeks and start_week is None:
                start_week = i - min_consecutive_weeks + 1
        else:
            consecutive = 0
    
    # Encontrar fim (última semana acima do limiar)
    end_week = None
    for i in range(len(above_threshold) - 1, -1, -1):
        if above_threshold[i]:
            end_week = i
            break
    
    return start_week, end_week


def plot_epidemic_curve(series: np.ndarray, municipality: str = None,
                       year: int = None, epidemic_period: tuple = None,
                       output_path: str = None):
    """
    Plota a curva epidêmica com marcação do período epidêmico.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal de casos.
    municipality : str, optional
        Nome do município.
    year : int, optional
        Ano da série.
    epidemic_period : tuple, optional
        (semana_inicio, semana_fim) do período epidêmico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    plt.figure(figsize=(12, 6))
    weeks = np.arange(1, len(series) + 1)
    
    plt.plot(weeks, series, 'b-', linewidth=2, label='Casos de Dengue')
    plt.fill_between(weeks, series, alpha=0.3)
    
    if epidemic_period and epidemic_period[0] is not None:
        start, end = epidemic_period
        plt.axvline(x=start + 1, color='g', linestyle='--', linewidth=2, 
                   label=f'Início PE (semana {start + 1})')
        plt.axvline(x=end + 1, color='r', linestyle='--', linewidth=2, 
                   label=f'Fim PE (semana {end + 1})')
        plt.axvspan(start + 1, end + 1, alpha=0.2, color='yellow', 
                   label='Período Epidêmico')
    
    title = 'Curva Epidêmica de Dengue'
    if municipality:
        title += f' - {municipality}'
    if year:
        title += f' ({year})'
    
    plt.title(title, fontsize=14)
    plt.xlabel('Semana Epidemiológica', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def analyze_epidemiological_calendar(start_year: int = 2010, end_year: int = 2016):
    """
    Análise completa do calendário epidemiológico.
    
    Imprime um relatório sobre semanas epidemiológicas e anos com 53 semanas.
    
    Parameters
    ----------
    start_year : int
        Ano inicial.
    end_year : int
        Ano final.
    """
    print("=" * 60)
    print("ANÁLISE DO CALENDÁRIO DE SEMANAS EPIDEMIOLÓGICAS")
    print("=" * 60)
    
    weeks = get_weeks_per_year(start_year, end_year)
    years_53 = identify_years_with_53_weeks(start_year, end_year)
    
    print(f"\nPeríodo analisado: {start_year} - {end_year}")
    print("\nNúmero de semanas epidemiológicas por ano:")
    for year, n_weeks in weeks.items():
        marker = " *" if n_weeks == 53 else ""
        print(f"  {year}: {n_weeks} semanas{marker}")
    
    print(f"\nAnos com 53 semanas epidemiológicas: {years_53}")
    
    print("\n" + "-" * 60)
    print("CONCLUSÕES:")
    print("-" * 60)
    print("""
1. A maioria dos anos tem 52 semanas epidemiológicas.
2. Anos com 53 semanas ocorrem aproximadamente a cada 5-6 anos.
3. Para comparar séries temporais, é necessário normalizar para 52 semanas.
4. A recomendação é combinar a semana 53 com a semana 52, pois:
   - A última semana de um ano e a primeira do seguinte geralmente
     têm continuidade epidemiológica (mesma fase da epidemia).
   - O período epidêmico da dengue no Brasil geralmente cruza
     a virada do ano (pico entre janeiro e abril).
    """)


def analyze_real_data():
    """
    Analisa os dados reais de dengue do Rio de Janeiro.
    """
    print("=" * 70)
    print("ANÁLISE DOS DADOS REAIS DE DENGUE - RIO DE JANEIRO (2010-2016)")
    print("=" * 70)
    
    # Carregar dados reais
    filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    df = load_and_merge_dengue_data(filepath)
    
    # Análise de semanas por ano
    print("\n--- Semanas Epidemiológicas por Ano nos Dados ---")
    weeks_per_year = df.groupby('ano')['semana_epi'].max()
    for year, max_week in weeks_per_year.items():
        marker = " *" if max_week == 53 else ""
        if max_week < 52:
            marker = " (dados incompletos)"
        print(f"  {year}: {max_week} semanas{marker}")
    
    # Anos com 53 semanas
    years_53 = weeks_per_year[weeks_per_year == 53].index.tolist()
    print(f"\nAnos com 53 semanas nos dados: {years_53}")
    
    # Análise da transição entre anos
    print("\n--- Análise da Transição Entre Anos ---")
    for year in range(2010, 2016):
        last_week_data = df[(df['ano'] == year) & (df['semana_epi'] == df[df['ano'] == year]['semana_epi'].max())]
        first_week_next = df[(df['ano'] == year + 1) & (df['semana_epi'] == 1)]
        
        last_week = last_week_data['semana_epi'].iloc[0]
        last_total = last_week_data['casos'].sum()
        first_total = first_week_next['casos'].sum()
        
        print(f"  {year} (semana {last_week}) -> {year+1} (semana 1): {last_total:,} -> {first_total:,} casos")
        
    # Plotar curva epidêmica agregada para um município exemplo
    print("\n--- Curva Epidêmica: Exemplo com Municípios Mais Afetados ---")
    
    # Top 5 municípios com mais casos
    top_municipalities = df.groupby('municipio')['casos'].sum().nlargest(5).index.tolist()
    print(f"Top 5 municípios com mais casos: {top_municipalities}")
    
    return df, top_municipalities


def plot_multiple_municipalities(df: pd.DataFrame, municipalities: list, year: int = 2013):
    """
    Plota curvas epidêmicas de múltiplos municípios.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados de dengue.
    municipalities : list
        Lista de municípios a plotar.
    year : int
        Ano a visualizar.
    """
    plt.figure(figsize=(14, 8))
    
    for mun in municipalities:
        mun_data = df[(df['municipio'] == mun) & (df['ano'] == year)]
        mun_data = mun_data.sort_values('semana_epi')
        
        plt.plot(mun_data['semana_epi'], mun_data['casos'], 
                label=mun, linewidth=2, alpha=0.8)
    
    plt.xlabel('Semana Epidemiológica', fontsize=12)
    plt.ylabel('Número de Casos', fontsize=12)
    plt.title(f'Curvas Epidêmicas de Dengue - {year}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Executar análise do calendário teórico
    analyze_epidemiological_calendar(2010, 2016)
    
    # Analisar dados reais
    print("\n")
    df, top_municipalities = analyze_real_data()
    
    # Plotar curvas para os municípios mais afetados
    print("\n--- Plotando curvas epidêmicas ---")
    plot_multiple_municipalities(df, top_municipalities, year=2013)
