"""
Tarefa 2: Normalização de Dados

Este módulo implementa:
1. Normalização 1: Por população do Censo 2010 (taxa de incidência)
   a) Por ano solar (janeiro a dezembro)
   b) Por Período Epidêmico (PE)

2. Normalização 2: Por total de infectados no período (área unitária)
   a) Por ano solar
   b) Por Período Epidêmico (PE)

Nota: Os dados de população do Censo 2010 precisam ser obtidos do IBGE
ou de outra fonte oficial. Este módulo inclui dados de população 
estimados para os municípios do RJ.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from tarefa0_carregar_dados import load_and_merge_dengue_data, get_yearly_time_series

# Dados de população do Censo 2010 para municípios do RJ
# Fonte: IBGE - Censo Demográfico 2010
# Nota: Esta é uma amostra dos principais municípios. 
# Para análise completa, importar dados oficiais do IBGE.
POPULACAO_CENSO_2010 = {
    'Angra dos Reis': 169511,
    'Aperibé': 10213,
    'Araruama': 112008,
    'Areal': 11423,
    'Armação dos Búzios': 27560,
    'Arraial do Cabo': 27715,
    'Barra do Piraí': 94778,
    'Barra Mansa': 177861,
    'Belford Roxo': 469332,
    'Bom Jardim': 25333,
    'Bom Jesus do Itabapoana': 35411,
    'Cabo Frio': 186227,
    'Cachoeiras de Macacu': 54273,
    'Cambuci': 14827,
    'Campos dos Goytacazes': 463731,
    'Cantagalo': 19830,
    'Cardoso Moreira': 12600,
    'Carmo': 17434,
    'Casimiro de Abreu': 35347,
    'Comendador Levy Gasparian': 8180,
    'Conceição de Macabu': 21211,
    'Cordeiro': 20430,
    'Duas Barras': 10930,
    'Duque de Caxias': 855048,
    'Engenheiro Paulo de Frontin': 13237,
    'Guapimirim': 51483,
    'Iguaba Grande': 22851,
    'Itaboraí': 218008,
    'Itaguaí': 109091,
    'Italva': 14027,
    'Itaocara': 22899,
    'Itaperuna': 95841,
    'Itatiaia': 28783,
    'Japeri': 95492,
    'Laje do Muriaé': 7487,
    'Macaé': 206728,
    'Macuco': 5269,
    'Magé': 227322,
    'Mangaratiba': 36456,
    'Maricá': 127461,
    'Mendes': 17935,
    'Mesquita': 168376,
    'Miguel Pereira': 24642,
    'Miracema': 26843,
    'Natividade': 15082,
    'Nilópolis': 157425,
    'Niterói': 487562,
    'Nova Friburgo': 182082,
    'Nova Iguaçu': 796257,
    'Paracambi': 47124,
    'Paraíba do Sul': 41084,
    'Paraty': 37533,
    'Paty do Alferes': 26359,
    'Petrópolis': 295917,
    'Pinheiral': 22719,
    'Piraí': 26314,
    'Porciúncula': 17760,
    'Porto Real': 16592,
    'Quatis': 12793,
    'Queimados': 137962,
    'Quissamã': 20242,
    'Resende': 119769,
    'Rio Bonito': 55551,
    'Rio Claro': 17425,
    'Rio das Flores': 8561,
    'Rio das Ostras': 105676,
    'Rio de Janeiro': 6320446,
    'Santa Maria Madalena': 10321,
    'Santo Antônio de Pádua': 40589,
    'São Fidélis': 37543,
    'São Francisco de Itabapoana': 41354,
    'São Gonçalo': 999728,
    'São João da Barra': 32747,
    'São João de Meriti': 458673,
    'São José de Ubá': 7003,
    'São José do Vale do Rio Preto': 20251,
    'São Pedro da Aldeia': 87875,
    'São Sebastião do Alto': 8895,
    'Sapucaia': 17525,
    'Saquarema': 74234,
    'Seropédica': 78186,
    'Silva Jardim': 21349,
    'Sumidouro': 14900,
    'Tanguá': 30732,
    'Teresópolis': 163746,
    'Trajano de Moraes': 10289,
    'Três Rios': 77432,
    'Valença': 71843,
    'Varre-Sai': 9475,
    'Vassouras': 34410,
    'Volta Redonda': 257803,
}


def normalize_by_population(cases: np.ndarray, population: int, 
                           scale: int = 100000) -> np.ndarray:
    """
    Normaliza casos pela população (taxa de incidência).
    
    A taxa de incidência é calculada como:
    Taxa = (Casos / População) * 100.000
    
    Isso permite comparar municípios de diferentes tamanhos.
    
    Parameters
    ----------
    cases : np.ndarray
        Array com número de casos por período.
    population : int
        População do município (Censo 2010).
    scale : int
        Fator de escala (padrão: 100.000 habitantes).
    
    Returns
    -------
    np.ndarray
        Taxa de incidência por 100.000 habitantes.
    """
    if population <= 0:
        raise ValueError("População deve ser maior que zero")
    
    return (cases / population) * scale

def normalize_by_total_infected(series: np.ndarray) -> np.ndarray:
    """
    Normaliza série temporal pelo total de infectados.
    
    Após a normalização, a soma da série (área sob a curva) será 1.
    Isso permite comparar o formato das curvas epidêmicas,
    identificando municípios com dinâmicas sincronizadas.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal de casos.
    
    Returns
    -------
    np.ndarray
        Série normalizada com área unitária.
    """
    total = np.sum(series)
    if total == 0:
        return np.zeros_like(series)
    return series / total


def verify_unit_area(series: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Verifica se a série normalizada tem área unitária.
    
    Parameters
    ----------
    series : np.ndarray
        Série temporal normalizada.
    tolerance : float
        Tolerância para comparação numérica.
    
    Returns
    -------
    bool
        True se a área é aproximadamente 1.
    """
    area = np.sum(series)
    return np.abs(area - 1.0) < tolerance


def extract_solar_year_data(df: pd.DataFrame, year: int,
                           date_col: str = 'data',
                           cases_col: str = 'casos',
                           municipality_col: str = 'municipio') -> pd.DataFrame:
    """
    Extrai dados de um ano solar (1 de janeiro a 31 de dezembro).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados completos.
    year : int
        Ano solar desejado.
    date_col : str
        Nome da coluna de data.
    cases_col : str
        Nome da coluna de casos.
    municipality_col : str
        Nome da coluna de município.
    
    Returns
    -------
    pd.DataFrame
        Dados filtrados para o ano solar.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        mask = df[date_col].dt.year == year
        return df[mask]
    else:
        # Se não houver coluna de data, usar coluna de ano
        if 'ano' in df.columns:
            return df[df['ano'] == year]
        else:
            raise ValueError("Não foi possível identificar coluna de data ou ano")


def extract_epidemic_period_data(df: pd.DataFrame, start_week: int, end_week: int,
                                start_year: int = None, end_year: int = None,
                                week_col: str = 'semana_epi',
                                year_col: str = 'ano') -> pd.DataFrame:
    """
    Extrai dados do período epidêmico.
    
    O período epidêmico pode cruzar a virada do ano (ex: semana 40 a semana 20).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados completos.
    start_week : int
        Semana epidemiológica de início.
    end_week : int
        Semana epidemiológica de fim.
    start_year : int, optional
        Ano de início (se o período cruza anos).
    end_year : int, optional
        Ano de fim (se o período cruza anos).
    week_col : str
        Nome da coluna de semana epidemiológica.
    year_col : str
        Nome da coluna de ano.
    
    Returns
    -------
    pd.DataFrame
        Dados filtrados para o período epidêmico.
    """
    df = df.copy()
    
    if start_year is None or end_year is None:
        # Período dentro do mesmo ano
        mask = (df[week_col] >= start_week) & (df[week_col] <= end_week)
    else:
        if start_year == end_year:
            mask = ((df[year_col] == start_year) & 
                   (df[week_col] >= start_week) & 
                   (df[week_col] <= end_week))
        else:
            # Período cruza anos
            mask = (((df[year_col] == start_year) & (df[week_col] >= start_week)) |
                   ((df[year_col] == end_year) & (df[week_col] <= end_week)))
    
    return df[mask]


def calculate_incidence_rates(df: pd.DataFrame, population_data: Dict[str, int],
                             municipality_col: str = 'municipio',
                             cases_col: str = 'casos',
                             scale: int = 100000) -> pd.DataFrame:
    """
    Calcula taxa de incidência para todos os municípios.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados de casos.
    population_data : dict
        Dicionário {município: população}.
    municipality_col : str
        Nome da coluna de município.
    cases_col : str
        Nome da coluna de casos.
    scale : int
        Fator de escala (padrão: 100.000 habitantes).
    
    Returns
    -------
    pd.DataFrame
        DataFrame com taxas de incidência por município.
    """
    results = []
    
    for municipality in df[municipality_col].unique():
        mun_data = df[df[municipality_col] == municipality]
        total_cases = mun_data[cases_col].sum()
        
        if municipality in population_data:
            population = population_data[municipality]
            incidence_rate = (total_cases / population) * scale
        else:
            population = np.nan
            incidence_rate = np.nan
        
        results.append({
            'municipio': municipality,
            'casos_totais': total_cases,
            'populacao': population,
            'taxa_incidencia': incidence_rate
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('taxa_incidencia', ascending=False)
    
    return result_df


def normalize_all_series_by_population(time_series: Dict[str, np.ndarray],
                                       population_data: Dict[str, int],
                                       scale: int = 100000) -> Dict[str, np.ndarray]:
    """
    Normaliza todas as séries temporais pela população.
    
    Parameters
    ----------
    time_series : dict
        Dicionário {município: série_temporal}.
    population_data : dict
        Dicionário {município: população}.
    scale : int
        Fator de escala.
    
    Returns
    -------
    dict
        Dicionário {município: série_normalizada}.
    """
    normalized = {}
    
    for municipality, series in time_series.items():
        if municipality in population_data:
            population = population_data[municipality]
            normalized[municipality] = normalize_by_population(series, population, scale)
        else:
            print(f"Aviso: População não encontrada para {municipality}")
            normalized[municipality] = series
    
    return normalized


def normalize_all_series_by_total(time_series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normaliza todas as séries temporais pelo total de infectados.
    
    Parameters
    ----------
    time_series : dict
        Dicionário {município: série_temporal}.
    
    Returns
    -------
    dict
        Dicionário {município: série_normalizada} com área unitária.
    """
    normalized = {}
    
    for municipality, series in time_series.items():
        normalized[municipality] = normalize_by_total_infected(series)
        
        # Verificar área unitária
        if not verify_unit_area(normalized[municipality]):
            print(f"Aviso: {municipality} não tem área unitária após normalização")
    
    return normalized


def identify_most_affected_municipalities(incidence_df: pd.DataFrame, 
                                          top_n: int = 10) -> pd.DataFrame:
    """
    Identifica os municípios mais afetados pela dengue.
    
    Parameters
    ----------
    incidence_df : pd.DataFrame
        DataFrame com taxas de incidência (saída de calculate_incidence_rates).
    top_n : int
        Número de municípios a retornar.
    
    Returns
    -------
    pd.DataFrame
        Top N municípios mais afetados.
    """
    return incidence_df.head(top_n)


def plot_incidence_comparison(incidence_df: pd.DataFrame, top_n: int = 20,
                             output_path: str = None):
    """
    Plota gráfico de barras comparando taxas de incidência.
    
    Parameters
    ----------
    incidence_df : pd.DataFrame
        DataFrame com taxas de incidência.
    top_n : int
        Número de municípios a mostrar.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    top_municipalities = incidence_df.head(top_n)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(top_municipalities)), 
                   top_municipalities['taxa_incidencia'],
                   color='coral')
    
    plt.yticks(range(len(top_municipalities)), 
               top_municipalities['municipio'])
    plt.xlabel('Taxa de Incidência (por 100.000 hab.)', fontsize=12)
    plt.ylabel('Município', fontsize=12)
    plt.title(f'Top {top_n} Municípios Mais Afetados pela Dengue', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def plot_normalized_curves(normalized_series: Dict[str, np.ndarray],
                          municipalities: list = None,
                          title: str = "Curvas Epidêmicas Normalizadas",
                          output_path: str = None):
    """
    Plota curvas normalizadas para comparação visual de sincronização.
    
    Parameters
    ----------
    normalized_series : dict
        Dicionário {município: série_normalizada}.
    municipalities : list, optional
        Lista de municípios a plotar. Se None, plota todos.
    title : str
        Título do gráfico.
    output_path : str, optional
        Caminho para salvar o gráfico.
    """
    if municipalities is None:
        municipalities = list(normalized_series.keys())
    
    plt.figure(figsize=(14, 8))
    
    for municipality in municipalities:
        if municipality in normalized_series:
            series = normalized_series[municipality]
            weeks = np.arange(1, len(series) + 1)
            plt.plot(weeks, series, label=municipality, linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Semana Epidemiológica', fontsize=12)
    plt.ylabel('Proporção de Casos', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def print_normalization_summary(original_series: Dict[str, np.ndarray],
                               normalized_series: Dict[str, np.ndarray]):
    """
    Imprime resumo da normalização mostrando que séries têm área unitária.
    
    Parameters
    ----------
    original_series : dict
        Séries originais.
    normalized_series : dict
        Séries normalizadas.
    """
    print("=" * 60)
    print("VERIFICAÇÃO DE NORMALIZAÇÃO (ÁREA UNITÁRIA)")
    print("=" * 60)
    
    for municipality in normalized_series:
        original_sum = np.sum(original_series.get(municipality, [0]))
        normalized_sum = np.sum(normalized_series[municipality])
        is_unit = verify_unit_area(normalized_series[municipality])
        
        status = "✓" if is_unit else "✗"
        print(f"{status} {municipality}: Original = {original_sum:.0f} casos, "
              f"Normalizado = {normalized_sum:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 2: NORMALIZAÇÃO DE DADOS DE DENGUE - RJ")
    print("=" * 70)
    
    # Carregar dados reais
    filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    df = load_and_merge_dengue_data(filepath)
    
    # Análise para o ano de 2013 (ano com muitos casos)
    year = 2013
    print(f"\n--- Análise para o ano {year} ---")
    
    # Calcular total de casos por município
    year_data = df[df['ano'] == year]
    totals = year_data.groupby('municipio')['casos'].sum().reset_index()
    totals.columns = ['municipio', 'total_casos']
    
    # Adicionar população
    totals['populacao'] = totals['municipio'].map(POPULACAO_CENSO_2010)
    
    # Calcular taxa de incidência
    totals['taxa_incidencia'] = (totals['total_casos'] / totals['populacao']) * 100000
    totals = totals.dropna()  # Remover municípios sem dados de população
    totals = totals.sort_values('taxa_incidencia', ascending=False)
    
    print(f"\n--- NORMALIZAÇÃO 1: Taxa de Incidência (por 100.000 hab) ---")
    print(f"Top 15 Municípios Mais Afetados em {year}:")
    print(totals.head(15).to_string(index=False))
    
    # Extrair séries temporais para normalização
    time_series = get_yearly_time_series(df, year, normalize_to_52=True)
    
    # Filtrar apenas municípios com dados de população
    time_series_filtered = {mun: series for mun, series in time_series.items() 
                           if mun in POPULACAO_CENSO_2010}
    
    print(f"\n--- NORMALIZAÇÃO 2: Área Unitária ---")
    normalized_total = normalize_all_series_by_total(time_series_filtered)
    
    # Verificar área unitária
    print("\nVerificação de área unitária (primeiros 10 municípios):")
    for i, (mun, series) in enumerate(list(normalized_total.items())[:10]):
        area = np.sum(series)
        is_unit = verify_unit_area(series)
        status = "✓" if is_unit else "✗"
        print(f"  {status} {mun}: Área = {area:.10f}")
    
    # Plotar curvas normalizadas para os municípios mais afetados
    print(f"\n--- Plotando curvas normalizadas ---")
    top_municipalities = totals.head(10)['municipio'].tolist()
    
    # Filtrar séries normalizadas para top municípios
    top_normalized = {mun: normalized_total[mun] for mun in top_municipalities 
                      if mun in normalized_total}
    
    plot_normalized_curves(top_normalized, 
                          title=f"Curvas Normalizadas (Área Unitária) - Top 10 Municípios - {year}")
    
    print("\n" + "=" * 70)
    print("CONCLUSÕES:")
    print("=" * 70)
    print("""
1. A normalização por população permite identificar municípios
   proporcionalmente mais afetados, independente do tamanho.

2. A normalização por total de infectados (área unitária) permite
   comparar o FORMATO das curvas epidêmicas.

3. Municípios com curvas similares indicam sincronização na
   dinâmica da epidemia.

4. Para responder quais municípios são mais sincronizados,
   utilize a Tarefa 3 para calcular distâncias entre as curvas.
    """)
