"""
Utility functions for data loading and processing.
Funções utilitárias para carregar e processar dados de dengue.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from epiweeks import Week, Year


def load_dengue_data(filepath: str = None) -> pd.DataFrame:
    """
    Carrega dados de dengue do arquivo Excel.
    
    Parameters
    ----------
    filepath : str, optional
        Caminho para o arquivo Excel. Se None, usa o caminho padrão.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com os dados de dengue.
    """
    if filepath is None:
        # Caminho padrão relativo ao diretório do projeto
        filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    
    df = pd.read_excel(filepath)
    return df


def get_epidemiological_weeks_per_year(start_year: int = 2010, end_year: int = 2016) -> dict:
    """
    Retorna o número de semanas epidemiológicas para cada ano.
    
    Parameters
    ----------
    start_year : int
        Ano inicial.
    end_year : int
        Ano final.
    
    Returns
    -------
    dict
        Dicionário com ano como chave e número de semanas como valor.
    """
    weeks_per_year = {}
    for year in range(start_year, end_year + 1):
        # Usando a biblioteca epiweeks para determinar o número de semanas
        epi_year = Year(year)
        weeks_per_year[year] = epi_year.totalweeks()
    return weeks_per_year


def get_years_with_53_weeks(start_year: int = 2010, end_year: int = 2016) -> list:
    """
    Retorna lista de anos que têm 53 semanas epidemiológicas.
    
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
    weeks_per_year = get_epidemiological_weeks_per_year(start_year, end_year)
    return [year for year, weeks in weeks_per_year.items() if weeks == 53]


def load_population_data(filepath: str = None) -> pd.DataFrame:
    """
    Carrega dados de população do Censo 2010.
    
    Parameters
    ----------
    filepath : str, optional
        Caminho para o arquivo de população. Se None, extrai do arquivo de dengue.
    
    Returns
    -------
    pd.DataFrame
        DataFrame com os dados de população por município.
    """
    # Os dados de população podem estar no próprio arquivo de dengue
    # ou em um arquivo separado. Ajustar conforme estrutura real dos dados.
    if filepath is None:
        filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    
    # Tentar carregar aba específica de população, se existir
    try:
        df = pd.read_excel(filepath, sheet_name="Populacao")
    except ValueError:
        # Se não existir aba de população, retornar None
        # A população pode estar em outra fonte
        df = None
    
    return df


def prepare_time_series(df: pd.DataFrame, municipality_col: str = 'municipio',
                        week_col: str = 'semana_epi', cases_col: str = 'casos') -> dict:
    """
    Prepara séries temporais por município.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com os dados de dengue.
    municipality_col : str
        Nome da coluna de município.
    week_col : str
        Nome da coluna de semana epidemiológica.
    cases_col : str
        Nome da coluna de casos.
    
    Returns
    -------
    dict
        Dicionário com município como chave e série temporal como valor.
    """
    time_series = {}
    for municipality in df[municipality_col].unique():
        mun_data = df[df[municipality_col] == municipality].copy()
        mun_data = mun_data.sort_values(week_col)
        time_series[municipality] = mun_data[cases_col].values
    
    return time_series
