"""
Tarefa 0: Carregamento e Unificação dos Dados

Este módulo lida com o arquivo Excel mal formatado que contém:
- 7 abas (uma para cada ano de 2010-2016)
- Primeira linha contém dados de "Angra dos Reis" como nomes de colunas
- Cada aba tem: código IBGE, nome município, casos por semana epidemiológica, total

O objetivo é unificar todos os dados em um único DataFrame organizado.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suprimir avisos de extensão desconhecida do openpyxl
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def load_and_merge_dengue_data(filepath: str = None) -> pd.DataFrame:
    """
    Carrega e unifica os dados de dengue de todas as abas do Excel.
    
    O arquivo Excel tem uma formatação problemática onde:
    - A primeira linha de dados (Angra dos Reis) está sendo usada como cabeçalho
    - Os nomes das colunas são números representando casos
    - Precisamos reconstruir a estrutura correta
    
    Parameters
    ----------
    filepath : str, optional
        Caminho para o arquivo Excel. Se None, usa o caminho padrão.
    
    Returns
    -------
    pd.DataFrame
        DataFrame unificado com colunas:
        - codigo_ibge: Código IBGE do município
        - municipio: Nome do município
        - ano: Ano
        - semana_epi: Semana epidemiológica (1-53)
        - casos: Número de casos de dengue
    """
    if filepath is None:
        # Tenta ../data/ para scripts, ../../data/ para notebooks, senão raiz do projeto
        data_path = Path(__file__).parent.parent / "data" / "Dengue_Brasil_2010-2016_RJ.xlsx"
        if data_path.exists():
            filepath = data_path
        else:
            alt_path = Path(__file__).parent.parent.parent / "data" / "Dengue_Brasil_2010-2016_RJ.xlsx"
            if alt_path.exists():
                filepath = alt_path
            else:
                filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    
    xlsx = pd.ExcelFile(filepath)
    all_data = []
    
    for sheet_name in xlsx.sheet_names:
        year = int(sheet_name)
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        
        # A estrutura real é:
        # Coluna 0: Código IBGE
        # Coluna 1: Nome do município
        # Colunas 2 até n-1: Casos por semana epidemiológica
        # Última coluna: Total (vamos ignorar, podemos recalcular)
        
        # Determinar número de semanas epidemiológicas
        # (total de colunas - 3: código, nome, total)
        n_cols = df.shape[1]
        
        # Para 2016, há colunas extras vazias que precisamos ignorar
        # Identificar colunas com dados reais (não NaN)
        if year == 2016:
            # 2016 tem apenas 32 semanas de dados
            # Precisamos identificar onde os dados terminam
            last_data_col = 2
            for col in range(2, n_cols - 1):
                if df[col].isna().all():
                    break
                last_data_col = col
            n_weeks = last_data_col - 2 + 1
        else:
            # Anos completos: colunas 2 até n-2 (excluindo total)
            n_weeks = n_cols - 3
        
        print(f"Ano {year}: {n_weeks} semanas epidemiológicas detectadas")
        
        # Processar cada linha (município)
        for idx in range(len(df)):
            codigo_ibge = df.iloc[idx, 0]
            municipio = df.iloc[idx, 1]
            
            # Extrair casos por semana
            for week in range(1, n_weeks + 1):
                col_idx = week + 1  # +2 porque col 0=código, col 1=nome, então semana 1 está na col 2
                casos = df.iloc[idx, col_idx]
                
                # Converter para int, tratando NaN como 0
                if pd.isna(casos):
                    casos = 0
                else:
                    casos = int(casos)
                
                all_data.append({
                    'codigo_ibge': codigo_ibge,
                    'municipio': municipio,
                    'ano': year,
                    'semana_epi': week,
                    'casos': casos
                })
    
    result_df = pd.DataFrame(all_data)
    
    # Ordenar por município, ano e semana
    result_df = result_df.sort_values(['municipio', 'ano', 'semana_epi']).reset_index(drop=True)
    
    return result_df


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas resumidas por município e ano.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    
    Returns
    -------
    pd.DataFrame
        Estatísticas por município e ano.
    """
    summary = df.groupby(['municipio', 'ano']).agg({
        'casos': ['sum', 'mean', 'max', 'std'],
        'semana_epi': 'count'
    }).reset_index()
    
    # Achatar nomes das colunas
    summary.columns = ['municipio', 'ano', 'total_casos', 'media_semanal', 
                       'pico_semanal', 'desvio_padrao', 'n_semanas']
    
    return summary


def get_municipalities_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna lista de municípios com seus códigos IBGE.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    
    Returns
    -------
    pd.DataFrame
        Lista de municípios únicos com códigos.
    """
    municipalities = df[['codigo_ibge', 'municipio']].drop_duplicates()
    municipalities = municipalities.sort_values('municipio').reset_index(drop=True)
    return municipalities


def get_time_series_by_municipality(df: pd.DataFrame, normalize_weeks: bool = True) -> dict:
    """
    Extrai séries temporais por município.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    normalize_weeks : bool
        Se True, padroniza todas as séries para 52 semanas por ano.
    
    Returns
    -------
    dict
        Dicionário {município: array de casos por semana}.
    """
    time_series = {}
    
    for municipio in df['municipio'].unique():
        mun_data = df[df['municipio'] == municipio].copy()
        
        # Ordenar por ano e semana
        mun_data = mun_data.sort_values(['ano', 'semana_epi'])
        
        if normalize_weeks:
            # Padronizar para 52 semanas por ano
            series = []
            for year in sorted(mun_data['ano'].unique()):
                year_data = mun_data[mun_data['ano'] == year]
                cases = year_data['casos'].values
                
                if len(cases) > 52:
                    # Ano com 53 semanas: combinar semana 53 com semana 52
                    cases_52 = cases[:52].copy()
                    cases_52[51] += cases[52]
                    series.extend(cases_52)
                elif len(cases) < 52:
                    # Ano incompleto (ex: 2016): preencher com zeros
                    cases_padded = np.zeros(52)
                    cases_padded[:len(cases)] = cases
                    series.extend(cases_padded)
                else:
                    series.extend(cases)
            
            time_series[municipio] = np.array(series)
        else:
            time_series[municipio] = mun_data['casos'].values
    
    return time_series


def get_yearly_time_series(df: pd.DataFrame, year: int, normalize_to_52: bool = True) -> dict:
    """
    Extrai séries temporais por município para um ano específico.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    year : int
        Ano desejado.
    normalize_to_52 : bool
        Se True, normaliza para 52 semanas.
    
    Returns
    -------
    dict
        Dicionário {município: array de casos por semana}.
    """
    year_data = df[df['ano'] == year]
    time_series = {}
    
    for municipio in year_data['municipio'].unique():
        mun_data = year_data[year_data['municipio'] == municipio].copy()
        mun_data = mun_data.sort_values('semana_epi')
        cases = mun_data['casos'].values
        
        if normalize_to_52 and len(cases) > 52:
            # Combinar semana 53 com semana 52
            cases_52 = cases[:52].copy()
            cases_52[51] += cases[52]
            cases = cases_52
        elif normalize_to_52 and len(cases) < 52:
            # Preencher com zeros
            cases_padded = np.zeros(52)
            cases_padded[:len(cases)] = cases
            cases = cases_padded
        
        time_series[municipio] = cases
    
    return time_series


def save_unified_dataset(df: pd.DataFrame, output_path: str = None):
    """
    Salva o dataset unificado em formato CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    output_path : str, optional
        Caminho para salvar o arquivo. Se None, usa caminho padrão.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "dengue_unificado.csv"
    
    # Criar diretório se não existir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset salvo em: {output_path}")


def analyze_weeks_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analisa quantas semanas epidemiológicas cada ano tem nos dados.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame unificado de casos de dengue.
    
    Returns
    -------
    pd.DataFrame
        Número de semanas por ano.
    """
    weeks_per_year = df.groupby('ano')['semana_epi'].max().reset_index()
    weeks_per_year.columns = ['ano', 'max_semana']
    
    # Adicionar informação se tem 52 ou 53 semanas
    weeks_per_year['tem_53_semanas'] = weeks_per_year['max_semana'] == 53
    
    return weeks_per_year


if __name__ == "__main__":
    print("=" * 70)
    print("TAREFA 0: Carregamento e Unificação dos Dados de Dengue")
    print("=" * 70)
    
    # Carregar dados
    print("\n--- Carregando dados do arquivo Excel ---")
    filepath = Path(__file__).parent.parent / "Dengue_Brasil_2010-2016_RJ.xlsx"
    df = load_and_merge_dengue_data(filepath)
    
    print(f"\n--- Resumo do Dataset Unificado ---")
    print(f"Total de registros: {len(df):,}")
    print(f"Municípios: {df['municipio'].nunique()}")
    print(f"Anos: {df['ano'].min()} - {df['ano'].max()}")
    print(f"Colunas: {list(df.columns)}")
    
    print(f"\n--- Primeiras linhas ---")
    print(df.head(10).to_string())
    
    print(f"\n--- Análise de Semanas por Ano ---")
    weeks_analysis = analyze_weeks_per_year(df)
    print(weeks_analysis.to_string(index=False))
    
    print(f"\n--- Lista de Municípios ---")
    municipalities = get_municipalities_list(df)
    print(f"Total de municípios: {len(municipalities)}")
    print(municipalities.head(20).to_string(index=False))
    
    print(f"\n--- Estatísticas por Município e Ano ---")
    summary = get_summary_statistics(df)
    print(summary.head(20).to_string(index=False))
    
    # Salvar dataset unificado
    print(f"\n--- Salvando dataset unificado ---")
    save_unified_dataset(df)
    
    print("\n" + "=" * 70)
    print("DADOS CARREGADOS E UNIFICADOS COM SUCESSO!")
    print("=" * 70)
