#!/usr/bin/env python3
"""
Generate visualizations for the Dengue analysis presentation.
Creates heatmaps from distance matrices and other plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure matplotlib for better output
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'

def load_distance_matrix(filepath):
    """Load distance matrix from CSV."""
    df = pd.read_csv(filepath, index_col=0)
    return df

def create_heatmap(df, title, output_path, cmap='YlOrRd'):
    """Create a heatmap visualization of a distance matrix."""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Create heatmap with seaborn
    sns.heatmap(
        df, 
        cmap=cmap,
        center=None,
        square=True,
        linewidths=0,
        cbar_kws={'label': 'Distância', 'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Município', fontsize=12)
    ax.set_ylabel('Município', fontsize=12)
    
    # Rotate labels for readability
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_histogram(df, title, output_path, color='steelblue'):
    """Create histogram of distance distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get upper triangle values (excluding diagonal)
    values = df.values[np.triu_indices(len(df), k=1)]
    
    ax.hist(values, bins=50, color=color, edgecolor='white', alpha=0.8)
    ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {values.mean():.3f}')
    ax.axvline(np.median(values), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {np.median(values):.3f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Distância', fontsize=12)
    ax.set_ylabel('Frequência', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_comparison_boxplot(df_l1, df_l2, output_path):
    """Create boxplot comparing L1 and L2 distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    values_l1 = df_l1.values[np.triu_indices(len(df_l1), k=1)]
    values_l2 = df_l2.values[np.triu_indices(len(df_l2), k=1)]
    
    bp = ax.boxplot(
        [values_l1, values_l2], 
        labels=['L1 (Manhattan)', 'L2 (Euclidiana)'],
        patch_artist=True
    )
    
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Comparação das Distribuições de Distância (2013)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Distância', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

def find_top_pairs(df, n=10, ascending=True):
    """Find top n most/least similar pairs."""
    # Get upper triangle
    pairs = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            pairs.append({
                'municipio1': df.index[i],
                'municipio2': df.columns[j],
                'distancia': df.iloc[i, j]
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('distancia', ascending=ascending)
    return pairs_df.head(n)

def create_top_pairs_bar(df, title, output_path, ascending=True):
    """Create bar chart of top pairs."""
    top_pairs = find_top_pairs(df, n=15, ascending=ascending)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [f"{row['municipio1'][:15]}\n↔\n{row['municipio2'][:15]}" 
              for _, row in top_pairs.iterrows()]
    values = top_pairs['distancia'].values
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values))) if not ascending else plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
    
    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Distância', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

def create_stats_summary(df_l1, df_l2, output_path):
    """Create visual summary of statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (df, name, color) in zip(axes, [(df_l1, 'L1 (Manhattan)', '#3498db'), (df_l2, 'L2 (Euclidiana)', '#e74c3c')]):
        values = df.values[np.triu_indices(len(df), k=1)]
        
        stats = {
            'Mínimo': values.min(),
            'Máximo': values.max(),
            'Média': values.mean(),
            'Mediana': np.median(values),
            'Desvio Padrão': values.std()
        }
        
        bars = ax.bar(stats.keys(), stats.values(), color=color, edgecolor='white', alpha=0.8)
        ax.set_title(f'Estatísticas - {name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {output_path}")

def main():
    print("=" * 60)
    print("Gerando visualizações para a apresentação de Dengue")
    print("=" * 60)
    
    # Load distance matrices
    print("\n📊 Carregando matrizes de distância...")
    df_l1 = load_distance_matrix(OUTPUT_DIR / "matriz_distancia_L1.csv")
    df_l2 = load_distance_matrix(OUTPUT_DIR / "matriz_distancia_L2.csv")
    print(f"   Dimensão: {df_l1.shape[0]} x {df_l1.shape[1]} municípios")
    
    # Generate heatmaps
    print("\n🔥 Gerando heatmaps...")
    create_heatmap(df_l1, "Matriz de Distância L1 (Manhattan) - 2013", 
                   OUTPUT_DIR / "heatmap_L1.png", cmap='Blues')
    create_heatmap(df_l2, "Matriz de Distância L2 (Euclidiana) - 2013", 
                   OUTPUT_DIR / "heatmap_L2.png", cmap='Reds')
    
    # Generate histograms
    print("\n📈 Gerando histogramas...")
    create_histogram(df_l1, "Distribuição das Distâncias L1 (Manhattan) - 2013", 
                    OUTPUT_DIR / "hist_L1.png", color='#3498db')
    create_histogram(df_l2, "Distribuição das Distâncias L2 (Euclidiana) - 2013", 
                    OUTPUT_DIR / "hist_L2.png", color='#e74c3c')
    
    # Generate comparison
    print("\n📊 Gerando comparações...")
    create_comparison_boxplot(df_l1, df_l2, OUTPUT_DIR / "boxplot_comparison.png")
    create_stats_summary(df_l1, df_l2, OUTPUT_DIR / "stats_summary.png")
    
    # Generate top pairs charts
    print("\n🏆 Gerando gráficos de top pares...")
    create_top_pairs_bar(df_l1, "Top 15 Pares Mais Sincronizados (L1)", 
                        OUTPUT_DIR / "top_pairs_L1.png", ascending=True)
    create_top_pairs_bar(df_l2, "Top 15 Pares Mais Sincronizados (L2)", 
                        OUTPUT_DIR / "top_pairs_L2.png", ascending=True)
    
    print("\n" + "=" * 60)
    print("✅ Todas as visualizações foram geradas com sucesso!")
    print("=" * 60)
    
    # List generated files
    print("\nArquivos gerados:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  📄 {f.name}")

if __name__ == "__main__":
    main()
