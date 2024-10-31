
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gc


def filterdf(df, col1, val1, col2, val2):
    """
    Generate a filtered dataframe by two variables.
    """
    return df[(df[col1] == val1) & (df[col2] == val2)]



def update(original_df, filtered_df):
    """
    Update the original dataframe with the changes made in a filtered dataframe.
    """
    original_df.loc[filtered_df.index, :] = filtered_df
    return original_df



def binary_categorical_view(dataframe):
    categorical_columns = ['air_conditioner', 'chimney', 'garden', 'storage_room', 'swimming_pool', 'terrace']
    houses_vis = dataframe.copy()
    houses_vis[categorical_columns] = houses_vis[categorical_columns].replace({0: 'No', 1: 'Yes'})
    
    plt.figure(figsize=(16, 12))
    
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(3, 3, i)
        # Asignar hue y desactivar la leyenda
        ax = sns.countplot(x=houses_vis[column], hue=houses_vis[column], order=['No', 'Yes'], legend=False, palette="pastel")
        plt.title(f'{column.capitalize()}')
        plt.xlabel(column.capitalize())
        plt.ylabel('')
        plt.xticks(rotation=0)
        
        total = len(houses_vis[column])
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() - 0.05 * total
            ax.annotate(percentage, (x, y), ha='center', va='center', color='black', fontsize=12, weight='bold')
        
        max_height = max([p.get_height() for p in ax.patches], default=0)
        if max_height > 0:
            ax.set_ylim(0, max_height * 1.2)
    
    plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
    plt.show()

def categorical_features_view(dataframe):
    categorical_columns = ['room_num', 'bath_num', 'condition']
    houses_vis = dataframe.copy()
    
    plt.figure(figsize=(18, 24))
    
    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(4, 2, i)
        # Asignamos 'hue' y desactivamos la leyenda
        ax = sns.countplot(x=houses_vis[column], hue=houses_vis[column], legend=False, palette="pastel")
        plt.title(f'Datos de {column.capitalize()}')
        plt.xlabel(column.capitalize())
        plt.ylabel('')
        plt.xticks(rotation=0)
        
        total = len(houses_vis[column])
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y + 500), ha='center', va='bottom', color='black', fontsize=12, weight='bold')
        
        max_height = max([p.get_height() for p in ax.patches])
        ax.set_ylim(0, max_height * 1.15)
    
    plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
    plt.show()

# Boxplots

def boxplot_view(dataframe, column):
    plt.figure(figsize=(12, 2))
    plt.boxplot(dataframe[column], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=True)
    plt.title(f'Boxplot of {column.capitalize()} (With Outliers)')
    plt.xlabel(column.capitalize())
    plt.grid(True)
    plt.show()

def boxplot_view_wo(dataframe, column):  
    plt.figure(figsize=(12, 2))  
    plt.boxplot(dataframe[column], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=False)
    plt.title(f'Boxplot of {column.capitalize()} (Without Outliers)')
    plt.xlabel(column.capitalize())
    plt.grid(True)  # Mostrar cuadrícula
    plt.show()

def boxplot_with_mean(dataframe, group_col, target_col, show_outliers=True, figsize=(12, 8), log_transform=False):
    # Crear una copia del DataFrame para evitar SettingWithCopyWarning
    df_copy = dataframe.copy()

    if log_transform:
        # Aplicar transformación logarítmica a la variable objetivo
        df_copy[target_col] = np.log1p(df_copy[target_col])  # Usar log1p para manejar valores de 0
    
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_copy, x=group_col, y=target_col, showfliers=show_outliers)

    # Calcular y agregar la media a cada caja
    means = df_copy.groupby(group_col)[target_col].mean().values
    for i, mean in enumerate(means):
        ax.text(i, mean, f'{mean:.2f}', horizontalalignment='center', size='medium', color='black', weight='semibold')

    plt.title(f'Boxplot of {target_col} by {group_col} with Mean Values')
    plt.xlabel(group_col)
    plt.ylabel(target_col)
    plt.grid(True)
    plt.show()

def plot_histogram(df, column, bins=20, kde=True, figsize=(12, 6)):
    plt.figure(figsize=figsize)
    sns.histplot(df[column], bins=bins, kde=kde, color=sns.color_palette("dark")[2])
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')  
    plt.show()
