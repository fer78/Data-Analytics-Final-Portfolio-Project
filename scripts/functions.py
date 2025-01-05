
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import folium

from IPython.display import display


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


import matplotlib.pyplot as plt
import seaborn as sns


def binary_categorical_view(dataframe):
    categorical_columns = ['air_conditioner', 'chimney', 'garden', 'storage_room', 'swimming_pool', 'terrace']
    houses_vis = dataframe.copy()
    houses_vis[categorical_columns] = houses_vis[categorical_columns].replace({0: 'No', 1: 'Yes'})

    plt.figure(figsize=(16, 12))

    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(3, 3, i)

        ax = sns.countplot(x=houses_vis[column], order=['No', 'Yes'], palette="pastel")
        plt.title(f'{column.capitalize()}')
        plt.xlabel(column.capitalize())
        plt.ylabel('')
        plt.xticks(rotation=0)

        total = len(houses_vis[column])
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height() + 500
            ax.annotate(percentage, (x, y), ha='center', va='bottom', color='black', fontsize=12, weight='bold')

        max_height = max([p.get_height() for p in ax.patches], default=0)
        if max_height > 0:
            ax.set_ylim(0, max_height * 1.2)

    plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
    plt.show()


def categorical_features_view(dataframe):
    categorical_columns = ['room_num', 'bath_num', 'condition']
    houses_vis = dataframe.copy()

    plt.figure(figsize=(18, 12))

    for i, column in enumerate(categorical_columns, 1):
        plt.subplot(2, 2, i)

        ax = sns.countplot(x=houses_vis[column], palette="pastel", order=sorted(houses_vis[column].unique()))
        plt.title(f'Datos de {column.capitalize()}')
        plt.xlabel(column.capitalize())
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=0)

        total = len(houses_vis[column])
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y + 500), ha='center', va='bottom', color='black', fontsize=10, weight='bold')

        max_height = max([p.get_height() for p in ax.patches])
        ax.set_ylim(0, max_height * 1.2)

    plt.tight_layout(pad=4.0, w_pad=4.0, h_pad=4.0)
    plt.show()


def boxplot_view(dataframe, column):
    plt.figure(figsize=(10, 2))
    plt.boxplot(dataframe[column], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=True)
    plt.title(f'Boxplot of {column.capitalize()} (With Outliers)')
    plt.xlabel(column.capitalize())
    plt.show()

def boxplot_view_wo(dataframe, column):  
    plt.figure(figsize=(10, 2))
    plt.boxplot(dataframe[column], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), showfliers=False)
    plt.title(f'Boxplot of {column.capitalize()} (Without Outliers)')
    plt.xlabel(column.capitalize())
    plt.show()


def bivariate_distribution(dataframe, group_col, target_col, show_outliers=True, show_summary=True, figsize=(10, 6)):
    """
    Displays a boxplot and a summary table of a variable grouped by another variable's values.
    """
    df_copy = dataframe.copy()

    # Boxplot
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_copy, x=group_col, y=target_col, showfliers=show_outliers, palette="pastel")
    plt.title(f'Boxplot of {target_col} by {group_col} with Mean Values')
    plt.xlabel(group_col)
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.show()

    # Summary Statistics table
    if show_summary:
        summary_stats = df_copy.groupby(group_col)[target_col].agg(
            mean=lambda x: round(x.mean(), 2),
            Q1=lambda x: x.quantile(0.25),
            Q3=lambda x: x.quantile(0.75),
            std_dev=lambda x: round(x.std(), 2)
        ).reset_index()
        summary_stats = summary_stats.sort_values(by='mean', ascending=False).reset_index(drop=True)
        print(f"Summary Table of {target_col} by {group_col} sorted by Mean:")
        print(summary_stats)


def plot_rooms_bathrooms_distribution(df):
    """
    Displays the distribution of rooms and bathrooms
    """
    features = ['room_num', 'bath_num']
    plt.figure(figsize=(10, 6))

    for i, feature in enumerate(features, 1):
        plt.subplot(1, 2, i)
        sns.countplot(data=df, x=feature, palette="pastel")
        plt.title(f'{feature.capitalize()} Distribution')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_category_histograms(df, numeric_col, category_col, bins=20, figsize=(18, 6), color_palette="pastel"):
    categories = df[category_col].unique()
    colors = sns.color_palette(color_palette, len(categories))

    plt.figure(figsize=figsize)
    for i, (category, color) in enumerate(zip(categories, colors), 1):
        plt.subplot(1, len(categories), i)
        sns.histplot(df[df[category_col] == category][numeric_col], bins=bins, kde=True, color=color)
        plt.title(f'Distribution of {numeric_col} for {category} Category')
        plt.xlabel(numeric_col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_histogram(df, column, bins=20, kde=True, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    sns.histplot(df[column], bins=bins, kde=kde, color=sns.color_palette("dark")[2])
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')  
    plt.show()


def plot_binary_categorical_relationships(dataframe, target_variable, figsize=(18, 12)):
    """
    Create boxplot charts to display the relationship between a numerical variable and predefined binary variables.
    """
    binary_variables = ['air_conditioner', 'chimney', 'garden', 'storage_room', 'swimming_pool', 'terrace']
    df_copy = dataframe.copy()
    df_copy[binary_variables] = df_copy[binary_variables].replace({0: 'No', 1: 'Yes'})
    plt.figure(figsize=figsize)

    n_cols = 3
    n_rows = (len(binary_variables) + n_cols - 1) // n_cols

    for i, binary_var in enumerate(binary_variables, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(data=df_copy, x=binary_var, y=target_variable, palette="pastel")
        plt.title(f'{target_variable.capitalize()} Distribution by {binary_var.capitalize()} Availability')
        plt.xlabel(binary_var.capitalize())
        plt.ylabel(target_variable.capitalize())

    plt.tight_layout()
    plt.show()

    # Summary table
    summary_table = {}
    for binary_var in binary_variables:
        means = df_copy.groupby(binary_var)[target_variable].mean()
        summary_table[binary_var.capitalize()] = [round(means.get('Yes', 0), 2), round(means.get('No', 0), 2)]

    summary_df = pd.DataFrame(summary_table, index=['Yes', 'No'])

    print(f"\nSummary Table of Mean {target_variable.capitalize()} by Binary Categorical Variables")
    print(summary_df)


def correlation_heatmap_by_size_category(df):
    """
    heat map of correlation by size category
    """
    size_categories = df['size_category'].unique()
    rows = (len(size_categories) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))

    # Asegurar que `axes` sea siempre una lista para un solo heatmap
    if rows == 1:
        axes = [axes] if isinstance(axes, plt.Axes) else axes.flatten()
    else:
        axes = axes.flatten()

    for i, size_category in enumerate(size_categories):
        filtered_df = df[df['size_category'] == size_category]
        corr = filtered_df[['price', 'm2_real', 'room_num', 'bath_num']].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[i])
        axes[i].set_title(f'Correlation Matrix for {size_category} Apartments')

    # delete empty subplots if necessary
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def calculate_price_per_m2(df, price_col, m2_col, group_col):
    """
    Calculate price per square meters based on the media.
    """
    summary_table = df.groupby(group_col).agg(
        mean_price=(price_col, 'mean'),
        mean_m2=(m2_col, 'mean')
    ).reset_index()

    summary_table['price_m2_ratio'] = (summary_table['mean_price'] / summary_table['mean_m2']).round(2)
    summary_table = summary_table.sort_values(by='price_m2_ratio').reset_index(drop=True)

    return summary_table

