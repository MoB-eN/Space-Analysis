# Django imports
from django.shortcuts import render
from django.template.defaulttags import register

# Data analysis and visualization imports
import pandas as pd
import numpy as np
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import base64

# Custom template filter to access dictionary items in Django templates
@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

def set_visualization_style():
    """
    Set global matplotlib style configurations for consistent dark-themed visualizations.
    Configures colors, fonts, grid lines, and other visual elements for all plots.
    """
    plt.style.use('dark_background')
    
    # Define custom color scheme for visualizations
    colors = {
        'background': '#0C0F1D',  # Dark background
        'figure_bg': '#161B2E',   # Slightly lighter background for figures
        'primary': '#4F46E5',     # Primary color for main elements
        'secondary': '#818CF8',   # Secondary color for supporting elements
        'accent1': '#10B981',     # Accent colors for additional elements
        'accent2': '#3B82F6',
        'accent3': '#F59E0B',
        'text': '#E5E7EB',        # Text color
        'grid': '#374151'         # Grid line color
    }
    
    # Apply the custom style configurations
    plt.rcParams.update({
        'figure.facecolor': colors['figure_bg'],
        'axes.facecolor': colors['figure_bg'],
        'axes.edgecolor': colors['grid'],
        'axes.labelcolor': colors['text'],
        'axes.grid': True,
        'grid.color': colors['grid'],
        'grid.alpha': 0.2,
        'text.color': colors['text'],
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'figure.dpi': 100,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': colors['figure_bg'],
        'legend.edgecolor': colors['grid'],
        'legend.fontsize': 10,
        'legend.framealpha': 0.8
    })
    return colors

def analyze_column(df, column_name):
    """
    Perform comprehensive analysis of a single column in the dataset.
    
    This method provides both statistical analysis and visualizations for a given column.
    For numeric columns, it calculates descriptive statistics and creates various plots
    (histogram, box plot, violin plot, Q-Q plot, time series, and ECDF).
    For categorical columns, it provides value counts and creates bar and pie charts.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        column_name (str): Name of the column to analyze
    
    Returns:
        dict: Analysis results including statistics and visualizations
    """
    set_visualization_style()
    analysis = {}
    
    # Validate column existence
    if column_name not in df.columns:
        return {'error': f"Column '{column_name}' not found in the dataset."}

    series = df[column_name]
    
    # Calculate basic statistics
    analysis['type'] = str(series.dtype)
    analysis['missing'] = series.isnull().sum()
    analysis['missing_percent'] = round(analysis['missing'] / len(series) * 100, 2)
    analysis['unique'] = series.nunique()
    analysis['unique_percent'] = round(analysis['unique'] / len(series) * 100, 2)

    # Remove missing values for analysis
    series = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        # Calculate detailed statistics for numeric columns
        analysis.update({
            'mean': series.mean(),
            'median': series.median(),
            'mode': series.mode().iloc[0] if not series.mode().empty else None,
            'std': series.std(),
            'var': series.var(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'min': series.min(),
            'max': series.max(),
            'range': series.max() - series.min(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75),
            'iqr': series.quantile(0.75) - series.quantile(0.25)
        })

        # Create multiple visualizations for numeric data
        fig, axes = plt.subplots(3, 2, figsize=(13, 15))
        
        # Distribution plots
        sns.histplot(series, kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Histogram with KDE")
        
        sns.boxplot(x=series, ax=axes[0, 1])
        axes[0, 1].set_title("Box Plot")
        
        sns.violinplot(x=series, ax=axes[1, 0])
        axes[1, 0].set_title("Violin Plot")
        
        # Statistical plots
        stats.probplot(series, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot")
        
        sns.lineplot(x=range(len(series)), y=series.values, ax=axes[2, 0])
        axes[2, 0].set_title("Time Series Plot")
        
        sns.ecdfplot(series, ax=axes[2, 1])
        axes[2, 1].set_title("Empirical CDF")
        
        plt.tight_layout()
        
    else:
        # Analysis for categorical columns
        analysis.update({
            'top_values': series.value_counts().head(10).to_dict(),
            'top_percent': (series.value_counts(normalize=True) * 100).head(10).to_dict()
        })

        # Create visualizations for categorical data
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Bar plot of top categories
        sns.countplot(y=series, order=series.value_counts().index[:10], ax=axes[0])
        axes[0].set_title("Top 10 Categories")
        
        # Pie chart of category distribution
        series.value_counts().head(10).plot.pie(ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title("Category Distribution")
        
        plt.tight_layout()

    # Save visualizations as SVG
    buffer = BytesIO()
    fig.savefig(buffer, format='svg', bbox_inches='tight')
    buffer.seek(0)
    analysis['visualizations'] = buffer.getvalue().decode('utf-8')
    buffer.close()
    plt.close(fig)

    return analysis

def generate_xy_visualization(data, x_column, y_column):
    """
    Generate comprehensive bivariate analysis between two columns.
    
    This method creates various visualizations and statistical analyses for the relationship
    between two variables. For numeric pairs, it includes correlation analysis and various
    plots. For categorical pairs, it performs chi-square analysis and creates appropriate
    visualizations.
    
    Args:
        data (pandas.DataFrame): Input dataframe
        x_column (str): Name of the first column
        y_column (str): Name of the second column
    
    Returns:
        tuple: (plot_data, analysis_results)
    """
    analysis = {}
    
    # Validate column existence
    if x_column not in data.columns or y_column not in data.columns:
        return "Error: One or both columns not found in dataset.", {}

    # Clean data by removing missing values
    clean_data = data[[x_column, y_column]].dropna()
    
    # Check if both columns are numeric
    x_is_numeric = pd.api.types.is_numeric_dtype(clean_data[x_column])
    y_is_numeric = pd.api.types.is_numeric_dtype(clean_data[y_column])

    fig = plt.figure(figsize=(12, 20))
    gs = fig.add_gridspec(2, 2)

    if x_is_numeric and y_is_numeric:
        # Analysis for numeric columns
        pearson_corr, p_value = stats.pearsonr(clean_data[x_column], clean_data[y_column])
        spearman_corr, spearman_p = stats.spearmanr(clean_data[x_column], clean_data[y_column])
        
        analysis.update({
            'pearson_correlation': pearson_corr,
            'pearson_p_value': p_value,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p
        })

        # Create various plots for numeric data
        ax1 = fig.add_subplot(gs[0, 0])
        sns.regplot(data=clean_data, x=x_column, y=y_column, ax=ax1, color='#4F46E5')
        ax1.set_title(f'Scatter Plot with Regression Line: {x_column} vs {y_column}')

        ax2 = fig.add_subplot(gs[0, 1])
        plt.hexbin(clean_data[x_column], clean_data[y_column], gridsize=5, cmap='viridis')
        ax2.set_title(f'Hexbin Plot: {x_column} vs {y_column}')

        ax3 = fig.add_subplot(gs[1, 0])
        sns.kdeplot(data=clean_data, x=x_column, y=y_column, ax=ax3, cmap='viridis')
        ax3.set_title(f'2D KDE Plot: {x_column} vs {y_column}')

        # Create residual plot
        ax4 = fig.add_subplot(gs[1, 1])
        z = np.polyfit(clean_data[x_column], clean_data[y_column], 1)
        p = np.poly1d(z)
        residuals = clean_data[y_column] - p(clean_data[x_column])
        sns.scatterplot(x=clean_data[x_column], y=residuals, ax=ax4, color='#E53E3E')
        ax4.axhline(y=0, color='k', linestyle='--')
        ax4.set_title('Residual Plot')

    else:
        # Analysis for categorical columns
        contingency = pd.crosstab(clean_data[x_column], clean_data[y_column])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        analysis.update({
            'chi2_statistic': chi2,
            'chi2_p_value': p_value,
            'degrees_of_freedom': dof
        })

        # Create visualizations for categorical data
        ax1 = fig.add_subplot(gs[0, :])
        sns.heatmap(contingency, annot=True, fmt='d', cmap='viridis', ax=ax1)
        ax1.set_title(f'Contingency Table Heatmap: {x_column} vs {y_column}')

        ax2 = fig.add_subplot(gs[1, :])
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0)
        contingency_pct.plot(kind='bar', stacked=True, ax=ax2)
        ax2.set_title(f'Stacked Bar Plot: {x_column} vs {y_column}')
        ax2.legend(title=y_column, bbox_to_anchor=(1.05, 1))

    # Save and return visualizations
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    plot_data = buf.getvalue().decode('utf-8')
    buf.close()
    plt.close(fig)

    return plot_data, analysis

def upload_file(request):
    """
    Handle file upload and data analysis requests.
    
    This Django view function handles three main operations:
    1. File upload: Processes CSV files and stores them in the session
    2. Single column analysis: Analyzes individual columns when requested
    3. Two-column analysis: Performs bivariate analysis between two columns
    
    Args:
        request (HttpRequest): Django request object
    
    Returns:
        HttpResponse: Rendered template with analysis results
    """
    context = {}

    try:
        if request.method == 'POST':
            # Handle file upload
            if 'file' in request.FILES:
                file = request.FILES['file']
                if file.name.endswith('.csv'):
                    # Read and store CSV file
                    df = pd.read_csv(file)
                    request.session['df'] = df.to_json()
                    context.update({
                        'data_preview': df.head().values.tolist(),
                        'columns': df.columns.tolist(),
                        'basic_info': {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'memory': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                            'total_missing': df.isnull().sum().sum()
                        }
                    })
                else:
                    context['error'] = "Please upload a CSV file."

            # Handle single column analysis request
            elif 'analyze_column' in request.POST:
                column_name = request.POST.get('analyze_column')
                if column_name and 'df' in request.session:
                    df = pd.read_json(request.session['df'])
                    column_analysis = analyze_column(df, column_name)
                    if 'error' in column_analysis:
                        context['error'] = column_analysis['error']
                    else:
                        context.update({
                            'column_analysis': column_analysis,
                            'selected_column': column_name,
                            'columns': df.columns.tolist()
                        })
                else:
                    context['error'] = "Please select a column to analyze."

            # Handle two-column analysis request
            elif 'analyze_xy' in request.POST:
                x_column = request.POST.get('x_column')
                y_column = request.POST.get('y_column')

                if x_column and y_column and 'df' in request.session:
                    df = pd.read_json(request.session['df'])
                    main_plot, heatmap = generate_xy_visualization(df, x_column, y_column)
                    context.update({
                        'xy_visualization': main_plot,
                        'selected_x': x_column,
                        'selected_y': y_column,
                        'columns': df.columns.tolist()
                    })
                else:
                    context['error'] = "Please select both X and Y columns for analysis."

        return render(request, 'core/upload.html', context)
    except Exception as e:
        context['error'] = str(e)
        return render(request, 'core/upload.html', context)