import pandas as pd
import numpy as np

def analyze_dataset(df):
    """Analyze the uploaded dataset"""
    analysis = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return analysis

def generate_summary_stats(df, column):
    """Generate summary statistics for a specific column"""
    if pd.api.types.is_numeric_dtype(df[column]):
        return {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max()
        }
    else:
        return {
            'unique_values': df[column].nunique(),
            'most_common': df[column].mode()[0],
            'value_counts': df[column].value_counts().head().to_dict()
        }