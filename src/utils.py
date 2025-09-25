"""
Utility functions for the data cleaning application.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import os


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def format_number(number: int) -> str:
    """
    Format large numbers with thousand separators.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{number:,}"


def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is supported.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file type is supported
    """
    if not filename:
        return False
    
    supported_extensions = ['.csv', '.xlsx', '.xls']
    _, ext = os.path.splitext(filename.lower())
    return ext in supported_extensions


def create_sample_data() -> pd.DataFrame:
    """
    Create a sample dataset for demonstration purposes.
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    n_rows = 1000
    
    # Create sample data with various data types and quality issues
    data = {
        'id': range(1, n_rows + 1),
        'name': [f'Person_{i}' for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, n_rows),
        'salary': np.random.normal(50000, 15000, n_rows),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_rows),
        'hire_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'performance_score': np.random.uniform(1, 10, n_rows),
        'is_active': np.random.choice([True, False], n_rows, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    # Missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_rows), replace=False)
    df.loc[missing_indices, 'salary'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_rows), replace=False)
    df.loc[missing_indices, 'department'] = np.nan
    
    # Duplicate rows
    duplicate_indices = np.random.choice(df.index, size=50, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Outliers in salary
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'salary'] = np.random.uniform(200000, 500000, 20)
    
    # Mixed case in names
    mixed_case_indices = np.random.choice(df.index, size=100, replace=False)
    df.loc[mixed_case_indices, 'name'] = df.loc[mixed_case_indices, 'name'].str.upper()
    
    return df


def generate_data_report(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary containing data quality metrics
    """
    report = {
        'basic_info': {
            'rows': len(data),
            'columns': len(data.columns),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'memory_usage_formatted': format_file_size(data.memory_usage(deep=True).sum())
        },
        'data_types': {
            'numeric': len(data.select_dtypes(include=[np.number]).columns),
            'categorical': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(data.select_dtypes(include=['datetime64']).columns),
            'boolean': len(data.select_dtypes(include=['bool']).columns)
        },
        'missing_data': {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': len(data.columns[data.isnull().any()]),
            'complete_rows': len(data.dropna())
        },
        'duplicates': {
            'duplicate_rows': data.duplicated().sum(),
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100
        }
    }
    
    # Add column-specific information
    report['columns'] = {}
    for col in data.columns:
        col_info = {
            'dtype': str(data[col].dtype),
            'non_null_count': data[col].count(),
            'null_count': data[col].isnull().sum(),
            'unique_count': data[col].nunique(),
            'unique_percentage': (data[col].nunique() / len(data)) * 100
        }
        
        if data[col].dtype in ['int64', 'float64']:
            col_info.update({
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            })
        
        report['columns'][col] = col_info
    
    return report


def save_temp_file(data: pd.DataFrame, file_format: str = 'csv') -> str:
    """
    Save DataFrame to a temporary file.
    
    Args:
        data: DataFrame to save
        file_format: Format to save ('csv' or 'xlsx')
        
    Returns:
        Path to temporary file
    """
    if file_format == 'csv':
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(temp_file.name, index=False)
    elif file_format == 'xlsx':
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        data.to_excel(temp_file.name, index=False)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'xlsx'.")
    
    return temp_file.name


def validate_column_operation(data: pd.DataFrame, column: str, operation: str) -> Tuple[bool, str]:
    """
    Validate if a column operation can be performed.
    
    Args:
        data: Input DataFrame
        column: Column name
        operation: Operation to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    if column not in data.columns:
        return False, f"Column '{column}' not found in dataset."
    
    if operation in ['mean', 'median'] and data[column].dtype not in ['int64', 'float64']:
        return False, f"Cannot calculate {operation} for non-numeric column '{column}'."
    
    if operation == 'mode' and data[column].empty:
        return False, f"Cannot calculate mode for empty column '{column}'."
    
    return True, "Operation is valid."


def get_column_suggestions(data: pd.DataFrame, operation_type: str) -> List[str]:
    """
    Get column suggestions based on operation type.
    
    Args:
        data: Input DataFrame
        operation_type: Type of operation ('numeric', 'categorical', 'all')
        
    Returns:
        List of suggested column names
    """
    if operation_type == 'numeric':
        return list(data.select_dtypes(include=[np.number]).columns)
    elif operation_type == 'categorical':
        return list(data.select_dtypes(include=['object', 'category']).columns)
    else:
        return list(data.columns)


def format_operation_summary(operation: str, parameters: Dict[str, Any], 
                           rows_before: int, rows_after: int) -> str:
    """
    Format a summary of a data operation.
    
    Args:
        operation: Name of the operation
        parameters: Parameters used
        rows_before: Number of rows before operation
        rows_after: Number of rows after operation
        
    Returns:
        Formatted summary string
    """
    summary = f"**{operation}**\n"
    summary += f"- Rows: {format_number(rows_before)} → {format_number(rows_after)}"
    
    if rows_before != rows_after:
        change = rows_after - rows_before
        summary += f" ({change:+,})"
    
    if parameters:
        summary += "\n- Parameters: "
        param_strs = [f"{k}={v}" for k, v in parameters.items()]
        summary += ", ".join(param_strs)
    
    return summary
