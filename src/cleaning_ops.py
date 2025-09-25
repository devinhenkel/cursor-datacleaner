"""
Data cleaning operations for the data cleaning application.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Performs data cleaning operations on datasets.
    """
    
    def __init__(self):
        pass
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = "delete", 
                            columns: Optional[List[str]] = None, 
                            knn_neighbors: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Handle missing values using various strategies.
        
        Args:
            data: Input DataFrame
            strategy: Strategy to use ('delete', 'mean', 'median', 'mode', 'knn')
            columns: Specific columns to process (None for all columns)
            knn_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            Tuple of (cleaned_data, message)
        """
        try:
            result_data = data.copy()
            
            if columns is None:
                columns = list(data.columns)
            
            # Check if there are any missing values
            missing_before = result_data[columns].isnull().sum().sum()
            if missing_before == 0:
                return result_data, "No missing values found in the specified columns."
            
            if strategy == "delete":
                # Remove rows with any missing values in specified columns
                result_data = result_data.dropna(subset=columns)
                message = f"Removed {len(data) - len(result_data)} rows with missing values."
                
            elif strategy == "mean":
                # Fill missing values with mean (numeric columns only)
                numeric_cols = [col for col in columns if data[col].dtype in ['int64', 'float64']]
                for col in numeric_cols:
                    mean_value = result_data[col].mean()
                    result_data[col].fillna(mean_value, inplace=True)
                message = f"Filled missing values with mean for {len(numeric_cols)} numeric columns."
                
            elif strategy == "median":
                # Fill missing values with median (numeric columns only)
                numeric_cols = [col for col in columns if data[col].dtype in ['int64', 'float64']]
                for col in numeric_cols:
                    median_value = result_data[col].median()
                    result_data[col].fillna(median_value, inplace=True)
                message = f"Filled missing values with median for {len(numeric_cols)} numeric columns."
                
            elif strategy == "mode":
                # Fill missing values with mode (all columns)
                for col in columns:
                    mode_value = result_data[col].mode()
                    if len(mode_value) > 0:
                        result_data[col].fillna(mode_value[0], inplace=True)
                message = f"Filled missing values with mode for {len(columns)} columns."
                
            elif strategy == "knn":
                # KNN imputation for numeric columns only
                numeric_cols = [col for col in columns if data[col].dtype in ['int64', 'float64']]
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    result_data[numeric_cols] = imputer.fit_transform(result_data[numeric_cols])
                    message = f"Applied KNN imputation (k={knn_neighbors}) to {len(numeric_cols)} numeric columns."
                else:
                    message = "No numeric columns found for KNN imputation."
            
            missing_after = result_data[columns].isnull().sum().sum()
            message += f" Missing values reduced from {missing_before} to {missing_after}."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error handling missing values: {str(e)}"
    
    def remove_duplicates(self, data: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         keep: str = 'first') -> Tuple[pd.DataFrame, str]:
        """
        Remove duplicate rows from the dataset.
        
        Args:
            data: Input DataFrame
            columns: Columns to consider for duplicates (None for all columns)
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Tuple of (cleaned_data, message)
        """
        try:
            duplicates_before = data.duplicated(subset=columns).sum()
            
            if duplicates_before == 0:
                return data, "No duplicate rows found."
            
            result_data = data.drop_duplicates(subset=columns, keep=keep)
            
            duplicates_removed = len(data) - len(result_data)
            message = f"Removed {duplicates_removed} duplicate rows. Dataset now has {len(result_data)} rows."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error removing duplicates: {str(e)}"
    
    def remove_outliers(self, data: pd.DataFrame, method: str = "iqr", 
                       columns: Optional[List[str]] = None,
                       z_threshold: float = 3.0) -> Tuple[pd.DataFrame, str]:
        """
        Remove outliers from numerical columns.
        
        Args:
            data: Input DataFrame
            method: Method to use ('iqr' or 'zscore')
            columns: Columns to process (None for all numeric columns)
            z_threshold: Threshold for z-score method
            
        Returns:
            Tuple of (cleaned_data, message)
        """
        try:
            result_data = data.copy()
            
            # Get numeric columns if not specified
            if columns is None:
                columns = list(data.select_dtypes(include=[np.number]).columns)
            else:
                # Filter to only numeric columns
                columns = [col for col in columns if data[col].dtype in ['int64', 'float64']]
            
            if len(columns) == 0:
                return data, "No numeric columns found for outlier removal."
            
            outliers_removed = 0
            
            if method == "iqr":
                for col in columns:
                    Q1 = result_data[col].quantile(0.25)
                    Q3 = result_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (result_data[col] < lower_bound) | (result_data[col] > upper_bound)
                    outliers_removed += outliers_mask.sum()
                    result_data = result_data[~outliers_mask]
                
                message = f"Removed {outliers_removed} outliers using IQR method from {len(columns)} columns."
                
            elif method == "zscore":
                for col in columns:
                    z_scores = np.abs((result_data[col] - result_data[col].mean()) / result_data[col].std())
                    outliers_mask = z_scores > z_threshold
                    outliers_removed += outliers_mask.sum()
                    result_data = result_data[~outliers_mask]
                
                message = f"Removed {outliers_removed} outliers using Z-score method (threshold={z_threshold}) from {len(columns)} columns."
            
            message += f" Dataset now has {len(result_data)} rows."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error removing outliers: {str(e)}"
    
    def convert_data_types(self, data: pd.DataFrame, 
                          conversions: Dict[str, str]) -> Tuple[pd.DataFrame, str]:
        """
        Convert data types of specified columns.
        
        Args:
            data: Input DataFrame
            conversions: Dictionary mapping column names to target data types
            
        Returns:
            Tuple of (converted_data, message)
        """
        try:
            result_data = data.copy()
            successful_conversions = []
            failed_conversions = []
            
            for column, target_type in conversions.items():
                if column not in data.columns:
                    failed_conversions.append(f"{column} (column not found)")
                    continue
                
                try:
                    if target_type == "int":
                        result_data[column] = pd.to_numeric(result_data[column], errors='coerce').astype('Int64')
                    elif target_type == "float":
                        result_data[column] = pd.to_numeric(result_data[column], errors='coerce')
                    elif target_type == "str":
                        result_data[column] = result_data[column].astype(str)
                    elif target_type == "bool":
                        result_data[column] = result_data[column].astype(bool)
                    elif target_type == "datetime":
                        result_data[column] = pd.to_datetime(result_data[column], errors='coerce')
                    else:
                        failed_conversions.append(f"{column} (unsupported type: {target_type})")
                        continue
                    
                    successful_conversions.append(f"{column} → {target_type}")
                    
                except Exception as e:
                    failed_conversions.append(f"{column} ({str(e)})")
            
            message = f"Successfully converted {len(successful_conversions)} columns: {', '.join(successful_conversions)}"
            if failed_conversions:
                message += f". Failed conversions: {', '.join(failed_conversions)}"
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error converting data types: {str(e)}"
    
    def rename_columns(self, data: pd.DataFrame, 
                      column_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, str]:
        """
        Rename columns in the dataset.
        
        Args:
            data: Input DataFrame
            column_mapping: Dictionary mapping old names to new names
            
        Returns:
            Tuple of (renamed_data, message)
        """
        try:
            result_data = data.copy()
            
            # Check which columns exist
            existing_columns = {old: new for old, new in column_mapping.items() 
                              if old in data.columns}
            missing_columns = {old: new for old, new in column_mapping.items() 
                             if old not in data.columns}
            
            if existing_columns:
                result_data = result_data.rename(columns=existing_columns)
                renamed_list = [f"'{old}' → '{new}'" for old, new in existing_columns.items()]
                message = f"Renamed {len(existing_columns)} columns: {', '.join(renamed_list)}"
                
                if missing_columns:
                    missing_list = [f"'{old}'" for old in missing_columns.keys()]
                    message += f". Columns not found: {', '.join(missing_list)}"
            else:
                message = "No columns were renamed (columns not found)."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error renaming columns: {str(e)}"
    
    def filter_rows(self, data: pd.DataFrame, column: str, 
                   filter_type: str, value: Union[str, int, float]) -> Tuple[pd.DataFrame, str]:
        """
        Filter rows based on conditions.
        
        Args:
            data: Input DataFrame
            column: Column to filter on
            filter_type: Type of filter ('equal', 'not_equal', 'greater_than', 'less_than', 'contains')
            value: Value to filter by
            
        Returns:
            Tuple of (filtered_data, message)
        """
        try:
            if column not in data.columns:
                return data, f"Column '{column}' not found in dataset."
            
            result_data = data.copy()
            original_rows = len(data)
            
            if filter_type == "equal":
                result_data = result_data[result_data[column] == value]
            elif filter_type == "not_equal":
                result_data = result_data[result_data[column] != value]
            elif filter_type == "greater_than":
                try:
                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') > float(value)]
                except (ValueError, TypeError):
                    return data, f"Cannot apply 'greater_than' filter to non-numeric column or value."
            elif filter_type == "less_than":
                try:
                    result_data = result_data[pd.to_numeric(result_data[column], errors='coerce') < float(value)]
                except (ValueError, TypeError):
                    return data, f"Cannot apply 'less_than' filter to non-numeric column or value."
            elif filter_type == "contains":
                if data[column].dtype == 'object':
                    result_data = result_data[result_data[column].astype(str).str.contains(str(value), na=False)]
                else:
                    return data, f"'Contains' filter can only be applied to text columns."
            else:
                return data, f"Unknown filter type: {filter_type}"
            
            filtered_rows = len(result_data)
            removed_rows = original_rows - filtered_rows
            
            message = f"Applied {filter_type} filter on column '{column}' with value '{value}'. "
            message += f"Kept {filtered_rows} rows, removed {removed_rows} rows."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error filtering rows: {str(e)}"
    
    def standardize_text(self, data: pd.DataFrame, 
                        columns: Optional[List[str]] = None,
                        operations: List[str] = ['strip', 'lower']) -> Tuple[pd.DataFrame, str]:
        """
        Standardize text in specified columns.
        
        Args:
            data: Input DataFrame
            columns: Columns to standardize (None for all text columns)
            operations: List of operations ('strip', 'lower', 'upper', 'title')
            
        Returns:
            Tuple of (standardized_data, message)
        """
        try:
            result_data = data.copy()
            
            if columns is None:
                columns = list(data.select_dtypes(include=['object']).columns)
            else:
                columns = [col for col in columns if col in data.columns and data[col].dtype == 'object']
            
            if len(columns) == 0:
                return data, "No text columns found for standardization."
            
            for col in columns:
                for operation in operations:
                    if operation == 'strip':
                        result_data[col] = result_data[col].astype(str).str.strip()
                    elif operation == 'lower':
                        result_data[col] = result_data[col].astype(str).str.lower()
                    elif operation == 'upper':
                        result_data[col] = result_data[col].astype(str).str.upper()
                    elif operation == 'title':
                        result_data[col] = result_data[col].astype(str).str.title()
            
            message = f"Applied text standardization ({', '.join(operations)}) to {len(columns)} columns."
            
            return result_data, message
            
        except Exception as e:
            return data, f"Error standardizing text: {str(e)}"
    
    def detect_data_issues(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect common data quality issues.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary containing detected issues
        """
        try:
            issues = {
                'missing_values': {},
                'duplicates': 0,
                'potential_outliers': {},
                'mixed_types': {},
                'high_cardinality': {},
                'low_variance': []
            }
            
            # Missing values
            missing = data.isnull().sum()
            issues['missing_values'] = {col: count for col, count in missing.items() if count > 0}
            
            # Duplicates
            issues['duplicates'] = data.duplicated().sum()
            
            # Potential outliers (using IQR method)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                if outliers > 0:
                    issues['potential_outliers'][col] = outliers
            
            # High cardinality columns (more than 50% unique values)
            for col in data.columns:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio > 0.5 and data[col].nunique() > 10:
                    issues['high_cardinality'][col] = {
                        'unique_count': data[col].nunique(),
                        'unique_ratio': unique_ratio
                    }
            
            # Low variance numeric columns
            for col in numeric_cols:
                if data[col].var() < 0.01:  # Very low variance
                    issues['low_variance'].append(col)
            
            return issues
            
        except Exception as e:
            return {'error': f"Error detecting data issues: {str(e)}"}
