"""
DataHandler class for managing data operations in the data cleaning application.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import io
import tempfile
import os


class DataHandler:
    """
    Manages data loading, processing, and state management for the data cleaning application.
    """
    
    def __init__(self):
        self.original_data: Optional[pd.DataFrame] = None
        self.current_data: Optional[pd.DataFrame] = None
        self.filename: Optional[str] = None
        self.operations_history: list = []
    
    def load_data(self, file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Load data from CSV or XLSX file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (success, message, preview_data)
        """
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            
            if ext == '.csv':
                data = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            else:
                return False, "Unsupported file format. Please upload CSV or XLSX files.", None
            
            if data.empty:
                return False, "The uploaded file is empty.", None
            
            # Store data
            self.original_data = data.copy()
            self.current_data = data.copy()
            self.filename = os.path.basename(file_path)
            self.operations_history = []
            
            # Create preview (first 10 rows)
            preview = data.head(10)
            
            success_msg = f"Successfully loaded {data.shape[0]} rows and {data.shape[1]} columns from {self.filename}"
            return True, success_msg, preview
            
        except Exception as e:
            return False, f"Error loading file: {str(e)}", None
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata about the current dataset.
        
        Returns:
            Dictionary containing dataset metadata
        """
        if self.current_data is None:
            return None
        
        data = self.current_data
        
        metadata = {
            'filename': self.filename,
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().to_dict(),
            'total_missing': data.isnull().sum().sum(),
            'duplicate_rows': data.duplicated().sum()
        }
        
        return metadata
    
    def get_preview(self, n_rows: int = 10) -> Optional[pd.DataFrame]:
        """
        Get preview of current data.
        
        Args:
            n_rows: Number of rows to preview
            
        Returns:
            DataFrame with preview data
        """
        if self.current_data is None:
            return None
        
        return self.current_data.head(n_rows)
    
    def reset_data(self) -> bool:
        """
        Reset current data to original state.
        
        Returns:
            True if reset successful
        """
        if self.original_data is None:
            return False
        
        self.current_data = self.original_data.copy()
        self.operations_history = []
        return True
    
    def add_operation_to_history(self, operation: str, parameters: Dict[str, Any] = None):
        """
        Add an operation to the history log.
        
        Args:
            operation: Name of the operation performed
            parameters: Parameters used for the operation
        """
        history_entry = {
            'operation': operation,
            'parameters': parameters or {},
            'timestamp': pd.Timestamp.now(),
            'rows_before': self.current_data.shape[0] if self.current_data is not None else 0,
            'cols_before': self.current_data.shape[1] if self.current_data is not None else 0
        }
        self.operations_history.append(history_entry)
    
    def update_history_after_operation(self):
        """Update the last history entry with post-operation statistics."""
        if self.operations_history and self.current_data is not None:
            self.operations_history[-1]['rows_after'] = self.current_data.shape[0]
            self.operations_history[-1]['cols_after'] = self.current_data.shape[1]
    
    def get_operations_summary(self) -> str:
        """
        Get a summary of all operations performed.
        
        Returns:
            String summary of operations
        """
        if not self.operations_history:
            return "No operations performed yet."
        
        summary = "Operations History:\n"
        for i, op in enumerate(self.operations_history, 1):
            summary += f"{i}. {op['operation']}"
            if op.get('rows_before') is not None and op.get('rows_after') is not None:
                summary += f" ({op['rows_before']} → {op['rows_after']} rows)"
            summary += f" at {op['timestamp'].strftime('%H:%M:%S')}\n"
        
        return summary
    
    def export_to_csv(self) -> Optional[str]:
        """
        Export current data to CSV format.
        
        Returns:
            CSV content as string or None if no data
        """
        if self.current_data is None:
            return None
        
        output = io.StringIO()
        self.current_data.to_csv(output, index=False)
        return output.getvalue()
    
    def export_to_excel(self) -> Optional[bytes]:
        """
        Export current data to Excel format.
        
        Returns:
            Excel content as bytes or None if no data
        """
        if self.current_data is None:
            return None
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            self.current_data.to_excel(writer, sheet_name='Cleaned_Data', index=False)
        
        return output.getvalue()
    
    def is_data_loaded(self) -> bool:
        """Check if data is currently loaded."""
        return self.current_data is not None
    
    def get_column_names(self) -> list:
        """Get list of current column names."""
        if self.current_data is None:
            return []
        return list(self.current_data.columns)
    
    def get_numeric_columns(self) -> list:
        """Get list of numeric column names."""
        if self.current_data is None:
            return []
        return list(self.current_data.select_dtypes(include=[np.number]).columns)
    
    def get_categorical_columns(self) -> list:
        """Get list of categorical/object column names."""
        if self.current_data is None:
            return []
        return list(self.current_data.select_dtypes(include=['object', 'category']).columns)
