"""
Exploratory Data Analysis functions for the data cleaning application.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Optional, Tuple, Dict, Any


class EDAAnalyzer:
    """
    Performs exploratory data analysis operations on datasets.
    """
    
    def __init__(self):
        # Set style for matplotlib plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Generate summary statistics for numerical columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (summary_stats_df, message)
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "No numerical columns found in the dataset."
            
            # Generate comprehensive summary statistics
            summary = numeric_data.describe()
            
            # Add additional statistics
            additional_stats = pd.DataFrame({
                'variance': numeric_data.var(),
                'skewness': numeric_data.skew(),
                'kurtosis': numeric_data.kurtosis(),
                'missing_count': numeric_data.isnull().sum(),
                'missing_percentage': (numeric_data.isnull().sum() / len(numeric_data)) * 100
            }).T
            
            # Combine all statistics
            full_summary = pd.concat([summary, additional_stats])
            
            message = f"Summary statistics generated for {len(numeric_data.columns)} numerical columns."
            return full_summary, message
            
        except Exception as e:
            return None, f"Error generating summary statistics: {str(e)}"
    
    def generate_value_counts(self, data: pd.DataFrame, max_unique: int = 20) -> Tuple[Optional[Dict], str]:
        """
        Generate value counts for categorical columns.
        
        Args:
            data: Input DataFrame
            max_unique: Maximum number of unique values to show counts for
            
        Returns:
            Tuple of (value_counts_dict, message)
        """
        try:
            categorical_data = data.select_dtypes(include=['object', 'category'])
            
            if categorical_data.empty:
                return None, "No categorical columns found in the dataset."
            
            value_counts = {}
            
            for col in categorical_data.columns:
                unique_count = data[col].nunique()
                
                if unique_count <= max_unique:
                    counts = data[col].value_counts().head(20)
                    value_counts[col] = {
                        'counts': counts.to_dict(),
                        'total_unique': unique_count,
                        'most_common': counts.index[0] if len(counts) > 0 else None,
                        'most_common_count': counts.iloc[0] if len(counts) > 0 else 0
                    }
                else:
                    # For high cardinality columns, show top values only
                    counts = data[col].value_counts().head(10)
                    value_counts[col] = {
                        'counts': counts.to_dict(),
                        'total_unique': unique_count,
                        'most_common': counts.index[0] if len(counts) > 0 else None,
                        'most_common_count': counts.iloc[0] if len(counts) > 0 else 0,
                        'note': f"Showing top 10 values out of {unique_count} unique values"
                    }
            
            message = f"Value counts generated for {len(value_counts)} categorical columns."
            return value_counts, message
            
        except Exception as e:
            return None, f"Error generating value counts: {str(e)}"
    
    def create_missing_values_plot(self, data: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Create visualization for missing values.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (plot_html, message)
        """
        try:
            missing_data = data.isnull().sum()
            missing_percentage = (missing_data / len(data)) * 100
            
            if missing_data.sum() == 0:
                return None, "No missing values found in the dataset."
            
            # Create plotly bar chart
            fig = go.Figure()
            
            # Add bars for missing counts
            fig.add_trace(go.Bar(
                x=missing_data.index,
                y=missing_data.values,
                name='Missing Count',
                marker_color='lightcoral',
                text=[f'{count}<br>({pct:.1f}%)' for count, pct in zip(missing_data.values, missing_percentage.values)],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Missing Values by Column',
                xaxis_title='Columns',
                yaxis_title='Missing Count',
                template='plotly_white',
                height=500
            )
            
            # Convert to HTML with embedded plotly
            plot_html = fig.to_html(include_plotlyjs=True, config={'displayModeBar': True})
            
            total_missing = missing_data.sum()
            message = f"Missing values visualization created. Total missing values: {total_missing}"
            
            return plot_html, message
            
        except Exception as e:
            return None, f"Error creating missing values plot: {str(e)}"
    
    def create_correlation_matrix(self, data: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Create correlation matrix heatmap for numerical columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (plot_html, message)
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                return None, "Need at least 2 numerical columns to create correlation matrix."
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            # Create plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Correlation Matrix of Numerical Variables',
                template='plotly_white',
                height=600,
                width=600
            )
            
            # Convert to HTML with embedded plotly
            plot_html = fig.to_html(include_plotlyjs=True, config={'displayModeBar': True})
            
            message = f"Correlation matrix created for {len(numeric_data.columns)} numerical columns."
            
            return plot_html, message
            
        except Exception as e:
            return None, f"Error creating correlation matrix: {str(e)}"
    
    def create_histograms(self, data: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Create histograms for all numerical columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (plot_html, message)
        """
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                return None, "No numerical columns found for histogram creation."
            
            # Calculate number of rows and columns for subplots
            n_cols = min(3, len(numeric_data.columns))
            n_rows = (len(numeric_data.columns) + n_cols - 1) // n_cols
            
            # Create subplots
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=numeric_data.columns.tolist(),
                vertical_spacing=0.08,
                horizontal_spacing=0.05
            )
            
            # Add histograms
            for i, col in enumerate(numeric_data.columns):
                row = i // n_cols + 1
                col_pos = i % n_cols + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=numeric_data[col],
                        name=col,
                        showlegend=False,
                        marker_color='skyblue',
                        opacity=0.7
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title='Distribution of Numerical Variables',
                template='plotly_white',
                height=300 * n_rows,
                showlegend=False
            )
            
            # Convert to HTML with embedded plotly
            plot_html = fig.to_html(include_plotlyjs=True, config={'displayModeBar': True})
            
            message = f"Histograms created for {len(numeric_data.columns)} numerical columns."
            
            return plot_html, message
            
        except Exception as e:
            return None, f"Error creating histograms: {str(e)}"
    
    def create_data_overview_plot(self, data: pd.DataFrame) -> Tuple[Optional[str], str]:
        """
        Create a comprehensive data overview visualization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (plot_html, message)
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Data Types Distribution', 'Missing Values by Column', 
                              'Column Cardinality', 'Data Quality Score'],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # 1. Data types distribution
            dtype_counts = data.dtypes.value_counts()
            fig.add_trace(
                go.Pie(
                    labels=dtype_counts.index.astype(str),
                    values=dtype_counts.values,
                    name="Data Types"
                ),
                row=1, col=1
            )
            
            # 2. Missing values
            missing_counts = data.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            if len(missing_cols) > 0:
                fig.add_trace(
                    go.Bar(
                        x=missing_cols.index,
                        y=missing_cols.values,
                        name="Missing Values",
                        marker_color='lightcoral'
                    ),
                    row=1, col=2
                )
            
            # 3. Column cardinality (unique values per column)
            cardinality = data.nunique().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(
                    x=cardinality.index,
                    y=cardinality.values,
                    name="Unique Values",
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
            
            # 4. Data quality score
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            quality_score = ((total_cells - missing_cells) / total_cells) * 100
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=quality_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Completeness %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Dataset Overview Dashboard',
                template='plotly_white',
                height=800,
                showlegend=False
            )
            
            # Convert to HTML with embedded plotly
            plot_html = fig.to_html(include_plotlyjs=True, config={'displayModeBar': True})
            
            message = f"Data overview dashboard created for dataset with {data.shape[0]} rows and {data.shape[1]} columns."
            
            return plot_html, message
            
        except Exception as e:
            return None, f"Error creating data overview: {str(e)}"
