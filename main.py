"""
Data Cleaning Application - Main Gradio Interface
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import tempfile
import os

# Import our custom modules
from src.data_handler import DataHandler
from src.eda_functions import EDAAnalyzer
from src.cleaning_ops import DataCleaner
from src.utils import format_file_size, format_number, validate_file_type, create_sample_data


class DataCleaningApp:
    """Main application class for the data cleaning interface."""
    
    def __init__(self):
        self.data_handler = DataHandler()
        self.eda_analyzer = EDAAnalyzer()
        self.data_cleaner = DataCleaner()
    
    def load_file(self, file) -> Tuple[pd.DataFrame, str, str]:
        """Load uploaded file and return preview, metadata, and message."""
        if file is None:
            return None, "Please upload a file.", ""
        
        try:
            success, message, preview = self.data_handler.load_data(file.name)
            
            if success:
                metadata = self.data_handler.get_metadata()
                metadata_str = self._format_metadata(metadata)
                return preview, message, metadata_str
            else:
                return None, message, ""
                
        except Exception as e:
            return None, f"Error loading file: {str(e)}", ""
    
    def load_sample_data(self) -> Tuple[pd.DataFrame, str, str]:
        """Load sample data for demonstration."""
        try:
            sample_data = create_sample_data()
            
            # Save to temporary file and load through data handler
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            sample_data.to_csv(temp_file.name, index=False)
            
            success, message, preview = self.data_handler.load_data(temp_file.name)
            
            if success:
                metadata = self.data_handler.get_metadata()
                metadata_str = self._format_metadata(metadata)
                return preview, f"Sample data loaded successfully! {message}", metadata_str
            else:
                return None, message, ""
                
        except Exception as e:
            return None, f"Error loading sample data: {str(e)}", ""
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for display."""
        if not metadata:
            return ""
        
        info = f"""
**Dataset Information:**
- **Filename:** {metadata['filename']}
- **Shape:** {format_number(metadata['shape'][0])} rows × {metadata['shape'][1]} columns
- **Memory Usage:** {format_file_size(metadata['memory_usage'])}
- **Missing Values:** {format_number(metadata['total_missing'])} total
- **Duplicate Rows:** {format_number(metadata['duplicate_rows'])}

**Column Types:**
"""
        dtype_counts = {}
        for col, dtype in metadata['dtypes'].items():
            dtype_str = str(dtype)
            dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
        
        for dtype, count in dtype_counts.items():
            info += f"- {dtype}: {count} columns\n"
        
        return info
    
    def generate_summary_stats(self) -> Tuple[str, str]:
        """Generate summary statistics."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        stats, message = self.eda_analyzer.generate_summary_statistics(self.data_handler.current_data)
        
        if stats is not None:
            # Convert DataFrame to HTML for display
            html_output = f"""
            <div style="margin: 20px;">
                <h3>📈 Summary Statistics</h3>
                {stats.to_html(classes='table table-striped', table_id='summary-stats')}
            </div>
            """
            return html_output, message
        else:
            return message, ""
    
    def generate_value_counts(self) -> Tuple[str, str]:
        """Generate value counts for categorical columns."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        value_counts, message = self.eda_analyzer.generate_value_counts(self.data_handler.current_data)
        
        if value_counts is None:
            return message, ""
        
        # Format value counts for display as HTML
        output = """
        <div style="margin: 20px;">
            <h3>📋 Value Counts for Categorical Columns</h3>
        """
        
        for col, info in value_counts.items():
            output += f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                <h4>{col}</h4>
                <p><strong>Total unique values:</strong> {info['total_unique']}</p>
            """
            
            if 'note' in info:
                output += f"<p><em>{info['note']}</em></p>"
            
            output += "<ul>"
            for value, count in list(info['counts'].items())[:10]:
                output += f"<li>{value}: {format_number(count)}</li>"
            output += "</ul></div>"
        
        output += "</div>"
        return output, message
    
    def create_missing_values_plot(self) -> Tuple[str, str]:
        """Create missing values visualization."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        plot_html, message = self.eda_analyzer.create_missing_values_plot(self.data_handler.current_data)
        
        if plot_html is None:
            return message, ""
        
        # Wrap in a div to ensure proper display
        wrapped_html = f"""
        <div style="width: 100%; height: 600px; overflow: auto;">
            {plot_html}
        </div>
        """
        
        return wrapped_html, message
    
    def create_correlation_matrix(self) -> Tuple[str, str]:
        """Create correlation matrix."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        plot_html, message = self.eda_analyzer.create_correlation_matrix(self.data_handler.current_data)
        
        if plot_html is None:
            return message, ""
        
        # Wrap in a div to ensure proper display
        wrapped_html = f"""
        <div style="width: 100%; height: 700px; overflow: auto;">
            {plot_html}
        </div>
        """
        
        return wrapped_html, message
    
    def create_histograms(self) -> Tuple[str, str]:
        """Create histograms."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        plot_html, message = self.eda_analyzer.create_histograms(self.data_handler.current_data)
        
        if plot_html is None:
            return message, ""
        
        # Wrap in a div to ensure proper display
        wrapped_html = f"""
        <div style="width: 100%; height: 800px; overflow: auto;">
            {plot_html}
        </div>
        """
        
        return wrapped_html, message
    
    def create_data_overview(self) -> Tuple[str, str]:
        """Create data overview dashboard."""
        if not self.data_handler.is_data_loaded():
            return "Please load data first.", ""
        
        plot_html, message = self.eda_analyzer.create_data_overview_plot(self.data_handler.current_data)
        
        if plot_html is None:
            return message, ""
        
        # Wrap in a div to ensure proper display
        wrapped_html = f"""
        <div style="width: 100%; height: 900px; overflow: auto;">
            {plot_html}
        </div>
        """
        
        return wrapped_html, message
    
    def handle_missing_values(self, strategy: str, knn_neighbors: int) -> Tuple[pd.DataFrame, str, str]:
        """Handle missing values."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        self.data_handler.add_operation_to_history("Handle Missing Values", 
                                                  {"strategy": strategy, "knn_neighbors": knn_neighbors})
        
        self.data_handler.current_data, message = self.data_cleaner.handle_missing_values(
            self.data_handler.current_data, strategy, knn_neighbors=knn_neighbors
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def remove_duplicates(self, keep: str) -> Tuple[pd.DataFrame, str, str]:
        """Remove duplicate rows."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        self.data_handler.add_operation_to_history("Remove Duplicates", {"keep": keep})
        
        self.data_handler.current_data, message = self.data_cleaner.remove_duplicates(
            self.data_handler.current_data, keep=keep
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def remove_outliers(self, method: str, z_threshold: float) -> Tuple[pd.DataFrame, str, str]:
        """Remove outliers."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        self.data_handler.add_operation_to_history("Remove Outliers", 
                                                  {"method": method, "z_threshold": z_threshold})
        
        self.data_handler.current_data, message = self.data_cleaner.remove_outliers(
            self.data_handler.current_data, method=method, z_threshold=z_threshold
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def convert_data_type(self, column: str, target_type: str) -> Tuple[pd.DataFrame, str, str]:
        """Convert data type of a column."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        if not column:
            return self.data_handler.get_preview(), "Please select a column.", self._format_metadata(self.data_handler.get_metadata())
        
        self.data_handler.add_operation_to_history("Convert Data Type", 
                                                  {"column": column, "target_type": target_type})
        
        conversions = {column: target_type}
        self.data_handler.current_data, message = self.data_cleaner.convert_data_types(
            self.data_handler.current_data, conversions
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def rename_column(self, old_name: str, new_name: str) -> Tuple[pd.DataFrame, str, str]:
        """Rename a column."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        if not old_name or not new_name:
            return self.data_handler.get_preview(), "Please provide both old and new column names.", self._format_metadata(self.data_handler.get_metadata())
        
        self.data_handler.add_operation_to_history("Rename Column", 
                                                  {"old_name": old_name, "new_name": new_name})
        
        column_mapping = {old_name: new_name}
        self.data_handler.current_data, message = self.data_cleaner.rename_columns(
            self.data_handler.current_data, column_mapping
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def filter_data(self, column: str, filter_type: str, value: str) -> Tuple[pd.DataFrame, str, str]:
        """Filter data based on conditions."""
        if not self.data_handler.is_data_loaded():
            return None, "Please load data first.", ""
        
        if not column or not value:
            return self.data_handler.get_preview(), "Please select a column and provide a filter value.", self._format_metadata(self.data_handler.get_metadata())
        
        self.data_handler.add_operation_to_history("Filter Data", 
                                                  {"column": column, "filter_type": filter_type, "value": value})
        
        self.data_handler.current_data, message = self.data_cleaner.filter_rows(
            self.data_handler.current_data, column, filter_type, value
        )
        
        self.data_handler.update_history_after_operation()
        
        preview = self.data_handler.get_preview()
        metadata = self._format_metadata(self.data_handler.get_metadata())
        
        return preview, message, metadata
    
    def reset_data(self) -> Tuple[pd.DataFrame, str, str]:
        """Reset data to original state."""
        if not self.data_handler.is_data_loaded():
            return None, "No data loaded.", ""
        
        success = self.data_handler.reset_data()
        if success:
            preview = self.data_handler.get_preview()
            metadata = self._format_metadata(self.data_handler.get_metadata())
            return preview, "Data reset to original state.", metadata
        else:
            return None, "Error resetting data.", ""
    
    def export_csv(self) -> Optional[str]:
        """Export current data as CSV."""
        if not self.data_handler.is_data_loaded():
            return None
        
        csv_content = self.data_handler.export_to_csv()
        if csv_content:
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_file.write(csv_content)
            temp_file.close()
            return temp_file.name
        return None
    
    def export_excel(self) -> Optional[str]:
        """Export current data as Excel."""
        if not self.data_handler.is_data_loaded():
            return None
        
        excel_content = self.data_handler.export_to_excel()
        if excel_content:
            temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
            temp_file.write(excel_content)
            temp_file.close()
            return temp_file.name
        return None
    
    def get_column_choices(self) -> List[str]:
        """Get current column names for dropdowns."""
        if self.data_handler.is_data_loaded():
            return self.data_handler.get_column_names()
        return []
    
    def get_operations_history(self) -> str:
        """Get operations history."""
        return self.data_handler.get_operations_summary()


def create_interface():
    """Create and configure the Gradio interface."""
    
    app = DataCleaningApp()
    
    with gr.Blocks(title="Data Cleaning Application", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧹 Data Cleaning Application")
        gr.Markdown("Upload your CSV or XLSX files to explore, clean, and export your data with ease!")
        
        # State variables
        current_data = gr.State()
        
        with gr.Tab("📁 Data Upload & Preview"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload CSV or XLSX File",
                        file_types=[".csv", ".xlsx", ".xls"]
                    )
                    sample_btn = gr.Button("Load Sample Data", variant="secondary")
                    reset_btn = gr.Button("Reset to Original", variant="outline")
                
                with gr.Column(scale=2):
                    metadata_display = gr.Markdown("Upload a file to see dataset information.")
            
            with gr.Row():
                preview_table = gr.Dataframe(
                    label="Data Preview (First 10 rows)",
                    interactive=False,
                    wrap=True
                )
            
            upload_message = gr.Textbox(label="Status", interactive=False)
        
        with gr.Tab("📊 Exploratory Data Analysis"):
            with gr.Row():
                summary_btn = gr.Button("📈 Summary Statistics")
                value_counts_btn = gr.Button("📋 Value Counts")
                missing_btn = gr.Button("❓ Missing Values")
                correlation_btn = gr.Button("🔗 Correlation Matrix")
                histogram_btn = gr.Button("📊 Histograms")
                overview_btn = gr.Button("🎯 Data Overview")
            
            eda_output = gr.HTML(label="Analysis Output")
            eda_message = gr.Textbox(label="Analysis Status", interactive=False)
        
        with gr.Tab("🧽 Data Cleaning"):
            with gr.Accordion("Missing Values", open=True):
                with gr.Row():
                    missing_strategy = gr.Dropdown(
                        ["delete", "mean", "median", "mode", "knn"],
                        label="Missing Value Strategy",
                        value="delete"
                    )
                    knn_neighbors = gr.Number(
                        label="KNN Neighbors (for KNN method)",
                        value=5,
                        minimum=1,
                        maximum=20
                    )
                    handle_missing_btn = gr.Button("Handle Missing Values")
            
            with gr.Accordion("Duplicates & Outliers"):
                with gr.Row():
                    duplicate_keep = gr.Dropdown(
                        ["first", "last"],
                        label="Keep Which Duplicate",
                        value="first"
                    )
                    remove_duplicates_btn = gr.Button("Remove Duplicates")
                
                with gr.Row():
                    outlier_method = gr.Dropdown(
                        ["iqr", "zscore"],
                        label="Outlier Detection Method",
                        value="iqr"
                    )
                    z_threshold = gr.Number(
                        label="Z-Score Threshold",
                        value=3.0,
                        minimum=1.0,
                        maximum=5.0
                    )
                    remove_outliers_btn = gr.Button("Remove Outliers")
            
            with gr.Accordion("Column Operations"):
                with gr.Row():
                    convert_column = gr.Dropdown(
                        label="Column to Convert",
                        choices=[],
                        interactive=True
                    )
                    target_dtype = gr.Dropdown(
                        ["int", "float", "str", "bool", "datetime"],
                        label="Target Data Type",
                        value="str"
                    )
                    convert_btn = gr.Button("Convert Data Type")
                
                with gr.Row():
                    old_column_name = gr.Textbox(label="Old Column Name")
                    new_column_name = gr.Textbox(label="New Column Name")
                    rename_btn = gr.Button("Rename Column")
            
            with gr.Accordion("Data Filtering"):
                with gr.Row():
                    filter_column = gr.Dropdown(
                        label="Column to Filter",
                        choices=[],
                        interactive=True
                    )
                    filter_type = gr.Dropdown(
                        ["equal", "not_equal", "greater_than", "less_than", "contains"],
                        label="Filter Type",
                        value="equal"
                    )
                    filter_value = gr.Textbox(label="Filter Value")
                    filter_btn = gr.Button("Apply Filter")
            
            cleaning_message = gr.Textbox(label="Cleaning Status", interactive=False)
        
        with gr.Tab("📤 Export & History"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Export Data")
                    export_csv_btn = gr.Button("📄 Download as CSV")
                    export_excel_btn = gr.Button("📊 Download as Excel")
                    
                    csv_download = gr.File(label="CSV Download", visible=False)
                    excel_download = gr.File(label="Excel Download", visible=False)
                
                with gr.Column():
                    gr.Markdown("### Operations History")
                    history_display = gr.Textbox(
                        label="Operations Performed",
                        interactive=False,
                        lines=10
                    )
                    refresh_history_btn = gr.Button("🔄 Refresh History")
        
        # Event handlers
        def update_column_choices():
            choices = app.get_column_choices()
            return gr.Dropdown(choices=choices), gr.Dropdown(choices=choices)
        
        # File upload events
        file_upload.change(
            fn=app.load_file,
            inputs=[file_upload],
            outputs=[preview_table, upload_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        sample_btn.click(
            fn=app.load_sample_data,
            outputs=[preview_table, upload_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        reset_btn.click(
            fn=app.reset_data,
            outputs=[preview_table, upload_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        # EDA events
        summary_btn.click(
            fn=app.generate_summary_stats,
            outputs=[eda_output, eda_message]
        )
        
        value_counts_btn.click(
            fn=app.generate_value_counts,
            outputs=[eda_output, eda_message]
        )
        
        missing_btn.click(
            fn=app.create_missing_values_plot,
            outputs=[eda_output, eda_message]
        )
        
        correlation_btn.click(
            fn=app.create_correlation_matrix,
            outputs=[eda_output, eda_message]
        )
        
        histogram_btn.click(
            fn=app.create_histograms,
            outputs=[eda_output, eda_message]
        )
        
        overview_btn.click(
            fn=app.create_data_overview,
            outputs=[eda_output, eda_message]
        )
        
        # Cleaning events
        handle_missing_btn.click(
            fn=app.handle_missing_values,
            inputs=[missing_strategy, knn_neighbors],
            outputs=[preview_table, cleaning_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        remove_duplicates_btn.click(
            fn=app.remove_duplicates,
            inputs=[duplicate_keep],
            outputs=[preview_table, cleaning_message, metadata_display]
        )
        
        remove_outliers_btn.click(
            fn=app.remove_outliers,
            inputs=[outlier_method, z_threshold],
            outputs=[preview_table, cleaning_message, metadata_display]
        )
        
        convert_btn.click(
            fn=app.convert_data_type,
            inputs=[convert_column, target_dtype],
            outputs=[preview_table, cleaning_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        rename_btn.click(
            fn=app.rename_column,
            inputs=[old_column_name, new_column_name],
            outputs=[preview_table, cleaning_message, metadata_display]
        ).then(
            fn=update_column_choices,
            outputs=[convert_column, filter_column]
        )
        
        filter_btn.click(
            fn=app.filter_data,
            inputs=[filter_column, filter_type, filter_value],
            outputs=[preview_table, cleaning_message, metadata_display]
        )
        
        # Export events
        export_csv_btn.click(
            fn=app.export_csv,
            outputs=[csv_download]
        )
        
        export_excel_btn.click(
            fn=app.export_excel,
            outputs=[excel_download]
        )
        
        refresh_history_btn.click(
            fn=app.get_operations_history,
            outputs=[history_display]
        )
    
    return demo


def main():
    """Main function to launch the application."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
