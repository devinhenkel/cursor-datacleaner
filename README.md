# 🧹 Data Cleaning Application

A comprehensive web-based data cleaning application built with Gradio that allows users to upload CSV/XLSX files, explore data through various EDA functions, and perform data cleaning operations with customizable parameters.

## ✨ Features

### 📁 Data Upload & Preview
- Upload CSV or XLSX files
- Load sample data for demonstration
- Preview first 10 rows of dataset
- Display comprehensive dataset metadata (shape, data types, missing values, duplicates)
- Reset data to original state

### 📊 Exploratory Data Analysis (EDA)
- **Summary Statistics**: Descriptive statistics for numerical columns including mean, median, std, skewness, kurtosis
- **Value Counts**: Frequency analysis for categorical columns with configurable display limits
- **Missing Values Visualization**: Interactive heatmap showing missing data patterns
- **Correlation Matrix**: Heatmap of correlations between numerical variables
- **Histograms**: Distribution plots for all numerical columns
- **Data Overview Dashboard**: Comprehensive dashboard with data types, missing values, cardinality, and quality score

### 🧽 Data Cleaning Operations
- **Missing Value Handling**: Multiple strategies (Delete, Mean, Median, Mode, KNN imputation)
- **Duplicate Removal**: Remove duplicate rows with options to keep first/last occurrence
- **Outlier Detection & Removal**: IQR and Z-score methods with configurable thresholds
- **Data Type Conversion**: Convert columns to int, float, str, bool, or datetime
- **Column Renaming**: Rename columns with validation
- **Data Filtering**: Filter rows based on conditions (equal, not equal, greater than, less than, contains)

### 📤 Export & History
- **Export Options**: Download cleaned data as CSV or Excel files
- **Operations History**: Track all operations performed with parameters and timestamps
- **Progress Tracking**: Real-time updates on dataset changes

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cursor-datacleaner
```

2. Install dependencies:
```bash
pip install -e .
```

## 🎯 Usage

### Running the Application
```bash
python main.py
```

The application will launch at `http://localhost:7860`

### Using the Interface

1. **Upload Data**: 
   - Click "Upload CSV or XLSX File" to upload your data
   - Or click "Load Sample Data" to try the application with demo data

2. **Explore Data**:
   - Go to the "Exploratory Data Analysis" tab
   - Click any of the analysis buttons to generate insights

3. **Clean Data**:
   - Navigate to the "Data Cleaning" tab
   - Use the accordion sections to access different cleaning operations
   - Configure parameters and apply operations

4. **Export Results**:
   - Go to the "Export & History" tab
   - Download your cleaned data as CSV or Excel
   - Review the operations history

## 🏗️ Project Structure

```
cursor-datacleaner/
├── main.py                 # Main Gradio application
├── src/
│   ├── __init__.py
│   ├── data_handler.py     # Core data management class
│   ├── eda_functions.py    # EDA analysis functions
│   ├── cleaning_ops.py     # Data cleaning operations
│   └── utils.py           # Utility functions
├── pyproject.toml         # Project dependencies
└── README.md              # This file
```

## 📋 Dependencies

- **gradio>=4.0.0**: Web interface framework
- **pandas>=2.0.0**: Data manipulation and analysis
- **numpy>=1.24.0**: Numerical computing
- **matplotlib>=3.7.0**: Static plotting
- **seaborn>=0.12.0**: Statistical data visualization
- **scikit-learn>=1.3.0**: Machine learning tools (for KNN imputation)
- **openpyxl>=3.1.0**: Excel file handling
- **plotly>=5.15.0**: Interactive plotting

## 🎨 Interface Overview

### Data Upload & Preview Tab
- File upload component with format validation
- Sample data generator for testing
- Live preview table showing first 10 rows
- Comprehensive metadata display

### EDA Tab
- Six analysis buttons for different types of exploration
- Interactive visualizations using Plotly
- Real-time status messages

### Data Cleaning Tab
- Organized accordion sections for different operations
- Parameter controls with validation
- Immediate feedback on operations

### Export & History Tab
- Download buttons for CSV and Excel export
- Operations history with timestamps
- Data transformation tracking

## 🔧 Key Features

### Robust Data Handling
- Supports CSV and Excel formats
- Memory-efficient processing
- State management (original vs. current data)
- Comprehensive error handling

### Interactive Visualizations
- Plotly-based charts and graphs
- Responsive design
- Export-ready visualizations

### Flexible Cleaning Operations
- Configurable parameters for all operations
- Multiple strategies for missing value handling
- Advanced outlier detection methods
- Safe data type conversions

### User Experience
- Intuitive tabbed interface
- Real-time feedback and status updates
- Progress tracking for all operations
- Comprehensive history logging

## 🚦 Performance

- **File Upload**: < 5 seconds for typical files
- **EDA Operations**: < 30 seconds for datasets < 10MB
- **Cleaning Operations**: < 60 seconds for datasets < 10MB
- **Memory Efficient**: Optimized for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues, questions, or contributions, please open an issue on the repository.

---

Built with ❤️ using Gradio, Pandas, and Plotly
