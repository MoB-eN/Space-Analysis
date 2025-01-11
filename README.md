# Data Analysis and Visualization Web App

This Django-based web application provides users with a robust platform for data analysis and visualization. Users can upload CSV files and analyze their data through a range of statistical methods and visualizations.

## Features

1. **File Upload:**
   - Users can upload a CSV file for analysis.
   - The application previews the first few rows of the dataset and provides basic information such as the number of rows, columns, memory usage, and total missing values.

2. **Single Column Analysis:**
   - Performs descriptive statistical analysis (mean, median, mode, etc.) for numeric columns.
   - Generates visualizations such as histograms, box plots, violin plots, Q-Q plots, and empirical CDFs.
   - Provides the frequency distribution for categorical columns, including top values and their percentages.

3. **Two-Column Analysis:**
   - Supports X vs Y analysis for numeric and categorical columns.
   - For numeric pairs, it calculates Pearson and Spearman correlations and generates scatter plots, hexbin plots, 2D KDE plots, and residual plots.
   - For categorical pairs, it performs a chi-squared test and visualizes the results with heatmaps and stacked bar plots.

4. **Visualization Style:**
   - Implements a custom dark-themed visualization style using `matplotlib`.

## Project Structure

```
project_root/
|-- core/
|   |-- views.py     # Contains all logic for file upload and analysis.
|   |-- templates/
|       |-- core/
|           |-- upload.html  # HTML template for file upload and analysis display.
|-- static/
|   |-- css/         # CSS files for styling the web app.
|-- manage.py        # Django project management script.
|-- requirements.txt # Python dependencies.
```

## Dependencies

- Python 3.8+
- Django
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Django development server:
   ```bash
   python manage.py runserver
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

## Code Breakdown

### Key Functions

#### `analyze_column(df, column_name)`
Analyzes a single column of the dataset. For numeric columns, it provides statistics and visualizations. For categorical columns, it calculates frequency distributions and visualizes them.

#### `generate_xy_visualization(data, x_column, y_column)`
Generates analysis and visualizations for two columns, handling both numeric and categorical pairs. Includes advanced visualizations like scatter plots, hexbin plots, and heatmaps.

#### `upload_file(request)`
Handles file uploads, column selection for analysis, and rendering results in the `upload.html` template.

### Helper Functions

- `set_visualization_style()`: Configures a custom dark theme for all visualizations.
- `get_item(dictionary, key)`: A Django template filter to retrieve dictionary values by key.

## Visualizations

The application generates various types of visualizations:

1. **Histograms and KDE plots** for numeric distributions.
2. **Box plots and violin plots** for spread and density visualization.
3. **Scatter plots with regression lines** for X vs Y numeric data.
4. **Heatmaps** for categorical data relationships.

## Error Handling

- Checks if uploaded files are in CSV format.
- Validates column names before performing analysis.
- Provides user-friendly error messages for missing or invalid data.

## Future Improvements

- Add support for other file formats (e.g., Excel).
- Enable export of analysis results and visualizations.
- Include advanced machine learning-based insights.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Libraries like Pandas, Matplotlib, and Seaborn for enabling data manipulation and visualization.
- Django for providing a robust web development framework.

---

Thank you for using the Data Analysis and Visualization Web App. Feel free to contribute or raise issues on the project repository!

