# QuLab: Portfolio AI Performance & Benchmarking Dashboard

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

The **QuLab: Portfolio AI Performance & Benchmarking Dashboard** is a sophisticated Streamlit application designed to empower Private Equity (PE) Portfolio Managers with a systematic and data-driven framework for assessing and optimizing AI performance across their portfolio companies.

In today's dynamic investment landscape, AI is a critical driver of value, but its impact needs to be systematically quantified and managed. This application addresses this need by guiding PE Portfolio Managers through a complete AI performance review cycle. Users can load synthetic portfolio data, compute key AI readiness and financial impact metrics, benchmark companies against peers and industry standards, track progress over time, and identify "Centers of Excellence" for best practice scaling, as well as "Companies for Review" that require immediate strategic attention. Finally, the dashboard helps assess how a company's AI capabilities enhance its exit readiness and potential valuation.

Each stage of this application is crafted to facilitate data-driven decisions that optimize a fund's overall AI strategy and maximize risk-adjusted returns, providing a clear narrative from initial data overview to strategic exit planning.

## Features

This application provides a multi-stage workflow, offering a comprehensive suite of analytical tools and visualizations:

1.  **Initializing Portfolio Data**:
    *   **Synthetic Data Generation**: Dynamically generate realistic synthetic portfolio data based on customizable parameters (number of companies, number of historical quarters).
    *   **Data Overview**: Display the raw data structure, descriptive statistics, and data types for quick understanding.

2.  **Calculating PE Org-AI-R Scores**:
    *   **Customizable AI Readiness Score**: Calculate a proprietary "PE Org-AI-R Score" based on Idiosyncratic Readiness, Systematic Opportunity, and Synergy components.
    *   **Weighted Parameters**: Adjust `alpha` (weight for Idiosyncratic Readiness) and `beta` (Synergy Coefficient) to align the score calculation with specific strategic priorities.
    *   **Formula Transparency**: Display the underlying mathematical formula for the Org-AI-R score.

3.  **Benchmarking AI Performance**:
    *   **Within-Portfolio Benchmarking**: Assess a company's AI readiness relative to others in the fund using Org-AI-R Percentiles.
    *   **Cross-Industry Benchmarking**: Evaluate performance against industry peers using Industry-Adjusted Z-Scores.
    *   **Interactive Visualizations**: Bar charts showing Org-AI-R scores and scatter plots comparing Org-AI-R with Z-scores for a selected quarter.

4.  **AI Investment & EBITDA Impact**:
    *   **AI Investment Efficiency (AIE)**: Quantify the efficiency of AI capital deployment by measuring the impact on Org-AI-R per unit of investment.
    *   **Attributed EBITDA Impact**: Estimate the percentage and absolute EBITDA growth directly attributable to improvements in AI readiness.
    *   **Visual Analysis**: Scatter plot comparing AI Investment (log scale) against AI Investment Efficiency, sized by Attributed EBITDA Impact.

5.  **Tracking Progress Over Time**:
    *   **Time-Series Trajectories**: Visualize the historical progress of selected companies' Org-AI-R scores and AI Investment Efficiency.
    *   **Portfolio Average Comparison**: Overlay portfolio-wide average trends to contextualize individual company performance.
    *   **Multi-Company Selection**: Interactively select multiple companies to track their performance across quarters.

6.  **Actionable Insights: Centers of Excellence & Companies for Review**:
    *   **Customizable Thresholds**: Define dynamic thresholds for Org-AI-R scores and EBITDA impact to identify key categories.
    *   **Centers of Excellence (CoE)**: Automatically identify high-performing companies whose AI best practices can be scaled across the portfolio.
    *   **Companies for Review**: Pinpoint underperforming companies that require immediate strategic attention or resource reallocation.
    *   **Strategic Quadrant Plot**: A scatter plot visualizing all companies, highlighting CoE and Review companies, along with the defined thresholds for quick strategic identification.

7.  **Exit-Readiness & Valuation**:
    *   **Exit-AI-R Score**: Calculate a specialized "Exit-AI-R Score" based on Visible, Documented, and Sustainable AI capabilities, crucial for exit planning.
    *   **Weighted Exit Factors**: Adjust `w1`, `w2`, `w3` parameters to prioritize different aspects of AI readiness that buyers might value most.
    *   **Projected Exit Multiple**: Quantify the potential uplift in valuation multiples driven by a company's AI readiness.
    *   **Valuation Impact Visualization**: Scatter plot showing the relationship between Exit-AI-R Score and Projected Exit Multiple.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   **Python**: Version 3.8 or higher is recommended.
*   **pip**: Python package installer.

### Installation

1.  **Clone the repository** (assuming this code is part of a Git repository):
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *If not using Git, save the provided `streamlit_app.py` file to a directory of your choice.*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
    On Windows:
    ```bash
    .\venv\Scripts\activate
    ```
    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in your project directory with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    numpy>=1.24.0
    matplotlib>=3.7.0
    seaborn>=0.13.0
    scipy>=1.10.0
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Save the application code**:
    Save the provided Python code as `streamlit_app.py` (or any other `.py` file) in your project directory.

2.  **Run the Streamlit application**:
    Ensure your virtual environment is active, then run:
    ```bash
    streamlit run streamlit_app.py
    ```
    This will open the application in your default web browser (usually at `http://localhost:8501`).

3.  **Basic Interaction**:
    *   **Global Portfolio Setup (Sidebar)**: On the left sidebar, adjust the `Number of Portfolio Companies` and `Number of Quarters (History)`.
    *   **Generate New Data**: Click the "Generate New Portfolio Data" button to create a fresh synthetic dataset based on your chosen parameters. This is the first step to populate the dashboard.
    *   **Navigate Stages (Sidebar)**: Use the "Portfolio Review Stages" radio buttons in the sidebar to move between different analytical sections of the dashboard.
    *   **Interactive Controls**: Within each page, utilize sliders, selectboxes, and buttons to customize calculations and visualizations (e.g., `alpha` and `beta` for Org-AI-R, thresholds for actionable insights).

## Project Structure

```
.
├── streamlit_app.py      # Main Streamlit application code
├── requirements.txt      # List of Python dependencies
└── README.md             # This README file
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: The open-source app framework used to build and deploy the interactive web application.
*   **Pandas**: For robust data manipulation and analysis.
*   **NumPy**: For numerical operations, especially in data generation and calculations.
*   **Matplotlib**: A foundational library for creating static, interactive, and animated visualizations in Python.
*   **Seaborn**: Built on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.
*   **SciPy**: Used for scientific computing, specifically for `zscore` calculation.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to good practices and includes appropriate comments and documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You would need to create a `LICENSE` file in your repository with the MIT License text)*

## Contact

For any questions, feedback, or further information, please contact:

*   **Your Name/Organization**: [Your Name or QuLab]
*   **Email**: [your.email@example.com]
*   **GitHub**: [https://github.com/your-username](https://github.com/your-username)
