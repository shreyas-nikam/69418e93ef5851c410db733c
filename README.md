This README provides a comprehensive guide to the **QuLab: Portfolio AI Performance & Benchmarking Dashboard**, a Streamlit application designed for Private Equity Portfolio Managers.

---

# QuLab: Portfolio AI Performance & Benchmarking Dashboard

## Project Title and Description

**QuLab: Portfolio AI Performance & Benchmarking Dashboard**

Welcome to QuLab, your essential toolkit as a Private Equity Portfolio Manager! This Streamlit application is designed to provide a robust, data-driven framework for systematically evaluating and enhancing the AI performance across your diverse portfolio companies. In today's landscape, AI is not merely a buzzword but a critical lever for driving growth, efficiency, and ultimately, maximizing exit valuations.

This dashboard guides you through an end-to-end AI performance review cycle, mirroring the real-world tasks of a Portfolio Manager: from ingesting data and calculating core AI readiness metrics to benchmarking, quantifying financial impact, tracking progress, identifying actionable insights, and assessing AI's contribution to exit valuation. It offers interactive controls, allowing you to fine-tune assumptions and immediately see the impact on your portfolio, empowering you to make data-driven decisions that optimize your fund's overall AI strategy and maximize risk-adjusted returns.

## Features

The application provides a multi-page interactive experience, each focusing on a critical aspect of AI portfolio management:

1.  **Initializing Portfolio Data: The Bedrock for AI Performance Tracking**
    *   Generates synthetic portfolio data for a configurable number of companies and quarters.
    *   Provides an overview of the data structure, descriptive statistics, and data types to ensure data integrity.

2.  **Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment**
    *   Calculates the Organizational AI Readiness (Org-AI-R) score using a customizable formula.
    *   Allows adjustment of weights (`alpha`, `beta`) for Idiosyncratic Readiness, Systematic Opportunity, and Synergy to align with strategic priorities.
    *   Displays the latest Org-AI-R scores across the portfolio.

3.  **Benchmarking AI Performance: Identifying Relative AI Standing**
    *   Calculates within-portfolio percentile ranks and industry-adjusted Z-scores for Org-AI-R scores.
    *   Visualizes company performance against portfolio averages and industry peers using interactive bar and scatter plots.
    *   Allows selection of specific quarters for benchmarking focus.

4.  **AI Investment & EBITDA Impact: Quantifying Financial Returns**
    *   Measures AI Investment Efficiency, indicating how effectively capital is deployed for AI initiatives.
    *   Calculates Attributed EBITDA Impact (percentage and absolute), quantifying the direct financial uplift from improved AI maturity.
    *   Visualizes the relationship between AI investment, efficiency, and EBITDA impact.

5.  **Tracking Progress Over Time: Visualizing Trajectories**
    *   Monitors historical trends of Org-AI-R scores and AI Investment Efficiency for selected companies.
    *   Compares individual company trajectories against the overall portfolio average to identify consistent improvers or decliners.

6.  **Actionable Insights: Centers of Excellence & Companies for Review**
    *   Allows dynamic configuration of thresholds for Org-AI-R score and Attributed EBITDA Impact.
    *   Automatically identifies "Centers of Excellence" (high performers) and "Companies for Review" (underperformers).
    *   Provides a visual segmentation of the portfolio to inform strategic resource allocation and intervention strategies.

7.  **Exit-Readiness & Valuation: Maximizing Strategic Exits**
    *   Calculates an Exit-AI-R score based on Visible, Documented, and Sustainable AI capabilities.
    *   Enables adjustment of weighting factors (`w1`, `w2`, `w3`) to prioritize aspects most valued by potential acquirers.
    *   Projects exit valuation multiples by adding an AI premium based on the Exit-AI-R score.
    *   Visualizes the relationship between Exit-AI-R and projected multiples, aiding in crafting a compelling exit narrative.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository** (if hosted on GitHub):
    ```bash
    git clone https://github.com/your-username/quslab-ai-portfolio-dashboard.git
    cd quslab-ai-portfolio-dashboard
    ```
    *(If not from a repository, ensure `app.py` is in your working directory.)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies**:
    Create a `requirements.txt` file in the same directory as `app.py` with the following content:
    ```
    streamlit==1.32.0
    pandas==2.2.1
    numpy==1.26.4
    plotly==5.19.0
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Ensure your virtual environment is activated and you are in the directory containing `app.py`.
    ```bash
    streamlit run app.py
    ```

2.  **Access the application**:
    A new tab will automatically open in your web browser pointing to `http://localhost:8501` (or another port if 8501 is in use).

3.  **Interact with the Dashboard**:
    *   **Sidebar Controls**:
        *   **Global Portfolio Setup**: Adjust the `Number of Portfolio Companies` and `Number of Quarters (History)`. Click "Generate New Portfolio Data" to create a fresh synthetic dataset.
        *   **Portfolio Review Stages**: Navigate between the different analysis pages using the radio buttons.
    *   **Page-Specific Controls**: On each page, you will find sliders and select boxes to fine-tune calculations (e.g., weights for Org-AI-R or Exit-AI-R, thresholds for actionable insights) and customize visualizations. The dashboard dynamically updates as you interact with these controls.

## Project Structure

The project is structured for clarity and ease of use, with all core logic consolidated into a single Streamlit application file.

```
.
├── app.py                  # Main Streamlit application containing all logic and UI components
├── requirements.txt        # Lists all Python dependencies required to run the application
└── README.md               # This README file
```

## Technology Stack

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application and user interface.
*   **Pandas**: For efficient data manipulation, analysis, and DataFrame operations.
*   **NumPy**: For numerical operations and generating synthetic data.
*   **Plotly**: For creating interactive and insightful data visualizations (scatter plots, bar charts, line charts).

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please consider the following:

1.  **Fork the repository**.
2.  **Create a new branch** for your feature or fix.
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **Implement your changes**.
4.  **Test your changes** thoroughly.
5.  **Commit your changes** with a descriptive message.
    ```bash
    git commit -m "feat: Add new feature X"
    ```
6.  **Push your branch** to your forked repository.
    ```bash
    git push origin feature/your-feature-name
    ```
7.  **Open a Pull Request** against the `main` branch of the original repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (Note: A `LICENSE` file would typically be included in the repository.)

## Contact

For questions, feedback, or collaborations, please reach out:

*   **Your Name/Organization**: Quant University Lab
*   **GitHub**: [github.com/your-github-profile](https://github.com/QuantUniversity) (Example)
*   **Email**: [info@quantuniversity.com](mailto:info@quantuniversity.com) (Example)

---