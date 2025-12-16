id: 69418e93ef5851c410db733c_documentation
summary: Portfolio AI Performance & Benchmarking Dashboard Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Building an AI Performance & Benchmarking Dashboard for Private Equity

## Introduction: Optimizing AI Value Creation in Private Equity
Duration: 0:10:00

<aside class="positive">
This codelab is designed for developers who want to understand how to build a comprehensive analytics application using Streamlit. It focuses on a real-world use case in Private Equity (PE), demonstrating how AI performance can be quantified, benchmarked, and used to drive strategic decision-making. You will learn about data generation, complex metric calculation, interactive visualizations, and generating actionable insights.
</aside>

Welcome to QuLab, where you'll explore a powerful Streamlit application designed for Private Equity (PE) Portfolio Managers. In today's competitive landscape, Artificial Intelligence (AI) is no longer a luxury but a critical driver of value creation, operational efficiency, and enhanced exit valuations for portfolio companies. However, quantifying the impact of AI and identifying areas for strategic intervention remains a significant challenge.

This application provides an end-to-end framework to tackle this challenge. It allows a Portfolio Manager to:
*   **Systematically assess** the AI readiness of portfolio companies.
*   **Benchmark** their performance against peers and industry standards.
*   **Quantify the financial impact** and efficiency of AI investments.
*   **Track progress** over time to monitor strategic initiatives.
*   **Identify 'Centers of Excellence'** for scaling best practices and 'Companies for Review' requiring intervention.
*   **Evaluate AI's contribution** to a company's exit readiness and potential valuation uplift.

By walking through this codelab, developers will gain insights into:
1.  **Streamlit Application Development:** Building interactive dashboards with dynamic data.
2.  **Data Generation & Management:** Creating synthetic datasets for complex scenarios and managing application state with `st.session_state`.
3.  **Financial Modeling & AI Metrics:** Implementing custom formulas for AI readiness, investment efficiency, and valuation impact.
4.  **Data Visualization:** Leveraging `plotly.express` and `plotly.graph_objects` for insightful and interactive charts.
5.  **Strategic Decision Support:** Translating analytical metrics into actionable business recommendations.

### Application Architecture and Data Flow

The application follows a modular architecture, where core data processing logic is encapsulated in Python functions, and Streamlit handles the interactive UI. The data flow is sequential, building upon calculated metrics in previous steps.

The overall flow of the application is as follows:

<aside class="positive">
Understanding this high-level architecture is crucial for grasping how the various components of the Streamlit application work together to deliver comprehensive AI performance analytics. Each box represents a key stage in the data processing pipeline, progressively enriching the portfolio data with strategic insights.
</aside>

```mermaid
graph TD
    A[Start: Generate Synthetic Portfolio Data] --> B{Data Initialization & Overview};
    B --&gt; C[Calculate Org-AI-R Scores];
    C --&gt; D[Benchmark AI Performance];
    D --&gt; E[Calculate AI Investment Efficiency & EBITDA Impact];
    E --&gt; F[Track Progress Over Time];
    F --&gt; G[Identify Actionable Insights (CoE & Review)];
    G --&gt; H[Evaluate Exit-Readiness & Valuation Impact];
    H --&gt; I[End: Strategic Decision Support & Reporting];

    subgraph Data Input
        A
    end

    subgraph Core Calculations
        C
        D
        E
        H
    end

    subgraph Visualization & Insights
        B
        F
        G
        I
    end
```

**Detailed Breakdown:**

1.  **Data Generation:** Synthetic portfolio data for companies across various industries and quarters is generated, simulating key AI-related readiness scores and financial metrics.
2.  **Org-AI-R Calculation:** Based on user-defined weights, each company's Organizational AI Readiness (Org-AI-R) score is computed, combining idiosyncratic strengths, systematic opportunities, and synergies.
3.  **Benchmarking:** Org-AI-R scores are then benchmarked using percentile ranks (within portfolio) and industry-adjusted Z-scores (cross-portfolio).
4.  **Financial Impact:** AI Investment Efficiency and Attributed EBITDA Impact are calculated, linking AI initiatives to tangible financial outcomes.
5.  **Time-Series Analysis:** The historical trends of Org-AI-R and AI Investment Efficiency are visualized.
6.  **Actionable Segmentation:** Companies are categorized into 'Centers of Excellence' or 'Companies for Review' based on configurable thresholds.
7.  **Exit Strategy:** Exit-AI-R scores and Projected Exit Multiples are calculated, assessing how AI capabilities enhance a company's attractiveness and valuation during an exit.

This interactive dashboard allows a Portfolio Manager to fine-tune assumptions, explore scenarios, and derive actionable insights for optimizing AI value creation within their fund.

## Setting Up Your Development Environment
Duration: 0:05:00

Before you dive into the code, you need to set up your local development environment. This involves installing Python and the necessary libraries.

### Prerequisites

*   **Python 3.8+**: Ensure you have a recent version of Python installed. You can download it from [python.org](https://www.python.org/downloads/).
*   **pip**: Python's package installer, usually comes with Python.

### 1. Create a Virtual Environment (Recommended)

It's a good practice to use a virtual environment to manage dependencies for your project.

```console
python -m venv venv
```

### 2. Activate the Virtual Environment

*   **On Windows:**
    ```console
    .\venv\Scripts\activate
    ```
*   **On macOS/Linux:**
    ```console
    source venv/bin/activate
    ```

### 3. Install Required Libraries

With your virtual environment activated, install all the necessary Python packages using pip.

```console
pip install streamlit pandas numpy plotly warnings
```

<aside class="positive">
The `warnings` library is typically part of Python's standard library and doesn't usually require a separate `pip install`. However, including it in the `pip install` command is harmless and ensures all explicit dependencies are covered.
</aside>

### 4. Save the Application Code

Create a file named `app.py` in your project directory and paste the entire Streamlit application code provided in the problem description into it.

### 5. Run the Streamlit Application

Navigate to your project directory in the terminal (with the virtual environment activated) and run the application:

```console
streamlit run app.py
```

This command will open a new tab in your web browser displaying the Streamlit application. You should see the dashboard with the initial portfolio data loaded.

## Initializing Portfolio Data: The Bedrock for AI Performance Tracking
Duration: 0:15:00

The first step in any analytical journey is to ensure you have robust and relevant data. In this application, we begin by generating synthetic portfolio data that serves as the foundation for all subsequent AI performance assessments. This allows us to simulate a diverse portfolio of companies with various AI readiness indicators and financial metrics.

### Understanding the Data Generation Process

The `load_portfolio_data` function is responsible for creating this synthetic dataset. It simulates key characteristics of portfolio companies over several quarters.

```python
def load_portfolio_data(num_companies=10, num_quarters=5):
    """
    Generates synthetic portfolio data for the specified number of companies and quarters.
    """
    company_names = [f"Company {i+1}" for i in range(num_companies)]
    industries = ['Tech', 'Healthcare', 'Finance', 'Retail', 'Manufacturing']
    quarters = [f"Q{i+1}" for i in range(num_quarters)]

    data = []
    for company_id, company_name in enumerate(company_names):
        industry = np.random.choice(industries)
        baseline_ebitda = np.random.uniform(50, 500) * 1e6 # in millions
        baseline_multiple = np.random.uniform(5, 15) # For exit valuation
        ai_premium_coefficient = np.random.uniform(0.05, 0.25) # For exit valuation

        for q_idx, quarter in enumerate(quarters):
            # Simulate historical data with some trends
            idiosyncratic_readiness = np.random.uniform(30, 90) + q_idx * np.random.uniform(0.5, 2.0)
            systematic_opportunity = np.random.uniform(20, 80) + q_idx * np.random.uniform(0.2, 1.5)
            synergy = np.random.uniform(0.01, 0.05) * (idiosyncratic_readiness * systematic_opportunity) / 100

            ai_investment = np.random.uniform(0.1, 10) * 1e6 # in millions
            ebitda_impact = np.random.uniform(0.5, 8.0) # percentage increase
            gamma_coefficient = np.random.uniform(0.1, 0.5)

            # Exit readiness components
            visible_ai = np.random.uniform(22, 92) + q_idx * np.random.uniform(0.3, 1.8)
            documented_ai = np.random.uniform(25, 95) + q_idx * np.random.uniform(0.4, 2.0)
            sustainable_ai = np.random.uniform(20, 90) + q_idx * np.random.uniform(0.5, 2.2)

            data.append({
                'CompanyID': company_id,
                'CompanyName': company_name,
                'Industry': industry,
                'Quarter': quarter,
                'IdiosyncraticReadiness': np.clip(idiosyncratic_readiness + np.random.normal(0, 5), 0, 100),
                'SystematicOpportunity': np.clip(systematic_opportunity + np.random.normal(0, 5), 0, 100),
                'Synergy': np.clip(synergy + np.random.normal(0, 1), 0, 100),
                'AI_Investment': np.clip(ai_investment + np.random.normal(0, 1e6), 0, 20e6),
                'EBITDA_Impact': np.clip(ebitda_impact + np.random.normal(0, 0.5), 0.1, 15.0),
                'BaselineEBITDA': baseline_ebitda,
                'BaselineMultiple': baseline_multiple,
                'AI_PremiumCoefficient': ai_premium_coefficient,
                'GammaCoefficient': gamma_coefficient,
                'Visible': np.clip(visible_ai + np.random.normal(0, 5), 0, 100),
                'Documented': np.clip(documented_ai + np.random.normal(0, 5), 0, 100),
                'Sustainable': np.clip(sustainable_ai + np.random.normal(0, 5), 0, 100)
            })

    df = pd.DataFrame(data)
    df['Quarter'] = pd.Categorical(df['Quarter'], categories=quarters, ordered=True)
    # ... (initialization of other columns) ...
    return df
```

Key features of the generated data:
*   **Company-specific metrics:** `IdiosyncraticReadiness`, `AI_Investment`, `BaselineEBITDA`, etc.
*   **Industry-specific context:** `SystematicOpportunity`.
*   **Time-series aspect:** Data generated for multiple `Quarter`s.
*   **Exit-related metrics:** `Visible`, `Documented`, `Sustainable` AI capabilities, `BaselineMultiple`, `AI_PremiumCoefficient`.

### Interacting with the Application (Page 1)

In the Streamlit application, navigate to the sidebar on the left.
You'll see:
*   **Number of Portfolio Companies:** Set the number of companies to simulate (e.g., 10).
*   **Number of Quarters (History):** Define how many historical quarters the data should cover (e.g., 5).

<aside class="positive">
The `st.session_state` is crucial here. It stores the `portfolio_df` across different pages and reruns of the application, ensuring data persistence. When you click "Generate New Portfolio Data", `st.session_state.portfolio_df` is updated, and calculation flags (`org_ai_r_recalculated`, `exit_ai_r_recalculated`) are reset to trigger re-computation in subsequent steps.
</aside>

After generating the data, the main panel for "1. Initializing Portfolio Data" displays:
*   **Overview of Generated Portfolio Data:** The first few rows of the DataFrame (`st.dataframe(st.session_state.portfolio_df.head())`).
*   **Descriptive Statistics:** A statistical summary of numerical columns (`st.dataframe(st.session_state.portfolio_df.describe())`).
*   **Data Information:** Details on column names, data types, and non-null values (`st.session_state.portfolio_df.info(buf=buffer)`). This is rendered as `st.text`.

This initial review helps ensure the data's integrity and readiness for analysis, mimicking a Portfolio Manager's essential data quality checks.

## Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment
Duration: 0:20:00

With the foundational data in place, the next crucial step is to quantify the AI maturity of each portfolio company. This is achieved through the Organizational AI Readiness (Org-AI-R) score. This score provides a structured, measurable assessment of a company's ability to leverage AI for value creation, moving beyond subjective evaluations.

### The Org-AI-R Score Formula

The `calculate_org_ai_r` function implements the core logic for this metric.

```python
def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
    """
    Calculates the Organizational AI Readiness (Org-AI-R) score for each company.
    """
    df_copy = df.copy()
    df_copy['Org_AI_R_Score'] = (
        alpha * df_copy['IdiosyncraticReadiness'] +
        (1 - alpha) * df_copy['SystematicOpportunity'] +
        beta * df_copy['Synergy']
    )
    df_copy['Org_AI_R_Score'] = np.clip(df_copy['Org_AI_R_Score'], 0, 100)
    # ... (update industry means/stds) ...
    return df_copy
```

The formula for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:

$$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$

Where:
*   $V^R_{org,j}(t)$: **Idiosyncratic Readiness**. This represents company-specific capabilities at time $t$, such as data infrastructure, AI talent pool, leadership commitment, and internal AI-driven processes. These are factors largely controllable by the company.
*   $H^R_{org,k}(t)$: **Systematic Opportunity**. This captures the industry-level AI potential at time $t$, reflecting broader market adoption rates, disruption potential within the sector, and the competitive AI landscape. These are external factors influencing the company's AI context.
*   $\alpha$: **Weight for Idiosyncratic Readiness**. This parameter allows you to adjust how much importance is placed on a company's internal, controllable AI capabilities ($V^R_{org,j}$) versus the external industry potential ($H^R_{org,k}$). A higher $\alpha$ means prioritizing internal strengths.
*   $\beta$: **Synergy Coefficient**. This coefficient quantifies the additional value derived from the interplay and alignment between a company's idiosyncratic readiness and the systematic opportunity in its industry. It reflects how well a company can capitalize on market potential with its internal capabilities.

### Interacting with the Application (Page 2)

Navigate to "2. Calculating Org-AI-R Scores" in the sidebar.
Here, you'll find interactive sliders:
*   **Weight for Idiosyncratic Readiness ($\alpha$):** Adjust this slider (e.g., from 0.55 to 0.70).
*   **Synergy Coefficient ($\beta$):** Adjust this slider (e.g., from 0.08 to 0.25).

<aside class="negative">
If Org-AI-R scores are not calculated, subsequent pages that depend on this metric (like benchmarking or EBITDA impact) will display warnings. Always ensure calculations are performed in the correct order.
</aside>

Clicking the **"Recalculate Org-AI-R Scores"** button triggers the `calculate_org_ai_r` function with your chosen parameters. The application then displays:
*   **Latest Quarter's PE Org-AI-R Scores:** A table showing the calculated scores for the most recent quarter, sorted to highlight leaders and laggards. This is crucial for a Portfolio Manager's initial assessment of AI maturity across the fund.

This step allows the Portfolio Manager to calibrate the Org-AI-R score calculation to reflect their fund's specific strategic emphasis, whether on internal capabilities or broader market opportunities.

## Benchmarking AI Performance: Identifying Relative AI Standing
Duration: 0:15:00

An absolute Org-AI-R score is informative, but its true value emerges when benchmarked against peers. This step allows you to understand how a company's AI performance stacks up against other holdings in the portfolio and within its specific industry. This comparative analysis helps identify true AI leaders and areas for improvement.

### Benchmarking Metrics

The `calculate_benchmarks` function computes two key metrics:

```python
def calculate_benchmarks(df):
    """
    Calculates percentile rank and Z-score for Org-AI-R scores.
    """
    df_copy = df.copy()
    df_copy['Org_AI_R_Percentile'] = df_copy.groupby('Quarter')['Org_AI_R_Score'].rank(pct=True) * 100
    df_copy['Org_AI_R_Z_Score'] = df_copy.apply(
        lambda row: (row['Org_AI_R_Score'] - row['IndustryMeanOrgAIR']) / row['IndustryStdDevOrgAIR']
        if row['IndustryStdDevOrgAIR'] != 0 else 0, axis=1
    )
    return df_copy
```

1.  **Within-Portfolio Benchmarking (Percentile Rank):**
    $$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$
    This metric indicates a company's standing relative to all other fund holdings. For example, a 90th percentile means it outperforms 90% of its peers within the portfolio. This helps identify internal champions.

2.  **Cross-Portfolio Benchmarking (Industry-Adjusted Z-Score):**
    $$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
    This score shows how much a company's Org-AI-R deviates from its industry's mean ($\mu_k$), in terms of standard deviations ($\sigma_k$). Positive values suggest outperformance relative to industry peers, while negative values signal underperformance. This provides context-specific performance.

### Interacting with the Application (Page 3)

Navigate to "3. Benchmarking AI Performance" in the sidebar.
You will:
*   **Select Quarter for Benchmarking:** Choose a specific quarter (e.g., the latest one) using a `st.selectbox` to focus the analysis.

The page then displays:
*   **Org-AI-R Benchmarks Table:** A table showing `CompanyName`, `Industry`, `Org_AI_R_Score`, `Org_AI_R_Percentile`, and `Org_AI_R_Z_Score` for the selected quarter.
*   **Bar Chart: Org-AI-R Scores by Company:** Visualizes individual company Org-AI-R scores, with an overlaid red dashed line representing the portfolio average. This uses `plotly.express.bar`.

    ```python
    # Example plot code snippet
    fig1 = px.bar(...)
    fig1.add_hline(y=portfolio_average_org_ai_r, line_dash="dash", line_color="red", annotation_text=f"Portfolio Average: {portfolio_average_org_ai_r:.2f}")
    st.plotly_chart(fig1, use_container_width=True)
    ```

*   **Scatter Plot: Org-AI-R Score vs. Industry-Adjusted Z-Score:** This plot uses `plotly.express.scatter` to visualize relative performance.
    *   `x-axis`: `Org_AI_R_Score`
    *   `y-axis`: `Org_AI_R_Z_Score`
    *   `color`: `Industry`
    *   `size`: `Org_AI_R_Percentile` (larger points indicate higher within-portfolio ranking).
    *   It includes vertical and horizontal lines representing the portfolio mean Org-AI-R and industry mean Z-score, respectively.

    ```python
    # Example plot code snippet
    fig2 = px.scatter(...)
    fig2.add_vline(x=portfolio_mean_org_ai_r, line_dash="dash", line_color="orange", annotation_text=f"Portfolio Mean Org-AI-R: {portfolio_mean_org_ai_r:.2f}")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Industry Mean Z-Score (0)")
    st.plotly_chart(fig2, use_container_width=True)
    ```

This visual and numerical benchmarking allows a Portfolio Manager to quickly identify which companies are truly excelling in AI within their context and which might need focused attention.

## AI Investment Efficiency and EBITDA Impact: Quantifying Financial Returns
Duration: 0:20:00

Quantifying the financial impact of AI investments is paramount for a Portfolio Manager. This step assesses how efficiently companies convert their AI expenditures into real business value, specifically in terms of EBITDA growth. This provides critical insights into capital deployment strategies and highlights which companies are generating the most value from their AI spend.

### Core Financial Impact Metrics

The `calculate_aie_ebitda` function computes these crucial metrics.

```python
def calculate_aie_ebitda(df):
    """
    Calculates AI Investment Efficiency and Attributed EBITDA Impact.
    """
    df_sorted = df.sort_values(by=['CompanyID', 'Quarter'])
    df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID')['Org_AI_R_Score'].diff().fillna(0)

    # AI Investment Efficiency (AIE_j)
    df_sorted['AI_Investment_Efficiency'] = df_sorted.apply(
        lambda row: (row['Delta_Org_AI_R'] * row['EBITDA_Impact']) / (row['AI_Investment'] / 1e6)
        if row['AI_Investment'] > 0 and row['Delta_Org_AI_R'] > 0 and row['EBITDA_Impact'] > 0 else 0, axis=1
    )
    df_sorted.loc[df_sorted['Delta_Org_AI_R'] <= 0, 'AI_Investment_Efficiency'] = 0

    # Attributed EBITDA Impact (%)
    df_sorted['Attributed_EBITDA_Impact_Pct'] = df_sorted.apply(
        lambda row: row['GammaCoefficient'] * row['Delta_Org_AI_R'] * row['SystematicOpportunity'] / 100
        if row['Delta_Org_AI_R'] > 0 else 0, axis=1
    )
    df_sorted['Attributed_EBITDA_Impact_Absolute'] = (df_sorted['Attributed_EBITDA_Impact_Pct'] / 100) * df_sorted['BaselineEBITDA']
    return df_sorted
```

1.  **AI Investment Efficiency ($\text{AIE}_j$):**
    $$ \text{AIE}_j = \frac{\Delta\text{Org-AI-R}_j \cdot \text{EBITDA Impact}_j}{\text{AI Investment}_j \text{ (in millions)}} $$
    This metric measures the combined impact (Org-AI-R points and baseline EBITDA Impact percentage) generated per million dollars of AI investment. A higher AIE indicates more efficient capital deployment for AI initiatives. `$\Delta\text{Org-AI-R}_j$` represents the change in Org-AI-R score from the previous quarter, and `EBITDA Impact_j` is the baseline percentage increase in EBITDA.

2.  **Attributed EBITDA Impact ($\Delta\text{EBITDA}\%$):**
    $$ \Delta\text{EBITDA}\% = \text{GammaCoefficient} \cdot \Delta\text{Org-AI-R} \cdot H^R_{org,k}/100 $$
    This is the estimated percentage increase in EBITDA directly attributed to the change in a company's Org-AI-R score, factoring in its industry's systematic opportunity ($H^R_{org,k}$) and a `GammaCoefficient`. The `GammaCoefficient` acts as a scaling factor, reflecting the sensitivity of EBITDA to AI readiness changes.

### Interacting with the Application (Page 4)

Navigate to "4. AI Investment & EBITDA Impact" in the sidebar.
The page automatically calculates and displays these metrics for the latest quarter:
*   **AI Investment Efficiency and Attributed EBITDA Impact Table:** A table listing `CompanyName`, `AI_Investment`, `Delta_Org_AI_R`, `AI_Investment_Efficiency`, `Attributed_EBITDA_Impact_Pct`, and `Attributed_EBITDA_Impact_Absolute`.
*   **Scatter Plot: AI Investment vs. Efficiency (Highlighting EBITDA Impact):** This plot helps visualize the trade-offs.
    *   `x-axis`: `AI_Investment_M` (AI Investment in Millions, using `log_x=True` for better visualization of wide ranges).
    *   `y-axis`: `AI_Investment_Efficiency`.
    *   `color`: `Industry`.
    *   `size`: `Attributed_EBITDA_Impact_Pct` (larger points indicate greater attributed EBITDA impact).

    ```python
    # Example plot code snippet
    fig = px.scatter(
        latest_quarter_df,
        x='AI_Investment_M',
        y='AI_Investment_Efficiency',
        color='Industry',
        size='Attributed_EBITDA_Impact_Pct',
        log_x=True
        # ... other parameters ...
    )
    st.plotly_chart(fig, use_container_width=True)
    ```

This visualization allows a Portfolio Manager to identify companies that are highly efficient with lower investments, those generating significant financial uplift, or those achieving both, informing capital allocation decisions.

## Tracking Progress Over Time: Visualizing Trajectories
Duration: 0:15:00

Current metrics offer a snapshot, but understanding the historical trajectory of a portfolio company's AI performance is crucial for assessing the effectiveness of long-term strategies and identifying sustainable trends. This step provides time-series visualizations of key AI performance indicators.

### Monitoring Trends

This section re-uses the already calculated metrics (`Org_AI_R_Score` and `AI_Investment_Efficiency`) and presents them in a time-series context. There isn't a new calculation function for this specific page; instead, it aggregates and plots existing data.

<aside class="positive">
Tracking progress over time is essential for strategic planning. It helps a Portfolio Manager understand if interventions are working, if companies are maintaining their competitive edge, or if new challenges are emerging.
</aside>

### Interacting with the Application (Page 5)

Navigate to "5. Tracking Progress Over Time" in the sidebar.
You will:
*   **Select Companies to Track:** Use a `st.multiselect` widget to choose up to 5 companies for clearer visualization of their trends.

The page then displays two line charts:

1.  **Org-AI-R Score Trajectory Over Time:**
    *   This `plotly.express.line` chart shows the `Org_AI_R_Score` for selected companies across quarters.
    *   It overlays a line representing the **"Portfolio Average"** Org-AI-R score for context.
    *   Markers are used for individual data points to highlight specific quarter values.

    ```python
    # Example plot code snippet
    plot_df_org_ai_r = pd.concat([df_filtered_companies[['Quarter', 'Org_AI_R_Score', 'CompanyName']], portfolio_avg_org_ai_r])
    fig1 = px.line(
        plot_df_org_ai_r,
        x='Quarter',
        y='Org_AI_R_Score',
        color='CompanyName',
        markers=True,
        title='Org-AI-R Score Trajectory Over Time'
        # ... other parameters ...
    )
    st.plotly_chart(fig1, use_container_width=True)
    ```

2.  **AI Investment Efficiency Trajectory Over Time:**
    *   Similarly, this `plotly.express.line` chart visualizes the `AI_Investment_Efficiency` for selected companies.
    *   It also includes an overlaid **"Portfolio Average"** line for comparative analysis.

    ```python
    # Example plot code snippet
    plot_df_aie = pd.concat([df_filtered_companies[['Quarter', 'AI_Investment_Efficiency', 'CompanyName']], portfolio_avg_aie])
    fig2 = px.line(
        plot_df_aie,
        x='Quarter',
        y='AI_Investment_Efficiency',
        color='CompanyName',
        markers=True,
        title='AI Investment Efficiency Trajectory Over Time'
        # ... other parameters ...
    )
    st.plotly_chart(fig2, use_container_width=True)
    ```

These visualizations help a Portfolio Manager spot consistent improvers, decliners, or companies that deviate significantly from the average, informing decisions on which companies warrant deeper investigation or targeted support.

## Actionable Insights: Centers of Excellence & Companies for Review
Duration: 0:20:00

A critical responsibility of a Portfolio Manager is to translate analytical insights into actionable strategies. This step allows for the strategic segmentation of the portfolio, identifying high-performing 'Centers of Excellence' to scale best practices, and 'Companies for Review' that require immediate strategic intervention.

### Identifying Action Categories

The `identify_actionable_insights` function applies user-defined thresholds to categorize companies.

```python
def identify_actionable_insights(df, org_ai_r_threshold_coe=75, ebitda_impact_threshold_coe=3,
                                 org_ai_r_threshold_review=50, ebitda_impact_threshold_review=1.0):
    """
    Identifies Centers of Excellence and Companies for Review based on thresholds.
    """
    latest_data = df.loc[df.groupby('CompanyID')['Quarter'].idxmax()].copy()

    # Centers of Excellence
    centers_of_excellence = latest_data[
        (latest_data['Org_AI_R_Score'] > org_ai_r_threshold_coe) &
        (latest_data['Attributed_EBITDA_Impact_Pct'] > ebitda_impact_threshold_coe)
    ].sort_values(by='Org_AI_R_Score', ascending=False)

    # Companies for Review
    companies_for_review = latest_data[
        (latest_data['Org_AI_R_Score'] <= org_ai_r_threshold_review) |
        (latest_data['Attributed_EBITDA_Impact_Pct'] <= ebitda_impact_threshold_review)
    ].sort_values(by='Org_AI_R_Score', ascending=True)

    return centers_of_excellence, companies_for_review
```

*   **Centers of Excellence (CoE):** Companies demonstrating both high `Org-AI-R Scores` (above `coe_org_ai_r_threshold_coe`) and significant `Attributed_EBITDA_Impact_Pct` (above `coe_ebitda_threshold_coe`). These companies are benchmarks for best practices.
*   **Companies for Review:** Companies with lower `Org-AI_R_Scores` (below or equal to `review_org_ai_r_threshold_review`) OR minimal `Attributed_EBITDA_Impact_Pct` (below or equal to `review_ebitda_threshold_review`). These indicate areas requiring strategic intervention.

### Interacting with the Application (Page 6)

Navigate to "6. Actionable Insights: CoE & Review" in the sidebar.
You will find several `st.slider` widgets to define your strategic thresholds:
*   **Org-AI-R Score Threshold for Center of Excellence:** (e.g., default 75)
*   **Attributed EBITDA Impact (%) Threshold for Center of Excellence:** (e.g., default 3.0%)
*   **Org-AI-R Score Threshold for Companies for Review:** (e.g., default 50)
*   **Attributed EBITDA Impact (%) Threshold for Companies for Review:** (e.g., default 1.0%)

Clicking **"Re-evaluate Actionable Insights"** will re-segment the companies based on the new thresholds.

The page then displays:
*   **Centers of Excellence Table:** Lists companies meeting the CoE criteria.
*   **Companies for Review Table:** Lists companies meeting the review criteria.
*   **Scatter Plot: Portfolio AI Performance Segmentation:** This interactive `plotly.express.scatter` plot visually segments the portfolio.
    *   `x-axis`: `Org_AI_R_Score`
    *   `y-axis`: `Attributed_EBITDA_Impact_Pct`
    *   `color`: `Category` (CoE, Company for Review, Normal).
    *   `size`: `AI_Investment_Efficiency`.
    *   Dynamically adjusting vertical and horizontal lines represent the defined thresholds for both CoE and Review categories.
    *   Specific markers and text labels (`go.Scatter` with `mode='text'`) are added for CoE and Review companies, making them immediately identifiable.

    ```python
    # Example plot code snippet for segmentation
    fig = px.scatter(
        latest_quarter_df,
        x='Org_AI_R_Score',
        y='Attributed_EBITDA_Impact_Pct',
        color='Category',
        size='AI_Investment_Efficiency',
        # ... other parameters ...
    )
    # Add dynamic threshold lines
    fig.add_vline(x=coe_org_ai_r_threshold_val, line_dash="dash", line_color="green", annotation_text=f"CoE Org-AI-R > {coe_org_ai_r_threshold_val}")
    fig.add_hline(y=coe_ebitda_threshold_val, line_dash="dash", line_color="green", annotation_text=f"CoE EBITDA > {coe_ebitda_threshold_val}%")
    # ... add lines for review thresholds ...

    # Add text labels for identified companies
    for _, row in centers_of_excellence.iterrows():
        fig.add_trace(go.Scatter(mode='markers+text', text=[row['CompanyName']], ...))
    # ... similar for companies_for_review ...

    st.plotly_chart(fig, use_container_width=True)
    ```

This segmentation provides immediate strategic clarity, enabling a Portfolio Manager to allocate resources, provide targeted support, or scale best practices across the fund.

## Evaluating Exit-Readiness and Potential Valuation Impact
Duration: 0:20:00

For a Private Equity fund, preparing for a successful exit is a paramount concern. AI capabilities can significantly enhance a company's attractiveness to potential acquirers and consequently impact its valuation multiple. This final analytical step assesses a company's 'buyer-friendly' AI assets and their contribution to its projected exit multiple.

### Exit-AI-R Score and Valuation Model

The `calculate_exit_readiness_and_valuation` function is responsible for these calculations.

```python
def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
    """
    Calculates Exit-AI-R Score and Projected Exit Multiple.
    """
    df_copy = df.copy()
    df_copy['Exit_AI_R_Score'] = (
        w1 * df_copy['Visible'] + w2 * df_copy['Documented'] + w3 * df_copy['Sustainable']
    )
    df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100)

    # Multiple_j = Multiple_base,k + AI Premium Coefficient * Exit-AI-R_j/100
    df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
    df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
    return df_copy
```

1.  **Exit-AI-R Score:**
    $$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$
    This score quantifies how well a company's AI capabilities are positioned to attract buyers and command a premium.
    *   $\text{Visible}_j$: **Visible AI Capabilities** (e.g., product features, tech stack).
    *   $\text{Documented}_j$: **Documented AI Impact** (e.g., ROI reports, IP).
    *   $\text{Sustainable}_j$: **Sustainable AI Capabilities** (e.g., talent pipeline, scalable infrastructure).
    *   $w_1, w_2, w_3$: **Weighting Factors**. These configurable weights prioritize different aspects of AI capability that buyers might value most.

2.  **Multiple Attribution Model (Projected Exit Multiple):**
    $$ \text{Projected Exit Multiple}_j = \text{BaselineMultiple}_{k} + \text{AI Premium Coefficient} \cdot \text{Exit-AI-R}_j/100 $$
    This model translates the `Exit-AI-R Score` into a potential uplift on the company's `BaselineMultiple`. The `AI Premium Coefficient` acts as a sensitivity factor, showing how much the exit multiple is boosted by a higher Exit-AI-R score.

### Interacting with the Application (Page 7)

Navigate to "7. Exit-Readiness & Valuation" in the sidebar.
You will find `st.slider` widgets to adjust the weighting factors for the Exit-AI-R score:
*   **Weight for Visible AI Capabilities ($w_1$):** (e.g., default 0.35)
*   **Weight for Documented AI Impact ($w_2$):** (e.g., default 0.40)
*   **Weight for Sustainable AI Capabilities ($w_3$):** (e.g., default 0.25)

Clicking **"Recalculate Exit-Readiness & Valuation"** re-computes the scores and projected multiples based on your adjusted weights.

The page then displays:
*   **Latest Quarter's Exit-Readiness and Projected Valuation Impact Table:** A table showing `CompanyName`, `Exit_AI_R_Score`, `BaselineMultiple`, `AI_Premium_Multiple_Additive`, and `Projected_Exit_Multiple`.
*   **Scatter Plot: Exit-AI-R Score vs. Projected Exit Multiple:** This `plotly.express.scatter` plot visualizes the relationship.
    *   `x-axis`: `Exit_AI_R_Score`
    *   `y-axis`: `Projected_Exit_Multiple`
    *   `color`: `Industry`
    *   `size`: `Attributed_EBITDA_Impact_Pct` (showing companies with strong AI readiness and financial performance).
    *   Company names are added as text labels using `go.Scatter` with `mode='text'`.

    ```python
    # Example plot code snippet
    fig = px.scatter(
        latest_quarter_df,
        x='Exit_AI_R_Score',
        y='Projected_Exit_Multiple',
        color='Industry',
        size='Attributed_EBITDA_Impact_Pct',
        # ... other parameters ...
    )
    # Add company names as text labels
    for _, row in latest_quarter_df.iterrows():
        fig.add_trace(go.Scatter(mode='text', text=[row['CompanyName']], ...))
    st.plotly_chart(fig, use_container_width=True)
    