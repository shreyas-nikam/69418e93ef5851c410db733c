
# Portfolio AI Performance & Benchmarking Dashboard

As a **Portfolio Manager** or **Quant Analyst**, overseeing a fund with diverse portfolio companies, it's critical to have a systematic way to track, compare, and benchmark the AI maturity and value creation progress of each holding. This notebook provides a hands-on workflow to achieve that, moving beyond mere theoretical understanding to practical application. We will load data, calculate key AI readiness metrics, benchmark performance, track progress over time, and identify actionable insights to optimize our fund's overall AI strategy and drive superior risk-adjusted returns.

## 0. Setup: Installing Libraries and Importing Dependencies

Before we begin our analysis, we need to ensure all necessary libraries are installed and imported. These tools will enable us to load and manipulate data, perform complex calculations, and visualize our findings effectively.

### Installing Required Libraries

```python
!pip install pandas numpy scipy matplotlib seaborn tabulate
```

### Importing Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, zscore
from tabulate import tabulate # For clean table printing
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Libraries installed and imported successfully!")
```

## 1. Initializing the Portfolio Overview: Data Loading for AI Performance Tracking

As a Portfolio Manager, my first step is always to gather the relevant data for my fund's holdings. To systematically track, compare, and benchmark the AI maturity and value creation across my portfolio, I need to load a comprehensive dataset. This dataset will include historical and current PE Org-AI-R scores, AI investment data, and documented EBITDA impacts for each portfolio company. This initial data load forms the bedrock of all subsequent analyses and strategic decisions.

```python
def load_portfolio_data(num_companies=10, num_quarters=5):
    """
    Generates synthetic portfolio data for AI performance and benchmarking.
    """
    np.random.seed(42) # for reproducibility
    
    company_ids = [f'C{i:02d}' for i in range(1, num_companies + 1)]
    company_names = [f'Company {i}' for i in range(1, num_companies + 1)]
    
    industries = ['Manufacturing', 'Healthcare', 'Retail', 'Business Services', 'Technology']
    
    data = []
    for company_id, company_name in zip(company_ids, company_names):
        industry = np.random.choice(industries)
        
        # Industry-specific Systematic Opportunity (HR_org,k)
        if industry == 'Manufacturing':
            H_R_org_k_base = 72
        elif industry == 'Healthcare':
            H_R_org_k_base = 78
        elif industry == 'Retail':
            H_R_org_k_base = 75
        elif industry == 'Business Services':
            H_R_org_k_base = 80
        elif industry == 'Technology':
            H_R_org_k_base = 85
        else:
            H_R_org_k_base = 70 # Default

        # Baseline Multiple
        if industry == 'Manufacturing':
            baseline_multiple = 6.5
        elif industry == 'Healthcare':
            baseline_multiple = 8.0
        elif industry == 'Retail':
            baseline_multiple = 7.0
        elif industry == 'Business Services':
            baseline_multiple = 9.0
        elif industry == 'Technology':
            baseline_multiple = 12.0
        else:
            baseline_multiple = 7.5

        # AI Premium Coefficient delta
        ai_premium_coeff = np.random.uniform(1.0, 3.0)

        # Gamma for EBITDA attribution
        gamma_coeff = np.random.uniform(0.02, 0.05)
        
        current_idiosyncratic_readiness = np.random.uniform(30, 80)
        current_synergy = np.random.uniform(40, 90) # percentage units
        current_ebitda_impact = np.random.uniform(0.5, 5.0) # initial % impact

        for q in range(1, num_quarters + 1):
            quarter = f'Q{q}'
            
            # Simulate progress over time
            idiosyncratic_readiness = current_idiosyncratic_readiness + np.random.uniform(-5, 8)
            idiosyncratic_readiness = np.clip(idiosyncratic_readiness, 0, 100)
            
            systematic_opportunity = H_R_org_k_base + np.random.uniform(-3, 3)
            systematic_opportunity = np.clip(systematic_opportunity, 0, 100)

            synergy = current_synergy + np.random.uniform(-5, 5)
            synergy = np.clip(synergy, 0, 100)
            
            ai_investment = np.random.uniform(500000, 5000000)
            
            ebitda_impact = current_ebitda_impact + np.random.uniform(-0.5, 1.0)
            ebitda_impact = np.clip(ebitda_impact, 0.0, 10.0) # percentage
            
            baseline_ebitda = np.random.uniform(10_000_000, 100_000_000)

            # Exit readiness components
            visible = np.random.uniform(30, 90)
            documented = np.random.uniform(30, 90)
            sustainable = np.random.uniform(30, 90)

            data.append([
                company_id, company_name, industry, quarter,
                idiosyncratic_readiness, systematic_opportunity, synergy,
                ai_investment, ebitda_impact, baseline_ebitda,
                H_R_org_k_base, H_R_org_k_base * 0.15, # Industry Mean/StdDev placeholder for Z-score
                visible, documented, sustainable, baseline_multiple, ai_premium_coeff, gamma_coeff
            ])
            
            current_idiosyncratic_readiness = idiosyncratic_readiness
            current_synergy = synergy
            current_ebitda_impact = ebitda_impact

    df = pd.DataFrame(data, columns=[
        'CompanyID', 'CompanyName', 'Industry', 'Quarter',
        'IdiosyncraticReadiness', 'SystematicOpportunity', 'Synergy',
        'AI_Investment', 'EBITDA_Impact', 'BaselineEBITDA',
        'IndustryMeanOrgAIR', 'IndustryStdDevOrgAIR',
        'Visible', 'Documented', 'Sustainable',
        'BaselineMultiple', 'AI_PremiumCoefficient', 'GammaCoefficient'
    ])
    
    # Simulate IndustryMeanOrgAIR and IndustryStdDevOrgAIR more realistically
    # This will be updated later when Org-AI-R is calculated
    df['IndustryMeanOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('mean')
    df['IndustryStdDevOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('std').fillna(5) # fillna for single-company industries

    # Save to CSV
    df.to_csv('portfolio_data.csv', index=False)
    return df

# Load the data
portfolio_df = load_portfolio_data(num_companies=10, num_quarters=5)

# Display the first few rows and information about the dataset
print("Portfolio Data Head:")
print(portfolio_df.head())
print("\nPortfolio Data Info:")
portfolio_df.info()
```

## 2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment

The core of our AI performance tracking is the PE Org-AI-R score. As a Portfolio Manager, I need to compute this score for each company and for each quarter to understand their current AI maturity and how it evolves over time. This score is a parametric framework that objectively assesses organizational AI readiness, allowing for systematic comparison across diverse portfolio holdings. It quantifies enterprise AI opportunity by combining organization-specific capabilities (Idiosyncratic Readiness), industry-level AI potential (Systematic Opportunity), and the synergy between them.

The formula for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:

$$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$

where:
*   $V^R_{org,j}(t)$ represents Idiosyncratic Readiness, reflecting the organization-specific capabilities of company $j$ at time $t$. This is normalized to $[0, 100]$.
*   $H^R_{org,k}(t)$ denotes Systematic Opportunity, representing the industry-level AI potential for industry $k$ at time $t$. This is also normalized to $[0, 100]$.
*   $\alpha \in [0.55, 0.70]$ is the weight assigned to organizational factors ($V^R_{org,j}$) versus market factors ($H^R_{org,k}$). We will use a default value of $0.6$.
*   $\beta \in [0.08, 0.25]$ is the Synergy coefficient, quantifying the additional value derived from the interplay between idiosyncratic readiness and systematic opportunity. We will use a default value of $0.15$.
*   $\text{Synergy}(V^R_{org,j}, H^R_{org,k})$ is a percentage unit in $[0, 100]$ representing the integration and alignment of idiosyncratic readiness with systematic opportunity.

```python
def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
    """
    Calculates the PE Org-AI-R Score for each company.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'IdiosyncraticReadiness',
                       'SystematicOpportunity', and 'Synergy' columns.
    alpha (float): Weight for Idiosyncratic Readiness (alpha in [0.55, 0.70]).
    beta (float): Synergy coefficient (beta in [0.08, 0.25]).

    Returns:
    pd.DataFrame: DataFrame with 'Org_AI_R_Score' column added.
    """
    if not (0.55 <= alpha <= 0.70):
        raise ValueError("Alpha must be between 0.55 and 0.70")
    if not (0.08 <= beta <= 0.25):
        raise ValueError("Beta must be between 0.08 and 0.25")

    df['Org_AI_R_Score'] = (
        alpha * df['IdiosyncraticReadiness'] +
        (1 - alpha) * df['SystematicOpportunity'] +
        beta * df['Synergy']
    )
    # Ensure scores are within a reasonable range, e.g., 0-100 as per documentation
    df['Org_AI_R_Score'] = np.clip(df['Org_AI_R_Score'], 0, 100)
    return df

# Apply the function to calculate Org-AI-R scores
portfolio_df = calculate_org_ai_r(portfolio_df, alpha=0.6, beta=0.15)

# Display the latest quarter's Org-AI-R scores for all companies
latest_quarter_df = portfolio_df.loc[portfolio_df.groupby('CompanyID')['Quarter'].idxmax()]
print("\nLatest Quarter's PE Org-AI-R Scores:")
print(tabulate(latest_quarter_df[['CompanyName', 'Industry', 'Org_AI_R_Score']].sort_values(by='Org_AI_R_Score', ascending=False),
               headers='keys', tablefmt='pipe', showindex=False))
```

This table gives me a quick overview of the current AI readiness across my portfolio. The `Org_AI_R_Score` acts as a standardized metric, allowing me to quantitatively assess each company's position relative to others, which is invaluable for initial strategic framing.

## 3. Benchmarking Portfolio Companies: Identifying Relative AI Performance

Understanding a company's standalone Org-AI-R score is a good start, but as a Quant Analyst, I need to know how each company performs relative to its peers. Benchmarking is crucial for identifying leaders, laggards, and setting realistic targets. I will compute two types of benchmarks: "within-portfolio" percentile rankings and "cross-portfolio" industry-adjusted Z-scores.

*   **Within-Portfolio Benchmarking (Definition 6):** This shows a company's standing relative to all other companies within our fund's portfolio.
    $$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{PortfolioSize}} \right) \times 100 $$
    Here, $\text{Rank}(\text{Org-AI-R}_j)$ is the rank of company $j$'s Org-AI-R score among all companies in the portfolio (lower rank for lower score, higher rank for higher score), and $\text{PortfolioSize}$ is the total number of companies in the portfolio.

*   **Cross-Portfolio Benchmarking (Definition 6):** This adjusts for industry-specific AI potential, comparing a company to its industry peers based on normalized scores.
    $$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
    Here, $\text{Org-AI-R}_j$ is the score for company $j$, $\mu_k$ is the mean Org-AI-R score for industry $k$, and $\sigma_k$ is the standard deviation of Org-AI-R scores for industry $k$. My synthetic data already contains `IndustryMeanOrgAIR` and `IndustryStdDevOrgAIR` to simulate these industry-specific benchmarks.

```python
def calculate_benchmarks(df):
    """
    Calculates within-portfolio percentile rankings and cross-portfolio Z-scores.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Org_AI_R_Score', 'Industry',
                       'IndustryMeanOrgAIR', and 'IndustryStdDevOrgAIR' columns.

    Returns:
    pd.DataFrame: DataFrame with 'Org_AI_R_Percentile' and 'Org_AI_R_Z_Score' columns added.
    """
    # Calculate within-portfolio percentile for each quarter
    df['Org_AI_R_Percentile'] = df.groupby('Quarter')['Org_AI_R_Score'].rank(pct=True) * 100

    # Calculate cross-portfolio (industry-adjusted) Z-score
    # Using existing IndustryMeanOrgAIR and IndustryStdDevOrgAIR from the synthetic data
    # Ensure standard deviation is not zero to avoid division errors
    df['Org_AI_R_Z_Score'] = df.apply(
        lambda row: (row['Org_AI_R_Score'] - row['IndustryMeanOrgAIR']) / row['IndustryStdDevOrgAIR']
        if row['IndustryStdDevOrgAIR'] != 0 else 0, axis=1
    )
    return df

# Apply the function to calculate benchmarks
portfolio_df = calculate_benchmarks(portfolio_df)

# Display the latest quarter's benchmarks
latest_quarter_df = portfolio_df.loc[portfolio_df.groupby('CompanyID')['Quarter'].idxmax()]
print("\nLatest Quarter's Org-AI-R Benchmarks:")
print(tabulate(latest_quarter_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Org_AI_R_Percentile', 'Org_AI_R_Z_Score']].sort_values(by='Org_AI_R_Score', ascending=False),
               headers='keys', tablefmt='pipe', showindex=False))

# Visualization: Bar chart of Org-AI-R scores with Z-score indication
plt.figure(figsize=(12, 7))
sns.barplot(x='CompanyName', y='Org_AI_R_Score', hue='Industry', data=latest_quarter_df.sort_values(by='Org_AI_R_Score', ascending=False), palette='viridis')
plt.axhline(y=latest_quarter_df['Org_AI_R_Score'].mean(), color='r', linestyle='--', label='Portfolio Average Org-AI-R')
plt.title('Latest Quarter Org-AI-R Scores by Company (with Portfolio Average)')
plt.xlabel('Company Name')
plt.ylabel('Org-AI-R Score')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Industry')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 7))
sns.scatterplot(x='Org_AI_R_Score', y='Org_AI_R_Z_Score', hue='Industry', size='Org_AI_R_Percentile', sizes=(50, 400),
                data=latest_quarter_df, palette='coolwarm', legend='full')
plt.axvline(x=latest_quarter_df['Org_AI_R_Score'].mean(), color='gray', linestyle='--', label='Portfolio Mean Org-AI-R')
plt.axhline(y=0, color='gray', linestyle='-', label='Industry Mean Z-Score')
plt.title('Org-AI-R Score vs. Industry-Adjusted Z-Score (Latest Quarter)')
plt.xlabel('Org-AI-R Score')
plt.ylabel('Org-AI-R Z-Score (Industry-Adjusted)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

These benchmarks are invaluable. The `Org_AI_R_Percentile` tells me a company's relative standing within our own fund, while the `Org_AI_R_Z_Score` normalizes for industry-specific opportunities. Companies with high Z-scores are truly outperforming their industry, signaling potential best practices. Conversely, negative Z-scores or low percentiles highlight areas where deeper review and intervention might be needed. This data drives my strategic resource allocation decisions.

## 4. Assessing AI Investment Efficiency and EBITDA Attribution

As a Portfolio Manager, I need to go beyond just scores and understand the financial impact of AI initiatives. It's not enough for companies to improve their Org-AI-R scores; those improvements must translate into tangible value. This section focuses on two key metrics: **AI Investment Efficiency (AIE)** and **EBITDA Attribution**.

**AI Investment Efficiency (AIE)** helps me evaluate how effectively capital investments in AI translate into capability growth and financial impact. A higher AIE indicates a more efficient use of AI investment dollars.
The formula for AI Investment Efficiency for company $j$ over a period $T$ is:

$$ \text{AIE}_j = \frac{\Delta\text{Org-AI-R}_j}{\text{AI Investment}_j} \times \text{EBITDA Impact}_j $$

Here, $\Delta\text{Org-AI-R}_j$ is the change in Org-AI-R score for company $j$ over the period, $\text{AI Investment}_j$ is the total AI investment during that period, and $\text{EBITDA Impact}_j$ is the percentage increase in EBITDA attributed to AI initiatives.

**EBITDA Attribution (%)** directly quantifies the financial upside. I use a model that links Org-AI-R improvement to EBITDA enhancement, calibrated by a value creation coefficient $\gamma$.
The formula for EBITDA Attribution percentage is:

$$ \Delta\text{EBITDA}\% = \gamma \cdot \Delta\text{Org-AI-R} \cdot H^R_{org,k}/100 $$

Here, $\gamma$ is the value creation coefficient (estimated from historical data, typically $\gamma \in [0.02, 0.05]$), $\Delta\text{Org-AI-R}$ is the change in Org-AI-R score, and $H^R_{org,k}$ is the systematic opportunity for the company's industry.

```python
def calculate_aie_ebitda(df):
    """
    Calculates AI Investment Efficiency and Attributed EBITDA Impact.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'CompanyID', 'Quarter', 'Org_AI_R_Score',
                       'AI_Investment', 'EBITDA_Impact', 'BaselineEBITDA',
                       'IndustryMeanOrgAIR', and 'GammaCoefficient' columns.

    Returns:
    pd.DataFrame: DataFrame with 'Delta_Org_AI_R', 'AI_Investment_Efficiency',
                  'Attributed_EBITDA_Impact_Pct', and 'Attributed_EBITDA_Impact_Absolute' columns added.
    """
    df_sorted = df.sort_values(by=['CompanyID', 'Quarter'])

    # Calculate Delta Org-AI-R
    df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID')['Org_AI_R_Score'].diff().fillna(0)

    # Calculate AI Investment Efficiency
    # Handle potential division by zero for AI_Investment
    df_sorted['AI_Investment_Efficiency'] = df_sorted.apply(
        lambda row: (row['Delta_Org_AI_R'] / row['AI_Investment']) * row['EBITDA_Impact'] * 1000000 # Scaling for readability (per $M investment)
        if row['AI_Investment'] > 0 and row['Delta_Org_AI_R'] > 0 else 0,
        axis=1
    )
    
    # Calculate Attributed EBITDA Impact %
    # Using IndustryMeanOrgAIR as a proxy for H_R_org,k and GammaCoefficient from synthetic data
    df_sorted['Attributed_EBITDA_Impact_Pct'] = df_sorted.apply(
        lambda row: row['GammaCoefficient'] * row['Delta_Org_AI_R'] * row['IndustryMeanOrgAIR'] / 100
        if row['Delta_Org_AI_R'] > 0 else 0,
        axis=1
    )
    
    # Calculate Attributed EBITDA Impact Absolute
    df_sorted['Attributed_EBITDA_Impact_Absolute'] = (df_sorted['Attributed_EBITDA_Impact_Pct'] / 100) * df_sorted['BaselineEBITDA']
    
    return df_sorted

# Apply the function
portfolio_df = calculate_aie_ebitda(portfolio_df)

# Display the latest quarter's AIE and EBITDA attribution
latest_quarter_df = portfolio_df.loc[portfolio_df.groupby('CompanyID')['Quarter'].idxmax()]
print("\nLatest Quarter's AI Investment Efficiency and Attributed EBITDA Impact:")
print(tabulate(latest_quarter_df[['CompanyName', 'Industry', 'Delta_Org_AI_R', 'AI_Investment', 'AI_Investment_Efficiency', 'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute']]
               .sort_values(by='AI_Investment_Efficiency', ascending=False),
               headers='keys', tablefmt='pipe', floatfmt=(".2f", ".2f", ".2f", ",.0f", ".2f", ".2f", ",.0f"), showindex=False))

# Visualization: Scatter plot of AI Investment vs. Efficiency
plt.figure(figsize=(12, 7))
sns.scatterplot(x='AI_Investment', y='AI_Investment_Efficiency', hue='Industry', size='Attributed_EBITDA_Impact_Pct', sizes=(50, 500),
                data=latest_quarter_df[latest_quarter_df['AI_Investment_Efficiency'] > 0], palette='plasma', legend='full')
plt.title('AI Investment vs. Efficiency (Latest Quarter, Highlighting EBITDA Impact)')
plt.xlabel('AI Investment ($)')
plt.ylabel('AI Investment Efficiency (Points per $M investment)')
plt.xscale('log') # Use log scale for investment for better distribution
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

This analysis provides critical insights into the financial returns of our AI initiatives. Companies with high `AI_Investment_Efficiency` are effectively leveraging their investments, which might indicate a robust implementation strategy or a strong foundational AI capability. The `Attributed_EBITDA_Impact_Absolute` allows me to directly quantify the monetary value generated by AI improvements, enabling me to justify continued investment and prioritize high-impact projects.

## 5. Tracking Progress Over Time: Visualizing Trajectories

As a Portfolio Manager, current metrics are important, but understanding the trajectory of performance over time is essential for strategic decision-making. I need to observe how Org-AI-R scores and AI Investment Efficiency evolve for individual companies and the entire fund. This time-series analysis allows me to track quarterly progress, identify sustained improvements or declines, and assess the long-term impact of our AI strategies.

```python
def plot_time_series(df, metric_col, title, ylabel):
    """
    Generates line charts for a given metric over time for all companies and the portfolio average.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'CompanyID', 'CompanyName', 'Quarter', and the metric column.
    metric_col (str): The name of the column to plot (e.g., 'Org_AI_R_Score').
    title (str): Title of the plot.
    ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(14, 8))
    
    # Plot individual company trajectories
    sns.lineplot(x='Quarter', y=metric_col, hue='CompanyName', marker='o', data=df)
    
    # Plot portfolio average trajectory
    portfolio_avg = df.groupby('Quarter')[metric_col].mean().reset_index()
    sns.lineplot(x='Quarter', y=metric_col, data=portfolio_avg, color='black', linestyle='--', marker='X', linewidth=2, label='Portfolio Average')
    
    plt.title(title)
    plt.xlabel('Quarter')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Company', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot Org-AI-R Score trajectory
plot_time_series(portfolio_df, 'Org_AI_R_Score', 'Org-AI-R Score Trajectory Over Time', 'Org-AI-R Score')

# Plot AI Investment Efficiency trajectory (only for positive efficiency values)
plot_time_series(portfolio_df[portfolio_df['AI_Investment_Efficiency'] > 0], 'AI_Investment_Efficiency',
                 'AI Investment Efficiency Trajectory Over Time', 'AI Investment Efficiency (Points per $M investment)')
```

These time-series charts provide a dynamic view of our portfolio's AI journey. I can clearly see which companies are consistently improving their `Org_AI_R_Score` and which are struggling. The `AI_Investment_Efficiency` plots help me understand if our investments are yielding sustained, improving returns. Deviations from the portfolio average, or sudden drops, trigger deeper investigation, allowing me to intervene proactively and course-correct.

## 6. Identifying Centers of Excellence and Companies for Review

A key responsibility of a Portfolio Manager is to leverage successes and address weaknesses across the fund. This means systematically identifying "Centers of Excellence" – high-performing companies whose AI best practices can be transferred – and flagging "companies for review" that require additional strategic guidance or intervention. This activity is guided by specific performance thresholds.

**Center of Excellence Criteria (from Section 6.3):**
*   Org-AI-R Score $> 75$
*   Demonstrated EBITDA impact $> 3\%$

```python
def identify_actionable_insights(df, org_ai_r_threshold=75, ebitda_impact_threshold=3):
    """
    Identifies Centers of Excellence and companies requiring review based on thresholds.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'CompanyName', 'Org_AI_R_Score', and 'EBITDA_Impact'.
    org_ai_r_threshold (int): Minimum Org-AI-R score for Centers of Excellence.
    ebitda_impact_threshold (float): Minimum EBITDA impact (%) for Centers of Excellence.

    Returns:
    tuple: (pd.DataFrame) Centers of Excellence, (pd.DataFrame) Companies for Review.
    """
    # Filter for the latest quarter's data
    latest_data = df.loc[df.groupby('CompanyID')['Quarter'].idxmax()]

    # Identify Centers of Excellence
    centers_of_excellence = latest_data[
        (latest_data['Org_AI_R_Score'] > org_ai_r_threshold) &
        (latest_data['EBITDA_Impact'] > ebitda_impact_threshold)
    ].sort_values(by='Org_AI_R_Score', ascending=False)

    # Identify Companies for Review (e.g., low Org-AI-R or low EBITDA impact)
    # Define a lower threshold for review
    review_org_ai_r_threshold = 50
    review_ebitda_impact_threshold = 1.0

    companies_for_review = latest_data[
        (latest_data['Org_AI_R_Score'] <= review_org_ai_r_threshold) |
        (latest_data['EBITDA_Impact'] <= review_ebitda_impact_threshold)
    ].sort_values(by='Org_AI_R_Score', ascending=True)

    return centers_of_excellence, companies_for_review

# Identify centers of excellence and companies for review
centers_of_excellence_df, companies_for_review_df = identify_actionable_insights(
    portfolio_df, org_ai_r_threshold=75, ebitda_impact_threshold=3
)

print("\n--- Centers of Excellence (Org-AI-R > 75 AND EBITDA Impact > 3%) ---")
if not centers_of_excellence_df.empty:
    print(tabulate(centers_of_excellence_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']],
                   headers='keys', tablefmt='pipe', floatfmt=(".2f"), showindex=False))
else:
    print("No companies currently meet the Centers of Excellence criteria.")

print("\n--- Companies for Deeper Review (Org-AI-R <= 50 OR EBITDA Impact <= 1%) ---")
if not companies_for_review_df.empty:
    print(tabulate(companies_for_review_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']],
                   headers='keys', tablefmt='pipe', floatfmt=(".2f"), showindex=False))
else:
    print("No companies currently identified for deeper review.")

# Visualization: Scatter plot highlighting CoE and Review Companies
latest_data = portfolio_df.loc[portfolio_df.groupby('CompanyID')['Quarter'].idxmax()]
plt.figure(figsize=(12, 7))
sns.scatterplot(x='Org_AI_R_Score', y='EBITDA_Impact', hue='Industry', size='AI_Investment_Efficiency', sizes=(50, 500),
                data=latest_data, palette='viridis', legend='full', alpha=0.8)

# Highlight Centers of Excellence
if not centers_of_excellence_df.empty:
    plt.scatter(centers_of_excellence_df['Org_AI_R_Score'], centers_of_excellence_df['EBITDA_Impact'],
                color='green', marker='*', s=1000, label='Center of Excellence', edgecolors='black', linewidth=1.5, zorder=5)
    for i, row in centers_of_excellence_df.iterrows():
        plt.text(row['Org_AI_R_Score'], row['EBITDA_Impact'], row['CompanyName'],
                 horizontalalignment='right', verticalalignment='bottom', fontsize=9, color='green', weight='bold')

# Highlight Companies for Review
if not companies_for_review_df.empty:
    plt.scatter(companies_for_review_df['Org_AI_R_Score'], companies_for_review_df['EBITDA_Impact'],
                color='red', marker='X', s=500, label='Company for Review', edgecolors='black', linewidth=1.5, zorder=5)
    for i, row in companies_for_review_df.iterrows():
        plt.text(row['Org_AI_R_Score'], row['EBITDA_Impact'], row['CompanyName'],
                 horizontalalignment='left', verticalalignment='top', fontsize=9, color='red', weight='bold')


plt.axhline(y=3, color='gray', linestyle=':', label='EBITDA Impact Threshold (3%)')
plt.axvline(x=75, color='gray', linestyle=':', label='Org-AI-R Threshold (75)')
plt.title('Portfolio AI Performance: Org-AI-R Score vs. EBITDA Impact (Latest Quarter)')
plt.xlabel('Org-AI-R Score')
plt.ylabel('EBITDA Impact (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Company Type / Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

This targeted identification is critical for my role. I can now direct operating partners to specific Centers of Excellence for documenting and transferring best practices across the fund. Simultaneously, I can allocate resources and attention to companies flagged for review, developing tailored improvement plans to address their specific challenges and bring them up to par. This proactive approach ensures continuous value creation across the entire portfolio.

## 7. Evaluating Exit-Readiness and Potential Valuation Impact

As a Portfolio Manager, preparing for a successful exit is always on my mind. For companies nearing this stage, understanding how their AI capabilities will be perceived by potential buyers is paramount for maximizing valuation. This involves assessing the **Exit-Readiness Score** and then modeling its **potential impact on the exit multiple**.

The **Exit-Readiness Score** quantifies how "buyer-friendly" a company's AI capabilities are. It focuses on aspects visible to buyers, documented impact, and the sustainability of these capabilities.
The formula for the Exit-Readiness Score for portfolio company $j$ preparing for exit is:

$$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$

where:
*   $\text{Visible}_j$: AI capabilities apparent to buyers (e.g., product features, technology stack).
*   $\text{Documented}_j$: Quantified AI impact with an audit trail.
*   $\text{Sustainable}_j$: Embedded capabilities versus one-time projects.
*   The weights are: $w_1 = 0.35$ (Visible), $w_2 = 0.40$ (Documented), $w_3 = 0.25$ (Sustainable).

The **Multiple Attribution Model** then translates this Exit-AI-R score into a potential uplift in the company's valuation multiple:
$$ \text{Multiple}_j = \text{Multiple}_{base,k} + \delta \cdot \text{Exit-AI-R}_j/100 $$
Here, $\text{Multiple}_{base,k}$ is the baseline industry multiple for industry $k$, and $\delta$ is the AI premium coefficient (estimated to be in the range $[1.0, 3.0]$ turns of EBITDA). My synthetic data includes `BaselineMultiple` and `AI_PremiumCoefficient` ($\delta$) for each company.

```python
def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
    """
    Calculates Exit-Readiness Score and Projected Exit Multiple.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Visible', 'Documented', 'Sustainable',
                       'BaselineMultiple', and 'AI_PremiumCoefficient' columns.
    w1 (float): Weight for Visible AI capabilities.
    w2 (float): Weight for Documented AI impact.
    w3 (float): Weight for Sustainable AI capabilities.

    Returns:
    pd.DataFrame: DataFrame with 'Exit_AI_R_Score' and 'Projected_Exit_Multiple' columns added.
    """
    df_copy = df.copy() # Avoid modifying the original DataFrame directly

    # Calculate Exit-AI-R Score
    df_copy['Exit_AI_R_Score'] = (
        w1 * df_copy['Visible'] +
        w2 * df_copy['Documented'] +
        w3 * df_copy['Sustainable']
    )
    df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100) # Ensure within 0-100 range

    # Calculate Projected Exit Multiple using the AI Premium Coefficient (delta)
    df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
    df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
    
    return df_copy

# Apply the function to calculate exit readiness and valuation impact
portfolio_df = calculate_exit_readiness_and_valuation(portfolio_df, w1=0.35, w2=0.40, w3=0.25)

# Display the latest quarter's exit readiness and projected multiples
latest_quarter_df = portfolio_df.loc[portfolio_df.groupby('CompanyID')['Quarter'].idxmax()]
print("\nLatest Quarter's Exit-Readiness and Projected Valuation Impact:")
print(tabulate(latest_quarter_df[['CompanyName', 'Industry', 'Exit_AI_R_Score', 'BaselineMultiple', 'Projected_Exit_Multiple']]
               .sort_values(by='Projected_Exit_Multiple', ascending=False),
               headers='keys', tablefmt='pipe', floatfmt=(".2f", ".2f", ".2f", ".2f", ".2f"), showindex=False))

# Visualization: Scatter plot of Exit-AI-R Score vs. Projected Exit Multiple
plt.figure(figsize=(12, 7))
sns.scatterplot(x='Exit_AI_R_Score', y='Projected_Exit_Multiple', hue='Industry', size='EBITDA_Impact', sizes=(50, 500),
                data=latest_quarter_df, palette='Spectral', legend='full', alpha=0.8)

for i, row in latest_quarter_df.iterrows():
    plt.text(row['Exit_AI_R_Score'], row['Projected_Exit_Multiple'], row['CompanyName'],
             horizontalalignment='left', verticalalignment='bottom', fontsize=8)

plt.title('Exit-AI-R Score vs. Projected Exit Multiple (Latest Quarter)')
plt.xlabel('Exit-AI-R Score')
plt.ylabel('Projected Exit Multiple')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

This analysis provides critical data for our exit planning strategy. By quantifying the `Exit_AI_R_Score` and its projected impact on the valuation multiple, I can build a compelling, evidence-based AI narrative for potential buyers. This allows me to highlight the embedded and sustainable AI value creation, leading to potentially higher valuations and better returns for the fund. It also guides where to focus pre-exit efforts to enhance the `Visible`, `Documented`, and `Sustainable` aspects of a company's AI capabilities.
