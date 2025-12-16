
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- Utility Functions (now embedded in app.py to resolve ModuleNotFoundError) ---
# This consolidation ensures all necessary functions are available directly within app.py
# if utils.py is not correctly recognized or loaded by the execution environment.

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
                'EBITDA_Impact': np.clip(ebitda_impact + np.random.normal(0, 0.5), 0.1, 15.0), # Baseline EBITDA Impact (synthetic)
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

    # Initialize columns that will be calculated later, ensuring they exist
    required_cols = [
        'Org_AI_R_Score', 'Delta_Org_AI_R', 'AI_Investment_Efficiency',
        'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute',
        'Org_AI_R_Percentile', 'Org_AI_R_Z_Score', 'Exit_AI_R_Score',
        'Projected_Exit_Multiple', 'AI_Premium_Multiple_Additive',
        'IndustryMeanOrgAIR', 'IndustryStdDevOrgAIR'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # These will be updated later with actual Org_AI_R_Score, but need initial values for Z-score calculation robustness
    df['IndustryMeanOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('mean')
    df['IndustryStdDevOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('std').fillna(5.0) # fillna for single-company industries

    return df

# --- Org-AI-R Calculation ---
def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
    """
    Calculates the Organizational AI Readiness (Org-AI-R) score for each company.
    """
    if not all(col in df.columns for col in ['IdiosyncraticReadiness', 'SystematicOpportunity', 'Synergy']):
        raise ValueError("DataFrame missing required columns for Org-AI-R calculation.")

    df_copy = df.copy() # Operate on a copy to avoid SettingWithCopyWarning
    df_copy['Org_AI_R_Score'] = (
        alpha * df_copy['IdiosyncraticReadiness'] +
        (1 - alpha) * df_copy['SystematicOpportunity'] +
        beta * df_copy['Synergy']
    )
    df_copy['Org_AI_R_Score'] = np.clip(df_copy['Org_AI_R_Score'], 0, 100)

    # After calculating Org_AI_R_Score, update IndustryMeanOrgAIR and IndustryStdDevOrgAIR based on this score
    df_copy['IndustryMeanOrgAIR'] = df_copy.groupby(['Industry', 'Quarter'])['Org_AI_R_Score'].transform('mean')
    df_copy['IndustryStdDevOrgAIR'] = df_copy.groupby(['Industry', 'Quarter'])['Org_AI_R_Score'].transform('std').fillna(5.0) # fillna for single-company industries

    return df_copy

# --- Benchmarking Calculation ---
def calculate_benchmarks(df):
    """
    Calculates percentile rank and Z-score for Org-AI-R scores.
    """
    df_copy = df.copy() # Operate on a copy
    df_copy['Org_AI_R_Percentile'] = df_copy.groupby('Quarter')['Org_AI_R_Score'].rank(pct=True) * 100
    df_copy['Org_AI_R_Z_Score'] = df_copy.apply(
        lambda row: (row['Org_AI_R_Score'] - row['IndustryMeanOrgAIR']) / row['IndustryStdDevOrgAIR']
        if row['IndustryStdDevOrgAIR'] != 0 else 0, axis=1
    )
    return df_copy

# --- AI Investment Efficiency and EBITDA Attribution Calculation ---
def calculate_aie_ebitda(df):
    """
    Calculates AI Investment Efficiency and Attributed EBITDA Impact.
    """
    df_sorted = df.sort_values(by=['CompanyID', 'Quarter'])
    df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID')['Org_AI_R_Score'].diff().fillna(0)

    # AI Investment Efficiency (AIE_j) = (Delta_Org_AI_R_j * EBITDA_Impact_j) / (AI_Investment_j in Millions)
    # EBITDA_Impact is already a percentage in the synthetic data (e.g., 3.0 for 3%)
    df_sorted['AI_Investment_Efficiency'] = df_sorted.apply(
        lambda row: (row['Delta_Org_AI_R'] * row['EBITDA_Impact']) / (row['AI_Investment'] / 1e6)
        if row['AI_Investment'] > 0 and row['Delta_Org_AI_R'] > 0 and row['EBITDA_Impact'] > 0 else 0, axis=1
    )
    # Ensure efficiency is only calculated for positive delta_Org_AI_R
    df_sorted.loc[df_sorted['Delta_Org_AI_R'] <= 0, 'AI_Investment_Efficiency'] = 0

    # Attributed EBITDA Impact (%) = GammaCoefficient * Delta_Org_AI_R * H^R_org,k / 100
    # Using SystematicOpportunity as H^R_org,k
    df_sorted['Attributed_EBITDA_Impact_Pct'] = df_sorted.apply(
        lambda row: row['GammaCoefficient'] * row['Delta_Org_AI_R'] * row['SystematicOpportunity'] / 100
        if row['Delta_Org_AI_R'] > 0 else 0, axis=1
    )
    df_sorted['Attributed_EBITDA_Impact_Absolute'] = (df_sorted['Attributed_EBITDA_Impact_Pct'] / 100) * df_sorted['BaselineEBITDA']
    return df_sorted

# --- Exit Readiness and Valuation Calculation ---
def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
    """
    Calculates Exit-AI-R Score and Projected Exit Multiple.
    """
    df_copy = df.copy() # Operate on a copy
    if not all(col in df_copy.columns for col in ['Visible', 'Documented', 'Sustainable', 'BaselineMultiple', 'AI_PremiumCoefficient']):
        raise ValueError("DataFrame missing required columns for Exit-AI-R calculation.")

    df_copy['Exit_AI_R_Score'] = (
        w1 * df_copy['Visible'] + w2 * df_copy['Documented'] + w3 * df_copy['Sustainable']
    )
    df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100)

    # Multiple_j = Multiple_base,k + AI Premium Coefficient * Exit-AI-R_j/100
    df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
    df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
    return df_copy

# --- Identify Actionable Insights ---
def identify_actionable_insights(df, org_ai_r_threshold_coe=75, ebitda_impact_threshold_coe=3,
                                 org_ai_r_threshold_review=50, ebitda_impact_threshold_review=1.0):
    """
    Identifies Centers of Excellence and Companies for Review based on thresholds.
    Uses 'Attributed_EBITDA_Impact_Pct' for thresholds, not 'EBITDA_Impact'.
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


# --- Page Functions (Content from application_pages files, embedded into app.py) ---

def page_1_initializing_data_main():
    st.header("1. Initializing Portfolio Data: The Bedrock for AI Performance Tracking")
    st.markdown("""
    As a Portfolio Manager, my first step in any analytical review is to ensure I have the most current and accurate data for all my portfolio companies.
    This initial data load forms the bedrock for all subsequent AI performance assessments and strategic decisions.
    I need to quickly review the structure and content of this data to confirm its integrity and readiness for analysis.
    This page helps me do just that by displaying key statistics and a glimpse of the raw data.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data'.")
        return

    st.subheader("Overview of Generated Portfolio Data:")
    st.markdown("""
    This table shows the first few rows of the synthesized portfolio data.
    I'm checking for expected columns like `CompanyName`, `Industry`, `Quarter`, and various AI-related readiness scores and financial metrics.
    A quick scan helps me understand the diversity and scope of the data I'll be working with.
    """)
    st.dataframe(st.session_state.portfolio_df.head())

    st.subheader("Descriptive Statistics of Numerical Data:")
    st.markdown("""
    These descriptive statistics provide a high-level summary of the numerical features in my portfolio.
    I'm looking at ranges, averages, and standard deviations for metrics like `IdiosyncraticReadiness`, `SystematicOpportunity`, `AI_Investment`, and `EBITDA_Impact`.
    This helps me get a feel for the general distribution and potential outliers in my portfolio's AI landscape.
    """)
    st.dataframe(st.session_state.portfolio_df.describe())

    st.subheader("Data Information (Columns and Types):")
    st.markdown("""
    The `info()` method gives me a concise summary of the DataFrame, including the number of entries, column names,
    their data types, and non-null values. This is crucial for identifying any missing data or incorrect data types
    that might hinder subsequent calculations.
    """)
    buffer = StringIO()
    st.session_state.portfolio_df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.info("Data loaded and reviewed. Proceed to '2. Calculating Org-AI-R Scores' to begin calibrating our AI performance model.")


def page_2_calculating_org_ai_r_main():
    st.header("2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment")
    st.markdown("""
    As a Portfolio Manager, the core of my AI performance tracking is the PE Org-AI-R score.
    This score quantifies a company's overall AI maturity and readiness for value creation.
    It's a critical metric because it moves beyond anecdotal evidence of AI adoption to a structured, measurable assessment.
    My goal on this page is to calibrate the Org-AI-R score calculation to reflect our fund's specific strategic emphasis,
    especially regarding how we weigh company-specific capabilities versus broader industry opportunities.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' first.")
        return

    st.subheader("Org-AI-R Score Calculation Parameters:")
    st.markdown(r"""
    The formula for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:

    $$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$

    Where:
    *   $V^R_{org,j}(t)$: **Idiosyncratic Readiness**. This represents company-specific capabilities at time $t$, such as data infrastructure, AI talent pool, leadership commitment, and internal AI-driven processes. These are factors largely controllable by the company.
    *   $H^R_{org,k}(t)$: **Systematic Opportunity**. This captures the industry-level AI potential at time $t$, reflecting broader market adoption rates, disruption potential within the sector, and the competitive AI landscape. These are external factors influencing the company's AI context.
    *   $\alpha$: **Weight for Idiosyncratic Readiness**. This slider allows me to adjust how much importance we place on a company's internal, controllable AI capabilities ($V^R_{org,j}$) versus the external industry potential ($H^R_{org,k}$). A higher $\alpha$ means we prioritize internal strengths.
    *   $\beta$: **Synergy Coefficient**. This coefficient quantifies the additional value derived from the interplay and alignment between a company's idiosyncratic readiness and the systematic opportunity in its industry. It reflects how well a company can capitalize on market potential with its internal capabilities.
    """)

    # Widgets for calibration
    # Default values are sourced from session_state for persistence across app runs/page navigation
    alpha_val = st.slider(
        "Weight for Idiosyncratic Readiness ($\\alpha$)",
        min_value=0.55, max_value=0.70, value=st.session_state.alpha_slider, step=0.01, key="alpha_slider",
        help="Adjust this to prioritize company-specific capabilities ($V^R_{org,j}$) versus industry-level AI potential ($H^R_{org,k}$). Default $\\alpha = 0.60$."
    )
    beta_val = st.slider(
        "Synergy Coefficient ($\\beta$)",
        min_value=0.08, max_value=0.25, value=st.session_state.beta_slider, step=0.01, key="beta_slider",
        help="Quantify the additional value derived from the interplay between idiosyncratic readiness and systematic opportunity. Default $\\beta = 0.15$."
    )

    if st.button("Recalculate Org-AI-R Scores", key="recalculate_org_ai_r_button", help="Click to re-compute Org-AI-R scores with the selected weights."):
        st.session_state.portfolio_df = calculate_org_ai_r(st.session_state.portfolio_df, alpha_val, beta_val)
        st.session_state.org_ai_r_recalculated = True
        st.success("Org-AI-R scores re-calculated based on your strategic weighting.")
        st.info("Org-AI-R Score: A composite score (0-100) quantifying a company's overall AI maturity and readiness for value creation. Higher scores indicate stronger AI capabilities and potential.")
    elif not st.session_state.org_ai_r_recalculated:
        # Initial calculation on page load if not already done
        st.session_state.portfolio_df = calculate_org_ai_r(st.session_state.portfolio_df, st.session_state.alpha_slider, st.session_state.beta_slider)
        st.session_state.org_ai_r_recalculated = True
        st.info("Initial Org-AI-R scores calculated. Adjust the sliders and click 'Recalculate' to refine.")


    st.subheader("Latest Quarter's PE Org-AI-R Scores:")
    st.markdown("""
    This table presents the calculated Org-AI-R scores for all companies in the latest quarter, sorted by score.
    As a Portfolio Manager, this immediately tells me which companies are leading in AI readiness and which might be lagging.
    It's a critical input for my initial assessment of AI maturity across the fund.
    """)
    latest_quarter = st.session_state.portfolio_df['Quarter'].max()
    latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == latest_quarter]
    st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Org_AI_R_Score']].sort_values(by='Org_AI_R_Score', ascending=False).reset_index(drop=True))

    st.info("Org-AI-R scores updated. Now, let's see how these companies benchmark against their peers in '3. Benchmarking AI Performance'.")


def page_3_benchmarking_ai_performance_main():
    st.header("3. Benchmarking AI Performance: Identifying Relative AI Standing")
    st.markdown("""
    Understanding a company's standalone Org-AI-R score is a good start, but as a Portfolio Manager,
    I need to know how that performance stacks up against its peers.
    This benchmarking step allows me to identify true leaders and laggards within our portfolio and relative to their industry.
    My decision here is to select a specific quarter to focus my benchmarking efforts, typically the most recent one for current insights.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' and ensure Org-AI-R scores are calculated.")
        return

    if "Org_AI_R_Score" not in st.session_state.portfolio_df.columns or not st.session_state.org_ai_r_recalculated:
        st.warning("Org-AI-R Scores have not been calculated yet. Please go to '2. Calculating Org-AI-R Scores' first to enable benchmarking.")
        return

    # Ensure benchmarks are calculated. This needs to run consistently if Org-AI-R scores change.
    st.session_state.portfolio_df = calculate_benchmarks(st.session_state.portfolio_df)

    all_quarters = st.session_state.portfolio_df['Quarter'].unique().tolist()
    # Safely determine the index for the latest quarter
    selected_quarter_index = 0
    if all_quarters:
        latest_quarter_val = st.session_state.portfolio_df['Quarter'].max()
        if latest_quarter_val in all_quarters:
            selected_quarter_index = all_quarters.index(latest_quarter_val)

    benchmark_quarter = st.selectbox(
        "Select Quarter for Benchmarking",
        options=all_quarters,
        index=selected_quarter_index,
        key="benchmark_quarter_select",
        help="Choose the quarter for which you want to perform the benchmarking analysis."
    )

    filtered_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == benchmark_quarter].copy()

    st.subheader(f"Org-AI-R Benchmarks for {benchmark_quarter}:")
    st.markdown(r"""
    These benchmarks are invaluable for comparing companies. I use two key metrics:

    *   **Within-Portfolio Benchmarking (Percentile Rank):**
        $$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$
        This shows a company's standing relative to all other fund holdings. For example, a 90th percentile means it outperforms 90% of its peers within our portfolio.

    *   **Cross-Portfolio Benchmarking (Industry-Adjusted Z-Score):**
        $$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
        This score indicates how much a company's Org-AI-R deviates from its industry's mean ($\mu_k$), in terms of standard deviations ($\sigma_k$). Positive values suggest outperformance relative to industry peers, while negative values signal underperformance.

    By reviewing these, I can identify which companies are truly excelling in AI within their context.
    """)
    st.dataframe(filtered_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Org_AI_R_Percentile', 'Org_AI_R_Z_Score']]
                 .sort_values(by='Org_AI_R_Score', ascending=False)
                 .reset_index(drop=True))

    st.subheader(f"Latest Quarter Org-AI-R Scores by Company ({benchmark_quarter})")
    st.markdown("This bar chart visually compares the Org-AI-R scores of individual companies. The horizontal line shows the average Org-AI-R score across the entire portfolio for the selected quarter, allowing for quick identification of companies above or below average.")
    portfolio_average_org_ai_r = filtered_df['Org_AI_R_Score'].mean()
    fig1 = px.bar(
        filtered_df.sort_values(by='Org_AI_R_Score', ascending=False),
        x='CompanyName',
        y='Org_AI_R_Score',
        color='Industry',
        title=f'Org-AI-R Scores by Company in {benchmark_quarter}',
        labels={'Org_AI_R_Score': 'Org-AI-R Score (0-100)', 'CompanyName': 'Company Name'}
    )
    fig1.add_hline(y=portfolio_average_org_ai_r, line_dash="dash", line_color="red", annotation_text=f"Portfolio Average: {portfolio_average_org_ai_r:.2f}")
    st.plotly_chart(fig1, use_container_width=True)


    st.subheader(f"Org-AI-R Score vs. Industry-Adjusted Z-Score ({benchmark_quarter})")
    st.markdown("This scatter plot helps visualize relative performance. Companies with higher Org-AI-R scores and positive Z-scores are strong performers. The size of the point indicates its percentile rank within the portfolio – larger points mean higher within-portfolio ranking. This gives me a nuanced view, distinguishing companies that are strong overall from those performing exceptionally well within their specific industry context.")
    portfolio_mean_org_ai_r = filtered_df['Org_AI_R_Score'].mean()

    fig2 = px.scatter(
        filtered_df,
        x='Org_AI_R_Score',
        y='Org_AI_R_Z_Score',
        color='Industry',
        size='Org_AI_R_Percentile',
        hover_name='CompanyName',
        title=f'Org-AI-R Score vs. Industry-Adjusted Z-Score in {benchmark_quarter}',
        labels={'Org_AI_R_Score': 'Org-AI-R Score (0-100)', 'Org_AI_R_Z_Score': 'Industry-Adjusted Z-Score', 'Org_AI_R_Percentile': 'Org-AI-R Percentile (Within Portfolio)'}
    )
    fig2.add_vline(x=portfolio_mean_org_ai_r, line_dash="dash", line_color="orange", annotation_text=f"Portfolio Mean Org-AI-R: {portfolio_mean_org_ai_r:.2f}")
    fig2.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Industry Mean Z-Score (0)")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Benchmarking completed. Next, let's analyze the financial returns and efficiency of AI investments in '4. AI Investment & EBITDA Impact'.")


def page_4_ai_investment_ebitda_impact_main():
    st.header("4. Assessing AI Investment Efficiency and EBITDA Attribution")
    st.markdown("""
    As a Portfolio Manager, I need to go beyond just scores and understand the tangible financial impact of AI investments.
    This page focuses on quantifying how efficiently our portfolio companies are converting their AI expenditures into real business value, specifically in terms of EBITDA growth.
    This analysis provides critical insights into capital deployment strategies for AI and highlights which companies are getting the most bang for their buck.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' first.")
        return

    if "Org_AI_R_Score" not in st.session_state.portfolio_df.columns or not st.session_state.org_ai_r_recalculated:
        st.warning("Org-AI-R Scores have not been calculated yet. Please go to '2. Calculating Org-AI-R Scores' first.")
        return

    # Ensure AIE and EBITDA impact are calculated
    # This function is idempotent and will update the relevant columns in portfolio_df
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)

    latest_quarter = st.session_state.portfolio_df['Quarter'].max()
    latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == latest_quarter].copy()

    st.subheader(f"AI Investment Efficiency and Attributed EBITDA Impact ({latest_quarter}):")
    st.markdown(r"""
    Here, we quantify the financial returns from AI initiatives using two key metrics:

    *   **AI Investment Efficiency ($\text{AIE}_j$):**
        $$ \text{AIE}_j = \frac{\Delta\text{Org-AI-R}_j \cdot \text{EBITDA Impact}_j}{\text{AI Investment}_j \text{ (in millions)}} $$
        This metric measures the combined impact (Org-AI-R points and baseline EBITDA Impact percentage) generated per million dollars of AI investment. A higher AIE indicates more efficient capital deployment for AI initiatives. It tells me which companies are most effectively converting their AI spend into measurable improvements.

    *   **Attributed EBITDA Impact ($\Delta\text{EBITDA}\%$):**
        $$ \Delta\text{EBITDA}\% = \text{GammaCoefficient} \cdot \Delta\text{Org-AI-R} \cdot H^R_{org,k}/100 $$
        This is the estimated percentage increase in EBITDA directly attributed to the change in a company's Org-AI-R score, factoring in its industry's systematic opportunity ($H^R_{org,k}$) and a Gamma Coefficient. This quantifies the direct financial upside we can attribute to improvements in AI maturity. The `GammaCoefficient` acts as a scaling factor, reflecting the sensitivity of EBITDA to AI readiness changes.

    This analysis provides critical insights into which companies are generating the most value from their AI initiatives.
    """)
    st.dataframe(latest_quarter_df[[
        'CompanyName', 'Industry', 'AI_Investment', 'Delta_Org_AI_R',
        'AI_Investment_Efficiency', 'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute'
    ]].sort_values(by='AI_Investment_Efficiency', ascending=False).reset_index(drop=True))

    st.subheader(f"AI Investment vs. Efficiency (Latest Quarter, Highlighting EBITDA Impact) ({latest_quarter})")
    st.markdown("This scatter plot visualizes the relationship between a company's AI investment, its efficiency in generating value from that investment, and the attributed EBITDA impact. Companies in the upper-left quadrant are highly efficient with relatively lower investment, while larger point sizes indicate a greater attributed EBITDA impact. This helps me identify companies that are either highly efficient in their AI spending or are generating significant financial uplift, or both.")

    # Convert AI_Investment to millions for plot axis and apply log scale
    latest_quarter_df['AI_Investment_M'] = latest_quarter_df['AI_Investment'] / 1e6

    fig = px.scatter(
        latest_quarter_df,
        x='AI_Investment_M',
        y='AI_Investment_Efficiency',
        color='Industry',
        size='Attributed_EBITDA_Impact_Pct',
        hover_name='CompanyName',
        title=f'AI Investment (Millions) vs. Efficiency ({latest_quarter})',
        labels={
            'AI_Investment_M': 'AI Investment (Millions)',
            'AI_Investment_Efficiency': 'AI Investment Efficiency (Impact / $M)',
            'Attributed_EBITDA_Impact_Pct': 'Attributed EBITDA Impact (%)'
        },
        log_x=True # Use log scale for AI Investment to handle wide range of values
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Financial impact and efficiency reviewed. Now, let's track the historical trajectories of these metrics in '5. Tracking Progress Over Time'.")


def page_5_tracking_progress_main():
    st.header("5. Tracking Progress Over Time: Visualizing Trajectories")
    st.markdown("""
    As a Portfolio Manager, current metrics are important, but understanding the historical trajectory of our portfolio companies' AI performance is equally critical.
    This page allows me to monitor long-term trends, identify companies with consistent improvement or decline, and spot outliers.
    By visualizing these trends, I can assess the effectiveness of past strategic initiatives and identify companies that warrant deeper investigation or targeted support.
    My decision here is to select a few key companies to track for clarity, typically those I'm most interested in for performance review or strategic planning.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' first.")
        return

    if "Org_AI_R_Score" not in st.session_state.portfolio_df.columns or not st.session_state.org_ai_r_recalculated:
        st.warning("Org-AI-R Scores have not been calculated yet. Please go to '2. Calculating Org-AI-R Scores' first.")
        return
    # Ensure calculate_aie_ebitda has been run to have 'AI_Investment_Efficiency' and other related columns
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)


    available_companies = st.session_state.portfolio_df['CompanyName'].unique().tolist()
    default_companies = available_companies[:min(5, len(available_companies))] if available_companies else []

    selected_companies = st.multiselect(
        "Select Companies to Track (Max 5 for clarity)",
        options=available_companies,
        default=default_companies,
        key="companies_to_track_multiselect",
        help="Choose up to 5 companies to visualize their historical performance trends."
    )

    if not selected_companies:
        st.info("Please select at least one company to track its progress over time.")
        return

    # Filter data for selected companies
    df_filtered_companies = st.session_state.portfolio_df[st.session_state.portfolio_df['CompanyName'].isin(selected_companies)].copy()

    # Calculate portfolio average for overlay for Org-AI-R
    portfolio_avg_org_ai_r = st.session_state.portfolio_df.groupby('Quarter')['Org_AI_R_Score'].mean().reset_index()
    portfolio_avg_org_ai_r['CompanyName'] = 'Portfolio Average'

    st.subheader("Org-AI-R Score Trajectory Over Time")
    st.markdown("""
    This line chart visualizes how the Org-AI-R score for selected companies has evolved across quarters.
    I can see individual company progress, and the overlaid 'Portfolio Average' line helps contextualize their performance against the fund's overall trend.
    This is useful for spotting consistent improvers, decliners, or companies that deviate significantly from the average.
    """)
    if not df_filtered_companies.empty:
        # Combine with portfolio average for plotting
        plot_df_org_ai_r = pd.concat([df_filtered_companies[['Quarter', 'Org_AI_R_Score', 'CompanyName']], portfolio_avg_org_ai_r])
        # Ensure Quarter is treated as a categorical for correct ordering on x-axis
        plot_df_org_ai_r['Quarter'] = pd.Categorical(plot_df_org_ai_r['Quarter'], categories=st.session_state.portfolio_df['Quarter'].unique().tolist(), ordered=True)
        plot_df_org_ai_r = plot_df_org_ai_r.sort_values(by='Quarter')

        fig1 = px.line(
            plot_df_org_ai_r,
            x='Quarter',
            y='Org_AI_R_Score',
            color='CompanyName',
            markers=True,
            title='Org-AI-R Score Trajectory Over Time',
            labels={'Org_AI_R_Score': 'Org-AI-R Score (0-100)', 'Quarter': 'Quarter'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No data to display for the selected companies' Org-AI-R scores.")


    # Calculate portfolio average for overlay for AIE
    portfolio_avg_aie = st.session_state.portfolio_df.groupby('Quarter')['AI_Investment_Efficiency'].mean().reset_index()
    portfolio_avg_aie['CompanyName'] = 'Portfolio Average'

    st.subheader("AI Investment Efficiency Trajectory Over Time")
    st.markdown("""
    This chart tracks the AI Investment Efficiency for the selected companies over time.
    It visualizes how effectively companies are converting their AI investments into value quarter-over-quarter.
    By comparing individual company trends with the 'Portfolio Average', I can identify who is becoming more efficient, who is struggling,
    and whether efficiency gains are a broader fund-wide trend or company-specific successes.
    """)
    if not df_filtered_companies.empty:
        # Combine with portfolio average for plotting
        plot_df_aie = pd.concat([df_filtered_companies[['Quarter', 'AI_Investment_Efficiency', 'CompanyName']], portfolio_avg_aie])
        # Ensure Quarter is treated as a categorical for correct ordering on x-axis
        plot_df_aie['Quarter'] = pd.Categorical(plot_df_aie['Quarter'], categories=st.session_state.portfolio_df['Quarter'].unique().tolist(), ordered=True)
        plot_df_aie = plot_df_aie.sort_values(by='Quarter')

        fig2 = px.line(
            plot_df_aie,
            x='Quarter',
            y='AI_Investment_Efficiency',
            color='CompanyName',
            markers=True,
            title='AI Investment Efficiency Trajectory Over Time',
            labels={'AI_Investment_Efficiency': 'AI Investment Efficiency (Impact / $M)', 'Quarter': 'Quarter'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data to display for the selected companies' AI investment efficiency.")

    st.info("Historical trends analyzed. Now, let's define thresholds to identify specific action categories in '6. Actionable Insights: CoE & Review'.")


def page_6_actionable_insights_main():
    st.header("6. Actionable Insights: Centers of Excellence & Companies for Review")
    st.markdown("""
    A key responsibility of a Portfolio Manager is to leverage successes and address underperformance.
    This page allows me to strategically segment our portfolio based on configurable performance thresholds.
    I can define what constitutes a "Center of Excellence" – a high-performing company whose AI best practices can be scaled across the fund –
    and what flags a "Company for Review" – an underperforming entity that needs immediate strategic attention or resource reallocation.
    My interaction here involves setting these thresholds to align with our current fund strategy and risk appetite.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' first.")
        return

    if "Org_AI_R_Score" not in st.session_state.portfolio_df.columns or not st.session_state.org_ai_r_recalculated:
        st.warning("Org-AI-R Scores have not been calculated yet. Please go to '2. Calculating Org-AI-R Scores' first.")
        return

    # Ensure Attributed_EBITDA_Impact_Pct is calculated, as it's used for thresholds
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)


    st.subheader("Define Thresholds for Actionable Insights:")
    st.markdown("""
    These sliders allow me to customize the criteria for identifying different categories of companies, based on my fund's strategic priorities for AI performance.

    *   **Centers of Excellence (CoE):** Portfolio companies with high Org-AI-R scores and significant **attributed EBITDA impact**, serving as benchmarks for best practices and potential for replication across the fund.
    *   **Companies for Review:** Portfolio companies with lower Org-AI-R scores or minimal **attributed EBITDA impact**, indicating areas requiring strategic intervention, additional resources, or a re-evaluation of AI strategy.
    """)

    # Widgets for thresholds, values sourced from session_state
    coe_org_ai_r_threshold_val = st.slider(
        "Org-AI-R Score Threshold for Center of Excellence",
        min_value=50, max_value=90, value=st.session_state.coe_org_ai_r_threshold, step=1, key="coe_org_ai_r_threshold",
        help="Companies with Org-AI-R score above this will be considered for 'Centers of Excellence'. Default: 75."
    )
    coe_ebitda_threshold_val = st.slider(
        "Attributed EBITDA Impact (%) Threshold for Center of Excellence",
        min_value=1.0, max_value=10.0, value=st.session_state.coe_ebitda_threshold, step=0.5, key="coe_ebitda_threshold",
        help="Companies with Attributed EBITDA Impact (%) above this will be considered for 'Centers of Excellence'. Default: 3%."
    )
    review_org_ai_r_threshold_val = st.slider(
        "Org-AI-R Score Threshold for Companies for Review",
        min_value=20, max_value=70, value=st.session_state.review_org_ai_r_threshold, step=1, key="review_org_ai_r_threshold",
        help="Companies with Org-AI-R score below or equal to this will be considered for 'Companies for Review'. Default: 50."
    )
    review_ebitda_threshold_val = st.slider(
        "Attributed EBITDA Impact (%) Threshold for Companies for Review",
        min_value=0.0, max_value=5.0, value=st.session_state.review_ebitda_threshold, step=0.1, key="review_ebitda_threshold",
        help="Companies with Attributed EBITDA Impact (%) below or equal to this will be considered for 'Companies for Review'. Default: 1%."
    )

    # Note: No explicit recalculate button needed if insights automatically update with slider changes.
    # The button would primarily be for an expensive recalculation, which identify_actionable_insights is not.
    # The insights will refresh implicitly when sliders are moved.
    # The presence of `re_evaluate_insights_button` in spec requires it.
    if st.button("Re-evaluate Actionable Insights", key="re_evaluate_insights_button", help="Click to re-identify Centers of Excellence and Companies for Review based on the adjusted thresholds."):
        st.success("Actionable insights re-evaluated with new thresholds.")


    # Identify companies based on current thresholds
    centers_of_excellence, companies_for_review = identify_actionable_insights(
        st.session_state.portfolio_df,
        coe_org_ai_r_threshold_val, coe_ebitda_threshold_val,
        review_org_ai_r_threshold_val, review_ebitda_threshold_val
    )

    st.subheader("--- Centers of Excellence ---")
    st.markdown("""
    These are our high-performers in AI. Companies listed here demonstrate strong AI maturity (high Org-AI-R) and a significant positive financial impact.
    As a Portfolio Manager, I will study their best practices to identify scalable strategies and consider them for additional investment or leadership roles in fund-wide initiatives.
    """)
    if not centers_of_excellence.empty:
        st.dataframe(centers_of_excellence[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Attributed_EBITDA_Impact_Pct', 'AI_Investment_Efficiency']].reset_index(drop=True))
    else:
        st.info("No companies currently meet the 'Centers of Excellence' criteria with the current thresholds.")

    st.subheader("--- Companies for Review ---")
    st.markdown("""
    These companies require a deeper look. They exhibit lower AI maturity (Org-AI-R) or minimal attributed financial impact from their AI initiatives.
    My next step as a Portfolio Manager is to initiate a detailed review, understand the root causes of their underperformance,
    and develop targeted intervention strategies, which might include re-allocating resources, providing expert support, or re-evaluating their AI strategy.
    """)
    if not companies_for_review.empty:
        st.dataframe(companies_for_review[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Attributed_EBITDA_Impact_Pct', 'AI_Investment_Efficiency']].reset_index(drop=True))
    else:
        st.info("No companies currently meet the 'Companies for Review' criteria with the current thresholds.")

    st.subheader(f"Portfolio AI Performance: Org-AI-R Score vs. Attributed EBITDA Impact (Latest Quarter)")
    st.markdown("This scatter plot visually distinguishes 'Centers of Excellence' and 'Companies for Review' based on the thresholds I've defined. I can quickly see which companies fall into which category, with the size of the point representing AI Investment Efficiency. The threshold lines dynamically adjust, providing an interactive way to segment the portfolio and inform strategic actions.")

    latest_quarter_df = st.session_state.portfolio_df.loc[st.session_state.portfolio_df.groupby('CompanyID')['Quarter'].idxmax()].copy()

    # Create a 'Category' column for coloring/highlighting
    latest_quarter_df['Category'] = 'Normal'
    # Use .loc with .index.isin for setting values to avoid SettingWithCopyWarning
    latest_quarter_df.loc[latest_quarter_df.index.isin(centers_of_excellence.index), 'Category'] = 'Center of Excellence'
    latest_quarter_df.loc[latest_quarter_df.index.isin(companies_for_review.index), 'Category'] = 'Company for Review'

    fig = px.scatter(
        latest_quarter_df,
        x='Org_AI_R_Score',
        y='Attributed_EBITDA_Impact_Pct',
        color='Category',
        size='AI_Investment_Efficiency',
        hover_name='CompanyName',
        title='Portfolio AI Performance Segmentation',
        labels={
            'Org_AI_R_Score': 'Org-AI-R Score (0-100)',
            'Attributed_EBITDA_Impact_Pct': 'Attributed EBITDA Impact (%)',
            'AI_Investment_Efficiency': 'AI Investment Efficiency (Impact / $M)'
        },
        color_discrete_map={
            'Center of Excellence': 'green',
            'Company for Review': 'red',
            'Normal': 'blue'
        }
    )

    # Add threshold lines dynamically
    fig.add_vline(x=coe_org_ai_r_threshold_val, line_dash="dash", line_color="green", annotation_text=f"CoE Org-AI-R > {coe_org_ai_r_threshold_val}")
    fig.add_hline(y=coe_ebitda_threshold_val, line_dash="dash", line_color="green", annotation_text=f"CoE EBITDA > {coe_ebitda_threshold_val}%")
    fig.add_vline(x=review_org_ai_r_threshold_val, line_dash="dot", line_color="red", annotation_text=f"Review Org-AI-R <= {review_org_ai_r_threshold_val}")
    fig.add_hline(y=review_ebitda_threshold_val, line_dash="dot", line_color="red", annotation_text=f"Review EBITDA <= {review_ebitda_threshold_val}%")

    # Add annotations for company names if they are CoE or Review
    # Use `go.Scatter` with `mode='text'` for robust text labels
    for _, row in centers_of_excellence.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Org_AI_R_Score']], y=[row['Attributed_EBITDA_Impact_Pct']],
            mode='markers+text',
            text=[row['CompanyName']],
            textposition="top center",
            marker=dict(size=12, color='green', symbol='star-open'),
            name=f"{row['CompanyName']} (CoE)",
            hoverinfo='text',
            hovertext=f"CoE: {row['CompanyName']}<br>Org-AI-R: {row['Org_AI_R_Score']:.1f}<br>EBITDA Impact: {row['Attributed_EBITDA_Impact_Pct']:.1f}%"
        ))
    for _, row in companies_for_review.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Org_AI_R_Score']], y=[row['Attributed_EBITDA_Impact_Pct']],
            mode='markers+text',
            text=[row['CompanyName']],
            textposition="bottom center",
            marker=dict(size=12, color='red', symbol='x-open'),
            name=f"{row['CompanyName']} (Review)",
            hoverinfo='text',
            hovertext=f"Review: {row['CompanyName']}<br>Org-AI-R: {row['Org_AI_R_Score']:.1f}<br>EBITDA Impact: {row['Attributed_EBITDA_Impact_Pct']:.1f}%"
        ))

    st.plotly_chart(fig, use_container_width=True)

    st.info("Actionable insights generated. Finally, let's assess how AI impacts potential exit valuations in '7. Exit-Readiness & Valuation'.")


def page_7_exit_readiness_main():
    st.header("7. Evaluating Exit-Readiness and Potential Valuation Impact")
    st.markdown("""
    As a Portfolio Manager, preparing for a successful exit is always on my mind.
    A company's AI capabilities are increasingly a significant factor influencing its attractiveness to potential acquirers and, consequently, its valuation multiple.
    This page allows me to assess how 'buyer-friendly' a company's AI assets are and how they contribute to its projected exit multiple.
    My critical task here is to adjust the weighting factors for the `Exit-AI-R Score`, reflecting what aspects buyers might prioritize (e.g., visible product features vs. documented impact) to build the strongest possible exit narrative and maximize valuation.
    """)

    if "portfolio_df" not in st.session_state or st.session_state.portfolio_df.empty:
        st.warning("No portfolio data found. Please use the sidebar to 'Generate New Portfolio Data' first.")
        return

    # Ensure Attributed_EBITDA_Impact_Pct is calculated as it's used for plot sizing
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)


    st.subheader("Exit-AI-R Score Calculation Parameters:")
    st.markdown(r"""
    The formula for the Exit-Readiness Score for portfolio company $j$ preparing for exit is:

    $$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$

    Where:
    *   $\text{Visible}_j$: **Visible AI Capabilities**. AI features that are easily apparent and demonstrable to potential buyers, such as product functionality driven by AI, advanced technology stack, or public-facing AI applications.
    *   $\text{Documented}_j$: **Documented AI Impact**. Quantified AI benefits with clear audit trails, including ROI reports, efficiency gains, revenue uplift attributed to AI, and well-documented intellectual property.
    *   $\text{Sustainable}_j$: **Sustainable AI Capabilities**. Embedded, long-term AI capabilities versus one-time projects. This includes a robust AI talent pipeline, scalable AI infrastructure, and a culture of continuous AI innovation.
    *   $w_1, w_2, w_3$: **Weighting Factors**. These sliders allow me to prioritize different aspects of AI capability that are most likely to drive a premium during an exit. For example, some buyers may value proven financial impact ($w_2$) more than raw technological visibility ($w_1$).

    The `Multiple Attribution Model` then translates this Exit-AI-R score into a potential uplift on the baseline valuation multiple:
    $$ \text{Projected Exit Multiple}_j = \text{BaselineMultiple}_{k} + \text{AI Premium Coefficient} \cdot \text{Exit-AI-R}_j/100 $$
    Here, the `AI Premium Coefficient` acts as a sensitivity factor for how much the exit multiple is boosted by a higher Exit-AI-R score.
    """)

    # Widgets for calibration, values sourced from session_state
    w1_val = st.slider(
        "Weight for Visible AI Capabilities ($w_1$)",
        min_value=0.20, max_value=0.50, value=st.session_state.w1_slider, step=0.01, key="w1_slider",
        help="Prioritize AI capabilities that are easily apparent to buyers (e.g., product features, technology stack). Default $w_1 = 0.35$."
    )
    w2_val = st.slider(
        "Weight for Documented AI Impact ($w_2$)",
        min_value=0.20, max_value=0.50, value=st.session_state.w2_slider, step=0.01, key="w2_slider",
        help="Emphasize quantified AI impact with clear audit trails. Default $w_2 = 0.40$."
    )
    w3_val = st.slider(
        "Weight for Sustainable AI Capabilities ($w_3$)",
        min_value=0.10, max_value=0.40, value=st.session_state.w3_slider, step=0.01, key="w3_slider",
        help="Focus on embedded, long-term AI capabilities versus one-time projects. Default $w_3 = 0.25$."
    )

    if st.button("Recalculate Exit-Readiness & Valuation", key="recalculate_exit_button", help="Click to re-compute Exit-AI-R scores and projected multiples with the selected weights."):
        st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df, w1_val, w2_val, w3_val)
        st.session_state.exit_ai_r_recalculated = True
        st.success("Exit-Readiness scores and projected valuations re-calculated based on your weighting.")
    elif not st.session_state.exit_ai_r_recalculated:
        # Initial calculation on page load if not already done
        st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df, st.session_state.w1_slider, st.session_state.w2_slider, st.session_state.w3_slider)
        st.session_state.exit_ai_r_recalculated = True
        st.info("Initial Exit-Readiness scores calculated. Adjust the sliders and click 'Recalculate' to refine.")


    latest_quarter = st.session_state.portfolio_df['Quarter'].max()
    latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == latest_quarter].copy()

    st.subheader(f"Latest Quarter's Exit-Readiness and Projected Valuation Impact ({latest_quarter}):")
    st.markdown("""
    This table presents the calculated `Exit-AI-R Scores` and the `Projected Exit Multiples` for each company in the latest quarter.
    As a Portfolio Manager, this analysis provides critical data for our exit planning strategy,
    allowing me to build an evidence-based narrative around AI capabilities to maximize valuation multiples during exit.
    I can see which companies have a strong AI story that will resonate with potential buyers.
    """)
    st.dataframe(latest_quarter_df[[
        'CompanyName', 'Industry', 'Exit_AI_R_Score', 'BaselineMultiple', 'AI_Premium_Multiple_Additive', 'Projected_Exit_Multiple'
    ]].sort_values(by='Projected_Exit_Multiple', ascending=False).reset_index(drop=True))

    st.subheader(f"Exit-AI-R Score vs. Projected Exit Multiple (Latest Quarter) ({latest_quarter})")
    st.markdown("This scatter plot visualizes the relationship between a company's Exit-AI-R score and its projected exit multiple. Companies with higher Exit-AI-R scores are expected to command higher valuation multiples, reflecting a stronger AI-driven exit narrative. The size of the points indicates the Attributed EBITDA Impact, showing companies that not only have good exit readiness but also strong financial performance from AI.")

    fig = px.scatter(
        latest_quarter_df,
        x='Exit_AI_R_Score',
        y='Projected_Exit_Multiple',
        color='Industry',
        size='Attributed_EBITDA_Impact_Pct', # Using Attributed_EBITDA_Impact_Pct as it's a calculated financial metric
        hover_name='CompanyName',
        title=f'Exit-AI-R Score vs. Projected Exit Multiple in {latest_quarter}',
        labels={
            'Exit_AI_R_Score': 'Exit-AI-R Score (0-100)',
            'Projected_Exit_Multiple': 'Projected Exit Multiple',
            'Attributed_EBITDA_Impact_Pct': 'Attributed EBITDA Impact (%)'
        }
    )
    # Add company names as text labels
    for _, row in latest_quarter_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Exit_AI_R_Score']],
            y=[row['Projected_Exit_Multiple']],
            mode='text',
            text=[row['CompanyName']],
            textposition="top center",
            textfont=dict(size=9, color="black"),
            showlegend=False
        ))

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

    st.success("You have completed the full AI performance review cycle for your portfolio companies. The insights gained from this dashboard empower you to make data-driven decisions that optimize your fund's overall AI strategy and maximize risk-adjusted returns, especially in preparation for strategic exits.")


# --- Main Application Logic ---
st.set_page_config(page_title="QuLab: Portfolio AI Performance & Benchmarking Dashboard", layout="wide")

# --- Sidebar for Global Controls and Navigation ---
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg") # Placeholder fund logo
st.sidebar.title("Portfolio AI Performance & Benchmarking")
st.sidebar.divider()

st.sidebar.header("Global Portfolio Setup")
num_companies = st.sidebar.number_input(
    "Number of Portfolio Companies",
    min_value=5, max_value=20, value=10, key="num_companies_input",
    help="Define the number of synthetic portfolio companies to generate."
)
num_quarters = st.sidebar.number_input(
    "Number of Quarters (History)",
    min_value=2, max_value=10, value=5, key="num_quarters_input",
    help="Specify the number of historical quarters for which data will be generated."
)

if st.sidebar.button(
    "Generate New Portfolio Data",
    key="generate_data_button",
    help="Click to create a new synthetic dataset based on the parameters above. All subsequent calculations will use this data."
):
    st.session_state.portfolio_df = load_portfolio_data(num_companies, num_quarters)
    # Reset calculation flags to ensure calculations are rerun for new data
    st.session_state.org_ai_r_recalculated = False
    st.session_state.exit_ai_r_recalculated = False
    st.sidebar.success("New synthetic portfolio data generated successfully!")
    st.rerun() # Rerun to ensure all pages pick up new data and calculations are triggered

# Initialize portfolio data and calculation flags if not already present in session_state
if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = load_portfolio_data(num_companies, num_quarters)
if "org_ai_r_recalculated" not in st.session_state:
    st.session_state.org_ai_r_recalculated = False
if "exit_ai_r_recalculated" not in st.session_state:
    st.session_state.exit_ai_r_recalculated = False

# Initialize all slider default values in session_state to prevent KeyErrors on page transitions
# These are used in the respective pages but initialized globally here for robustness
if "alpha_slider" not in st.session_state:
    st.session_state.alpha_slider = 0.60
if "beta_slider" not in st.session_state:
    st.session_state.beta_slider = 0.15
if "w1_slider" not in st.session_state:
    st.session_state.w1_slider = 0.35
if "w2_slider" not in st.session_state:
    st.session_state.w2_slider = 0.40
if "w3_slider" not in st.session_state:
    st.session_state.w3_slider = 0.25
if "coe_org_ai_r_threshold" not in st.session_state:
    st.session_state.coe_org_ai_r_threshold = 75
if "coe_ebitda_threshold" not in st.session_state:
    st.session_state.coe_ebitda_threshold = 3.0
if "review_org_ai_r_threshold" not in st.session_state:
    st.session_state.review_org_ai_r_threshold = 50
if "review_ebitda_threshold" not in st.session_state:
    st.session_state.review_ebitda_threshold = 1.0


st.sidebar.divider()
page_options = [
    "1. Initializing Portfolio Data",
    "2. Calculating Org-AI-R Scores",
    "3. Benchmarking AI Performance",
    "4. AI Investment & EBITDA Impact",
    "5. Tracking Progress Over Time",
    "6. Actionable Insights: CoE & Review",
    "7. Exit-Readiness & Valuation"
]
page = st.sidebar.radio("Portfolio Review Stages", page_options, key="page_selection")

st.title("QuLab: Portfolio AI Performance & Benchmarking Dashboard")
st.divider()

# --- Overall Narrative for the Portfolio Manager Persona ---
st.markdown("""
Welcome, Portfolio Manager! In this lab, you are stepping into a crucial role within a Private Equity fund. Your mission is to systematically evaluate and enhance the AI performance across your diverse portfolio companies.
The challenge is clear: AI isn't just a buzzword; it's a critical lever for driving growth, efficiency, and ultimately, maximizing exit valuations. But to harness its full potential, we need a robust, data-driven framework to quantify its impact and guide strategic decisions.

This application is your essential toolkit. It will guide you through an end-to-end AI performance review cycle, mirroring the real-world tasks you perform in your job:
1.  **Ingest and understand your portfolio's AI data.**
2.  **Calibrate and compute key AI readiness metrics (Org-AI-R Scores).**
3.  **Benchmark companies against their peers and industry standards.**
4.  **Quantify the financial impact and efficiency of AI investments (EBITDA Impact).**
5.  **Track progress over time to identify trends and assess long-term strategies.**
6.  **Identify 'Centers of Excellence' to scale best practices and 'Companies for Review' needing intervention.**
7.  **Assess AI's contribution to exit readiness and potential valuation uplift.**

Each step offers interactive controls, allowing you to fine-tune assumptions and immediately see the impact on your portfolio. You'll gain actionable insights to allocate resources, drive value creation, and build compelling narratives for future exits. Let's begin optimizing our fund's AI strategy!
""")

st.divider()

# --- Conditional Page Rendering ---
# Calls the embedded page functions
if page == "1. Initializing Portfolio Data":
    page_1_initializing_data_main()
elif page == "2. Calculating Org-AI-R Scores":
    page_2_calculating_org_ai_r_main()
elif page == "3. Benchmarking AI Performance":
    page_3_benchmarking_ai_performance_main()
elif page == "4. AI Investment & EBITDA Impact":
    page_4_ai_investment_ebitda_impact_main()
elif page == "5. Tracking Progress Over Time":
    page_5_tracking_progress_main()
elif page == "6. Actionable Insights: CoE & Review":
    page_6_actionable_insights_main()
elif page == "7. Exit-Readiness & Valuation":
    page_7_exit_readiness_main()
