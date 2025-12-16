
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
from io import StringIO

# Suppress specific warnings for cleaner Streamlit output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
# Suppress FutureWarnings specifically from pandas.groupby(..., observed=False)
# This addresses warnings like "DataFrameGroupBy.apply operated on the grouping columns."
# and "The default of observed=False is deprecated and will be changed to True..."
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="QuLab: Portfolio AI Performance & Benchmarking Dashboard", layout="wide")

# --- Fund Branding & Title ---
# Removed the st.sidebar.write("### QuLab Fund Logo") as it was not in the original spec
# and might be causing issues with AppTest's internal element indexing.
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg") 
st.sidebar.divider()
st.sidebar.title("QuLab: Portfolio AI Performance & Benchmarking Dashboard")
st.divider()

# --- Application-wide Story Narrative ---
st.markdown("""
In this lab, we embark on a journey to empower Private Equity Portfolio Managers with a systematic and data-driven framework for assessing and optimizing AI performance across their portfolio companies.

As a Private Equity Portfolio Manager, I am constantly seeking opportunities to drive value across my fund's diverse portfolio. In today's landscape, AI is a critical lever for growth and efficiency, but its impact needs to be systematically quantified and managed. This application will guide me through a complete AI performance review cycle for my portfolio companies.

I'll start by loading the latest data, then compute key AI readiness and financial impact metrics. With these insights, I can benchmark companies against their peers and industry, track their progress over time, identify my "Centers of Excellence" whose AI best practices can be scaled, and pinpoint "Companies for Review" that need immediate strategic attention. Finally, as exit planning is always top of mind, I'll assess how a company's AI capabilities enhance its exit readiness and potential valuation.

Each step in this journey empowers me to make data-driven decisions that optimize our fund's overall AI strategy and maximize risk-adjusted returns.
""")
st.divider()

# --- Global Data Controls (Sidebar) ---
st.sidebar.header("Global Portfolio Setup")
num_companies = st.sidebar.number_input("Number of Portfolio Companies", min_value=5, max_value=20, value=10, key="num_companies_input")
num_quarters = st.sidebar.number_input("Number of Quarters (History)", min_value=2, max_value=10, value=5, key="num_quarters_input")

# --- Cached Functions for Data Generation and Calculations ---

@st.cache_data(ttl="2h")
def load_portfolio_data(num_companies_val, num_quarters_val):
    """Generates synthetic portfolio data for the specified number of companies and quarters."""
    np.random.seed(42) # For reproducibility
    companies = [f"Company {i+1}" for i in range(num_companies_val)]
    industries = ['Tech', 'Healthcare', 'Retail', 'Manufacturing', 'Finance']
    
    data = []
    for q_idx in range(num_quarters_val):
        # Generate quarter strings like Q1 FY24, Q2 FY24, etc.
        quarter = f"Q{q_idx+1} FY24" 
        for i, company_name in enumerate(companies):
            industry = np.random.choice(industries)
            
            # Core AI Readiness components (simulated 0-100 scale)
            idiosyncratic_readiness = np.random.uniform(30, 90) 
            systematic_opportunity = np.random.uniform(40, 95)  
            synergy = np.random.uniform(5, 20) 
            
            # AI Investment and Impact
            ai_investment = np.random.uniform(0.5, 10.0) * 1_000_000 # in millions
            ebitda_impact = np.random.uniform(0.5, 8.0) # Percentage impact
            baseline_ebitda = np.random.uniform(50, 500) * 1_000_000 # in millions
            
            # Exit Readiness Factors (simulated 0-100 scale)
            visible_ai = np.random.uniform(30, 90) 
            documented_ai = np.random.uniform(20, 85) 
            sustainable_ai = np.random.uniform(40, 90) 
            
            # Coefficients (simulated)
            gamma_coefficient = np.random.uniform(0.05, 0.15) # For EBITDA attribution
            ai_premium_coefficient = np.random.uniform(0.01, 0.05) # For exit multiple premium
            baseline_multiple = np.random.uniform(8.0, 15.0) # Industry baseline multiple

            data.append({
                'Quarter': quarter,
                'CompanyID': i,
                'CompanyName': company_name,
                'Industry': industry,
                'IdiosyncraticReadiness': idiosyncratic_readiness,
                'SystematicOpportunity': systematic_opportunity,
                'Synergy': synergy,
                'AI_Investment': ai_investment,
                'EBITDA_Impact': ebitda_impact,
                'BaselineEBITDA': baseline_ebitda,
                'Visible': visible_ai,
                'Documented': documented_ai,
                'Sustainable': sustainable_ai,
                'GammaCoefficient': gamma_coefficient,
                'AI_PremiumCoefficient': ai_premium_coefficient,
                'BaselineMultiple': baseline_multiple
            })
    
    df = pd.DataFrame(data)
    
    # Initialize IndustryMeanOrgAIR and IndustryStdDevOrgAIR using SystematicOpportunity as an initial proxy
    df['IndustryMeanOrgAIR'] = df.groupby(['Industry', 'Quarter'], observed=False)['SystematicOpportunity'].transform('mean')
    df['IndustryStdDevOrgAIR'] = df.groupby(['Industry', 'Quarter'], observed=False)['SystematicOpportunity'].transform('std').fillna(5) 
    
    # Ensure correct data types and order for quarters
    quarter_categories = [f"Q{q+1} FY24" for q in range(num_quarters_val)]
    df['Quarter'] = pd.Categorical(df['Quarter'], categories=quarter_categories, ordered=True)
    
    return df

@st.cache_data(ttl="2h")
def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
    """Calculates the PE Org-AI-R Score for each company based on input weights."""
    if df.empty:
        return df
    
    df_copy = df.copy() 
    
    # Calculate Org-AI-R Score
    df_copy['Org_AI_R_Score'] = (
        alpha * df_copy['IdiosyncraticReadiness'] +
        (1 - alpha) * df_copy['SystematicOpportunity'] +
        beta * df_copy['Synergy']
    )
    df_copy['Org_AI_R_Score'] = np.clip(df_copy['Org_AI_R_Score'], 0, 100) # Clip to a 0-100 range
    return df_copy

@st.cache_data(ttl="2h")
def calculate_benchmarks(df):
    """Calculates Org-AI-R Percentile and Industry-Adjusted Z-Score for benchmarking."""
    if df.empty or 'Org_AI_R_Score' not in df.columns:
        return df.assign(Org_AI_R_Percentile=0.0, Org_AI_R_Z_Score=0.0) 
    
    df_copy = df.copy()
    
    # Within-Portfolio Benchmarking (Percentile)
    df_copy['Org_AI_R_Percentile'] = df_copy.groupby('Quarter', observed=False)['Org_AI_R_Score'].rank(pct=True) * 100
    
    # Update IndustryMeanOrgAIR based on the newly calculated Org_AI_R_Score
    df_copy['IndustryMeanOrgAIR'] = df_copy.groupby(['Industry', 'Quarter'], observed=False)['Org_AI_R_Score'].transform('mean')
    
    # Cross-Portfolio Benchmarking (Z-Score) using transform
    # Handle cases with insufficient data for Z-score (std dev = 0 or single element)
    def safe_zscore_transform(series):
        if len(series) > 1 and series.std() > 0:
            return zscore(series)
        return pd.Series(0.0, index=series.index) # Return Series of 0.0 with original index
    
    df_copy['Org_AI_R_Z_Score'] = df_copy.groupby(['Industry', 'Quarter'], observed=False)['Org_AI_R_Score'].transform(safe_zscore_transform)

    return df_copy

@st.cache_data(ttl="2h")
def calculate_aie_ebitda(df):
    """Calculates AI Investment Efficiency and Attributed EBITDA Impact."""
    if df.empty or 'Org_AI_R_Score' not in df.columns:
        # Return a DataFrame with the expected columns if input is empty
        return df.assign(Delta_Org_AI_R=0.0, AI_Investment_Efficiency=0.0, 
                         Attributed_EBITDA_Impact_Pct=0.0, Attributed_EBITDA_Impact_Absolute=0.0)
    
    df_copy = df.copy()
    df_sorted = df_copy.sort_values(by=['CompanyID', 'Quarter'])
    
    # Calculate change in Org-AI-R from previous quarter for each company
    df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID', observed=False)['Org_AI_R_Score'].diff().fillna(0)
    
    # AI Investment Efficiency (AIE)
    df_sorted['AI_Investment_Efficiency'] = df_sorted.apply(
        lambda row: (row['Delta_Org_AI_R'] / row['AI_Investment']) * row['EBITDA_Impact'] * 1_000_000 # Scale for readability (e.g., impact points per million invested)
        if row['AI_Investment'] > 0 and row['Delta_Org_AI_R'] > 0 else 0, axis=1
    )
    
    # Attributed EBITDA Impact Percentage
    df_sorted['Attributed_EBITDA_Impact_Pct'] = df_sorted.apply(
        lambda row: row['GammaCoefficient'] * row['Delta_Org_AI_R'] * (row['IndustryMeanOrgAIR'] / 100) # H^R_org,k as a % from 100
        if row['Delta_Org_AI_R'] > 0 else 0, axis=1
    )
    df_sorted['Attributed_EBITDA_Impact_Absolute'] = (df_sorted['Attributed_EBITDA_Impact_Pct'] / 100) * df_sorted['BaselineEBITDA']
    
    return df_sorted

@st.cache_data(ttl="2h")
def identify_actionable_insights(df, org_ai_r_threshold_coe, ebitda_impact_threshold_coe,
                                 org_ai_r_threshold_review, ebitda_impact_threshold_review):
    """Identifies Centers of Excellence and Companies for Review based on user-defined thresholds."""
    if df.empty or 'Org_AI_R_Score' not in df.columns or 'EBITDA_Impact' not in df.columns:
        return pd.DataFrame(), pd.DataFrame() 
    
    # Get the latest data for each company (assuming 'Quarter' is ordered correctly)
    latest_data = df.loc[df.groupby('CompanyID', observed=False)['Quarter'].idxmax()]
    
    centers_of_excellence = latest_data[
        (latest_data['Org_AI_R_Score'] > org_ai_r_threshold_coe) &
        (latest_data['EBITDA_Impact'] > ebitda_impact_threshold_coe)
    ].sort_values(by='Org_AI_R_Score', ascending=False)
    
    companies_for_review = latest_data[
        (latest_data['Org_AI_R_Score'] <= org_ai_r_threshold_review) |
        (latest_data['EBITDA_Impact'] <= ebitda_impact_threshold_review)
    ].sort_values(by='Org_AI_R_Score', ascending=True)
    
    return centers_of_excellence, companies_for_review

@st.cache_data(ttl="2h")
def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
    """Calculates Exit-AI-R Score and Projected Exit Multiple based on user-defined weights."""
    if df.empty:
        return df.assign(Exit_AI_R_Score=0.0, AI_Premium_Multiple_Additive=0.0, Projected_Exit_Multiple=0.0)
    
    df_copy = df.copy() 
    
    # Calculate Exit-AI-R Score
    df_copy['Exit_AI_R_Score'] = (
        w1 * df_copy['Visible'] + w2 * df_copy['Documented'] + w3 * df_copy['Sustainable']
    )
    df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100)
    
    # Calculate AI Premium and Projected Exit Multiple
    df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
    df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
    
    return df_copy


# --- Session State Initialization and Global Data Generation Logic ---
if "portfolio_df" not in st.session_state:
    # Default values from the sidebar inputs
    default_num_companies = 10
    default_num_quarters = 5
    st.session_state.portfolio_df = load_portfolio_data(default_num_companies, default_num_quarters)
    # Perform initial calculations immediately after data load
    st.session_state.portfolio_df = calculate_org_ai_r(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_benchmarks(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df)
    st.session_state.page_selection = "1. Initializing Portfolio Data" # Ensure initial page is set

# Button to generate new data
if st.sidebar.button("Generate New Portfolio Data", key="generate_data_button", help="Click to create a new synthetic dataset based on the parameters above. All subsequent calculations will use this data."):
    st.session_state.portfolio_df = load_portfolio_data(num_companies, num_quarters)
    # Recalculate all metrics immediately after new data generation
    st.session_state.portfolio_df = calculate_org_ai_r(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_benchmarks(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df)
    st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df)
    st.success("New synthetic portfolio data generated successfully! All calculations have been re-run.")
    # Reset page selection to start from the beginning with new data
    st.session_state.page_selection = "1. Initializing Portfolio Data"
    st.rerun() # Rerun to update page with new data immediately


# --- Page Navigation (Sidebar) ---
# Ensure page_selection is initialized in session state
if "page_selection" not in st.session_state:
    st.session_state.page_selection = "1. Initializing Portfolio Data"

page_selection = st.sidebar.radio("Portfolio Review Stages", 
                                  ["1. Initializing Portfolio Data", 
                                   "2. Calculating Org-AI-R Scores", 
                                   "3. Benchmarking AI Performance", 
                                   "4. AI Investment & EBITDA Impact", 
                                   "5. Tracking Progress Over Time", 
                                   "6. Actionable Insights: CoE & Review", 
                                   "7. Exit-Readiness & Valuation"], 
                                  key="page_selection")

# --- Main Content Area: Dynamic Page Rendering ---

# Page 1: Initializing Portfolio Data
if page_selection == "1. Initializing Portfolio Data":
    st.header("1. Initializing Portfolio Data: Overview")
    st.markdown("""
    As a Portfolio Manager, my first step is to ensure I'm working with the most current and comprehensive data for my fund's holdings. This initial data load forms the bedrock for all subsequent AI performance analysis and strategic decision-making. I need to quickly grasp the structure and scope of the data before diving into calculations.
    """)
    
    st.subheader("Overview of Generated Portfolio Data (First 5 Rows):")
    if not st.session_state.portfolio_df.empty:
        st.dataframe(st.session_state.portfolio_df.head())
    else:
        # Render an empty dataframe with expected columns if data is empty for AppTest compatibility
        empty_df = pd.DataFrame(columns=['Quarter', 'CompanyID', 'CompanyName', 'Industry', 'IdiosyncraticReadiness', 
                                         'SystematicOpportunity', 'Synergy', 'AI_Investment', 'EBITDA_Impact', 
                                         'BaselineEBITDA', 'Visible', 'Documented', 'Sustainable', 
                                         'GammaCoefficient', 'AI_PremiumCoefficient', 'BaselineMultiple'])
        st.dataframe(empty_df.head())
        st.info("No portfolio data loaded. Please use the sidebar to generate data.")

    st.subheader("Descriptive Statistics of Portfolio Data:")
    if not st.session_state.portfolio_df.empty:
        st.dataframe(st.session_state.portfolio_df.describe())
        
        st.subheader("Data Information:")
        buffer = StringIO()
        st.session_state.portfolio_df.info(buf=buffer)
        st.text(buffer.getvalue())
    else:
        # Render an empty dataframe describe for AppTest compatibility
        empty_describe_df = pd.DataFrame().describe()
        st.dataframe(empty_describe_df)
        st.info("No portfolio data loaded. Descriptive statistics are not available.")

# Page 2: Calculating Org-AI-R Scores
elif page_selection == "2. Calculating Org-AI-R Scores":
    st.header("2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment")
    st.markdown("""
    The core of our AI performance tracking is the PE Org-AI-R score, a crucial metric that quantifies each portfolio company's overall AI maturity and readiness for value creation. As a Portfolio Manager, I need the flexibility to calibrate this score to reflect my fund's strategic priorities, emphasizing either internal capabilities or external market opportunities.
    """)
    st.markdown(r"""
    The formula for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:
    $$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$
    """)
    st.markdown(r"""
    where:
    *   $V^R_{org,j}(t)$: Idiosyncratic Readiness for company $j$ at time $t$. This represents company-specific capabilities related to AI, such as data infrastructure, talent, and leadership commitment.
    *   $H^R_{org,k}(t)$: Systematic Opportunity for industry $k$ at time $t$. This reflects industry-level AI potential, adoption rates, disruption potential, and competitive dynamics within the sector.
    *   $\alpha$: Weight for Idiosyncratic Readiness. This slider allows me to prioritize company-specific capabilities versus industry-level AI potential.
    *   $\beta$: Synergy Coefficient. This quantifies the additional value derived from the interplay between idiosyncratic readiness and systematic opportunity, reflecting how well a company is positioned to capitalize on industry trends given its internal strengths.
    *   $\text{Synergy}(V^R_{org,j}, H^R_{org,k})$: A term representing the alignment and integration between a company's idiosyncratic readiness and the systematic opportunity in its industry.
    """)
    
    st.markdown("---")

    # Widgets for Org-AI-R calculation parameters
    alpha = st.slider("Weight for Idiosyncratic Readiness ($\alpha$)", min_value=0.55, max_value=0.70, value=0.60, step=0.01, key="alpha_slider", help="Adjust this to prioritize company-specific capabilities ($V^R_{org,j}$) versus industry-level AI potential ($H^R_{org,k}$). Default $\alpha = 0.60$.")
    beta = st.slider("Synergy Coefficient ($\beta$)", min_value=0.08, max_value=0.25, value=0.15, step=0.01, key="beta_slider", help="Quantify the additional value derived from the interplay between idiosyncratic readiness and systematic opportunity. Default $\beta = 0.15$.")
    
    if st.button("Recalculate Org-AI-R Scores", key="recalculate_org_ai_r_button", help="Click to re-compute Org-AI-R scores with the selected weights."):
        st.session_state.portfolio_df = calculate_org_ai_r(st.session_state.portfolio_df, alpha, beta)
        st.session_state.portfolio_df = calculate_benchmarks(st.session_state.portfolio_df) # Recalculate benchmarks after Org-AI-R changes
        st.session_state.portfolio_df = calculate_aie_ebitda(st.session_state.portfolio_df) # Recalculate AIE/EBITDA impact
        st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df) # Re-calculate exit values to ensure consistency
        st.success("Org-AI-R scores and related metrics updated successfully!")
    
    st.subheader("Latest Quarter's PE Org-AI-R Scores:")
    if not st.session_state.portfolio_df.empty:
        latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == st.session_state.portfolio_df['Quarter'].max()]
        st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Org_AI_R_Score']].sort_values(by='Org_AI_R_Score', ascending=False).set_index('CompanyName'))
    else:
        # Render an empty dataframe with expected columns for AppTest compatibility
        empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'Org_AI_R_Score']).set_index('CompanyName')
        st.dataframe(empty_df)
        st.info("No Org-AI-R scores available. Please generate portfolio data first.")

# Page 3: Benchmarking AI Performance
elif page_selection == "3. Benchmarking AI Performance":
    st.header("3. Benchmarking Portfolio Companies: Identifying Relative AI Performance")
    st.markdown("""
    Understanding a company's standalone Org-AI-R score is a good start, but as a Portfolio Manager, I need to know how it performs relative to its peers. Benchmarking allows me to identify leaders and laggards, both within my portfolio and against industry standards, providing crucial context for strategic resource allocation.
    
    We use two key benchmarking metrics:
    *   **Within-Portfolio Benchmarking (Org-AI-R Percentile):** This tells me how a company stacks up against all other companies in my fund.
    *   **Cross-Portfolio Benchmarking (Org-AI-R Z-Score):** This adjusts for industry differences, showing me how a company's AI readiness deviates from its industry's average, helping me identify true outperformers or underperformers relative to their sector.
    """)
    st.markdown(r"""
    **Org-AI-R Percentile:**
    $$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$
    where $\text{Rank}(\text{Org-AI-R}_j)$ is the rank of company $j$'s Org-AI-R score within the portfolio (from lowest to highest), and $\text{Portfolio Size}$ is the total number of companies in the portfolio for that quarter.
    """)
    st.markdown(r"""
    **Org-AI-R Z-Score:**
    $$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
    where $\text{Org-AI-R}_j$ is the Org-AI-R score of company $j$, $\mu_k$ is the mean Org-AI-R score for industry $k$, and $\sigma_k$ is the standard deviation of Org-AI-R scores for industry $k$.
    """)
    st.markdown("""
    These benchmarks are invaluable for identifying best practices within our leading companies and pinpointing those that require targeted support to catch up to their industry peers.
    """)
    st.markdown("---")

    if not st.session_state.portfolio_df.empty:
        quarter_options = st.session_state.portfolio_df['Quarter'].unique().tolist()
        
        # Determine the index of the latest quarter for the selectbox default
        selected_quarter_index = len(quarter_options) - 1 if quarter_options else 0
        
        selected_quarter = st.selectbox("Select Quarter for Benchmarking", 
                                        options=quarter_options, 
                                        index=selected_quarter_index, # Use index for default
                                        key="benchmark_quarter_select")
        
        filtered_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == selected_quarter].copy()
        
        st.subheader(f"Org-AI-R Benchmarks for {selected_quarter}:")
        st.dataframe(filtered_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Org_AI_R_Percentile', 'Org_AI_R_Z_Score']]
                     .sort_values(by='Org_AI_R_Score', ascending=False).set_index('CompanyName'))

        # Visualization 1: Bar Chart - Latest Quarter Org-AI-R Scores by Company
        st.subheader(f"Org-AI-R Scores by Company for {selected_quarter}")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(x='CompanyName', y='Org_AI_R_Score', hue='Industry', data=filtered_df.sort_values(by='Org_AI_R_Score', ascending=False), ax=ax1, palette='viridis')
        portfolio_avg_org_ai_r = filtered_df['Org_AI_R_Score'].mean()
        ax1.axhline(portfolio_avg_org_ai_r, color='red', linestyle='--', label=f'Portfolio Average ({portfolio_avg_org_ai_r:.1f})')
        ax1.set_title(f'Org-AI-R Scores by Company ({selected_quarter})')
        ax1.set_ylabel('Org-AI-R Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='Industry', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # Visualization 2: Scatter Plot - Org-AI-R Score vs. Industry-Adjusted Z-Score
        st.subheader(f"Org-AI-R Score vs. Industry-Adjusted Z-Score for {selected_quarter}")
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        sns.scatterplot(x='Org_AI_R_Score', y='Org_AI_R_Z_Score', hue='Industry', size='Org_AI_R_Percentile', sizes=(50, 400), 
                        data=filtered_df, ax=ax2, palette='magma', alpha=0.8, legend='full')
        
        # Portfolio mean Org-AI-R (vertical line)
        portfolio_mean_org_ai_r = filtered_df['Org_AI_R_Score'].mean()
        ax2.axvline(portfolio_mean_org_ai_r, color='gray', linestyle=':', label=f'Portfolio Mean Org-AI-R ({portfolio_mean_org_ai_r:.1f})')
        
        # Industry mean Z-score (horizontal line at Y=0)
        ax2.axhline(0, color='black', linestyle='--', label='Industry Mean Z-Score (0)')
        
        ax2.set_title(f'Org-AI-R Score vs. Industry-Adjusted Z-Score ({selected_quarter})')
        ax2.set_xlabel('Org-AI-R Score')
        ax2.set_ylabel('Org-AI-R Z-Score (Industry-Adjusted)')
        
        # Combine legends to avoid duplicates for lines
        handles, labels = ax2.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax2.legend(unique_labels.values(), unique_labels.keys(), title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    else:
        # Render an empty dataframe with expected columns for AppTest compatibility
        empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'Org_AI_R_Score', 'Org_AI_R_Percentile', 'Org_AI_R_Z_Score']).set_index('CompanyName')
        st.dataframe(empty_df)
        st.info("No data available for benchmarking. Please generate portfolio data first.")

# Page 4: AI Investment & EBITDA Impact
elif page_selection == "4. AI Investment & EBITDA Impact":
    st.header("4. Assessing AI Investment Efficiency and EBITDA Attribution")
    st.markdown("""
    As a Portfolio Manager, I need to go beyond just scores and understand the tangible financial impact of AI initiatives. This section quantifies the return on AI investment and attributes EBITDA growth directly to improvements in AI readiness. This allows me to evaluate the effectiveness of capital deployment and identify areas where AI investments are truly moving the needle.
    """)
    st.markdown(r"""
    **AI Investment Efficiency ($\text{AIE}_j$):**
    $$ \text{AIE}_j = \left( \frac{\Delta\text{Org-AI-R}_j}{\text{AI Investment}_j} \right) \times \text{EBITDA Impact}_j \times C $$
    where $\Delta\text{Org-AI-R}_j$ is the change in Org-AI-R score for company $j$, $\text{AI Investment}_j$ is the AI investment for company $j$, $\text{EBITDA Impact}_j$ is the direct percentage EBITDA impact, and $C$ is a scaling constant (e.g., $1,000,000$ to represent impact points per million invested). A higher AIE indicates more efficient capital deployment for AI initiatives.
    """)
    st.markdown(r"""
    **Attributed EBITDA Impact Percentage ($\Delta\text{EBITDA}\%$):**
    $$ \Delta\text{EBITDA}\% = \gamma \cdot \Delta\text{Org-AI-R}_j \cdot (H^R_{org,k} / 100) $$
    where $\gamma$ is a scaling coefficient (`GammaCoefficient`), $\Delta\text{Org-AI-R}_j$ is the change in Org-AI-R score for company $j$, and $H^R_{org,k}$ is the systematic opportunity for industry $k$ (proxied by `IndustryMeanOrgAIR`). This estimates the percentage increase in EBITDA directly attributed to the change in Org-AI-R score, factoring in industry opportunity.
    """)
    st.markdown("""
    This analysis provides critical insights into which companies are most effectively translating their AI maturity into financial returns, guiding future investment decisions and resource allocation.
    """)
    st.markdown("---")

    if not st.session_state.portfolio_df.empty:
        latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == st.session_state.portfolio_df['Quarter'].max()]
        
        st.subheader("Latest Quarter's AI Investment Efficiency and Attributed EBITDA Impact:")
        st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'AI_Investment', 'Delta_Org_AI_R', 
                                        'AI_Investment_Efficiency', 'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute']]
                     .sort_values(by='AI_Investment_Efficiency', ascending=False).set_index('CompanyName'))

        # Visualization: Scatter Plot - AI Investment vs. Efficiency
        st.subheader("AI Investment vs. Efficiency (Latest Quarter, Highlighting Attributed EBITDA Impact)")
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        sns.scatterplot(x='AI_Investment', y='AI_Investment_Efficiency', hue='Industry', size='Attributed_EBITDA_Impact_Pct', 
                        sizes=(50, 400), data=latest_quarter_df, ax=ax3, palette='tab10', alpha=0.8, legend='full')
        ax3.set_xscale('log') # Log scale for AI Investment for better distribution visibility
        ax3.set_title('AI Investment vs. Efficiency (Latest Quarter)')
        ax3.set_xlabel('AI Investment (Log Scale)')
        ax3.set_ylabel('AI Investment Efficiency (Impact per M invested)')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        # Render an empty dataframe with expected columns for AppTest compatibility
        empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'AI_Investment', 'Delta_Org_AI_R', 
                                         'AI_Investment_Efficiency', 'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute']).set_index('CompanyName')
        st.dataframe(empty_df)
        st.info("No data available for AI investment and EBITDA impact. Please generate portfolio data first.")

# Page 5: Tracking Progress Over Time
elif page_selection == "5. Tracking Progress Over Time":
    st.header("5. Tracking Progress Over Time: Visualizing Trajectories")
    st.markdown("""
    As a Portfolio Manager, current metrics are important, but understanding the trajectory of performance over time is crucial for assessing long-term strategy effectiveness. This section allows me to monitor how individual companies are progressing in their AI journey and how efficiently they're utilizing their AI investments quarter-over-quarter.
    """)
    st.markdown("---")

    if not st.session_state.portfolio_df.empty:
        all_companies = st.session_state.portfolio_df['CompanyName'].unique().tolist()
        default_companies_to_track = all_companies[:min(5, len(all_companies))]

        selected_companies = st.multiselect("Select Companies to Track (Max 5 for clarity)", 
                                            options=all_companies, 
                                            default=default_companies_to_track, 
                                            key="companies_to_track_multiselect")
        
        # Filter data for selected companies, if none selected, show message
        if selected_companies:
            tracking_df = st.session_state.portfolio_df[st.session_state.portfolio_df['CompanyName'].isin(selected_companies)].copy()

            # Visualization 1: Line Chart - Org-AI-R Score Trajectory Over Time
            st.subheader("Org-AI-R Score Trajectory Over Time")
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            
            # Plot individual company lines
            sns.lineplot(x='Quarter', y='Org_AI_R_Score', hue='CompanyName', marker='o', data=tracking_df, ax=ax4, palette='deep')
            
            # Overlay portfolio average (added observed=False)
            portfolio_avg_df = st.session_state.portfolio_df.groupby('Quarter', observed=False)['Org_AI_R_Score'].mean().reset_index()
            sns.lineplot(x='Quarter', y='Org_AI_R_Score', data=portfolio_avg_df, ax=ax4, color='black', linestyle='--', label='Portfolio Average', marker='x')
            
            ax4.set_title('Org-AI-R Score Trajectory Over Time')
            ax4.set_xlabel('Quarter')
            ax4.set_ylabel('Org-AI-R Score')
            ax4.legend(title='Company / Average', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

            # Visualization 2: Line Chart - AI Investment Efficiency Trajectory Over Time
            st.subheader("AI Investment Efficiency Trajectory Over Time")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            
            # Plot individual company lines
            sns.lineplot(x='Quarter', y='AI_Investment_Efficiency', hue='CompanyName', marker='o', data=tracking_df, ax=ax5, palette='dark')
            
            # Overlay portfolio average (added observed=False)
            portfolio_avg_aie_df = st.session_state.portfolio_df.groupby('Quarter', observed=False)['AI_Investment_Efficiency'].mean().reset_index()
            sns.lineplot(x='Quarter', y='AI_Investment_Efficiency', data=portfolio_avg_aie_df, ax=ax5, color='black', linestyle='--', label='Portfolio Average', marker='x')
            
            ax5.set_title('AI Investment Efficiency Trajectory Over Time')
            ax5.set_xlabel('Quarter')
            ax5.set_ylabel('AI Investment Efficiency')
            ax5.legend(title='Company / Average', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig5)
            plt.close(fig5)
        else:
            st.info("Please select companies to track their progress.")
    else:
        st.info("No data available for tracking progress. Please generate portfolio data first.")

# Page 6: Actionable Insights: CoE & Review
elif page_selection == "6. Actionable Insights: CoE & Review":
    st.header("6. Identifying Centers of Excellence and Companies for Review")
    st.markdown("""
    A key responsibility of a Portfolio Manager is to leverage successes and address underperformance proactively. This section empowers me to define specific criteria for identifying my 'Centers of Excellence' – companies with outstanding AI performance that can serve as benchmarks – and 'Companies for Review' – those needing immediate strategic attention or resource reallocation.
    
    This targeted identification is critical for optimizing our fund's overall AI strategy and maximizing risk-adjusted returns by fostering best practices and mitigating risks.
    """)
    st.markdown("---")

    # Widgets for defining thresholds
    coe_org_ai_r_threshold = st.slider("Org-AI-R Score Threshold for Center of Excellence", min_value=50, max_value=90, value=75, step=1, key="coe_org_ai_r_threshold", help="Companies with Org-AI-R score above this will be considered for 'Centers of Excellence'. Default: 75.")
    coe_ebitda_threshold = st.slider("EBITDA Impact (%) Threshold for Center of Excellence", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="coe_ebitda_threshold", help="Companies with EBITDA Impact above this will be considered for 'Centers of Excellence'. Default: 3%.")
    review_org_ai_r_threshold = st.slider("Org-AI-R Score Threshold for Companies for Review", min_value=20, max_value=70, value=50, step=1, key="review_org_ai_r_threshold", help="Companies with Org-AI-R score below or equal to this will be considered for 'Companies for Review'. Default: 50.")
    review_ebitda_threshold = st.slider("EBITDA Impact (%) Threshold for Companies for Review", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="review_ebitda_threshold", help="Companies with EBITDA Impact below or equal to this will be considered for 'Companies for Review'. Default: 1%.")
    
    re_evaluate_insights = st.button("Re-evaluate Actionable Insights", key="re_evaluate_insights_button", help="Click to re-identify Centers of Excellence and Companies for Review based on the adjusted thresholds.")

    if not st.session_state.portfolio_df.empty:
        # Recalculate insights either on button click or if they haven't been calculated yet for the current state
        if re_evaluate_insights or "centers_of_excellence_df" not in st.session_state or "companies_for_review_df" not in st.session_state:
            st.session_state.centers_of_excellence_df, st.session_state.companies_for_review_df = \
                identify_actionable_insights(st.session_state.portfolio_df, 
                                             coe_org_ai_r_threshold, coe_ebitda_threshold,
                                             review_org_ai_r_threshold, review_ebitda_threshold)
            if re_evaluate_insights:
                st.success("Actionable insights re-evaluated with new thresholds.")

        # Display Centers of Excellence
        st.subheader("--- Centers of Excellence ---")
        st.markdown("""
        **Centers of Excellence:** Portfolio companies with high Org-AI-R scores and significant EBITDA impact, serving as benchmarks for best practices and potential for replication across the fund.
        """)
        if not st.session_state.centers_of_excellence_df.empty:
            st.dataframe(st.session_state.centers_of_excellence_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']]
                         .set_index('CompanyName'))
        else:
            # Render an empty dataframe with expected columns for AppTest compatibility
            empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']).set_index('CompanyName')
            st.dataframe(empty_df)
            st.info("No companies currently meet the 'Centers of Excellence' criteria with the current thresholds.")

        # Display Companies for Review
        st.subheader("--- Companies for Review ---")
        st.markdown("""
        **Companies for Review:** Portfolio companies with lower Org-AI-R scores or minimal EBITDA impact, indicating areas requiring strategic intervention, additional resources, or a re-evaluation of AI strategy.
        """)
        if not st.session_state.companies_for_review_df.empty:
            st.dataframe(st.session_state.companies_for_review_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']]
                         .set_index('CompanyName'))
        else:
            # Render an empty dataframe with expected columns for AppTest compatibility
            empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']).set_index('CompanyName')
            st.dataframe(empty_df)
            st.info("No companies currently meet the 'Companies for Review' criteria with the current thresholds.")

        # Visualization: Portfolio AI Performance: Org-AI-R Score vs. EBITDA Impact (Latest Quarter)
        st.subheader("Portfolio AI Performance: Org-AI-R Score vs. EBITDA Impact (Latest Quarter)")
        latest_data_for_plot = st.session_state.portfolio_df.loc[st.session_state.portfolio_df.groupby('CompanyID', observed=False)['Quarter'].idxmax()]
        
        fig6, ax6 = plt.subplots(figsize=(14, 8))
        
        sns.scatterplot(x='Org_AI_R_Score', y='EBITDA_Impact', hue='Industry', size='AI_Investment_Efficiency', 
                        sizes=(50, 400), data=latest_data_for_plot, ax=ax6, palette='viridis', alpha=0.7, legend='full')
        
        # Highlight CoE companies
        for index, row in st.session_state.centers_of_excellence_df.iterrows():
            ax6.scatter(row['Org_AI_R_Score'], row['EBITDA_Impact'], marker='*', s=1000, color='green', alpha=0.8, edgecolors='black')
            ax6.text(row['Org_AI_R_Score'] + 1, row['EBITDA_Impact'], row['CompanyName'], fontsize=9, ha='left', va='center', color='green', weight='bold')

        # Highlight Companies for Review
        for index, row in st.session_state.companies_for_review_df.iterrows():
            ax6.scatter(row['Org_AI_R_Score'], row['EBITDA_Impact'], marker='X', s=700, color='red', alpha=0.8, edgecolors='black')
            ax6.text(row['Org_AI_R_Score'] + 1, row['EBITDA_Impact'], row['CompanyName'], fontsize=9, ha='left', va='center', color='red', weight='bold')

        # Add threshold lines
        ax6.axvline(coe_org_ai_r_threshold, color='green', linestyle=':', label=f'CoE Org-AI-R Threshold ({coe_org_ai_r_threshold})')
        ax6.axhline(coe_ebitda_threshold, color='green', linestyle=':', label=f'CoE EBITDA Impact Threshold ({coe_ebitda_threshold}%)')
        ax6.axvline(review_org_ai_r_threshold, color='red', linestyle='--', label=f'Review Org-AI-R Threshold ({review_org_ai_r_threshold})')
        ax6.axhline(review_ebitda_threshold, color='red', linestyle='--', label=f'Review EBITDA Impact Threshold ({review_ebitda_threshold}%)')

        ax6.set_title('Portfolio AI Performance: Org-AI-R Score vs. EBITDA Impact (Latest Quarter)')
        ax6.set_xlabel('Org-AI-R Score')
        ax6.set_ylabel('EBITDA Impact (%)')
        ax6.grid(True, linestyle='--', alpha=0.6)
        
        # Consolidate legend for clarity
        handles, labels = ax6.get_legend_handles_labels()
        # Filter out duplicate labels for CoE/Review if they appear multiple times
        unique_labels_dict = {}
        for h, l in zip(handles, labels):
            # Prioritize a single entry for 'Center of Excellence' and 'Company for Review'
            if 'Center of Excellence' in l and 'Center of Excellence' not in unique_labels_dict:
                unique_labels_dict['Center of Excellence'] = h
            elif 'Company for Review' in l and 'Company for Review' not in unique_labels_dict:
                unique_labels_dict['Company for Review'] = h
            elif l not in unique_labels_dict:
                unique_labels_dict[l] = h
        
        ax6.legend(unique_labels_dict.values(), unique_labels_dict.keys(), title='Metrics & Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)
    else:
        st.info("No data available for actionable insights. Please generate portfolio data first.")


# Page 7: Exit-Readiness & Valuation
elif page_selection == "7. Exit-Readiness & Valuation":
    st.header("7. Evaluating Exit-Readiness and Potential Valuation Impact")
    st.markdown("""
    As a Portfolio Manager, preparing for a successful exit is always on my mind. The AI capabilities of our portfolio companies can significantly influence their attractiveness to potential acquirers and, consequently, their exit valuation. This section allows me to assess how "buyer-friendly" a company's AI capabilities are and quantify the potential premium on its exit multiple.
    """)
    st.markdown(r"""
    The formula for the Exit-Readiness Score for portfolio company $j$ preparing for exit is:
    $$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$
    """)
    st.markdown(r"""
    where:
    *   $\text{Visible}_j$: **Visible AI Capabilities:** AI features apparent to buyers (e.g., product functionality, technology stack).
    *   $\text{Documented}_j$: **Documented AI Impact:** Quantified AI benefits with an auditable trail.
    *   $\text{Sustainable}_j$: **Sustainable AI Capabilities:** Embedded, long-term AI capabilities vs. one-time projects.
    *   $w_1, w_2, w_3$: Weighting factors that allow me to emphasize different aspects buyers might prioritize for a stronger exit narrative.
    """)
    st.markdown(r"""
    The Multiple Attribution Model then translates this Exit-AI-R score into a potential valuation uplift:
    $$ \text{Multiple}_j = \text{Multiple}_{base,k} + \delta \cdot (\text{Exit-AI-R}_j / 100) $$
    """)
    st.markdown(r"""
    where:
    *   $\text{Multiple}_{base,k}$: The baseline industry valuation multiple for industry $k$.
    *   $\delta$: The AI Premium Coefficient (`AI_PremiumCoefficient`), which determines how much each point of Exit-AI-R score contributes to the valuation multiple.
    """)
    st.markdown("""
    This analysis provides critical data for our exit planning strategy, enabling us to highlight AI as a key value driver to maximize valuation multiples.
    """)
    st.markdown("---")

    # Widgets for Exit-AI-R parameters
    w1 = st.slider("Weight for Visible AI Capabilities ($w_1$)", min_value=0.20, max_value=0.50, value=0.35, step=0.01, key="w1_slider", help="Prioritize AI capabilities that are easily apparent to buyers (e.g., product features, technology stack). Default $w_1 = 0.35$.")
    w2 = st.slider("Weight for Documented AI Impact ($w_2$)", min_value=0.20, max_value=0.50, value=0.40, step=0.01, key="w2_slider", help="Emphasize quantified AI impact with clear audit trails. Default $w_2 = 0.40$.")
    w3 = st.slider("Weight for Sustainable AI Capabilities ($w_3$)", min_value=0.10, max_value=0.40, value=0.25, step=0.01, key="w3_slider", help="Focus on embedded, long-term AI capabilities versus one-time projects. Default $w_3 = 0.25$.")
    
    # Optional: Display a warning if weights don't sum to near 1
    total_weights = w1 + w2 + w3
    if not (0.99 <= total_weights <= 1.01): 
        st.warning(f"Note: Sum of weights (w1+w2+w3) is {total_weights:.2f}. Consider adjusting them to sum closer to 1 for typical weighted averages.")

    if st.button("Recalculate Exit-Readiness & Valuation", key="recalculate_exit_button", help="Click to re-compute Exit-AI-R scores and projected multiples with the selected weights."):
        st.session_state.portfolio_df = calculate_exit_readiness_and_valuation(st.session_state.portfolio_df, w1, w2, w3)
        st.success("Exit-Readiness scores and projected valuations updated successfully!")

    if not st.session_state.portfolio_df.empty:
        latest_quarter_df = st.session_state.portfolio_df[st.session_state.portfolio_df['Quarter'] == st.session_state.portfolio_df['Quarter'].max()]

        st.subheader("Latest Quarter's Exit-Readiness and Projected Valuation Impact:")
        st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Exit_AI_R_Score', 'BaselineMultiple', 'AI_Premium_Multiple_Additive', 'Projected_Exit_Multiple']]
                     .sort_values(by='Projected_Exit_Multiple', ascending=False).set_index('CompanyName'))

        # Visualization: Scatter Plot - Exit-AI-R Score vs. Projected Exit Multiple
        st.subheader("Exit-AI-R Score vs. Projected Exit Multiple (Latest Quarter)")
        fig7, ax7 = plt.subplots(figsize=(12, 7))
        sns.scatterplot(x='Exit_AI_R_Score', y='Projected_Exit_Multiple', hue='Industry', size='Attributed_EBITDA_Impact_Pct', 
                        sizes=(50, 400), data=latest_quarter_df, ax=ax7, palette='Spectral', alpha=0.8, legend='full')
        
        # Label points with CompanyName
        for i, row in latest_quarter_df.iterrows():
            ax7.text(row['Exit_AI_R_Score'] + 0.5, row['Projected_Exit_Multiple'] + 0.05, row['CompanyName'], 
                     fontsize=8, ha='left', va='bottom', alpha=0.7)

        ax7.set_title('Exit-AI-R Score vs. Projected Exit Multiple (Latest Quarter)')
        ax7.set_xlabel('Exit-AI-R Score')
        ax7.set_ylabel('Projected Exit Multiple (x)')
        ax7.grid(True, linestyle='--', alpha=0.6)
        ax7.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)
    else:
        # Render an empty dataframe with expected columns for AppTest compatibility
        empty_df = pd.DataFrame(columns=['CompanyName', 'Industry', 'Exit_AI_R_Score', 'BaselineMultiple', 'AI_Premium_Multiple_Additive', 'Projected_Exit_Multiple']).set_index('CompanyName')
        st.dataframe(empty_df)
        st.info("No data available for exit-readiness and valuation. Please generate portfolio data first.")

