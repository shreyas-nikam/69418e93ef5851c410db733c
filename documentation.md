id: 69418e93ef5851c410db733c_documentation
summary: Portfolio AI Performance & Benchmarking Dashboard Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: AI Performance & Benchmarking Dashboard for Private Equity

## 1. Introduction: Empowering PE Portfolio Managers with AI Analytics
Duration: 0:05:00

Welcome to the QuLab AI Performance & Benchmarking Dashboard! This codelab provides a comprehensive guide to understanding and utilizing a Streamlit application designed for Private Equity (PE) Portfolio Managers. In today's rapidly evolving landscape, Artificial Intelligence (AI) is no longer a niche technology but a critical driver of value, efficiency, and competitive advantage across diverse industries. For PE funds, systematically quantifying and managing AI's impact across a portfolio is paramount to optimizing returns and ensuring successful exits.

This application offers a structured, data-driven framework to:
*   **Assess AI Maturity:** Quantify each portfolio company's AI readiness (Org-AI-R Score).
*   **Benchmark Performance:** Compare companies against peers and industry standards.
*   **Quantify Financial Impact:** Attribute EBITDA growth and evaluate the efficiency of AI investments.
*   **Track Progress:** Monitor AI evolution and investment effectiveness over time.
*   **Identify Actionable Insights:** Pinpoint "Centers of Excellence" for best practice dissemination and "Companies for Review" needing strategic intervention.
*   **Enhance Exit Strategy:** Evaluate how AI capabilities influence exit readiness and potential valuation multiples.

By following this guide, developers will gain insights into how a full-stack data application can be built using Streamlit to address complex financial and strategic challenges, leveraging data science, interactive visualizations, and user-driven analysis.

<aside class="positive">
<b>The core value proposition</b> of this application is to transform qualitative assessments of AI into quantifiable metrics, enabling PE managers to make informed, data-backed decisions that drive portfolio value and mitigate risks.
</aside>

### Application Architecture Overview

The application follows a standard Streamlit pattern, utilizing cached functions for efficient data processing and dynamic UI elements for user interaction.

```mermaid
graph TD
    A[User Interaction: Sidebar Controls] --> B(Generate New Portfolio Data Button)
    B --> C{load_portfolio_data}
    C --> D(Synthetic Portfolio DataFrame)
    D --> E{calculate_org_ai_r}
    E --> F{calculate_benchmarks}
    F --> G{calculate_aie_ebitda}
    G --> H{calculate_exit_readiness_and_valuation}
    H --> I(Processed Portfolio DataFrame in Session State)
    I --> J{Page Navigation (Sidebar Radio Buttons)}
    J --> K{Dynamic Page Rendering}
    K --> L1[Page 1: Data Overview]
    K --> L2[Page 2: Org-AI-R Calculation]
    K --> L3[Page 3: Benchmarking]
    K --> L4[Page 4: AI Investment & EBITDA Impact]
    K --> L5[Page 5: Tracking Progress]
    K --> L6[Page 6: Actionable Insights]
    K --> L7[Page 7: Exit-Readiness & Valuation]
    L1 & L2 & L3 & L4 & L5 & L6 & L7 --> M(Display DataFrames & Matplotlib/Seaborn Visualizations)
```

The application leverages `st.cache_data` for performance optimization, ensuring that computationally intensive data generation and calculation functions are only rerun when their inputs change, leading to a smoother user experience.

## 2. Setting Up the Portfolio & Initializing Data
Duration: 0:10:00

The first step in any analytical journey is data acquisition and preparation. This application uses a synthetic data generation approach to simulate a PE fund's portfolio. This allows for reproducible analysis and experimentation without needing real, sensitive financial data.

On the sidebar, you'll find "Global Portfolio Setup" controls:
*   **Number of Portfolio Companies:** Adjust the total number of companies in the simulated portfolio.
*   **Number of Quarters (History):** Define how many historical quarters of data will be generated for each company.

These parameters directly influence the size and depth of the generated dataset.

### Data Generation Logic

The `load_portfolio_data` function is responsible for creating this synthetic dataset. It simulates various metrics crucial for AI performance assessment, including:
*   **AI Readiness Components:** `IdiosyncraticReadiness`, `SystematicOpportunity`, `Synergy`.
*   **Financials & AI Investment:** `AI_Investment`, `EBITDA_Impact`, `BaselineEBITDA`.
*   **Exit Readiness Factors:** `Visible`, `Documented`, `Sustainable`.
*   **Coefficients:** `GammaCoefficient`, `AI_PremiumCoefficient`, `BaselineMultiple`.

Each company's data is generated over the specified number of quarters, ensuring a time-series element for tracking progress. The `st.cache_data(ttl="2h")` decorator ensures that this function is only executed once for a given set of `num_companies` and `num_quarters` within a two-hour window, significantly speeding up subsequent interactions.

```python
@st.cache_data(ttl="2h")
def load_portfolio_data(num_companies_val, num_quarters_val):
    # ... (function implementation) ...
    df = pd.DataFrame(data)
    # ... (post-processing for categories and initial industry means) ...
    return df
```

### Initial Data View

When you navigate to "1. Initializing Portfolio Data" in the main content area, you'll see a snapshot of the generated data.

<aside class="positive">
Use the "Generate New Portfolio Data" button in the sidebar to re-run the data generation with updated `Number of Portfolio Companies` or `Number of Quarters` settings. This will refresh all subsequent calculations based on the new dataset.
</aside>

The application displays:
*   **Overview of Generated Portfolio Data (First 5 Rows):** A `st.dataframe` showing the initial rows of the `portfolio_df`. This helps quickly inspect the data structure.
*   **Descriptive Statistics:** A `st.dataframe` of `portfolio_df.describe()`, providing summary statistics (mean, std, min, max, etc.) for numerical columns.
*   **Data Information:** Output from `portfolio_df.info()`, detailing column names, non-null counts, and data types, which is crucial for data quality checks.

```python
# Displaying data overview
st.subheader("Overview of Generated Portfolio Data (First 5 Rows):")
st.dataframe(st.session_state.portfolio_df.head())

# Displaying descriptive statistics
st.subheader("Descriptive Statistics of Portfolio Data:")
st.dataframe(st.session_state.portfolio_df.describe())

# Displaying data info
st.subheader("Data Information:")
buffer = StringIO()
st.session_state.portfolio_df.info(buf=buffer)
st.text(buffer.getvalue())
```

This initial data exploration ensures that the foundation for our analysis is robust and understood.

## 3. Demystifying the PE Org-AI-R Score
Duration: 0:15:00

The PE Org-AI-R (Organizational AI Readiness) Score is a cornerstone metric in this application, designed to quantify a portfolio company's overall AI maturity and preparedness to leverage AI for value creation. It's a weighted composite score, allowing PE managers to tailor the assessment based on their strategic focus.

### The Org-AI-R Formula

The core calculation for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:

$$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$

Let's break down each component:
*   $V^R_{org,j}(t)$: **Idiosyncratic Readiness**. This represents company-specific AI capabilities, such as internal data infrastructure, AI talent pool, leadership's commitment to AI, and existing AI initiatives. In our synthetic data, this is represented by `IdiosyncraticReadiness`.
*   $H^R_{org,k}(t)$: **Systematic Opportunity**. This reflects the AI potential and adoption rates at an industry level, including disruption potential, competitive dynamics, and general market receptiveness to AI within that sector. In our synthetic data, this is proxied by `SystematicOpportunity` and later refined by `IndustryMeanOrgAIR`.
*   $\alpha$: **Weight for Idiosyncratic Readiness**. This adjustable parameter (`alpha` slider) allows the Portfolio Manager to prioritize internal, company-specific AI strengths ($\alpha$ closer to 1) or external, industry-level AI opportunities ($\alpha$ closer to 0).
*   $\beta$: **Synergy Coefficient**. This parameter (`beta` slider) quantifies the additional value created when a company's internal AI readiness (`IdiosyncraticReadiness`) aligns well with the external AI opportunities (`SystematicOpportunity`) in its industry. It measures how effectively the company can capitalize on industry trends given its internal strengths. In our synthetic data, this is represented by `Synergy`.

The `Org_AI_R_Score` is then clipped to a 0-100 range for ease of interpretation.

### `calculate_org_ai_r` Function

This function takes the portfolio DataFrame and the user-defined `alpha` and `beta` weights to compute the Org-AI-R score for each company.

```python
@st.cache_data(ttl="2h")
def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
    if df.empty:
        return df
    df_copy = df.copy() 
    df_copy['Org_AI_R_Score'] = (
        alpha * df_copy['IdiosyncraticReadiness'] +
        (1 - alpha) * df_copy['SystematicOpportunity'] +
        beta * df_copy['Synergy']
    )
    df_copy['Org_AI_R_Score'] = np.clip(df_copy['Org_AI_R_Score'], 0, 100)
    return df_copy
```

### Interactive Parameter Adjustment

On the "2. Calculating Org-AI-R Scores" page, you'll find:
*   **Weight for Idiosyncratic Readiness ($\alpha$) slider:** Adjust this to emphasize internal capabilities versus external opportunities.
*   **Synergy Coefficient ($\beta$) slider:** Control the impact of the synergy between internal readiness and external opportunity.

After adjusting the sliders, click the "Recalculate Org-AI-R Scores" button. This action not only recomputes the Org-AI-R scores but also triggers a cascade of recalculations for all dependent metrics (benchmarks, EBITDA impact, exit readiness) to ensure consistency across the dashboard.

<aside class="negative">
Changing these weights significantly alters how AI readiness is perceived across the portfolio. Always re-evaluate your chosen weights based on your fund's specific investment thesis and strategic priorities.
</aside>

The page then displays a `st.dataframe` showing the `CompanyName`, `Industry`, and the newly calculated `Org_AI_R_Score` for the latest quarter, sorted by score.

## 4. Benchmarking AI Performance
Duration: 0:15:00

While an individual Org-AI-R score provides a snapshot of a company's AI maturity, its true significance often lies in how it compares to peers. Benchmarking is essential for identifying leaders, laggards, and strategic opportunities within the portfolio and against broader industry standards.

This section utilizes two key benchmarking metrics:
1.  **Org-AI-R Percentile (Within-Portfolio Benchmarking):** Ranks a company's AI readiness relative to all other companies within the fund's portfolio for a given quarter.
2.  **Org-AI-R Z-Score (Cross-Portfolio/Industry-Adjusted Benchmarking):** Measures how many standard deviations a company's Org-AI-R score is from its industry's average, normalizing for industry-specific AI potentials.

### Benchmarking Formulas

**Org-AI-R Percentile:**
This is calculated by ranking each company's Org-AI-R score within the entire portfolio for a given quarter and expressing it as a percentile.
$$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$
where $\text{Rank}(\text{Org-AI-R}_j)$ is the rank of company $j$'s Org-AI-R score within the portfolio (from lowest to highest), and $\text{Portfolio Size}$ is the total number of companies in the portfolio for that quarter.

**Org-AI-R Z-Score:**
This score normalizes the Org-AI-R score relative to its industry.
$$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
where $\text{Org-AI-R}_j$ is the Org-AI-R score of company $j$, $\mu_k$ is the mean Org-AI-R score for industry $k$, and $\sigma_k$ is the standard deviation of Org-AI-R scores for industry $k$. A positive Z-score indicates above-average performance within its industry, while a negative score suggests underperformance.

### `calculate_benchmarks` Function

This function takes the DataFrame (with `Org_AI_R_Score` already calculated) and adds the percentile and Z-score metrics.

```python
@st.cache_data(ttl="2h")
def calculate_benchmarks(df):
    if df.empty or 'Org_AI_R_Score' not in df.columns:
        return df.assign(Org_AI_R_Percentile=0.0, Org_AI_R_Z_Score=0.0) 
    
    df_copy = df.copy()
    df_copy['Org_AI_R_Percentile'] = df_copy.groupby('Quarter', observed=False)['Org_AI_R_Score'].rank(pct=True) * 100
    df_copy['IndustryMeanOrgAIR'] = df_copy.groupby(['Industry', 'Quarter'], observed=False)['Org_AI_R_Score'].transform('mean')
    
    def safe_zscore_transform(series):
        if len(series) > 1 and series.std() > 0:
            return zscore(series)
        return pd.Series(0.0, index=series.index)
    
    df_copy['Org_AI_R_Z_Score'] = df_copy.groupby(['Industry', 'Quarter'], observed=False)['Org_AI_R_Score'].transform(safe_zscore_transform)
    return df_copy
```

### Benchmarking Visualization

On the "3. Benchmarking AI Performance" page:
*   **Select Quarter for Benchmarking:** A `st.selectbox` allows you to choose a specific quarter to view the benchmarking results.
*   **Org-AI-R Benchmarks Table:** A `st.dataframe` displays `CompanyName`, `Industry`, `Org_AI_R_Score`, `Org_AI_R_Percentile`, and `Org_AI_R_Z_Score`, sorted by Org-AI-R Score.

Two interactive visualizations provide deeper insights:
1.  **Org-AI-R Scores by Company (Bar Chart):** A `seaborn.barplot` shows each company's Org-AI-R score, colored by industry, with a horizontal line indicating the portfolio average. This helps in quick visual comparison of absolute scores.
2.  **Org-AI-R Score vs. Industry-Adjusted Z-Score (Scatter Plot):** A `seaborn.scatterplot` plots `Org_AI_R_Score` on the x-axis and `Org_AI_R_Z_Score` on the y-axis, with points sized by `Org_AI_R_Percentile` and colored by `Industry`. This powerful visualization identifies:
    *   **High-Performing Companies:** Upper-right quadrant (high Org-AI-R, positive Z-score).
    *   **Industry Laggards:** Lower-left quadrant (low Org-AI-R, negative Z-score).
    *   **Hidden Gems/Challenges:** Companies with high absolute Org-AI-R but negative Z-score (meaning they are strong but perhaps underperforming relative to their *very* strong industry), or vice-versa.

These benchmarks are invaluable for identifying best practices within leading companies and pinpointing those that require targeted support to catch up to their industry peers.

## 5. Quantifying AI Investment & EBITDA Impact
Duration: 0:15:00

Beyond readiness scores, a Private Equity Portfolio Manager needs to understand the tangible financial outcomes of AI initiatives. This section focuses on quantifying the efficiency of AI investments and attributing specific EBITDA impact to improvements in AI readiness. This allows for direct evaluation of capital allocation and identification of where AI investments are truly generating value.

### Key Financial Impact Metrics

**AI Investment Efficiency ($\text{AIE}_j$):**
This metric measures how effectively AI investments translate into improvements in AI readiness and overall EBITDA impact. A higher AIE signifies a more efficient deployment of capital for AI initiatives.

$$ \text{AIE}_j = \left( \frac{\Delta\text{Org-AI-R}_j}{\text{AI Investment}_j} \right) \times \text{EBITDA Impact}_j \times C $$
where:
*   $\Delta\text{Org-AI-R}_j$: The change in Org-AI-R score for company $j$ from the previous quarter.
*   $\text{AI Investment}_j$: The total AI investment for company $j$ in the current quarter.
*   $\text{EBITDA Impact}_j$: The direct percentage EBITDA impact reported for company $j$.
*   $C$: A scaling constant (e.g., $1,000,000$ in the code) to make the efficiency metric more readable, representing impact points per million invested.

**Attributed EBITDA Impact Percentage ($\Delta\text{EBITDA}\%$):**
This formula estimates the percentage increase in EBITDA directly attributable to the change in a company's Org-AI-R score, factoring in the systematic opportunity of its industry.

$$ \Delta\text{EBITDA}\% = \gamma \cdot \Delta\text{Org-AI-R}_j \cdot (H^R_{org,k} / 100) $$
where:
*   $\gamma$: The `GammaCoefficient`, a scaling factor.
*   $\Delta\text{Org-AI-R}_j$: The change in Org-AI-R score for company $j$.
*   $H^R_{org,k}$: The systematic opportunity for industry $k$, proxied by `IndustryMeanOrgAIR` (the mean Org-AI-R score for the industry). The division by 100 normalizes it as a percentage from 100.

The `Attributed_EBITDA_Impact_Absolute` is then calculated by applying this percentage to the `BaselineEBITDA`.

### `calculate_aie_ebitda` Function

This function sorts the DataFrame by `CompanyID` and `Quarter` to correctly calculate the quarter-over-quarter change in `Org_AI_R_Score`. It then computes `AI_Investment_Efficiency` and the two `Attributed_EBITDA_Impact` metrics.

```python
@st.cache_data(ttl="2h")
def calculate_aie_ebitda(df):
    if df.empty or 'Org_AI_R_Score' not in df.columns:
        return df.assign(Delta_Org_AI_R=0.0, AI_Investment_Efficiency=0.0, 
                         Attributed_EBITDA_Impact_Pct=0.0, Attributed_EBITDA_Impact_Absolute=0.0)
    
    df_copy = df.copy()
    df_sorted = df_copy.sort_values(by=['CompanyID', 'Quarter'])
    df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID', observed=False)['Org_AI_R_Score'].diff().fillna(0)
    
    # ... (AIE and Attributed EBITDA calculations) ...
    
    return df_sorted
```

### Financial Impact Visualization

On the "4. AI Investment & EBITDA Impact" page:
*   **Latest Quarter's AI Investment Efficiency and Attributed EBITDA Impact Table:** A `st.dataframe` shows these key financial metrics for the most recent quarter, sorted by `AI_Investment_Efficiency`. This helps identify which companies are getting the most "bang for their buck" from AI investments.
*   **AI Investment vs. Efficiency (Scatter Plot):** A `seaborn.scatterplot` visualizes `AI_Investment` (log-scaled on the x-axis for better distribution) against `AI_Investment_Efficiency` on the y-axis. The size of the points represents `Attributed_EBITDA_Impact_Pct`, and colors denote `Industry`. This plot helps to:
    *   Identify companies that achieve high efficiency with relatively low investment.
    *   Spot heavy investors who might be getting low efficiency.
    *   Understand the relationship between investment size, efficiency, and actual EBITDA impact.

This analysis provides critical insights into which companies are most effectively translating their AI maturity into financial returns, guiding future investment decisions and resource allocation strategies within the PE fund.

## 6. Tracking Progress and Trends Over Time
Duration: 0:10:00

Understanding current performance is vital, but in private equity, assessing trajectory and long-term trends is equally, if not more, critical. This section empowers Portfolio Managers to monitor how individual companies are progressing in their AI journey and how efficiently they're utilizing AI investments over multiple quarters.

### Dynamic Company Selection

On the "5. Tracking Progress Over Time" page:
*   **Select Companies to Track:** A `st.multiselect` widget allows you to choose up to five (for clarity in visualization) specific portfolio companies. This enables focused analysis on a subset of companies of particular interest.

Once companies are selected, the application filters the historical data for these companies and displays two line charts.

### Trend Visualizations

1.  **Org-AI-R Score Trajectory Over Time (Line Chart):** A `seaborn.lineplot` displays the `Org_AI_R_Score` for each selected company across all available quarters.
    *   Individual company lines are clearly differentiated by color.
    *   A dashed black line represents the `Portfolio Average` Org-AI-R score for each quarter, providing a benchmark against the overall fund's performance. This helps identify if a company is outperforming or lagging the portfolio trend.
2.  **AI Investment Efficiency Trajectory Over Time (Line Chart):** Similarly, a `seaborn.lineplot` tracks the `AI_Investment_Efficiency` for selected companies over time.
    *   This chart is crucial for understanding whether AI investments are becoming more or less efficient over time. Are companies improving their ROI on AI initiatives, or are they stagnating?
    *   Again, the `Portfolio Average` AIE is overlaid to provide contextual comparison.

```python
# Example for Org-AI-R Trajectory plot
sns.lineplot(x='Quarter', y='Org_AI_R_Score', hue='CompanyName', marker='o', data=tracking_df, ax=ax4, palette='deep')
portfolio_avg_df = st.session_state.portfolio_df.groupby('Quarter', observed=False)['Org_AI_R_Score'].mean().reset_index()
sns.lineplot(x='Quarter', y='Org_AI_R_Score', data=portfolio_avg_df, ax=ax4, color='black', linestyle='--', label='Portfolio Average', marker='x')
```

These visualizations enable quick identification of:
*   **Positive Trends:** Companies with consistently rising Org-AI-R scores or AIE.
*   **Stagnation or Decline:** Companies where AI progress has stalled or efficiency is decreasing.
*   **Outliers:** Companies performing significantly above or below the portfolio average, prompting further investigation.

Tracking these trajectories is essential for proactive management, allowing PE managers to intervene early, reallocate resources, or double down on successful strategies.

## 7. Actionable Insights: Centers of Excellence and Companies for Review
Duration: 0:15:00

A critical function of a PE Portfolio Manager is to derive actionable insights from data. This section is designed to automatically identify "Centers of Excellence" (CoE) and "Companies for Review" based on user-defined performance thresholds, streamlining strategic decision-making. This targeted identification is crucial for optimizing the fund's overall AI strategy, fostering best practices, and mitigating risks.

### Defining Actionable Thresholds

On the "6. Actionable Insights: CoE & Review" page, you can dynamically set the criteria using `st.slider` widgets:
*   **Org-AI-R Score Threshold for Center of Excellence:** Companies with an Org-AI-R score *above* this value are considered for CoE status.
*   **EBITDA Impact (%) Threshold for Center of Excellence:** Companies with an EBITDA Impact *above* this value are considered for CoE status.
*   **Org-AI-R Score Threshold for Companies for Review:** Companies with an Org-AI-R score *below or equal to* this value are flagged for review.
*   **EBITDA Impact (%) Threshold for Companies for Review:** Companies with an EBITDA Impact *below or equal to* this value are flagged for review.

Clicking the "Re-evaluate Actionable Insights" button (or on initial load) triggers the `identify_actionable_insights` function with these thresholds.

### `identify_actionable_insights` Function

This function filters the *latest quarter's* data to identify companies that meet the specified criteria.

```python
@st.cache_data(ttl="2h")
def identify_actionable_insights(df, org_ai_r_threshold_coe, ebitda_impact_threshold_coe,
                                 org_ai_r_threshold_review, ebitda_impact_threshold_review):
    if df.empty or 'Org_AI_R_Score' not in df.columns or 'EBITDA_Impact' not in df.columns:
        return pd.DataFrame(), pd.DataFrame() 
    
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
```

<aside class="positive">
Adjusting these thresholds allows you to tune the sensitivity of your identification process, adapting to different market conditions or fund strategies.
</aside>

### Displaying Insights

The page displays two `st.dataframe` tables:
*   **Centers of Excellence:** Lists companies that meet the CoE criteria, highlighting their `Org_AI_R_Score`, `EBITDA_Impact`, and `AI_Investment_Efficiency`. These companies are potential sources of best practices and scalable AI solutions.
*   **Companies for Review:** Lists companies that fall below the review thresholds, indicating potential underperformance in AI readiness or financial impact. These companies require immediate strategic attention, potentially including resource reallocation, a re-evaluation of their AI strategy, or focused support.

### Portfolio AI Performance Map (Scatter Plot)

A highly informative `seaborn.scatterplot` visualizes **Org-AI-R Score vs. EBITDA Impact** for the latest quarter.
*   Each company is plotted based on its Org-AI-R Score (x-axis) and EBITDA Impact (y-axis), with hue by `Industry` and size by `AI_Investment_Efficiency`.
*   **Threshold Lines:** Vertical and horizontal lines clearly mark the `Org-AI-R` and `EBITDA Impact` thresholds for both CoE (green, dotted) and Companies for Review (red, dashed).
*   **Highlighted Companies:** Companies identified as CoE are marked with a large green star ($\star$) and their names, while Companies for Review are marked with a large red 'X' and their names.

This visual map immediately conveys the strategic positioning of all portfolio companies, making it easy to spot:
*   **"Stars":** Companies in the top-right quadrant (high Org-AI-R, high EBITDA Impact).
*   **"Problem Children":** Companies in the bottom-left quadrant (low Org-AI-R, low EBITDA Impact).
*   **"Growth Opportunities":** Companies with high Org-AI-R but perhaps moderate EBITDA impact (potential to convert AI readiness into more financial gains).
*   **"Efficiency Challenges":** Companies with decent EBITDA impact but lower Org-AI-R, suggesting their impact might not be sustainable without improving underlying readiness.

This integrated approach provides a powerful tool for strategic portfolio management.

## 8. Enhancing Exit-Readiness & Maximizing Valuation
Duration: 0:15:00

For Private Equity, the ultimate goal is a successful exit. The AI capabilities of a portfolio company can profoundly influence its attractiveness to potential acquirers and, consequently, its exit valuation. This section allows PE managers to assess how "buyer-friendly" a company's AI assets are and to quantify the potential premium these assets could add to its exit multiple.

### Exit-AI-R Score

The Exit-Readiness Score (`Exit-AI-R`) quantifies how well a company's AI capabilities are positioned to drive a favorable exit. The formula for the Exit-Readiness Score for portfolio company $j$ preparing for exit is:

$$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$
where:
*   $\text{Visible}_j$: **Visible AI Capabilities**. This refers to AI features that are easily discernible and appealing to potential buyers, such as product functionality driven by AI, a well-defined technology stack, clear customer value propositions derived from AI, and a strong brand narrative around AI innovation.
*   $\text{Documented}_j$: **Documented AI Impact**. This measures the extent to which a company has quantified and auditable evidence of AI's financial or operational benefits, including ROI analyses, efficiency gains, and revenue uplift attributable to AI. This provides concrete evidence for due diligence.
*   $\text{Sustainable}_j$: **Sustainable AI Capabilities**. This assesses whether AI capabilities are deeply embedded and can deliver long-term value, rather than being one-off projects. It includes aspects like a robust AI strategy, scalable infrastructure, continuous innovation pipelines, and a culture of AI adoption.
*   $w_1, w_2, w_3$: **Weighting factors**. These adjustable weights allow the Portfolio Manager to emphasize different aspects that potential buyers might prioritize in their valuation model or during a due diligence process.

### Multiple Attribution Model

This model translates the `Exit-AI-R` score into a potential uplift in the company's valuation multiple.

$$ \text{Multiple}_j = \text{Multiple}_{base,k} + \delta \cdot (\text{Exit-AI-R}_j / 100) $$
where:
*   $\text{Multiple}_{base,k}$: The baseline industry valuation multiple for industry $k$, representing the average multiple for companies in that sector without a specific AI premium.
*   $\delta$: The `AI_PremiumCoefficient`, which is a scaling factor that determines how much each percentage point of the `Exit-AI-R` score contributes to an additive premium on the baseline valuation multiple.

The sum of $w_1, w_2, w_3$ should ideally be around 1 to represent a weighted average. The application provides a warning if the sum significantly deviates.

### `calculate_exit_readiness_and_valuation` Function

This function computes the `Exit_AI_R_Score` based on the provided weights and then calculates the `AI_Premium_Multiple_Additive` and the `Projected_Exit_Multiple`.

```python
@st.cache_data(ttl="2h")
def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
    if df.empty:
        return df.assign(Exit_AI_R_Score=0.0, AI_Premium_Multiple_Additive=0.0, Projected_Exit_Multiple=0.0)
    
    df_copy = df.copy() 
    df_copy['Exit_AI_R_Score'] = (
        w1 * df_copy['Visible'] + w2 * df_copy['Documented'] + w3 * df_copy['Sustainable']
    )
    df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100)
    
    df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
    df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
    
    return df_copy
```

### Interactive Parameter Adjustment and Visualization

On the "7. Exit-Readiness & Valuation" page:
*   **Weight sliders ($w_1, w_2, w_3$):** Adjust these to reflect which aspects of AI capabilities (Visible, Documented, Sustainable) are most important for your exit strategy.
*   **"Recalculate Exit-Readiness & Valuation" button:** Re-computes the scores and multiples based on the adjusted weights.

The page then displays:
*   **Latest Quarter's Exit-Readiness and Projected Valuation Impact Table:** A `st.dataframe` showing `CompanyName`, `Industry`, `Exit_AI_R_Score`, `BaselineMultiple`, `AI_Premium_Multiple_Additive`, and `Projected_Exit_Multiple`, sorted by projected multiple.
*   **Exit-AI-R Score vs. Projected Exit Multiple (Scatter Plot):** A `seaborn.scatterplot` plots `Exit_AI_R_Score` (x-axis) against `Projected_Exit_Multiple` (y-axis). Points are sized by `Attributed_EBITDA_Impact_Pct` and colored by `Industry`. Each company is labeled directly on the plot for easy identification.

This visualization provides a clear picture of which companies are best positioned for an AI-driven exit premium and helps identify areas for improvement in their AI story. This data is critical for developing a compelling exit narrative and maximizing valuation multiples during divestment.
