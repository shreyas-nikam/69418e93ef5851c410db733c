
## Streamlit Application Requirements Specification: Portfolio AI Performance & Benchmarking Dashboard

### 1. Application Overview

**Story Narrative:**
As a Private Equity Portfolio Manager, I am constantly seeking opportunities to drive value across my fund's diverse portfolio. In today's landscape, AI is a critical lever for growth and efficiency, but its impact needs to be systematically quantified and managed. This application will guide me through a complete AI performance review cycle for my portfolio companies. I'll start by loading the latest data, then compute key AI readiness and financial impact metrics. With these insights, I can benchmark companies against their peers and industry, track their progress over time, identify my "Centers of Excellence" whose AI best practices can be scaled, and pinpoint "Companies for Review" that need immediate strategic attention. Finally, as exit planning is always top of mind, I'll assess how a company's AI capabilities enhance its exit readiness and potential valuation. Each step in this journey empowers me to make data-driven decisions that optimize our fund's overall AI strategy and maximize risk-adjusted returns.

**Real-World Problem the Persona is Solving:**
The Portfolio Manager (or Quantitative Analyst) needs a rigorous, standardized, and dynamic framework to:
1.  **Assess Organizational AI Readiness:** Objectively quantify the AI maturity of portfolio companies.
2.  **Benchmark Performance:** Understand how companies perform relative to peers (within portfolio and industry-adjusted).
3.  **Quantify Financial Impact:** Link AI investments and readiness improvements to tangible EBITDA growth and efficiency.
4.  **Drive Strategic Intervention:** Identify high-performing companies for best practice transfer and underperforming ones for focused turnaround efforts.
5.  **Optimize Exit Strategy:** Build an evidence-based narrative around AI capabilities to maximize valuation multiples during exit.
The challenge is moving beyond anecdotal AI adoption to a measurable, actionable value lever that directly informs investment decisions and portfolio management.

**How the Streamlit App Helps the Persona Apply the Concepts:**
The Streamlit application transforms the complex PE Org-AI-R framework and its associated financial models into an intuitive, interactive dashboard. Instead of requiring manual data manipulation and calculation, the app provides a guided workflow. The Portfolio Manager can:
*   **Generate and ingest data** with simple controls.
*   **Adjust model parameters** (e.g., weighting factors for Org-AI-R and Exit-AI-R) to see immediate impacts on scores and rankings, reflecting their strategic priorities.
*   **Visualize performance** through dynamic charts and tables that highlight key trends and comparisons.
*   **Interactively define thresholds** for identifying leaders and laggards, allowing for tailored insights based on current fund strategy.
*   **Receive actionable insights** and recommendations presented contextually within the narrative, directly supporting their decision-making process for resource allocation, best practice transfer, and exit planning. The app shows *how* these concepts are applied to solve real problems, rather than just explaining them.

**Learning Goals (Applied Skills):**
By interacting with the app, the Portfolio Manager will gain applied skills in:
*   **Parametric AI Readiness Assessment:** Understanding and configuring the components of the PE Org-AI-R score.
*   **Relative Performance Benchmarking:** Interpreting `Org-AI-R Percentile` and `Org-AI-R Z-Score` to identify competitive positioning.
*   **AI Value Attribution:** Quantifying `AI Investment Efficiency` and `Attributed EBITDA Impact %` to evaluate the financial ROI of AI initiatives.
*   **Time-Series Performance Analysis:** Tracking key metrics over quarters to identify trends and assess long-term strategy effectiveness.
*   **Data-Driven Portfolio Segmentation:** Using configurable thresholds to categorize companies into 'Centers of Excellence' and 'Companies for Review'.
*   **AI-Driven Exit Valuation:** Calculating and understanding the drivers of `Exit-AI-R Score` and its impact on `Projected Exit Multiple`.
*   **Strategic Scenario Planning:** Modifying input parameters to explore "what-if" scenarios and inform tactical adjustments.

### 2. User Interface Requirements

The UI will be designed as a multi-page application, with each page representing a distinct step in the Portfolio Manager's quarterly review workflow, mirroring the narrative flow.

#### Layout & Navigation Structure
The application will use a persistent sidebar for global controls and page navigation, and the main content area for displaying step-specific narrative, inputs, outputs, and visualizations.

*   **Sidebar (`st.sidebar`):**
    *   **Fund Branding:** `st.image("fund_logo.png")` (placeholder) and `st.title("Portfolio AI Performance & Benchmarking")`.
    *   **Global Data Controls:**
        *   `st.header("Global Portfolio Setup")`
        *   `st.number_input("Number of Portfolio Companies", min_value=5, max_value=20, value=10, key="num_companies_input")`
        *   `st.number_input("Number of Quarters (History)", min_value=2, max_value=10, value=5, key="num_quarters_input")`
        *   `st.button("Generate New Portfolio Data", key="generate_data_button", help="Click to create a new synthetic dataset based on the parameters above. All subsequent calculations will use this data.")`
    *   **Page Navigation:** `st.radio("Portfolio Review Stages", ["1. Initializing Portfolio Data", "2. Calculating Org-AI-R Scores", "3. Benchmarking AI Performance", "4. AI Investment & EBITDA Impact", "5. Tracking Progress Over Time", "6. Actionable Insights: CoE & Review", "7. Exit-Readiness & Valuation"], key="page_selection")`

*   **Main Content Area:** For each selected page, the content will dynamically update.
    *   **Narrative Intro:** A `st.markdown` block providing the context and purpose of the current step from the persona's perspective.
    *   **Interactive Widgets:** Inputs relevant to the current step (sliders, selectboxes).
    *   **Calculated Outputs:** `st.dataframe` for tabular results.
    *   **Visualizations:** `st.pyplot` or `st.plotly_chart` for graphical insights.
    *   **Actionable Insights/Summary:** A `st.markdown` block interpreting the results for the persona.

#### Input Widgets and Controls

1.  **Page "1. Initializing Portfolio Data"**
    *   **Purpose:** Allow the Portfolio Manager to confirm the loaded data and understand its structure.
    *   **Action:** Initial review of the fund's holdings data.
    *   **Display:** `st.dataframe(st.session_state.portfolio_df.head())` and `st.dataframe(st.session_state.portfolio_df.describe())` to show data summary.

2.  **Page "2. Calculating Org-AI-R Scores"**
    *   **Purpose:** To enable the Portfolio Manager to calibrate the Org-AI-R score calculation based on their fund's strategic emphasis.
    *   **Action:** Calibrating the organizational AI readiness assessment model.
    *   **Widgets:**
        *   `st.slider("Weight for Idiosyncratic Readiness ($\alpha$)", min_value=0.55, max_value=0.70, value=0.60, step=0.01, key="alpha_slider", help="Adjust this to prioritize company-specific capabilities ($V^R_{org,j}$) versus industry-level AI potential ($H^R_{org,k}$). Default $\alpha = 0.60$.")`
        *   `st.slider("Synergy Coefficient ($\beta$)", min_value=0.08, max_value=0.25, value=0.15, step=0.01, key="beta_slider", help="Quantify the additional value derived from the interplay between idiosyncratic readiness and systematic opportunity. Default $\beta = 0.15$.")`
        *   `st.button("Recalculate Org-AI-R Scores", key="recalculate_org_ai_r_button", help="Click to re-compute Org-AI-R scores with the selected weights.")`

3.  **Page "3. Benchmarking AI Performance"**
    *   **Purpose:** To allow the Portfolio Manager to focus benchmarking on the most recent or a specific historical quarter.
    *   **Action:** Selecting a specific period for peer comparison.
    *   **Widgets:**
        *   `st.selectbox("Select Quarter for Benchmarking", options=st.session_state.portfolio_df['Quarter'].unique().tolist(), index=len(st.session_state.portfolio_df['Quarter'].unique()) - 1, key="benchmark_quarter_select")`

4.  **Page "4. AI Investment & EBITDA Impact"**
    *   **Purpose:** No direct interactive widgets beyond company/quarter selection. This page focuses on displaying calculated financial impact metrics derived from previous steps.
    *   **Action:** Reviewing the financial returns and efficiency of AI investments.

5.  **Page "5. Tracking Progress Over Time"**
    *   **Purpose:** To enable the Portfolio Manager to selectively track the historical trajectory of specific companies for deeper analysis.
    *   **Action:** Monitoring long-term trends and identifying individual company outliers.
    *   **Widgets:**
        *   `st.multiselect("Select Companies to Track (Max 5 for clarity)", options=st.session_state.portfolio_df['CompanyName'].unique().tolist(), default=st.session_state.portfolio_df['CompanyName'].unique().tolist()[:5], key="companies_to_track_multiselect")`

6.  **Page "6. Actionable Insights: CoE & Review"**
    *   **Purpose:** To allow the Portfolio Manager to customize the thresholds for identifying high-performing 'Centers of Excellence' and underperforming 'Companies for Review', aligning with current fund strategy or risk appetite.
    *   **Action:** Defining performance benchmarks for strategic intervention and resource allocation.
    *   **Widgets:**
        *   `st.slider("Org-AI-R Score Threshold for Center of Excellence", min_value=50, max_value=90, value=75, step=1, key="coe_org_ai_r_threshold", help="Companies with Org-AI-R score above this will be considered for 'Centers of Excellence'. Default: 75.")`
        *   `st.slider("EBITDA Impact (%) Threshold for Center of Excellence", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="coe_ebitda_threshold", help="Companies with EBITDA Impact above this will be considered for 'Centers of Excellence'. Default: 3%.")`
        *   `st.slider("Org-AI-R Score Threshold for Companies for Review", min_value=20, max_value=70, value=50, step=1, key="review_org_ai_r_threshold", help="Companies with Org-AI-R score below or equal to this will be considered for 'Companies for Review'. Default: 50.")`
        *   `st.slider("EBITDA Impact (%) Threshold for Companies for Review", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="review_ebitda_threshold", help="Companies with EBITDA Impact below or equal to this will be considered for 'Companies for Review'. Default: 1%.")`
        *   `st.button("Re-evaluate Actionable Insights", key="re_evaluate_insights_button", help="Click to re-identify Centers of Excellence and Companies for Review based on the adjusted thresholds.")`

7.  **Page "7. Exit-Readiness & Valuation"**
    *   **Purpose:** To enable the Portfolio Manager to adjust the weighting factors for the Exit-AI-R score, reflecting different aspects buyers might prioritize (visibility, documentation, sustainability) for a stronger exit narrative.
    *   **Action:** Tailoring the exit narrative to maximize valuation by emphasizing key AI attributes.
    *   **Widgets:**
        *   `st.slider("Weight for Visible AI Capabilities ($w_1$)", min_value=0.20, max_value=0.50, value=0.35, step=0.01, key="w1_slider", help="Prioritize AI capabilities that are easily apparent to buyers (e.g., product features, technology stack). Default $w_1 = 0.35$.")`
        *   `st.slider("Weight for Documented AI Impact ($w_2$)", min_value=0.20, max_value=0.50, value=0.40, step=0.01, key="w2_slider", help="Emphasize quantified AI impact with clear audit trails. Default $w_2 = 0.40$.")`
        *   `st.slider("Weight for Sustainable AI Capabilities ($w_3$)", min_value=0.10, max_value=0.40, value=0.25, step=0.01, key="w3_slider", help="Focus on embedded, long-term AI capabilities versus one-time projects. Default $w_3 = 0.25$.")`
        *   `st.button("Recalculate Exit-Readiness & Valuation", key="recalculate_exit_button", help="Click to re-compute Exit-AI-R scores and projected multiples with the selected weights.")`

#### Visualization Components
All visualizations will be dynamically updated based on selected filters and parameters.

*   **Page "3. Benchmarking AI Performance"**
    *   **Bar Chart:** `Latest Quarter Org-AI-R Scores by Company (with Portfolio Average)`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Bar chart, `x='CompanyName'`, `y='Org_AI_R_Score'`, `hue='Industry'`.
        *   **Outputs:** Displays individual company scores and a horizontal line indicating the portfolio average.
    *   **Scatter Plot:** `Org-AI-R Score vs. Industry-Adjusted Z-Score (Latest Quarter)`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Scatter plot, `x='Org_AI_R_Score'`, `y='Org_AI_R_Z_Score'`, `hue='Industry'`, `size='Org_AI_R_Percentile'`.
        *   **Outputs:** Visualizes relative performance, with point size reflecting within-portfolio ranking. Includes a vertical line for portfolio mean Org-AI-R and a horizontal line at $Y=0$ for industry mean Z-score.

*   **Page "4. AI Investment & EBITDA Impact"**
    *   **Scatter Plot:** `AI Investment vs. Efficiency (Latest Quarter, Highlighting EBITDA Impact)`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Scatter plot, `x='AI_Investment'` (log scale), `y='AI_Investment_Efficiency'`, `hue='Industry'`, `size='Attributed_EBITDA_Impact_Pct'`.
        *   **Outputs:** Shows the relationship between investment, efficiency, and attributed financial impact, highlighting which companies achieve more impact per dollar.

*   **Page "5. Tracking Progress Over Time"**
    *   **Line Chart:** `Org-AI-R Score Trajectory Over Time`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Line chart, `x='Quarter'`, `y='Org_AI_R_Score'`, `hue='CompanyName'`, `marker='o'`.
        *   **Outputs:** Shows individual company progress and an overlaid line for the overall portfolio average.
    *   **Line Chart:** `AI Investment Efficiency Trajectory Over Time`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Line chart, `x='Quarter'`, `y='AI_Investment_Efficiency'`, `hue='CompanyName'`, `marker='o'`.
        *   **Outputs:** Visualizes how efficiently companies are converting AI investments into value over time, alongside the portfolio average.

*   **Page "6. Actionable Insights: CoE & Review"**
    *   **Scatter Plot:** `Portfolio AI Performance: Org-AI-R Score vs. EBITDA Impact (Latest Quarter)`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Scatter plot, `x='Org_AI_R_Score'`, `y='EBITDA_Impact'`, `hue='Industry'`, `size='AI_Investment_Efficiency'`.
        *   **Outputs:** Visually distinguishes 'Centers of Excellence' (e.g., large green star markers with company names) and 'Companies for Review' (e.g., large red 'X' markers with company names) based on user-defined thresholds. Includes horizontal and vertical lines for the thresholds.

*   **Page "7. Exit-Readiness & Valuation"**
    *   **Scatter Plot:** `Exit-AI-R Score vs. Projected Exit Multiple (Latest Quarter)`
        *   **Library:** `seaborn` / `matplotlib.pyplot` or `plotly.express`
        *   **Format:** Scatter plot, `x='Exit_AI_R_Score'`, `y='Projected_Exit_Multiple'`, `hue='Industry'`, `size='EBITDA_Impact'`.
        *   **Outputs:** Labels points with `CompanyName`. Shows the relationship between a company's AI exit readiness and its potential valuation uplift.

#### Interactive Elements & Feedback Mechanisms
*   **Dynamic Calculations:** All calculations (Org-AI-R, benchmarks, AIE, Exit-AI-R) will automatically re-run and update downstream displays when their input parameters (sliders, number inputs) change, or when triggered by dedicated "Recalculate" buttons.
*   **"Generate New Portfolio Data" Button:** Clears all existing data in `st.session_state` and calls `load_portfolio_data` with new `num_companies` and `num_quarters` values. A success message like "New synthetic portfolio data generated successfully!" will appear.
*   **Table and Chart Updates:** All tables (`st.dataframe`) and charts (`st.pyplot`, `st.plotly_chart`) will instantly reflect changes in input parameters, selected quarters, or re-calculations.
*   **Conditional Display of Insights:** If no 'Centers of Excellence' or 'Companies for Review' are identified, specific messages like "No companies currently meet the Centers of Excellence criteria with the current thresholds." will be displayed instead of empty tables.
*   **Narrative Feedback:** After key interactions (e.g., recalculating Org-AI-R), a brief `st.info` or `st.success` message will confirm the action and suggest the next narrative step, e.g., "Org-AI-R scores updated. Now, let's see how these companies benchmark against their peers."

### 3. Additional Requirements

#### Annotations & Tooltips
Each interactive widget will have a `help` attribute providing context. Visualizations will use legends and informative titles. Inline `st.markdown` will provide narrative and conceptual explanations for calculated metrics.

*   **General:**
    *   **Org-AI-R Score:** "A composite score (0-100) quantifying a company's overall AI maturity and readiness for value creation. Higher scores indicate stronger AI capabilities and potential."
    *   **Idiosyncratic Readiness ($V^R_{org,j}$):** "Company-specific capabilities related to AI, such as data infrastructure, talent, and leadership commitment."
    *   **Systematic Opportunity ($H^R_{org,k}$):** "Industry-level AI potential, reflecting adoption rates, disruption potential, and competitive dynamics within the sector."
    *   **Synergy:** "The alignment and integration between a company's idiosyncratic readiness and the systematic opportunity in its industry."
    *   **AI Investment ($AI\_Investment_j$):** "Total capital invested by company $j$ in AI initiatives over a period."
    *   **EBITDA Impact ($EBITDA\_Impact_j$):** "Percentage increase in EBITDA directly attributed to AI initiatives for company $j$."
*   **Benchmarking Page:**
    *   **Org-AI-R Percentile:** "A company's percentile rank within the current portfolio, showing its standing relative to all other fund holdings. E.g., 90th percentile means outperforming 90% of peers."
    *   **Org-AI-R Z-Score:** "An industry-adjusted score showing how much a company's Org-AI-R deviates from its industry's mean, in terms of standard deviations. Positive values indicate outperformance relative to industry peers."
*   **AI Investment & EBITDA Impact Page:**
    *   **AI Investment Efficiency ($\text{AIE}_j$):** "Measures the impact ($Org\_AI\_R$ points $\times$ $EBITDA\_Impact$) generated per dollar of AI investment. A higher AIE indicates more efficient capital deployment for AI initiatives."
    *   **Attributed EBITDA Impact ($\Delta\text{EBITDA}\%$):** "The estimated percentage increase in EBITDA directly attributed to the change in Org-AI-R score, factoring in industry opportunity. This quantifies the financial upside of AI improvements."
*   **Actionable Insights Page:**
    *   **Centers of Excellence:** "Portfolio companies with high Org-AI-R scores and significant EBITDA impact, serving as benchmarks for best practices and potential for replication across the fund."
    *   **Companies for Review:** "Portfolio companies with lower Org-AI-R scores or minimal EBITDA impact, indicating areas requiring strategic intervention, additional resources, or a re-evaluation of AI strategy."
*   **Exit-Readiness Page:**
    *   **Exit-AI-R Score:** "Quantifies how 'buyer-friendly' a company's AI capabilities are, considering how 'Visible', 'Documented', and 'Sustainable' these capabilities are to potential acquirers. Higher scores imply a stronger AI-driven exit narrative."
    *   **Projected Exit Multiple:** "The estimated valuation multiple for a company, calculated by adding an AI premium (derived from the Exit-AI-R score) to its baseline industry multiple."
    *   **Visible AI Capabilities:** "AI features apparent to buyers (e.g., product functionality, technology stack)."
    *   **Documented AI Impact:** "Quantified AI benefits with an auditable trail."
    *   **Sustainable AI Capabilities:** "Embedded, long-term AI capabilities vs. one-time projects."

#### State Management Requirements
*   All user inputs from widgets (sliders, number inputs, selectboxes, multiselects) will be stored in `st.session_state` to ensure they persist across page navigations and re-runs.
*   The `portfolio_df` DataFrame, which is the central data store, will be initialized in `st.session_state` on application start and updated by the `load_portfolio_data` function. All subsequent calculations will operate on `st.session_state.portfolio_df`.
*   Any derived metrics (e.g., `Org_AI_R_Score`, `Org_AI_R_Percentile`, `AI_Investment_Efficiency`, `Exit_AI_R_Score`) will be added as new columns to the `portfolio_df` within `st.session_state`.
*   User progress through the narrative flow (e.g., selected page) will be maintained via `st.session_state`.
*   The application will gracefully handle cases where `portfolio_df` is not yet initialized (e.g., if "Generate New Portfolio Data" hasn't been clicked), by prompting the user or showing loading indicators.

### 4. Notebook Content and Code Requirements

The Streamlit application will directly integrate the logic and calculations from the Jupyter Notebook. Each logical section of the notebook will correspond to a Streamlit page or a set of interactive components and displays.

**Initialization and Dependencies:**
*   The initial `!pip install` commands will be handled by a `requirements.txt` file for Streamlit deployment.
*   All `import` statements (pandas, numpy, matplotlib, seaborn, scipy.stats, tabulate, warnings) will be at the top of the `app.py` script.

**Code Stubs and UI Integration:**

**1. Initializing the Portfolio Overview: Data Loading for AI Performance Tracking**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 1.
    *   `# Portfolio AI Performance & Benchmarking Dashboard`
    *   `As a Portfolio Manager... This initial data load forms the bedrock...`
*   **Notebook Code (`load_portfolio_data`):**
    ```python
    def load_portfolio_data(num_companies=10, num_quarters=5):
        # ... (synthetic data generation logic as in notebook) ...
        # Simulate IndustryMeanOrgAIR and IndustryStdDevOrgAIR more realistically
        df['IndustryMeanOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('mean')
        df['IndustryStdDevOrgAIR'] = df.groupby(['Industry', 'Quarter'])['SystematicOpportunity'].transform('std').fillna(5) # fillna for single-company industries
        return df
    ```
*   **Streamlit Integration:**
    *   This function will be called on app startup if `st.session_state.portfolio_df` is not present, using default `num_companies` and `num_quarters`.
    *   It will be explicitly called when `st.session_state.generate_data_button` is clicked, using values from `st.session_state.num_companies_input` and `st.session_state.num_quarters_input`.
    *   The returned DataFrame will be stored in `st.session_state.portfolio_df`.
    *   **Display:** `st.subheader("Overview of Generated Portfolio Data:")`, `st.dataframe(st.session_state.portfolio_df.head())`, `st.dataframe(st.session_state.portfolio_df.info(buf=StringIO()))` (using `StringIO` to capture output).

**2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 2, including the formula:
    *   `## 2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment`
    *   `The core of our AI performance tracking is the PE Org-AI-R score...`
    *   `The formula for the PE Org-AI-R Score for target or portfolio company $j$ in industry $k$ at time $t$ is:`
    *   `$$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$`
    *   `where:` (followed by bullet points explaining terms, ensuring $V^R_{org,j}(t)$, $H^R_{org,k}(t)$, $\alpha$, $\beta$, $\text{Synergy}$ are correctly formatted).
    *   `This table gives me a quick overview...`
*   **Notebook Code (`calculate_org_ai_r`):**
    ```python
    def calculate_org_ai_r(df, alpha=0.6, beta=0.15):
        # ... (logic as in notebook, with ValueError checks) ...
        df['Org_AI_R_Score'] = (
            alpha * df['IdiosyncraticReadiness'] +
            (1 - alpha) * df['SystematicOpportunity'] +
            beta * df['Synergy']
        )
        df['Org_AI_R_Score'] = np.clip(df['Org_AI_R_Score'], 0, 100)
        return df
    ```
*   **Streamlit Integration:**
    *   The `calculate_org_ai_r` function will be called in Page 2's logic, passing `st.session_state.portfolio_df`, `st.session_state.alpha_slider`, and `st.session_state.beta_slider`.
    *   The updated DataFrame will be stored back into `st.session_state.portfolio_df`.
    *   **Display:** `st.subheader("Latest Quarter's PE Org-AI-R Scores:")`, `st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Org_AI_R_Score']].sort_values(by='Org_AI_R_Score', ascending=False))` (using a dynamically filtered `latest_quarter_df`).

**3. Benchmarking Portfolio Companies: Identifying Relative AI Performance**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 3, including formulas:
    *   `## 3. Benchmarking Portfolio Companies: Identifying Relative AI Performance`
    *   `Understanding a company's standalone Org-AI-R score...`
    *   `*   **Within-Portfolio Benchmarking (Definition 6):**`
    *   `$$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$`
    *   `*   **Cross-Portfolio Benchmarking (Definition 6):**`
    *   `$$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$`
    *   `These benchmarks are invaluable...`
*   **Notebook Code (`calculate_benchmarks` and visualizations):**
    ```python
    def calculate_benchmarks(df):
        df['Org_AI_R_Percentile'] = df.groupby('Quarter')['Org_AI_R_Score'].rank(pct=True) * 100
        df['Org_AI_R_Z_Score'] = df.apply(
            lambda row: (row['Org_AI_R_Score'] - row['IndustryMeanOrgAIR']) / row['IndustryStdDevOrgAIR']
            if row['IndustryStdDevOrgAIR'] != 0 else 0, axis=1
        )
        return df
    # Visualization code (Bar chart, Scatter plot)
    ```
*   **Streamlit Integration:**
    *   The `calculate_benchmarks` function will be called, taking `st.session_state.portfolio_df`. The updated DF will be stored.
    *   Filter `st.session_state.portfolio_df` for the quarter selected by `st.session_state.benchmark_quarter_select`.
    *   **Display:** `st.subheader("Latest Quarter's Org-AI-R Benchmarks:")`, `st.dataframe(filtered_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'Org_AI_R_Percentile', 'Org_AI_R_Z_Score']].sort_values(by='Org_AI_R_Score', ascending=False))`.
    *   **Visualizations:** Use `st.pyplot()` for the bar chart and scatter plot, passing the filtered DataFrame.

**4. Assessing AI Investment Efficiency and EBITDA Attribution**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 4, including formulas:
    *   `## 4. Assessing AI Investment Efficiency and EBITDA Attribution`
    *   `As a Portfolio Manager, I need to go beyond just scores...`
    *   `The formula for AI Investment Efficiency for company $j$ over a period $T$ is:`
    *   `$$ \text{AIE}_j = \frac{\Delta\text{Org-AI-R}_j}{\text{AI Investment}_j} \times \text{EBITDA Impact}_j $$`
    *   `The formula for EBITDA Attribution percentage is:`
    *   `$$ \Delta\text{EBITDA}\% = \gamma \cdot \Delta\text{Org-AI-R} \cdot H^R_{org,k}/100 $$`
    *   `This analysis provides critical insights...`
*   **Notebook Code (`calculate_aie_ebitda` and visualization):**
    ```python
    def calculate_aie_ebitda(df):
        df_sorted = df.sort_values(by=['CompanyID', 'Quarter'])
        df_sorted['Delta_Org_AI_R'] = df_sorted.groupby('CompanyID')['Org_AI_R_Score'].diff().fillna(0)
        df_sorted['AI_Investment_Efficiency'] = df_sorted.apply(
            lambda row: (row['Delta_Org_AI_R'] / row['AI_Investment']) * row['EBITDA_Impact'] * 1000000 # Scaling for readability
            if row['AI_Investment'] > 0 and row['Delta_Org_AI_R'] > 0 else 0, axis=1
        )
        df_sorted['Attributed_EBITDA_Impact_Pct'] = df_sorted.apply(
            lambda row: row['GammaCoefficient'] * row['Delta_Org_AI_R'] * row['IndustryMeanOrgAIR'] / 100
            if row['Delta_Org_AI_R'] > 0 else 0, axis=1
        )
        df_sorted['Attributed_EBITDA_Impact_Absolute'] = (df_sorted['Attributed_EBITDA_Impact_Pct'] / 100) * df_sorted['BaselineEBITDA']
        return df_sorted
    # Visualization code (Scatter plot)
    ```
*   **Streamlit Integration:**
    *   The `calculate_aie_ebitda` function will be called, taking `st.session_state.portfolio_df`. The updated DF will be stored.
    *   Filter `st.session_state.portfolio_df` for the latest quarter.
    *   **Display:** `st.subheader("Latest Quarter's AI Investment Efficiency and Attributed EBITDA Impact:")`, `st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Delta_Org_AI_R', 'AI_Investment', 'AI_Investment_Efficiency', 'Attributed_EBITDA_Impact_Pct', 'Attributed_EBITDA_Impact_Absolute']].sort_values(by='AI_Investment_Efficiency', ascending=False))`
    *   **Visualization:** Use `st.pyplot()` for the scatter plot.

**5. Tracking Progress Over Time: Visualizing Trajectories**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 5:
    *   `## 5. Tracking Progress Over Time: Visualizing Trajectories`
    *   `As a Portfolio Manager, current metrics are important...`
    *   `These time-series charts provide a dynamic view...`
*   **Notebook Code (`plot_time_series` function):**
    ```python
    def plot_time_series(df, metric_col, title, ylabel, selected_companies=None):
        # ... (logic as in notebook, adapted to filter for selected_companies) ...
        plt.figure(figsize=(14, 8))
        if selected_companies:
            df = df[df['CompanyName'].isin(selected_companies)]
        sns.lineplot(x='Quarter', y=metric_col, hue='CompanyName', marker='o', data=df)
        # ... (portfolio average logic) ...
        plt.show()
    ```
*   **Streamlit Integration:**
    *   Call `plot_time_series` twice in Page 5, once for `Org_AI_R_Score` and once for `AI_Investment_Efficiency`.
    *   Pass `st.session_state.portfolio_df` and `st.session_state.companies_to_track_multiselect`.
    *   **Display:** `st.pyplot()` for each generated chart.

**6. Identifying Centers of Excellence and Companies for Review**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 6:
    *   `## 6. Identifying Centers of Excellence and Companies for Review`
    *   `A key responsibility of a Portfolio Manager is to leverage successes...`
    *   `**Center of Excellence Criteria (from Section 6.3):**`
    *   `*   Org-AI-R Score $> 75$`
    *   `*   Demonstrated EBITDA impact $> 3\%$ `
    *   `**Companies for Review Criteria:**`
    *   `*   Org-AI-R Score $\le 50$`
    *   `*   OR EBITDA impact $\le 1\%$ `
    *   `This targeted identification is critical...`
*   **Notebook Code (`identify_actionable_insights` and visualization):**
    ```python
    def identify_actionable_insights(df, org_ai_r_threshold_coe=75, ebitda_impact_threshold_coe=3,
                                     org_ai_r_threshold_review=50, ebitda_impact_threshold_review=1.0):
        latest_data = df.loc[df.groupby('CompanyID')['Quarter'].idxmax()]
        centers_of_excellence = latest_data[
            (latest_data['Org_AI_R_Score'] > org_ai_r_threshold_coe) &
            (latest_data['EBITDA_Impact'] > ebitda_impact_threshold_coe)
        ].sort_values(by='Org_AI_R_Score', ascending=False)
        companies_for_review = latest_data[
            (latest_data['Org_AI_R_Score'] <= org_ai_r_threshold_review) |
            (latest_data['EBITDA_Impact'] <= ebitda_impact_threshold_review)
        ].sort_values(by='Org_AI_R_Score', ascending=True)
        return centers_of_excellence, companies_for_review
    # Visualization code (Scatter plot highlighting CoE and Review Companies)
    ```
*   **Streamlit Integration:**
    *   Call `identify_actionable_insights` using `st.session_state.portfolio_df` and the slider values from Page 6.
    *   **Display:** `st.subheader("--- Centers of Excellence ---")`, `st.dataframe(centers_of_excellence_df[['CompanyName', 'Industry', 'Org_AI_R_Score', 'EBITDA_Impact', 'AI_Investment_Efficiency']])` (conditional display if empty). Same for "Companies for Deeper Review".
    *   **Visualization:** Use `st.pyplot()` for the scatter plot, ensuring CoE and Review companies are highlighted and labeled with company names based on the identified dataframes. The threshold lines should correspond to the slider values.

**7. Evaluating Exit-Readiness and Potential Valuation Impact**
*   **Notebook Markdown:** Displayed as `st.markdown` on Page 7, including formulas:
    *   `## 7. Evaluating Exit-Readiness and Potential Valuation Impact`
    *   `As a Portfolio Manager, preparing for a successful exit is always on my mind...`
    *   `The formula for the Exit-Readiness Score for portfolio company $j$ preparing for exit is:`
    *   `$$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$`
    *   `The Multiple Attribution Model then translates this Exit-AI-R score...`
    *   `$$ \text{Multiple}_j = \text{Multiple}_{base,k} + \delta \cdot \text{Exit-AI-R}_j/100 $$`
    *   `This analysis provides critical data for our exit planning strategy...`
*   **Notebook Code (`calculate_exit_readiness_and_valuation` and visualization):**
    ```python
    def calculate_exit_readiness_and_valuation(df, w1=0.35, w2=0.40, w3=0.25):
        df_copy = df.copy() # Operate on a copy to avoid SettingWithCopyWarning
        df_copy['Exit_AI_R_Score'] = (
            w1 * df_copy['Visible'] + w2 * df_copy['Documented'] + w3 * df_copy['Sustainable']
        )
        df_copy['Exit_AI_R_Score'] = np.clip(df_copy['Exit_AI_R_Score'], 0, 100)
        df_copy['AI_Premium_Multiple_Additive'] = df_copy['AI_PremiumCoefficient'] * df_copy['Exit_AI_R_Score'] / 100
        df_copy['Projected_Exit_Multiple'] = df_copy['BaselineMultiple'] + df_copy['AI_Premium_Multiple_Additive']
        return df_copy
    # Visualization code (Scatter plot)
    ```
*   **Streamlit Integration:**
    *   Call `calculate_exit_readiness_and_valuation` using `st.session_state.portfolio_df` and the slider values for $w_1$, $w_2$, $w_3$ from Page 7. The updated DF will be stored.
    *   Filter `st.session_state.portfolio_df` for the latest quarter.
    *   **Display:** `st.subheader("Latest Quarter's Exit-Readiness and Projected Valuation Impact:")`, `st.dataframe(latest_quarter_df[['CompanyName', 'Industry', 'Exit_AI_R_Score', 'BaselineMultiple', 'Projected_Exit_Multiple']].sort_values(by='Projected_Exit_Multiple', ascending=False))`
    *   **Visualization:** Use `st.pyplot()` for the scatter plot, ensuring company names are labeled on the points.

