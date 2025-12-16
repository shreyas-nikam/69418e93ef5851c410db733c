id: 69418e93ef5851c410db733c_user_guide
summary: Portfolio AI Performance & Benchmarking Dashboard User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Empowering Private Equity Portfolio Managers with AI Performance Insights

## 1. Initializing Your Portfolio Data
Duration: 05:00

Welcome to **QuLab: Portfolio AI Performance & Benchmarking Dashboard**! As a Private Equity Portfolio Manager, you understand that Artificial Intelligence (AI) is no longer just a buzzword—it's a critical driver of value, efficiency, and competitive advantage across your portfolio. This application is designed to be your indispensable tool for systematically assessing, benchmarking, and optimizing AI performance within your fund's holdings.

In this lab, you will learn to:
*   Systematically quantify and manage the impact of AI across your diverse portfolio.
*   Benchmark companies against their peers and industry standards.
*   Track progress over time, identifying "Centers of Excellence" and "Companies for Review."
*   Assess how AI capabilities enhance a company's exit readiness and potential valuation.

Each step in this journey empowers you to make data-driven decisions that optimize your fund's overall AI strategy and maximize risk-adjusted returns.

<aside class="positive">
<b>Important Context:</b> This application simulates a real-world scenario where you, as a PE Portfolio Manager, are seeking a systematic framework to understand and improve AI adoption and impact across your acquired companies. The metrics and models presented are designed to provide tangible, actionable insights.
</aside>

Your first step is to get an overview of your portfolio's data.

1.  **Adjust Portfolio Size and History:** On the left sidebar, you can define the `Number of Portfolio Companies` (between 5 and 20) and `Number of Quarters (History)` (between 2 and 10) for which you want to generate synthetic data.
    *   The application automatically generates a diverse set of companies across different industries with simulated AI readiness metrics, investment figures, and financial impacts over several quarters.
    *   Initially, the application loads with default values (10 companies, 5 quarters).

2.  **Generate New Data (Optional):** If you wish to create a fresh dataset based on your chosen parameters, click the **Generate New Portfolio Data** button in the sidebar. This will refresh all subsequent calculations and visualizations.

3.  **Review the Initial Data:**
    *   Scroll down to the "Overview of Generated Portfolio Data" section. Here, you'll see the first few rows of your loaded dataset, giving you a glimpse into the various metrics being tracked, such as `IdiosyncraticReadiness`, `SystematicOpportunity`, `AI_Investment`, and `BaselineEBITDA`.
    *   The "Descriptive Statistics" table provides a summary of the numerical data, including mean, standard deviation, min, and max values. This helps you quickly understand the range and distribution of your portfolio's AI-related metrics.
    *   The "Data Information" section gives you technical details about the DataFrame, such as column names, non-null counts, and data types, ensuring data integrity.

<aside class="positive">
Understanding your raw data is fundamental. It ensures you have the necessary building blocks for all strategic analyses, from measuring AI maturity to projecting exit valuations.
</aside>

## 2. Calculating PE Org-AI-R Scores
Duration: 07:00

The **PE Org-AI-R Score** is the cornerstone of our AI maturity assessment. It's a single, composite metric designed to quantify each portfolio company's overall AI maturity and its readiness to generate value from AI initiatives. As a Portfolio Manager, you need the flexibility to calibrate this score to reflect your fund's strategic priorities, emphasizing either internal capabilities or external market opportunities.

The Org-AI-R Score for a company $j$ in industry $k$ at time $t$ is calculated using the following formula:
$$ \text{PE Org-AI-R}_{j,t} = \alpha \cdot V^R_{org,j}(t) + (1 - \alpha) \cdot H^R_{org,k}(t) + \beta \cdot \text{Synergy}(V^R_{org,j}, H^R_{org,k}) $$

Let's break down what each component means:
*   $V^R_{org,j}(t)$: **Idiosyncratic Readiness** for company $j$ at time $t$. This represents the company's unique, internal capabilities related to AI. Think of factors like the strength of its data infrastructure, the talent of its AI team, its internal processes for AI adoption, and the leadership's commitment to AI.
*   $H^R_{org,k}(t)$: **Systematic Opportunity** for industry $k$ at time $t$. This reflects the broader, industry-level potential for AI. This includes factors like the overall AI adoption rates within the sector, the potential for AI to disrupt existing business models, and the competitive landscape for AI innovation.
*   $\alpha$: **Weight for Idiosyncratic Readiness.** This is a crucial slider that allows you to prioritize how much emphasis is placed on a company's internal capabilities ($V^R_{org,j}$) versus the external industry potential ($H^R_{org,k}$). A higher $\alpha$ means you value internal readiness more.
*   $\beta$: **Synergy Coefficient.** This quantifies the additional value or uplift derived from the effective interplay between a company's idiosyncratic readiness and the systematic opportunity in its industry. It reflects how well a company is positioned to capitalize on industry trends given its internal strengths and strategic alignment.
*   $\text{Synergy}(V^R_{org,j}, H^R_{org,k})$: A term representing the alignment and integration between a company's idiosyncratic readiness and the systematic opportunity in its industry.

1.  **Adjust the Weights ($\alpha$ and $\beta$):** Use the sliders under the formula explanation to adjust:
    *   `Weight for Idiosyncratic Readiness ($\alpha$)`: Ranges from 0.55 to 0.70.
    *   `Synergy Coefficient ($\beta$)`: Ranges from 0.08 to 0.25.
    Experiment with these values to see how they influence the overall Org-AI-R scores. For example, if you believe a company's internal AI capabilities are paramount, you might increase $\alpha$. If you think a company's ability to leverage industry trends is key, adjust $\beta$.

2.  **Recalculate Scores:** After adjusting the weights, click the **Recalculate Org-AI-R Scores** button. This will re-compute the Org-AI-R score for all companies based on your updated strategic emphasis.

3.  **Review Latest Quarter's Scores:** Observe the "Latest Quarter's PE Org-AI-R Scores" table. It displays each company's name, industry, and its newly calculated Org-AI-R score, sorted from highest to lowest. This immediate feedback helps you understand the relative AI maturity of your portfolio companies under your chosen strategic focus.

<aside class="positive">
<b>Actionable Insight:</b> The Org-AI-R Score provides a holistic view. Companies with higher scores are generally more mature in their AI adoption and better positioned to extract value. This helps in initial screening for potential "Centers of Excellence" or "Companies for Review."
</aside>

## 3. Benchmarking AI Performance
Duration: 08:00

While knowing a company's standalone Org-AI-R score is valuable, as a Portfolio Manager, you need to understand its performance in context. **Benchmarking** allows you to identify leaders and laggards, both within your own portfolio and against broader industry standards, providing crucial context for strategic resource allocation.

We utilize two key benchmarking metrics:

1.  **Within-Portfolio Benchmarking (Org-AI-R Percentile):**
    This metric tells you how a company stacks up against all other companies in your fund for a given quarter. A higher percentile means the company's AI readiness is stronger relative to its peers within your portfolio.
    $$ \text{Percentile}_j = \left( \frac{\text{Rank}(\text{Org-AI-R}_j)}{\text{Portfolio Size}} \right) \times 100 $$
    Here, $\text{Rank}(\text{Org-AI-R}_j)$ is the rank of company $j$'s Org-AI-R score within the portfolio (from lowest to highest), and $\text{Portfolio Size}$ is the total number of companies in the portfolio for that quarter.

2.  **Cross-Portfolio Benchmarking (Org-AI-R Z-Score):**
    This metric adjusts for industry differences, showing you how a company's AI readiness deviates from its *industry's average*. A positive Z-score indicates the company is outperforming its industry peers in AI readiness, while a negative score suggests underperformance relative to its sector average. This helps you identify true outperformers or underperformers relative to their sector, removing the bias of industry-specific AI opportunities.
    $$ Z_{j,k} = \frac{\text{Org-AI-R}_j - \mu_k}{\sigma_k} $$
    Here, $\text{Org-AI-R}_j$ is the Org-AI-R score of company $j$, $\mu_k$ is the mean Org-AI-R score for industry $k$, and $\sigma_k$ is the standard deviation of Org-AI-R scores for industry $k$.

1.  **Select a Quarter:** Use the "Select Quarter for Benchmarking" dropdown to choose a specific quarter you want to analyze. By default, it will show the latest quarter.

2.  **Review Benchmarking Data:** The table titled "Org-AI-R Benchmarks" displays each company's `Org_AI_R_Score`, its `Org_AI_R_Percentile` within the portfolio, and its `Org_AI_R_Z_Score` relative to its industry peers. Companies are sorted by Org-AI-R Score.

3.  **Analyze Visualizations:**
    *   **Org-AI-R Scores by Company:** This bar chart visually represents each company's Org-AI-R score for the selected quarter, grouped by industry. The red dashed line shows the overall portfolio average. This immediately highlights which companies are leading or lagging in raw AI maturity.
    *   **Org-AI-R Score vs. Industry-Adjusted Z-Score:** This scatter plot is powerful for comparative analysis.
        *   Companies to the right have higher Org-AI-R scores.
        *   Companies above the horizontal black dashed line (Z-score of 0) are outperforming their industry average.
        *   Companies below the horizontal black dashed line are underperforming their industry average.
        *   The size of the bubble represents the company's Org-AI-R Percentile within the portfolio, giving you a quick sense of its internal ranking.
        *   The vertical grey dotted line represents the portfolio's mean Org-AI-R.
        *   This plot helps you quickly identify:
            *   **High-performing leaders:** High Org-AI-R score AND high positive Z-score (top right quadrant).
            *   **Potential laggards:** Low Org-AI-R score AND low negative Z-score (bottom left quadrant).
            *   **Industry outliers:** Companies with high Z-scores despite potentially average Org-AI-R, indicating strong performance relative to their sector's typical AI maturity.

<aside class="positive">
<b>Actionable Insight:</b> Benchmarking is invaluable for strategic resource allocation. Identify companies with high Org-AI-R and Z-scores as potential "Centers of Excellence" whose best practices can be scaled. Companies with low scores and negative Z-scores might need targeted intervention or re-evaluation.
</aside>

## 4. AI Investment & EBITDA Impact
Duration: 08:00

Beyond just scores, as a Portfolio Manager, you need to understand the tangible financial impact of AI initiatives. This section quantifies the return on AI investment and attributes EBITDA growth directly to improvements in AI readiness. This allows you to evaluate the effectiveness of capital deployment and identify areas where AI investments are truly moving the needle.

We focus on two key financial impact metrics:

1.  **AI Investment Efficiency ($\text{AIE}_j$):**
    This metric measures how effectively a company is translating its AI investments into improvements in its AI readiness and subsequent financial impact. A higher AIE indicates more efficient capital deployment for AI initiatives.
    $$ \text{AIE}_j = \left( \frac{\Delta\text{Org-AI-R}_j}{\text{AI Investment}_j} \right) \times \text{EBITDA Impact}_j \times C $$
    Here, $\Delta\text{Org-AI-R}_j$ is the change in Org-AI-R score for company $j$ from the previous quarter, $\text{AI Investment}_j$ is the AI investment for company $j$ in the current quarter, $\text{EBITDA Impact}_j$ is the direct percentage EBITDA impact, and $C$ is a scaling constant (e.g., $1,000,000$ to represent impact points per million invested, making the numbers more readable).

2.  **Attributed EBITDA Impact Percentage ($\Delta\text{EBITDA}\%$):**
    This formula estimates the percentage increase in EBITDA directly attributed to the change in Org-AI-R score, factoring in the broader industry opportunity. This helps you quantify the financial uplift from improving AI readiness.
    $$ \Delta\text{EBITDA}\% = \gamma \cdot \Delta\text{Org-AI-R}_j \cdot (H^R_{org,k} / 100) $$
    Here, $\gamma$ is a scaling coefficient (`GammaCoefficient`), $\Delta\text{Org-AI-R}_j$ is the change in Org-AI-R score for company $j$, and $H^R_{org,k}$ is the systematic opportunity for industry $k$ (which is proxied by the `IndustryMeanOrgAIR`).

1.  **Review Financial Impact Data:** The table titled "Latest Quarter's AI Investment Efficiency and Attributed EBITDA Impact" displays a breakdown of these metrics for the most recent quarter.
    *   `AI_Investment`: The absolute amount invested in AI for the quarter.
    *   `Delta_Org_AI_R`: The change in Org-AI-R score from the previous quarter. This is crucial for understanding progress.
    *   `AI_Investment_Efficiency`: How much impact (scaled for readability) was generated per unit of investment.
    *   `Attributed_EBITDA_Impact_Pct`: The percentage of EBITDA growth directly attributed to AI readiness improvements.
    *   `Attributed_EBITDA_Impact_Absolute`: The absolute dollar value of EBITDA impact attributed to AI.

2.  **Analyze the AI Investment vs. Efficiency Scatter Plot:**
    This visualization plots `AI_Investment` (on a logarithmic scale to better represent varying investment sizes) against `AI_Investment_Efficiency`.
    *   Companies higher on the Y-axis are generating more impact per dollar invested, indicating greater efficiency.
    *   The size of the bubble represents the `Attributed_EBITDA_Impact_Pct`, showing which companies are not only efficient but also driving significant financial uplift.
    *   This plot helps you identify:
        *   **Highly efficient impact generators:** High `AI_Investment_Efficiency` with large bubble sizes.
        *   **High investment, low efficiency:** Companies spending a lot on AI but seeing minimal returns (far right on X-axis, low on Y-axis). These might need strategic review.
        *   **Hidden gems:** Companies with relatively low AI investment but surprisingly high efficiency.

<aside class="positive">
<b>Actionable Insight:</b> This analysis provides critical insights into which companies are most effectively translating their AI maturity into financial returns. It guides future investment decisions, allowing you to reallocate capital to more efficient initiatives or investigate underperforming ones.
</aside>

## 5. Tracking Progress Over Time
Duration: 06:00

As a Portfolio Manager, current metrics are important, but understanding the **trajectory of performance over time** is crucial for assessing long-term strategy effectiveness. This section allows you to monitor how individual companies are progressing in their AI journey and how efficiently they're utilizing their AI investments quarter-over-quarter. Observing trends helps confirm that strategic initiatives are yielding continuous improvements and identifies any stalls or regressions.

1.  **Select Companies to Track:** Use the "Select Companies to Track" multiselect box. You can choose up to 5 companies (recommended for clarity) to visualize their historical performance.

2.  **Analyze Org-AI-R Score Trajectory:**
    *   The first line chart, "Org-AI-R Score Trajectory Over Time," displays the Org-AI-R score for your selected companies across all available quarters.
    *   Each colored line represents a different company, while the black dashed line shows the overall portfolio average.
    *   Observe:
        *   **Upward trends:** Companies consistently improving their Org-AI-R score.
        *   **Plateaus or declines:** Companies that have stalled or regressed in their AI maturity.
        *   **Outperformance/Underperformance vs. Average:** How individual companies compare to the overall portfolio trend.

3.  **Analyze AI Investment Efficiency Trajectory:**
    *   The second line chart, "AI Investment Efficiency Trajectory Over Time," tracks the `AI_Investment_Efficiency` for your selected companies over time.
    *   This shows if companies are getting more efficient in their AI spending, less efficient, or remaining consistent.
    *   Observe:
        *   Are companies improving their ability to generate impact per million invested?
        *   Are there specific quarters where efficiency spiked or dropped significantly, potentially correlating with specific AI project launches or challenges?

<aside class="positive">
<b>Actionable Insight:</b> Tracking trends helps you validate long-term strategies. Continuous improvement in Org-AI-R and AI Investment Efficiency indicates healthy progress. Conversely, flatlining or declining trends signal a need for deeper investigation and potential strategic pivots.
</aside>

## 6. Actionable Insights: CoE & Review
Duration: 10:00

A key responsibility of a Portfolio Manager is to leverage successes and address underperformance proactively. This section empowers you to define specific criteria for identifying your **"Centers of Excellence"** – companies with outstanding AI performance that can serve as benchmarks for best practices – and **"Companies for Review"** – those needing immediate strategic attention or resource reallocation.

This targeted identification is critical for optimizing your fund's overall AI strategy and maximizing risk-adjusted returns by fostering best practices and mitigating risks.

1.  **Define Thresholds:** Use the sliders to set the criteria for identifying these groups:
    *   `Org-AI-R Score Threshold for Center of Excellence`: Companies above this score will be considered.
    *   `EBITDA Impact (%) Threshold for Center of Excellence`: Companies above this percentage impact will be considered.
    *   `Org-AI-R Score Threshold for Companies for Review`: Companies *below or equal to* this score will be considered.
    *   `EBITDA Impact (%) Threshold for Companies for Review`: Companies *below or equal to* this percentage impact will be considered.

2.  **Re-evaluate Insights:** Click the **Re-evaluate Actionable Insights** button after adjusting the thresholds to update the lists and visualizations.

3.  **Review Centers of Excellence:**
    *   This table lists portfolio companies that exceed *both* your defined Org-AI-R score and EBITDA impact thresholds.
    *   These are your top performers in AI maturity and financial impact. They should be studied for best practices and potentially scaled initiatives across the portfolio.

4.  **Review Companies for Review:**
    *   This table lists portfolio companies that fall *below or equal to either* your defined Org-AI-R score or EBITDA impact thresholds.
    *   These companies require immediate attention. They might need strategic intervention, additional resources, a re-evaluation of their AI strategy, or closer monitoring.

5.  **Analyze the Portfolio AI Performance Scatter Plot:**
    This scatter plot visually maps all companies in the latest quarter based on their `Org_AI_R_Score` and `EBITDA_Impact`.
    *   **Green Lines and Stars:** The green dotted lines represent your "Center of Excellence" thresholds. Companies marked with large green stars ($\ast$) are those identified as Centers of Excellence, meeting both criteria. Their names are also highlighted in green.
    *   **Red Lines and X's:** The red dashed lines represent your "Companies for Review" thresholds. Companies marked with large red X's are those identified for review, falling below one or both thresholds. Their names are also highlighted in red.
    *   The size of the bubble still represents `AI_Investment_Efficiency`, adding another dimension to your analysis.

<aside class="positive">
<b>Actionable Insight:</b> This visualization is a powerful decision-making tool. It helps you quickly identify where to allocate your time and resources: celebrate and leverage your green stars, and prioritize interventions for your red X's.
</aside>

## 7. Evaluating Exit-Readiness and Potential Valuation Impact
Duration: 09:00

As a Portfolio Manager, preparing for a successful exit is a constant focus. The AI capabilities of your portfolio companies can significantly influence their attractiveness to potential acquirers and, consequently, their exit valuation. This section allows you to assess how "buyer-friendly" a company's AI capabilities are and quantify the potential premium on its exit multiple.

We use two key models for this:

1.  **Exit-AI-R Score:** This score assesses a company's AI capabilities from a buyer's perspective, focusing on factors that enhance perceived value during an acquisition.
    $$ \text{Exit-AI-R}_j = w_1 \cdot \text{Visible}_j + w_2 \cdot \text{Documented}_j + w_3 \cdot \text{Sustainable}_j $$
    Let's define the components:
    *   $\text{Visible}_j$: **Visible AI Capabilities.** These are AI features that are easily apparent and demonstrable to buyers, such as product functionality powered by AI, a well-defined AI technology stack, or compelling AI-driven customer experiences.
    *   $\text{Documented}_j$: **Documented AI Impact.** This refers to quantified AI benefits with an auditable trail. Buyers want to see clear, verifiable evidence of how AI has driven revenue growth, cost savings, or operational efficiency, supported by data and metrics.
    *   $\text{Sustainable}_j$: **Sustainable AI Capabilities.** This assesses whether AI capabilities are deeply embedded and long-term, rather than one-off projects. This includes factors like a strong AI talent pipeline, robust data governance, scalable AI infrastructure, and a culture of continuous AI innovation.
    *   $w_1, w_2, w_3$: **Weighting factors.** These sliders allow you to emphasize different aspects that potential buyers might prioritize when evaluating a company's AI assets for a stronger exit narrative. Typically, these weights should sum close to 1.

2.  **Multiple Attribution Model:** This model translates the Exit-AI-R score into a potential valuation uplift, showing how strong AI capabilities can command a higher exit multiple.
    $$ \text{Multiple}_j = \text{Multiple}_{base,k} + \delta \cdot (\text{Exit-AI-R}_j / 100) $$
    Here:
    *   $\text{Multiple}_{base,k}$: The **baseline industry valuation multiple** for industry $k$. This is the standard multiple typically applied to companies in that sector.
    *   $\delta$: The **AI Premium Coefficient** (`AI_PremiumCoefficient`). This coefficient determines how much each point of Exit-AI-R score contributes to an additive premium on the baseline valuation multiple. A higher $\delta$ implies that strong AI capabilities are highly valued by the market.

1.  **Adjust Exit-AI-R Weights:** Use the sliders for $w_1$, $w_2$, and $w_3$ to reflect what you believe buyers would prioritize (e.g., more weight on `Documented AI Impact` if you think buyers are highly focused on proven ROI).
    <aside class="negative">
    <b>Warning:</b> While not strictly enforced, consider adjusting $w_1+w_2+w_3$ to sum close to 1 for typical weighted averages, though the model will function regardless.
    </aside>

2.  **Recalculate Exit-Readiness & Valuation:** Click the **Recalculate Exit-Readiness & Valuation** button to update the scores and projected multiples based on your chosen weights.

3.  **Review Exit-Readiness Data:** The table "Latest Quarter's Exit-Readiness and Projected Valuation Impact" displays:
    *   `Exit_AI_R_Score`: The calculated Exit-AI-R score.
    *   `BaselineMultiple`: The industry's baseline valuation multiple.
    *   `AI_Premium_Multiple_Additive`: The additional multiple attributed solely to the company's AI capabilities.
    *   `Projected_Exit_Multiple`: The total projected multiple, including the AI premium.

4.  **Analyze the Exit-AI-R Score vs. Projected Exit Multiple Scatter Plot:**
    This scatter plot visually connects a company's `Exit_AI_R_Score` with its `Projected_Exit_Multiple`.
    *   Companies further to the right have higher Exit-AI-R scores, indicating more buyer-friendly AI capabilities.
    *   Companies higher on the Y-axis command a higher projected exit multiple, suggesting stronger valuation potential.
    *   Each point is labeled with the company name, and the bubble size represents the `Attributed_EBITDA_Impact_Pct`, showing the underlying financial impact.
    *   This plot helps you identify:
        *   Companies with strong AI assets that are likely to fetch a premium valuation.
        *   Companies where improving visible, documented, and sustainable AI capabilities can directly translate into a higher exit multiple.

<aside class="positive">
<b>Actionable Insight:</b> This analysis provides critical data for your exit planning strategy. It enables you to prioritize investments in AI areas that directly enhance buyer appeal and valuation, helping you strategically position companies for maximum return at exit.
</aside>
