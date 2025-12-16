
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
import pytest

# Helper function to load the app and ensure initial data generation
def get_app_with_initial_data():
    """
    Loads the Streamlit app and runs it once to ensure initial data generation
    and session state setup.
    """
    at = AppTest.from_file("app.py")
    at.run() 
    # The app's initial logic ensures portfolio_df is generated on the first run.
    assert "portfolio_df" in at.session_state
    assert not at.session_state["portfolio_df"].empty
    return at

def test_initial_app_load_and_global_controls():
    """
    Tests the initial loading of the app, presence of main elements,
    and interaction with global sidebar controls for data generation.
    """
    at = get_app_with_initial_data()

    # 1. Verify initial page content and title
    assert at.markdown[0].value.startswith("In this lab, we embark on a journey"), "Initial narrative markdown not found"
    assert at.sidebar.image[0].exists, "Sidebar image not found"
    assert at.sidebar.title[0].value == "QuLab: Portfolio AI Performance & Benchmarking Dashboard", "Sidebar title incorrect"
    assert at.header[0].value == "1. Initializing Portfolio Data: Overview", "Initial page header incorrect"

    # 2. Verify initial dataframes on "1. Initializing Portfolio Data" page
    assert at.dataframe[0].exists, "First dataframe (df.head()) not found"
    assert at.dataframe[1].exists, "Second dataframe (df.describe()) not found"
    assert at.text[0].exists, "Text output (df.info()) not found"
    assert "Company 1" in at.dataframe[0].to_string(), "Expected company name not in dataframe head"

    # 3. Verify initial global controls (sidebar number inputs)
    assert at.sidebar.number_input[0].value == 10, "Default num_companies is incorrect"
    assert at.sidebar.number_input[1].value == 5, "Default num_quarters is incorrect"

    # 4. Test changing global controls and regenerating data
    at.sidebar.number_input[0].set_value(7).run() # Change num_companies to 7
    at.sidebar.number_input[1].set_value(3).run() # Change num_quarters to 3
    
    assert at.sidebar.number_input[0].value == 7, "num_companies slider value did not update"
    assert at.sidebar.number_input[1].value == 3, "num_quarters slider value did not update"

    at.sidebar.button[0].click().run() # Click "Generate New Portfolio Data"
    
    assert at.success[0].value == "New synthetic portfolio data generated successfully! All calculations have been re-run.", "Success message not displayed"
    assert at.session_state["portfolio_df"] is not None, "portfolio_df not in session_state after regeneration"
    assert len(at.session_state["portfolio_df"]["CompanyName"].unique()) == 7, "Number of companies in generated data is incorrect"
    assert len(at.session_state["portfolio_df"]["Quarter"].unique()) == 3, "Number of quarters in generated data is incorrect"

    # Since the app calls st.rerun(), the page should reset to "1. Initializing Portfolio Data"
    assert at.header[0].value == "1. Initializing Portfolio Data: Overview", "Page did not reset to initial view"
    assert "Company 7" in at.dataframe[0].to_string(), "New company count not reflected in dataframe head"
    assert "Company 8" not in at.dataframe[0].to_string(), "Old company count still reflected in dataframe head"


def test_page_navigation():
    """
    Tests that selecting different radio buttons in the sidebar correctly
    changes the main content area's header.
    """
    at = get_app_with_initial_data()

    page_headers = {
        "1. Initializing Portfolio Data": "1. Initializing Portfolio Data: Overview",
        "2. Calculating Org-AI-R Scores": "2. Calculating PE Org-AI-R Scores: The Foundation of AI Maturity Assessment",
        "3. Benchmarking AI Performance": "3. Benchmarking Portfolio Companies: Identifying Relative AI Performance",
        "4. AI Investment & EBITDA Impact": "4. Assessing AI Investment Efficiency and EBITDA Attribution",
        "5. Tracking Progress Over Time": "5. Tracking Progress Over Time: Visualizing Trajectories",
        "6. Actionable Insights: CoE & Review": "6. Identifying Centers of Excellence and Companies for Review",
        "7. Exit-Readiness & Valuation": "7. Evaluating Exit-Readiness and Potential Valuation Impact",
    }

    for radio_label, expected_header in page_headers.items():
        at.sidebar.radio[0].set_value(radio_label).run()
        assert at.header[0].value == expected_header, f"Failed to navigate to '{radio_label}'. Expected header '{expected_header}', got '{at.header[0].value}'"


def test_org_ai_r_calculation_page_interactions():
    """
    Tests the functionality of the "Calculating Org-AI-R Scores" page,
    including slider adjustments and recalculation.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("2. Calculating Org-AI-R Scores").run()

    # Verify initial state of sliders
    assert at.slider[0].value == 0.60, "Default alpha slider value incorrect"
    assert at.slider[1].value == 0.15, "Default beta slider value incorrect"
    
    # Get initial Org-AI-R score for a reference company from session state
    initial_df = at.session_state["portfolio_df"]
    company_name_ref = initial_df["CompanyName"].iloc[0]
    initial_org_ai_r = initial_df[initial_df["CompanyName"] == company_name_ref]["Org_AI_R_Score"].iloc[-1]

    # Change slider values
    new_alpha = 0.65
    new_beta = 0.20
    at.slider[0].set_value(new_alpha).run()
    at.slider[1].set_value(new_beta).run()
    
    assert at.slider[0].value == new_alpha, "Alpha slider value did not update"
    assert at.slider[1].value == new_beta, "Beta slider value did not update"

    # Click recalculate button
    at.button[0].click().run()
    assert at.success[0].value == "Org-AI-R scores and related metrics updated successfully!", "Recalculation success message not displayed"

    # Verify Org-AI-R score has changed for the reference company
    updated_df = at.session_state["portfolio_df"]
    updated_org_ai_r = updated_df[updated_df["CompanyName"] == company_name_ref]["Org_AI_R_Score"].iloc[-1]
    assert updated_org_ai_r != initial_org_ai_r, "Org-AI-R score did not change after recalculation"
    
    # Verify the displayed dataframe is present and contains Org_AI_R_Score
    assert at.dataframe[0].exists, "Org-AI-R scores dataframe not found"
    assert "Org_AI_R_Score" in at.dataframe[0].to_string(), "Org_AI_R_Score column not found in dataframe"
    assert not at.pyplot, "Unexpected plot found on Org-AI-R calculation page"


def test_benchmarking_page_interactions():
    """
    Tests the functionality of the "Benchmarking AI Performance" page,
    including selectbox interaction and plot presence.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("3. Benchmarking AI Performance").run()

    # Verify dataframe contains percentile and z-score columns
    assert at.dataframe[0].exists, "Benchmarking dataframe not found"
    df_str = at.dataframe[0].to_string()
    assert "Org_AI_R_Percentile" in df_str, "Org_AI_R_Percentile column not in dataframe"
    assert "Org_AI_R_Z_Score" in df_str, "Org_AI_R_Z_Score column not in dataframe"

    # Verify initial plots are present (2 plots: bar chart and scatter plot)
    assert len(at.pyplot) == 2, f"Expected 2 plots, but found {len(at.pyplot)}"
    
    # Test selectbox interaction (change quarter)
    quarter_options = at.selectbox[0].options
    if len(quarter_options) > 1:
        initial_quarter = at.selectbox[0].value
        # Select a different quarter
        new_quarter = quarter_options[0] if initial_quarter != quarter_options[0] else quarter_options[1]
        at.selectbox[0].set_value(new_quarter).run()
        assert at.selectbox[0].value == new_quarter, "Selected quarter in selectbox did not update"
        # The dataframe and plots should implicitly update with the new quarter data.
        assert len(at.pyplot) == 2, "Plots disappeared after changing quarter"


def test_ai_investment_ebitda_impact_page_interactions():
    """
    Tests the functionality of the "AI Investment & EBITDA Impact" page,
    verifying dataframe content and plot presence.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("4. AI Investment & EBITDA Impact").run()

    # Verify dataframe for AIE and Attributed EBITDA impact
    assert at.dataframe[0].exists, "AI Investment & EBITDA Impact dataframe not found"
    df_str = at.dataframe[0].to_string()
    assert "AI_Investment_Efficiency" in df_str, "AI_Investment_Efficiency column not in dataframe"
    assert "Attributed_EBITDA_Impact_Pct" in df_str, "Attributed_EBITDA_Impact_Pct column not in dataframe"
    assert "Attributed_EBITDA_Impact_Absolute" in df_str, "Attributed_EBITDA_Impact_Absolute column not in dataframe"

    # Verify plot is present
    assert len(at.pyplot) == 1, f"Expected 1 plot, but found {len(at.pyplot)}"


def test_tracking_progress_over_time_page_interactions():
    """
    Tests the functionality of the "Tracking Progress Over Time" page,
    including multiselect interaction and plot presence.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("5. Tracking Progress Over Time").run()

    # Verify initial multiselect values (should have default selected companies)
    assert at.multiselect[0].value is not None, "Multiselect for companies to track is empty"
    assert len(at.multiselect[0].value) > 0, "No default companies selected in multiselect"

    # Verify initial plots are present (2 plots: Org-AI-R trajectory and AIE trajectory)
    assert len(at.pyplot) == 2, f"Expected 2 plots, but found {len(at.pyplot)}"

    # Test changing selected companies in multiselect
    all_companies = at.session_state["portfolio_df"]['CompanyName'].unique().tolist()
    if len(all_companies) >= 2: # Ensure enough companies to select two
        # Select two specific companies
        selected_companies = [all_companies[0], all_companies[1]]
        at.multiselect[0].set_value(selected_companies).run()
        assert at.multiselect[0].value == selected_companies, "Multiselect value did not update correctly"
        assert len(at.pyplot) == 2, "Plots disappeared or changed count after changing multiselect"

    # Test selecting no companies (should show info message and no plots)
    at.multiselect[0].set_value([]).run()
    assert at.info[0].value == "Please select companies to track their progress.", "Info message for no companies not displayed"
    assert not at.pyplot, "Plots unexpectedly displayed when no companies are selected"


def test_actionable_insights_page_interactions():
    """
    Tests the functionality of the "Actionable Insights: CoE & Review" page,
    including slider adjustments, re-evaluation, and dataframe/plot presence.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("6. Actionable Insights: CoE & Review").run()

    # Verify initial slider values
    assert at.slider[0].value == 75, "Default CoE Org-AI-R threshold incorrect"
    assert at.slider[1].value == 3.0, "Default CoE EBITDA threshold incorrect"
    assert at.slider[2].value == 50, "Default Review Org-AI-R threshold incorrect"
    assert at.slider[3].value == 1.0, "Default Review EBITDA threshold incorrect"

    # Verify CoE and Companies for Review dataframes are present (initially calculated)
    assert at.dataframe[0].exists, "Centers of Excellence dataframe not found"
    assert at.dataframe[1].exists, "Companies for Review dataframe not found"
    
    # Verify plot is present
    assert len(at.pyplot) == 1, f"Expected 1 plot, but found {len(at.pyplot)}"

    # Change threshold sliders
    new_coe_org_ai_r = 80
    new_review_ebitda = 2.0
    at.slider[0].set_value(new_coe_org_ai_r).run() # Increase CoE Org-AI-R threshold
    at.slider[3].set_value(new_review_ebitda).run() # Increase Review EBITDA threshold
    
    assert at.slider[0].value == new_coe_org_ai_r, "CoE Org-AI-R threshold slider did not update"
    assert at.slider[3].value == new_review_ebitda, "Review EBITDA threshold slider did not update"

    # Click re-evaluate button
    at.button[0].click().run()
    assert at.success[0].value == "Actionable insights re-evaluated with new thresholds.", "Re-evaluation success message not displayed"

    # Verify dataframes still exist, their content should have changed based on new thresholds
    assert at.dataframe[0].exists, "Centers of Excellence dataframe missing after re-evaluation"
    assert at.dataframe[1].exists, "Companies for Review dataframe missing after re-evaluation"
    assert len(at.pyplot) == 1, "Plot disappeared after re-evaluation"


def test_exit_readiness_valuation_page_interactions():
    """
    Tests the functionality of the "Exit-Readiness & Valuation" page,
    including slider adjustments, recalculation, and dataframe/plot presence.
    """
    at = get_app_with_initial_data()
    at.sidebar.radio[0].set_value("7. Exit-Readiness & Valuation").run()

    # Verify initial slider values
    assert at.slider[0].value == 0.35, "Default w1 slider value incorrect"
    assert at.slider[1].value == 0.40, "Default w2 slider value incorrect"
    assert at.slider[2].value == 0.25, "Default w3 slider value incorrect"

    # Get initial Exit-AI-R score and projected multiple for a reference company
    initial_df = at.session_state["portfolio_df"]
    company_name_ref = initial_df["CompanyName"].iloc[0]
    initial_exit_ai_r = initial_df[initial_df["CompanyName"] == company_name_ref]["Exit_AI_R_Score"].iloc[-1]
    initial_projected_multiple = initial_df[initial_df["CompanyName"] == company_name_ref]["Projected_Exit_Multiple"].iloc[-1]

    # Change slider values to new valid weights summing close to 1
    new_w1 = 0.40
    new_w2 = 0.30
    new_w3 = 0.30 # Total 1.00
    at.slider[0].set_value(new_w1).run()
    at.slider[1].set_value(new_w2).run()
    at.slider[2].set_value(new_w3).run()
    
    assert at.slider[0].value == new_w1, "w1 slider value did not update"
    assert at.slider[1].value == new_w2, "w2 slider value did not update"
    assert at.slider[2].value == new_w3, "w3 slider value did not update"

    # Click recalculate button
    at.button[0].click().run()
    assert at.success[0].value == "Exit-Readiness scores and projected valuations updated successfully!", "Recalculation success message not displayed"

    # Verify Exit-AI-R score and Projected_Exit_Multiple have changed for the reference company
    updated_df = at.session_state["portfolio_df"]
    updated_exit_ai_r = updated_df[updated_df["CompanyName"] == company_name_ref]["Exit_AI_R_Score"].iloc[-1]
    updated_projected_multiple = updated_df[updated_df["CompanyName"] == company_name_ref]["Projected_Exit_Multiple"].iloc[-1]
    
    assert updated_exit_ai_r != initial_exit_ai_r, "Exit-AI-R score did not change after recalculation"
    assert updated_projected_multiple != initial_projected_multiple, "Projected Exit Multiple did not change after recalculation"

    # Verify dataframe display and content
    assert at.dataframe[0].exists, "Exit-readiness dataframe not found"
    df_str = at.dataframe[0].to_string()
    assert "Exit_AI_R_Score" in df_str, "Exit_AI_R_Score column not in dataframe"
    assert "Projected_Exit_Multiple" in df_str, "Projected_Exit_Multiple column not in dataframe"
    
    # Verify plot is present
    assert len(at.pyplot) == 1, f"Expected 1 plot, but found {len(at.pyplot)}"

    # Test the warning message for weights not summing to 1 (optional)
    at.slider[0].set_value(0.50).run() # Make sum > 1
    at.slider[1].set_value(0.50).run()
    at.slider[2].set_value(0.50).run()
    at.button[0].click().run() # Trigger recalculation which includes the warning logic
    assert at.warning[0].exists, "Warning message for weights sum not displayed"
    assert "sum of weights (w1+w2+w3) is 1.50" in at.warning[0].value, "Warning message content incorrect"

