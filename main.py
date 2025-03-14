import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
import io
import base64
from datetime import datetime
import os
import tempfile
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import time

# Register Montserrat font
try:
    pdfmetrics.registerFont(TTFont('Montserrat', 'Montserrat-Regular.ttf'))
    pdfmetrics.registerFont(TTFont('Montserrat-Bold', 'Montserrat-Bold.ttf'))
except:
    st.warning("Montserrat font not found. Using default fonts.")

# Set page config
st.set_page_config(
    page_title="Lindblom Coaching - Lactate Threshold Analysis",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

.main {
    background-color: #FFFFFF;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
    color: #E6754E;
}

.stButton>button {
    background-color: #E6754E;
    color: white;
    font-family: 'Montserrat', sans-serif;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}

.stButton>button:hover {
    background-color: #c45d3a;
}

.highlight {
    color: #E6754E;
    font-weight: 600;
}

.result-box {
    background-color: #f8f8f8;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #E6754E;
}

footer {
    font-family: 'Montserrat', sans-serif;
    font-size: 12px;
    color: #888888;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# Load logo
logo_path = "logo.png"  # Update this with your logo path

# Sidebar
with st.sidebar:
    try:
        st.image(logo_path, width=200)
    except:
        st.title("Lindblom Coaching")
    
    st.header("Lactate Threshold Analysis")
    
    # Test type selector
    test_type = st.radio("Select test type:", ["Cycling", "Running", "Swimming"])
    
    # Athlete info
    st.subheader("Athlete Information")
    athlete_name = st.text_input("Name")
    birth_date = st.date_input("Birth date")
    height = st.number_input("Height (cm)", value=180)
    weight = st.number_input("Weight (kg)", value=75.0)
    
    # Test date
    test_date = st.date_input("Test date")
    
    # Sport specific preferences
    st.subheader("Test Parameters")
    
    if test_type == "Cycling":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=0.8, format="%.1f")
        
    elif test_type == "Running":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=1.4, format="%.1f")
        treadmill_incline = st.number_input("Treadmill incline (%)", value=0.0, format="%.1f")
        
    elif test_type == "Swimming":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=1.4, format="%.1f")

    # Threshold calculation methods
    st.subheader("Threshold Methods")
    threshold_methods = st.multiselect(
        "Select methods to calculate:",
        [
            "Fixed 2.0 mmol/L", 
            "Fixed 4.0 mmol/L", 
            "OBLA (Onset of Blood Lactate Accumulation)",
            "LT (Lactate Threshold)",
            "IAT (Individual Anaerobic Threshold)",
            "Modified Dmax",
            "Log-Log",
            "Log-Exp-ModDmax",
            "Exponential Dmax"
        ],
        default=["IAT (Individual Anaerobic Threshold)", "Modified Dmax"]
    )

# Main content
st.title("Lactate Threshold Analysis Tool")

st.header(f"{test_type} Protocol Setup")

# Protocol setup tabs
tab1, tab2, tab3 = st.tabs(["Protocol Setup", "Data Input", "Results"])

with tab1:
    st.write(f"Indicate the sport that you would like to analyze lactate thresholds on: **{test_type}**")
    
    # Include heart rate data
    include_hr = st.toggle("Include heart rate data?", value=True)
    
    # Number of steps
    col1, col2 = st.columns([1, 4])
    with col1:
        num_steps = st.number_input("Indicate how many steps were done (including rest)", min_value=3, max_value=20, value=8, step=1)
    
    # Step length
    col1, col2 = st.columns([1, 4])
    with col1:
        step_length = st.number_input("Indicate the length (in minutes) of each step", min_value=1, max_value=10, value=4, step=1)
    
    # Starting load parameters based on sport type
    if test_type == "Cycling":
        col1, col2 = st.columns([1, 4])
        with col1:
            starting_load = st.number_input("Indicate the starting load (Watts)", min_value=0, max_value=500, value=100, step=10)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            load_increment = st.number_input("Indicate the step increase (Watts)", min_value=5, max_value=100, value=25, step=5)
            
        st.info("For cycling, input the intensity in watts.")
        
    elif test_type == "Running":
        col1, col2 = st.columns([1, 4])
        with col1:
            starting_speed = st.number_input("Indicate the starting speed (km/h)", min_value=5.0, max_value=20.0, value=8.0, step=0.5, format="%.1f")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            speed_increment = st.number_input("Indicate the step increase (km/h)", min_value=0.5, max_value=5.0, value=0.5, step=0.5, format="%.1f")
            
        st.info("For running, input the intensity in km/h.")
        
    elif test_type == "Swimming":
        col1, col2 = st.columns([1, 4])
        with col1:
            starting_speed = st.number_input("Indicate the starting speed (m/s)", min_value=0.5, max_value=2.0, value=0.8, step=0.1, format="%.1f")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            speed_increment = st.number_input("Indicate the step increase (m/s)", min_value=0.05, max_value=0.5, value=0.1, step=0.05, format="%.2f")
            
        st.info("For swimming, input the intensity in m/s.")
    
    # Last step completion
    last_step_completed = st.toggle("Was the last step fully completed?", value=False)
    
    if not last_step_completed:
        col1, col2 = st.columns([1, 4])
        with col1:
            last_step_time = st.text_input("Then, indicate how long it was (in the mm:ss format)", value="02:00")

    # Curve fitting method
    st.subheader("Choose the default fitting method")
    fitting_method = st.radio(
        "",
        ["3rd degree polynomial", "4th degree polynomial", "B-spline"],
        horizontal=True
    )
    
    # Include baseline value
    include_baseline = st.toggle("Include baseline value in the curve fitting?", value=False)
    
    # Log-log method settings
    log_log_portion = st.slider(
        "Would you like to use only a portion of the data to fit the Log-log method? Default to 0.75 (meaning to use the first 75% of the data). You may choose a value between 0.25 and 1.",
        min_value=0.25,
        max_value=1.0,
        value=0.75,
        step=0.05
    )
    
    # Generate protocol button
    generate_btn = st.button("Generate Test Protocol", type="primary")
    
    if generate_btn:
        st.success("Protocol generated! Please proceed to the Data Input tab.")
        
        # Create empty dataframe for data input
        if test_type == "Cycling":
            df_template = pd.DataFrame({
                "step": range(num_steps),
                "load_watts": [starting_load + i * load_increment for i in range(num_steps)],
                "length": [step_length] * num_steps,
                "heart_rate": [None] * num_steps,
                "lactate": [None] * num_steps,
                "rpe": [None] * num_steps
            })
            
            # Set rest values
            df_template.loc[0, "load_watts"] = 0
            df_template.loc[0, "heart_rate"] = rest_hr
            df_template.loc[0, "lactate"] = rest_lactate
            
        elif test_type == "Running":
            df_template = pd.DataFrame({
                "step": range(num_steps),
                "speed_kmh": [starting_speed + i * speed_increment for i in range(num_steps)],
                "length": [step_length] * num_steps,
                "heart_rate": [None] * num_steps,
                "lactate": [None] * num_steps,
                "rpe": [None] * num_steps
            })
            
            # Set rest values
            df_template.loc[0, "speed_kmh"] = 0
            df_template.loc[0, "heart_rate"] = rest_hr
            df_template.loc[0, "lactate"] = rest_lactate
            
        elif test_type == "Swimming":
            df_template = pd.DataFrame({
                "step": range(num_steps),
                "speed_ms": [starting_speed + i * speed_increment for i in range(num_steps)],
                "length": [step_length] * num_steps,
                "heart_rate": [None] * num_steps,
                "lactate": [None] * num_steps,
                "rpe": [None] * num_steps
            })
            
            # Set rest values
            df_template.loc[0, "speed_ms"] = 0
            df_template.loc[0, "heart_rate"] = rest_hr
            df_template.loc[0, "lactate"] = rest_lactate
        
        # Store in session state
        st.session_state['df_template'] = df_template

with tab2:
    if 'df_template' not in st.session_state:
        st.info("Please set up the protocol in the Protocol Setup tab first.")
    else:
        st.subheader("Data Input")
        
        df = st.session_state['df_template'].copy()
        
        # Create input fields for each row in the dataframe
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            disabled=["step"],
            column_config={
                "step": st.column_config.NumberColumn("Step", help="Step number"),
                "load_watts": st.column_config.NumberColumn("Watts", help="Power in watts") if test_type == "Cycling" else None,
                "speed_kmh": st.column_config.NumberColumn("Speed (km/h)", help="Running speed in km/h") if test_type == "Running" else None,
                "speed_ms": st.column_config.NumberColumn("Speed (m/s)", help="Swimming speed in m/s") if test_type == "Swimming" else None,
                "length": st.column_config.NumberColumn("Length (min)", help="Duration of step in minutes"),
                "heart_rate": st.column_config.NumberColumn("Heart Rate", help="Heart rate in bpm"),
                "lactate": st.column_config.NumberColumn("Lactate", help="Blood lactate in mmol/L", format="%.2f"),
                "rpe": st.column_config.NumberColumn("RPE (6-20)", help="Rating of perceived exertion (6-20 scale)", min_value=6, max_value=20)
            }
        )
        
        st.session_state['edited_df'] = edited_df
        
        # Calculate button
        calculate_btn = st.button("Calculate Thresholds", type="primary")
        
        if calculate_btn:
            if st.session_state['edited_df']['lactate'].isna().any():
                st.error("Please fill in all lactate values before calculating thresholds.")
            else:
                # Process data and calculate thresholds
                df = st.session_state['edited_df'].copy()
                
                # Remove rest values (step 0) for calculations
                df_calc = df[df['step'] > 0].copy()
                
                # Add dictionary to session state to store results
                st.session_state['threshold_results'] = {}
                
                # Store intensity column name based on test type
                if test_type == "Cycling":
                    intensity_col = "load_watts"
                    intensity_label = "Power (Watts)"
                elif test_type == "Running":
                    intensity_col = "speed_kmh"
                    intensity_label = "Speed (km/h)"
                else:  # Swimming
                    intensity_col = "speed_ms"
                    intensity_label = "Speed (m/s)"
                
                st.session_state['intensity_col'] = intensity_col
                st.session_state['intensity_label'] = intensity_label
                
                # Calculate thresholds based on selected methods
                if "Fixed 2.0 mmol/L" in threshold_methods:
                    # Linear interpolation for fixed 2.0 mmol/L
                    try:
                        f = interp1d(df_calc['lactate'], df_calc[intensity_col], bounds_error=False, fill_value="extrapolate")
                        threshold_2mmol = float(f(2.0))
                        
                        # Get heart rate at threshold
                        f_hr = interp1d(df_calc[intensity_col], df_calc['heart_rate'], bounds_error=False, fill_value="extrapolate")
                        hr_2mmol = int(f_hr(threshold_2mmol))
                        
                        st.session_state['threshold_results']['Fixed 2.0 mmol/L'] = {
                            'value': threshold_2mmol,
                            'hr': hr_2mmol,
                            'lactate': 2.0,
                            'method': 'Fixed 2.0 mmol/L'
                        }
                    except:
                        st.warning("Could not calculate Fixed 2.0 mmol/L threshold. Check your data.")
                
                if "Fixed 4.0 mmol/L" in threshold_methods:
                    # Linear interpolation for fixed 4.0 mmol/L
                    try:
                        f = interp1d(df_calc['lactate'], df_calc[intensity_col], bounds_error=False, fill_value="extrapolate")
                        threshold_4mmol = float(f(4.0))
                        
                        # Get heart rate at threshold
                        f_hr = interp1d(df_calc[intensity_col], df_calc['heart_rate'], bounds_error=False, fill_value="extrapolate")
                        hr_4mmol = int(f_hr(threshold_4mmol))
                        
                        st.session_state['threshold_results']['Fixed 4.0 mmol/L'] = {
                            'value': threshold_4mmol,
                            'hr': hr_4mmol,
                            'lactate': 4.0,
                            'method': 'Fixed 4.0 mmol/L'
                        }
                    except:
                        st.warning("Could not calculate Fixed 4.0 mmol/L threshold. Check your data.")
                
                if "OBLA (Onset of Blood Lactate Accumulation)" in threshold_methods:
                    # OBLA is typically 4.0 mmol/L
                    if "Fixed 4.0 mmol/L" in st.session_state['threshold_results']:
                        st.session_state['threshold_results']['OBLA'] = {
                            'value': st.session_state['threshold_results']['Fixed 4.0 mmol/L']['value'],
                            'hr': st.session_state['threshold_results']['Fixed 4.0 mmol/L']['hr'],
                            'lactate': 4.0,
                            'method': 'OBLA'
                        }
                
                if "LT (Lactate Threshold)" in threshold_methods:
                    try:
                        # LT is the first significant rise in lactate
                        # Typically defined as 1.0 mmol/L above baseline
                        baseline = df.loc[0, 'lactate']
                        target = baseline + 1.0
                        
                        f = interp1d(df_calc['lactate'], df_calc[intensity_col], bounds_error=False, fill_value="extrapolate")
                        threshold_lt = float(f(target))
                        
                        # Get heart rate at threshold
                        f_hr = interp1d(df_calc[intensity_col], df_calc['heart_rate'], bounds_error=False, fill_value="extrapolate")
                        hr_lt = int(f_hr(threshold_lt))
                        
                        st.session_state['threshold_results']['LT'] = {
                            'value': threshold_lt,
                            'hr': hr_lt,
                            'lactate': target,
                            'method': 'LT'
                        }
                    except:
                        st.warning("Could not calculate LT. Check your data.")
                
                if "IAT (Individual Anaerobic Threshold)" in threshold_methods:
                    try:
                        # IAT is typically LT + 1.5 mmol/L or baseline + 0.5 mmol/L
                        baseline = df.loc[0, 'lactate']
                        target = baseline + 0.5
                        
                        f = interp1d(df_calc['lactate'], df_calc[intensity_col], bounds_error=False, fill_value="extrapolate")
                        threshold_iat = float(f(target))
                        
                        # Get heart rate at threshold
                        f_hr = interp1d(df_calc[intensity_col], df_calc['heart_rate'], bounds_error=False, fill_value="extrapolate")
                        hr_iat = int(f_hr(threshold_iat))
                        
                        st.session_state['threshold_results']['IAT'] = {
                            'value': threshold_iat,
                            'hr': hr_iat,
                            'lactate': target,
                            'method': 'IAT'
                        }
                    except:
                        st.warning("Could not calculate IAT. Check your data.")
                
                if "Modified Dmax" in threshold_methods:
                    try:
                        # Modified Dmax method
                        # Find point with maximum perpendicular distance from the line
                        # connecting the first point with lactate > baseline + 0.4 mmol/L and the last point
                        
                        # Store intensity and lactate values
                        x = df_calc[intensity_col].values
                        y = df_calc['lactate'].values
                        
                        # Find first point with lactate > baseline + 0.4 mmol/L
                        baseline = df.loc[0, 'lactate']
                        threshold = baseline + 0.4
                        first_idx = np.where(y > threshold)[0][0]
                        
                        # Get coordinates of the first and last point
                        x1, y1 = x[first_idx], y[first_idx]
                        x2, y2 = x[-1], y[-1]
                        
                        # Calculate slope and intercept of the line
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        
                        # Calculate perpendicular distance for each point
                        distances = []
                        for i in range(len(x)):
                            # Distance from point to line formula: |Ax + By + C|/sqrt(A^2 + B^2)
                            # Line equation: y = slope*x + intercept or -slope*x + y - intercept = 0
                            A = -slope
                            B = 1
                            C = -intercept
                            distance = abs(A*x[i] + B*y[i] + C) / np.sqrt(A**2 + B**2)
                            distances.append(distance)
                        
                        # Find the point with maximum distance
                        max_idx = np.argmax(distances)
                        threshold_mdmax = x[max_idx]
                        
                        # Get heart rate and lactate at threshold
                        hr_mdmax = df_calc.iloc[max_idx]['heart_rate']
                        lactate_mdmax = df_calc.iloc[max_idx]['lactate']
                        
                        st.session_state['threshold_results']['Modified Dmax'] = {
                            'value': threshold_mdmax,
                            'hr': hr_mdmax,
                            'lactate': lactate_mdmax,
                            'method': 'Modified Dmax',
                            'first_idx': first_idx,  # Store for plotting
                            'max_idx': max_idx       # Store for plotting
                        }
                    except:
                        st.warning("Could not calculate Modified Dmax threshold. Check your data.")
                
                if "Log-Log" in threshold_methods:
                    try:
                        # Log-log method
                        # Take log of both intensity and lactate
                        # Fit a line to the log-log data
                        # Find the point where the slope changes dramatically
                        
                        # Store intensity and lactate values (skipping zero values)
                        mask = (df_calc[intensity_col] > 0) & (df_calc['lactate'] > 0)
                        x = df_calc.loc[mask, intensity_col].values
                        y = df_calc.loc[mask, 'lactate'].values
                        
                        # Take log of both
                        log_x = np.log(x)
                        log_y = np.log(y)
                        
                        # Use only a portion of the data for fitting as specified by user
                        cutoff_idx = int(len(log_x) * log_log_portion)
                        
                        # Fit polynomial to the log-log data
                        if fitting_method == "3rd degree polynomial":
                            poly_degree = 3
                        elif fitting_method == "4th degree polynomial":
                            poly_degree = 4
                        else:  # B-spline
                            poly_degree = 3  # Default to 3 for B-spline
                        
                        poly = np.polyfit(log_x[:cutoff_idx], log_y[:cutoff_idx], poly_degree)
                        
                        # Create a function from the polynomial
                        p = np.poly1d(poly)
                        
                        # Calculate the derivative
                        dp = np.polyder(p)
                        
                        # Find the log(intensity) where the derivative is 1.0
                        # This is often used as a threshold indicator in log-log plots
                        from scipy.optimize import fsolve
                        
                        def func(x):
                            return dp(x) - 1.0
                        
                        # Find the root (where derivative equals 1.0)
                        log_threshold = fsolve(func, log_x[len(log_x)//2])[0]
                        
                        # Convert back from log space
                        threshold_loglog = np.exp(log_threshold)
                        
                        # Get heart rate at threshold
                        f_hr = interp1d(df_calc[intensity_col], df_calc['heart_rate'], bounds_error=False, fill_value="extrapolate")
                        hr_loglog = int(f_hr(threshold_loglog))
                        
                        # Get lactate at threshold
                        f_lac = interp1d(df_calc[intensity_col], df_calc['lactate'], bounds_error=False, fill_value="extrapolate")
                        lactate_loglog = float(f_lac(threshold_loglog))
                        
                        st.session_state['threshold_results']['Log-Log'] = {
                            'value': threshold_loglog,
                            'hr': hr_loglog,
                            'lactate': lactate_loglog,
                            'method': 'Log-Log'
                        }
                    except:
                        st.warning("Could not calculate Log-Log threshold. Check your data.")
                
                # Add more methods as needed
                
                # Generate plots
                generate_plots(df, test_type)
                
                # Navigate to Results tab
                st.success("Thresholds calculated successfully! See results in the Results tab.")

def add_training_zones(df_plot, test_type):
    """Generate training zones visualization based on calculated thresholds"""
    
    intensity_col = st.session_state['intensity_col']
    intensity_label = st.session_state['intensity_label']
    
    # Determine training zones based on test type and threshold calculations
    zones = []
    
    # Set default zone colors
    zone_colors = {
        'Z1': 'rgba(173, 216, 230, 0.3)',  # Light blue
        'Z2': 'rgba(255, 255, 150, 0.3)',  # Light yellow
        'Z3': 'rgba(255, 200, 150, 0.3)',  # Light orange
        'Z4': 'rgba(255, 150, 150, 0.3)',  # Light red
        'Z5': 'rgba(200, 120, 120, 0.3)'   # Darker red
    }
    
    # Determine key threshold for zone calculation
    if 'IAT' in st.session_state['threshold_results']:
        # Use IAT as primary threshold
        iat_value = st.session_state['threshold_results']['IAT']['value']
        hr_iat = st.session_state['threshold_results']['IAT']['hr']
        
        if test_type == "Cycling":
            # Z1: Recovery - below 70% of IAT
            # Z2: Endurance - 70-95% of IAT
            # Z3: Tempo/Sweetspot - 95-105% of IAT
            # Z4: Threshold - 105-120% of IAT
            # Z5: VO2Max - above 120% of IAT
            
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.7 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 25, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.7 * iat_value, 'max': 0.95 * iat_value, 'hr_min': hr_iat - 25, 'hr_max': hr_iat - 10, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.95 * iat_value, 'max': 1.05 * iat_value, 'hr_min': hr_iat - 10, 'hr_max': hr_iat + 5, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.05 * iat_value, 'max': 1.2 * iat_value, 'hr_min': hr_iat + 5, 'hr_max': hr_iat + 15, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.2 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 15, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
        
        elif test_type == "Running":
            # Similar zones but adapted for running
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.75 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 20, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.75 * iat_value, 'max': 0.95 * iat_value, 'hr_min': hr_iat - 20, 'hr_max': hr_iat - 5, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.95 * iat_value, 'max': 1.05 * iat_value, 'hr_min': hr_iat - 5, 'hr_max': hr_iat + 8, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.05 * iat_value, 'max': 1.15 * iat_value, 'hr_min': hr_iat + 8, 'hr_max': hr_iat + 18, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.15 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 18, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
        
        elif test_type == "Swimming":
            # Similar zones but adapted for swimming
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.7 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 20, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.7 * iat_value, 'max': 0.9 * iat_value, 'hr_min': hr_iat - 20, 'hr_max': hr_iat - 5, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.9 * iat_value, 'max': 1.03 * iat_value, 'hr_min': hr_iat - 5, 'hr_max': hr_iat + 8, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.03 * iat_value, 'max': 1.12 * iat_value, 'hr_min': hr_iat + 8, 'hr_max': hr_iat + 18, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.12 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 18, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
    
    elif 'Modified Dmax' in st.session_state['threshold_results']:
        # Use Modified Dmax as primary threshold
        mdmax_value = st.session_state['threshold_results']['Modified Dmax']['value']
        hr_mdmax = st.session_state['threshold_results']['Modified Dmax']['hr']
        
        # Similar zone calculations as above but based on modified Dmax
        if test_type == "Cycling":
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.65 * mdmax_value, 'hr_min': 0, 'hr_max': hr_mdmax - 25, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.65 * mdmax_value, 'max': 0.9 * mdmax_value, 'hr_min': hr_mdmax - 25, 'hr_max': hr_mdmax - 10, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.9 * mdmax_value, 'max': 1.0 * mdmax_value, 'hr_min': hr_mdmax - 10, 'hr_max': hr_mdmax, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.0 * mdmax_value, 'max': 1.1 * mdmax_value, 'hr_min': hr_mdmax, 'hr_max': hr_mdmax + 10, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.1 * mdmax_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_mdmax + 10, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
        else:
            # Similar adjustments for running and swimming
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.7 * mdmax_value, 'hr_min': 0, 'hr_max': hr_mdmax - 20, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.7 * mdmax_value, 'max': 0.9 * mdmax_value, 'hr_min': hr_mdmax - 20, 'hr_max': hr_mdmax - 5, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.9 * mdmax_value, 'max': 1.0 * mdmax_value, 'hr_min': hr_mdmax - 5, 'hr_max': hr_mdmax, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.0 * mdmax_value, 'max': 1.1 * mdmax_value, 'hr_min': hr_mdmax, 'hr_max': hr_mdmax + 10, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.1 * mdmax_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_mdmax + 10, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
    
    else:
        # Fallback if no primary thresholds are available - use 4.0 mmol/L
        if 'Fixed 4.0 mmol/L' in st.session_state['threshold_results']:
            threshold_value = st.session_state['threshold_results']['Fixed 4.0 mmol/L']['value']
            hr_threshold = st.session_state['threshold_results']['Fixed 4.0 mmol/L']['hr']
            
            zones = [
                {'name': 'Z1', 'min': 0, 'max': 0.6 * threshold_value, 'hr_min': 0, 'hr_max': hr_threshold - 30, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.6 * threshold_value, 'max': 0.85 * threshold_value, 'hr_min': hr_threshold - 30, 'hr_max': hr_threshold - 15, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.85 * threshold_value, 'max': 0.95 * threshold_value, 'hr_min': hr_threshold - 15, 'hr_max': hr_threshold - 5, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 0.95 * threshold_value, 'max': 1.05 * threshold_value, 'hr_min': hr_threshold - 5, 'hr_max': hr_threshold + 5, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.05 * threshold_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_threshold + 5, 'hr_max': 220, 'color': zone_colors['Z5']}
            ]
    
    # Create training zones visualization
    fig_zones = go.Figure()
    
    # Add rectangles for each zone
    for zone in zones:
        fig_zones.add_shape(
            type="rect",
            x0=zone['min'],
            x1=zone['max'],
            y0=0,
            y1=max(df_plot['lactate']) * 1.1,
            fillcolor=zone['color'],
            line=dict(width=0),
            name=zone['name']
        )
    
    # Add lactate curve
    fig_zones.add_trace(go.Scatter(
        x=df_plot[intensity_col],
        y=df_plot['lactate'],
        mode='lines+markers',
        name='Lactate',
        line=dict(color='#E6754E', width=3),
        marker=dict(size=8)
    ))
    
    # Add heart rate if available
    if 'heart_rate' in df_plot.columns and not df_plot['heart_rate'].isna().all():
        # Normalize heart rate to lactate scale for visualization
        lactate_max = max(df_plot['lactate']) * 1.1
        hr_max = max(df_plot['heart_rate']) * 1.1
        normalized_hr = df_plot['heart_rate'] * (lactate_max / hr_max)
        
        fig_zones.add_trace(go.Scatter(
            x=df_plot[intensity_col],
            y=normalized_hr,
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#4EA1E6', width=3),
            marker=dict(size=8)
        ))
    
    # Add thresholds
    for method, result in st.session_state['threshold_results'].items():
        fig_zones.add_vline(
            x=result['value'],
            line_dash="dash",
            line_color="black",
            annotation_text=f"{method}: {result['value']:.1f}",
            annotation_position="top"
        )
    
    # Add zone labels
    for zone in zones:
        fig_zones.add_annotation(
            x=(zone['min'] + zone['max']) / 2,
            y=max(df_plot['lactate']) * 0.5,
            text=zone['name'],
            showarrow=False,
            font=dict(
                size=14,
                color="black"
            )
        )
    
    # Update layout
    fig_zones.update_layout(
        title='Training Zones',
        xaxis_title=intensity_label,
        yaxis_title='Lactate (mmol/L)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig_zones

def generate_plots(df, test_type):
    """Generate the lactate curve plots and store in session state"""
    
    # Remove rest values (step 0) for plotting
    df_plot = df[df['step'] > 0].copy()
    
    # Get intensity column name
    intensity_col = st.session_state['intensity_col']
    intensity_label = st.session_state['intensity_label']
    
    # Create matplotlib figure for PDF export
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lactate curve
    ax.plot(df_plot[intensity_col], df_plot['lactate'], 'o-', color='#E6754E', label='Lactate')
    
    # Plot heart rate if available
    if 'heart_rate' in df_plot.columns and not df_plot['heart_rate'].isna().all():
        ax2 = ax.twinx()
        ax2.plot(df_plot[intensity_col], df_plot['heart_rate'], 's-', color='#4EA1E6', label='Heart Rate')
        ax2.set_ylabel('Heart Rate (bpm)', color='#4EA1E6')
        ax2.tick_params(axis='y', labelcolor='#4EA1E6')
    
    # Set labels and title
    ax.set_xlabel(intensity_label)
    ax.set_ylabel('Lactate (mmol/L)', color='#E6754E')
    ax.tick_params(axis='y', labelcolor='#E6754E')
    ax.set_title('Lactate-to-performance-curve')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot thresholds
    for method, result in st.session_state['threshold_results'].items():
        if method == "Modified Dmax":
            # Get the indices for plotting
            first_idx = result['first_idx']
            max_idx = result['max_idx']
            
            # Plot the line connecting first and last point
            x_vals = [df_plot.iloc[first_idx][intensity_col], df_plot.iloc[-1][intensity_col]]
            y_vals = [df_plot.iloc[first_idx]['lactate'], df_plot.iloc[-1]['lactate']]
            ax.plot(x_vals, y_vals, '--', color='black')
            
            # Highlight the threshold point
            ax.plot(result['value'], result['lactate'], 'o', markersize=10, 
                   markerfacecolor='none', markeredgecolor='red', markeredgewidth=2)
            
            # Annotate the threshold
            ax.annotate(f"ModDmax: {result['value']:.1f}", 
                       (result['value'], result['lactate']),
                       xytext=(10, -20), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='red'))
            
        else:
            # Plot other thresholds as vertical lines
            ax.axvline(x=result['value'], linestyle='--', alpha=0.7, 
                      color='green', label=f"{method}: {result['value']:.1f}")
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if 'heart_rate' in df_plot.columns and not df_plot['heart_rate'].isna().all():
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax.legend(handles, labels, loc='upper left')
    
    # Save figure for PDF export
    st.session_state['matplotlib_fig'] = fig
    
    # Create interactive Plotly figure
    fig_plotly = go.Figure()
    
    # Add lactate curve
    fig_plotly.add_trace(go.Scatter(
        x=df_plot[intensity_col],
        y=df_plot['lactate'],
        mode='lines+markers',
        name='Lactate',
        line=dict(color='#E6754E', width=3),
        marker=dict(size=8)
    ))
    
    # Add heart rate if available
    if 'heart_rate' in df_plot.columns and not df_plot['heart_rate'].isna().all():
        fig_plotly.add_trace(go.Scatter(
            x=df_plot[intensity_col],
            y=df_plot['heart_rate'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#4EA1E6', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
    
    # Add thresholds
    for method, result in st.session_state['threshold_results'].items():
        if method == "Modified Dmax":
            # Get the indices for plotting
            first_idx = result['first_idx']
            max_idx = result['max_idx']
            
            # Add line connecting first point and last point
            fig_plotly.add_trace(go.Scatter(
                x=[df_plot.iloc[first_idx][intensity_col], df_plot.iloc[-1][intensity_col]],
                y=[df_plot.iloc[first_idx]['lactate'], df_plot.iloc[-1]['lactate']],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=False
            ))
            
            # Add threshold point
            fig_plotly.add_trace(go.Scatter(
                x=[result['value']],
                y=[result['lactate']],
                mode='markers',
                marker=dict(
                    size=12, 
                    symbol='circle-open',
                    color='red',
                    line=dict(width=2)
                ),
                name=f"ModDmax: {result['value']:.1f}"
            ))
        else:
            # Add vertical line for threshold
            fig_plotly.add_vline(
                x=result['value'],
                line_dash="dash",
                line_color="green",
                annotation_text=f"{method}: {result['value']:.1f}",
                annotation_position="top right"
            )
    
    # Update layout
    fig_plotly.update_layout(
        title='Lactate-to-performance-curve',
        xaxis_title=intensity_label,
        yaxis=dict(
            title='Lactate (mmol/L)',
            titlefont=dict(color='#E6754E'),
            tickfont=dict(color='#E6754E')
        ),
        yaxis2=dict(
            title='Heart Rate (bpm)',
            titlefont=dict(color='#4EA1E6'),
            tickfont=dict(color='#4EA1E6'),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    
    # Add training zones based on threshold methods
    fig_zones = add_training_zones(df_plot, test_type)
    
    # Store plotly figures in session state
    st.session_state['plotly_fig'] = fig_plotly
    st.session_state['plotly_zones'] = fig_zones

with tab3:
    if 'threshold_results' not in st.session_state:
        st.info("Please set up the protocol in the Protocol Setup tab and calculate thresholds in the Data Input tab.")
    else:
        st.header("Results")
        
        # Create tabs for different result views
        results_tab1, results_tab2, results_tab3 = st.tabs(["Summary", "Graphs", "Training Zones"])
        
        with results_tab1:
            # Display threshold results in a table
            st.subheader("Calculated Thresholds")
            
            # Create a dataframe from the results
            results_data = []
            for method, result in st.session_state['threshold_results'].items():
                if test_type == "Cycling":
                    results_data.append({
                        'Method': method,
                        'Power (Watts)': f"{result['value']:.1f}",
                        'Power (W/kg)': f"{result['value'] / weight:.2f}",
                        'Heart Rate (bpm)': f"{result['hr']}",
                        'Lactate (mmol/L)': f"{result['lactate']:.2f}"
                    })
                elif test_type == "Running":
                    # Calculate pace (min:sec per km)
                    pace_seconds = (1000 / result['value']) * 3.6
                    pace_min = int(pace_seconds // 60)
                    pace_sec = int(pace_seconds % 60)
                    
                    results_data.append({
                        'Method': method,
                        'Speed (km/h)': f"{result['value']:.1f}",
                        'Pace (min/km)': f"{pace_min}:{pace_sec:02d}",
                        'Heart Rate (bpm)': f"{result['hr']}",
                        'Lactate (mmol/L)': f"{result['lactate']:.2f}"
                    })
                elif test_type == "Swimming":
                    # Calculate pace (min:sec per 100m)
                    pace_seconds = (100 / result['value'])
                    pace_min = int(pace_seconds // 60)
                    pace_sec = int(pace_seconds % 60)
                    
                    results_data.append({
                        'Method': method,
                        'Speed (m/s)': f"{result['value']:.2f}",
                        'Pace (min/100m)': f"{pace_min}:{pace_sec:02d}",
                        'Heart Rate (bpm)': f"{result['hr']}",
                        'Lactate (mmol/L)': f"{result['lactate']:.2f}"
                    })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Training zones based on primary threshold method
            st.subheader("Training Zones")
            
            # Determine primary threshold method for zones
            primary_method = None
            if 'IAT' in st.session_state['threshold_results']:
                primary_method = 'IAT'
            elif 'Modified Dmax' in st.session_state['threshold_results']:
                primary_method = 'Modified Dmax'
            elif 'Fixed 4.0 mmol/L' in st.session_state['threshold_results']:
                primary_method = 'Fixed 4.0 mmol/L'
            elif len(st.session_state['threshold_results']) > 0:
                primary_method = list(st.session_state['threshold_results'].keys())[0]
            
            if primary_method:
                threshold_value = st.session_state['threshold_results'][primary_method]['value']
                hr_threshold = st.session_state['threshold_results'][primary_method]['hr']
                
                # Create training zones based on test type
                zones_data = []
                
                if test_type == "Cycling":
                    zones_data = [
                        {
                            'Zone': 'Z1 - Recovery',
                            'Power Range': f"below {int(0.7 * threshold_value)} W",
                            'Relative Power': f"below {(0.7 * threshold_value / weight):.2f} W/kg",
                            'Heart Rate': f"below {int(hr_threshold - 25)} bpm"
                        },
                        {
                            'Zone': 'Z2 - Endurance',
                            'Power Range': f"{int(0.7 * threshold_value)} - {int(0.95 * threshold_value)} W",
                            'Relative Power': f"{(0.7 * threshold_value / weight):.2f} - {(0.95 * threshold_value / weight):.2f} W/kg",
                            'Heart Rate': f"{int(hr_threshold - 25)} - {int(hr_threshold - 10)} bpm"
                        },
                        {
                            'Zone': 'Z3 - Tempo/Sweetspot',
                            'Power Range': f"{int(0.95 * threshold_value)} - {int(1.05 * threshold_value)} W",
                            'Relative Power': f"{(0.95 * threshold_value / weight):.2f} - {(1.05 * threshold_value / weight):.2f} W/kg",
                            'Heart Rate': f"{int(hr_threshold - 10)} - {int(hr_threshold + 5)} bpm"
                        },
                        {
                            'Zone': 'Z4 - Threshold',
                            'Power Range': f"{int(1.05 * threshold_value)} - {int(1.2 * threshold_value)} W",
                            'Relative Power': f"{(1.05 * threshold_value / weight):.2f} - {(1.2 * threshold_value / weight):.2f} W/kg",
                            'Heart Rate': f"{int(hr_threshold + 5)} - {int(hr_threshold + 15)} bpm"
                        },
                        {
                            'Zone': 'Z5 - VO2Max',
                            'Power Range': f"above {int(1.2 * threshold_value)} W",
                            'Relative Power': f"above {(1.2 * threshold_value / weight):.2f} W/kg",
                            'Heart Rate': f"above {int(hr_threshold + 15)} bpm"
                        }
                    ]
                elif test_type == "Running":
                    # Calculate pace ranges
                    def speed_to_pace(speed_kmh):
                        pace_seconds = (1000 / speed_kmh) * 3.6
                        pace_min = int(pace_seconds // 60)
                        pace_sec = int(pace_seconds % 60)
                        return f"{pace_min}:{pace_sec:02d}"
                    
                    zones_data = [
                        {
                            'Zone': 'Z1 - Recovery',
                            'Speed Range': f"below {(0.75 * threshold_value):.1f} km/h",
                            'Pace Range': f"slower than {speed_to_pace(0.75 * threshold_value)} min/km",
                            'Heart Rate': f"below {int(hr_threshold - 20)} bpm"
                        },
                        {
                            'Zone': 'Z2 - Endurance',
                            'Speed Range': f"{(0.75 * threshold_value):.1f} - {(0.95 * threshold_value):.1f} km/h",
                            'Pace Range': f"{speed_to_pace(0.95 * threshold_value)} - {speed_to_pace(0.75 * threshold_value)} min/km",
                            'Heart Rate': f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
                        },
                        {
                            'Zone': 'Z3 - Tempo/Sweetspot',
                            'Speed Range': f"{(0.95 * threshold_value):.1f} - {(1.05 * threshold_value):.1f} km/h",
                            'Pace Range': f"{speed_to_pace(1.05 * threshold_value)} - {speed_to_pace(0.95 * threshold_value)} min/km",
                            'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 8)} bpm"
                        },
                        {
                            'Zone': 'Z4 - Threshold',
                            'Speed Range': f"{(1.05 * threshold_value):.1f} - {(1.15 * threshold_value):.1f} km/h",
                            'Pace Range': f"{speed_to_pace(1.15 * threshold_value)} - {speed_to_pace(1.05 * threshold_value)} min/km",
                            'Heart Rate': f"{int(hr_threshold + 8)} - {int(hr_threshold + 18)} bpm"
                        },
                        {
                            'Zone': 'Z5 - VO2Max',
                            'Speed Range': f"above {(1.15 * threshold_value):.1f} km/h",
                            'Pace Range': f"faster than {speed_to_pace(1.15 * threshold_value)} min/km",
                            'Heart Rate': f"above {int(hr_threshold + 18)} bpm"
                        }
                    ]
                elif test_type == "Swimming":
                    # Calculate pace ranges (min:sec per 100m)
                    def speed_to_pace_100m(speed_ms):
                        pace_seconds = 100 / speed_ms
                        pace_min = int(pace_seconds // 60)
                        pace_sec = int(pace_seconds % 60)
                        return f"{pace_min}:{pace_sec:02d}"
                    
                    zones_data = [
                        {
                            'Zone': 'Z1 - Recovery',
                            'Speed Range': f"below {(0.7 * threshold_value):.2f} m/s",
                            'Pace Range': f"slower than {speed_to_pace_100m(0.7 * threshold_value)} min/100m",
                            'Heart Rate': f"below {int(hr_threshold - 20)} bpm"
                        },
                        {
                            'Zone': 'Z2 - Endurance',
                            'Speed Range': f"{(0.7 * threshold_value):.2f} - {(0.9 * threshold_value):.2f} m/s",
                            'Pace Range': f"{speed_to_pace_100m(0.9 * threshold_value)} - {speed_to_pace_100m(0.7 * threshold_value)} min/100m",
                            'Heart Rate': f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
                        },
                        {
                            'Zone': 'Z3 - Tempo/Sweetspot',
                            'Speed Range': f"{(0.9 * threshold_value):.2f} - {(1.03 * threshold_value):.2f} m/s",
                            'Pace Range': f"{speed_to_pace_100m(1.03 * threshold_value)} - {speed_to_pace_100m(0.9 * threshold_value)} min/100m",
                            'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 8)} bpm"
                        },
                        {
                            'Zone': 'Z4 - Threshold',
                            'Speed Range': f"{(1.03 * threshold_value):.2f} - {(1.12 * threshold_value):.2f} m/s",
                            'Pace Range': f"{speed_to_pace_100m(1.12 * threshold_value)} - {speed_to_pace_100m(1.03 * threshold_value)} min/100m",
                            'Heart Rate': f"{int(hr_threshold + 8)} - {int(hr_threshold + 18)} bpm"
                        },
                        {
                            'Zone': 'Z5 - VO2Max',
                            'Speed Range': f"above {(1.12 * threshold_value):.2f} m/s",
                            'Pace Range': f"faster than {speed_to_pace_100m(1.12 * threshold_value)} min/100m",
                            'Heart Rate': f"above {int(hr_threshold + 18)} bpm"
                        }
                    ]
                
                zones_df = pd.DataFrame(zones_data)
                st.dataframe(zones_df, use_container_width=True, hide_index=True)
            
            # Export as PDF
            st.subheader("Export Results")
            
            if st.button("Generate PDF Report", type="primary"):
                pdf_bytes = generate_pdf_report(test_type, athlete_name, birth_date, test_date, weight, height)
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"lactate_threshold_report_{athlete_name}_{test_date}.pdf",
                    mime="application/pdf"
                )
        
        with results_tab2:
            st.subheader("Lactate Curve and Thresholds")
            
            if 'plotly_fig' in st.session_state:
                st.plotly_chart(st.session_state['plotly_fig'], use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div class="result-box">
                <h4>Lactate Curve Explanation</h4>
                <p>The lactate curve shows the relationship between exercise intensity and blood lactate concentration. 
                The thresholds identified on this curve represent key physiological transitions:</p>
                <ul>
                <li><strong>Lactate Threshold (LT):</strong> The intensity at which lactate begins to accumulate in the blood at a faster rate than it can be removed.</li>
                <li><strong>Individual Anaerobic Threshold (IAT):</strong> The highest intensity that can be maintained without continuous lactate accumulation.</li>
                <li><strong>Modified Dmax:</strong> The point on the curve with the maximum perpendicular distance from the line connecting the first significant lactate rise and the last data point.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with results_tab3:
            st.subheader("Training Zones")
            
            if 'plotly_zones' in st.session_state:
                st.plotly_chart(st.session_state['plotly_zones'], use_container_width=True)
                
                # Add explanation
                st.markdown("""
                <div class="result-box">
                <h4>Training Zones Explanation</h4>
                <p>The training zones are calculated based on your primary threshold value:</p>
                <ul>
                <li><strong>Zone 1 (Recovery):</strong> Very light intensity training for active recovery.</li>
                <li><strong>Zone 2 (Endurance):</strong> Builds aerobic capacity and fat metabolism. Comfortable, conversational pace.</li>
                <li><strong>Zone 3 (Tempo/Sweetspot):</strong> Moderately hard effort that improves lactate clearance and endurance.</li>
                <li><strong>Zone 4 (Threshold):</strong> Hard effort at or near your lactate threshold that improves lactate tolerance.</li>
                <li><strong>Zone 5 (VO2Max):</strong> Very hard, intense efforts that develop maximum aerobic capacity.</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

def generate_pdf_report(test_type, athlete_name, birth_date, test_date, weight, height):
    """Generate a PDF report with the test results"""
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=inch/2, leftMargin=inch/2,
                           topMargin=inch/2, bottomMargin=inch/2)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontName='Montserrat-Bold',
        fontSize=16,
        textColor=colors.HexColor('#E6754E'),
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    heading_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Montserrat-Bold',
        fontSize=14,
        textColor=colors.HexColor('#E6754E'),
        spaceAfter=10
    )
    
    sub_heading_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Montserrat-Bold',
        fontSize=12,
        textColor=colors.HexColor('#E6754E'),
        spaceAfter=8
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='Montserrat',
        fontSize=10,
        spaceAfter=6
    )
    
    # Add title
    elements.append(Paragraph(f"Lactate Threshold Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add athlete information
    elements.append(Paragraph("Athlete Information", heading_style))
    
    # Format dates
    birth_date_str = birth_date.strftime("%Y-%m-%d")
    test_date_str = test_date.strftime("%Y-%m-%d")
    
    # Create table for athlete info
    athlete_data = [
        ["Name:", athlete_name],
        ["Birth Date:", birth_date_str],
        ["Test Date:", test_date_str],
        ["Height:", f"{height} cm"],
        ["Weight:", f"{weight} kg"],
        ["BMI:", f"{weight / ((height / 100) ** 2):.1f}"]
    ]
    
    athlete_table = Table(athlete_data, colWidths=[100, 350])
    athlete_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Montserrat'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('FONT', (0, 0), (0, -1), 'Montserrat-Bold')
    ]))
    
    elements.append(athlete_table)
    elements.append(Spacer(1, 12))
    
    # Add test results
    elements.append(Paragraph("Threshold Results", heading_style))
    
    # Create header row based on test type
    if test_type == "Cycling":
        result_header = ["Method", "Power (W)", "Power (W/kg)", "Heart Rate (bpm)", "Lactate (mmol/L)"]
    elif test_type == "Running":
        result_header = ["Method", "Speed (km/h)", "Pace (min/km)", "Heart Rate (bpm)", "Lactate (mmol/L)"]
    else:  # Swimming
        result_header = ["Method", "Speed (m/s)", "Pace (min/100m)", "Heart Rate (bpm)", "Lactate (mmol/L)"]
    
    # Create data rows
    result_data = [result_header]
    
    for method, result in st.session_state['threshold_results'].items():
        if test_type == "Cycling":
            result_data.append([
                method,
                f"{result['value']:.1f}",
                f"{result['value'] / weight:.2f}",
                f"{result['hr']}",
                f"{result['lactate']:.2f}"
            ])
        elif test_type == "Running":
            # Calculate pace
            pace_seconds = (1000 / result['value']) * 3.6
            pace_min = int(pace_seconds // 60)
            pace_sec = int(pace_seconds % 60)
            
            result_data.append([
                method,
                f"{result['value']:.1f}",
                f"{pace_min}:{pace_sec:02d}",
                f"{result['hr']}",
                f"{result['lactate']:.2f}"
            ])
        else:  # Swimming
            # Calculate pace
            pace_seconds = 100 / result['value']
            pace_min = int(pace_seconds // 60)
            pace_sec = int(pace_seconds % 60)
            
            result_data.append([
                method,
                f"{result['value']:.2f}",
                f"{pace_min}:{pace_sec:02d}",
                f"{result['hr']}",
                f"{result['lactate']:.2f}"
            ])
    
    # Create results table
    col_widths = [150, 80, 80, 80, 80]
    results_table = Table(result_data, colWidths=col_widths)
    results_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Montserrat'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6754E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONT', (0, 0), (-1, 0), 'Montserrat-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    elements.append(results_table)
    elements.append(Spacer(1, 12))
    
    # Add training zones
    elements.append(Paragraph("Training Zones", heading_style))
    
    # Determine primary threshold method for zones
    primary_method = None
    if 'IAT' in st.session_state['threshold_results']:
        primary_method = 'IAT'
    elif 'Modified Dmax' in st.session_state['threshold_results']:
        primary_method = 'Modified Dmax'
    elif 'Fixed 4.0 mmol/L' in st.session_state['threshold_results']:
        primary_method = 'Fixed 4.0 mmol/L'
    elif len(st.session_state['threshold_results']) > 0:
        primary_method = list(st.session_state['threshold_results'].keys())[0]
    
    elements.append(Paragraph(f"Based on {primary_method} threshold", normal_style))
    
    if primary_method:
        threshold_value = st.session_state['threshold_results'][primary_method]['value']
        hr_threshold = st.session_state['threshold_results'][primary_method]['hr']
        
        # Create zone header based on test type
        if test_type == "Cycling":
            zone_header = ["Zone", "Power Range", "Relative Power", "Heart Rate"]
            
            zone_data = [zone_header]
            zone_data.append([
                "Z1 - Recovery",
                f"below {int(0.7 * threshold_value)} W",
                f"below {(0.7 * threshold_value / weight):.2f} W/kg",
                f"below {int(hr_threshold - 25)} bpm"
            ])
            zone_data.append([
                "Z2 - Endurance",
                f"{int(0.7 * threshold_value)} - {int(0.95 * threshold_value)} W",
                f"{(0.7 * threshold_value / weight):.2f} - {(0.95 * threshold_value / weight):.2f} W/kg",
                f"{int(hr_threshold - 25)} - {int(hr_threshold - 10)} bpm"
            ])
            zone_data.append([
                "Z3 - Tempo/Sweetspot",
                f"{int(0.95 * threshold_value)} - {int(1.05 * threshold_value)} W",
                f"{(0.95 * threshold_value / weight):.2f} - {(1.05 * threshold_value / weight):.2f} W/kg",
                f"{int(hr_threshold - 10)} - {int(hr_threshold + 5)} bpm"
            ])
            zone_data.append([
                "Z4 - Threshold",
                f"{int(1.05 * threshold_value)} - {int(1.2 * threshold_value)} W",
                f"{(1.05 * threshold_value / weight):.2f} - {(1.2 * threshold_value / weight):.2f} W/kg",
                f"{int(hr_threshold + 5)} - {int(hr_threshold + 15)} bpm"
            ])
            zone_data.append([
                "Z5 - VO2Max",
                f"above {int(1.2 * threshold_value)} W",
                f"above {(1.2 * threshold_value / weight):.2f} W/kg",
                f"above {int(hr_threshold + 15)} bpm"
            ])
        elif test_type == "Running":
            # Calculate pace ranges
            def speed_to_pace(speed_kmh):
                pace_seconds = (1000 / speed_kmh) * 3.6
                pace_min = int(pace_seconds // 60)
                pace_sec = int(pace_seconds % 60)
                return f"{pace_min}:{pace_sec:02d}"
            
            zone_header = ["Zone", "Speed Range", "Pace Range", "Heart Rate"]
            
            zone_data = [zone_header]
            zone_data.append([
                "Z1 - Recovery",
                f"below {(0.75 * threshold_value):.1f} km/h",
                f"slower than {speed_to_pace(0.75 * threshold_value)} min/km",
                f"below {int(hr_threshold - 20)} bpm"
            ])
            zone_data.append([
                "Z2 - Endurance",
                f"{(0.75 * threshold_value):.1f} - {(0.95 * threshold_value):.1f} km/h",
                f"{speed_to_pace(0.95 * threshold_value)} - {speed_to_pace(0.75 * threshold_value)} min/km",
                f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
            ])
            zone_data.append([
                "Z3 - Tempo/Sweetspot",
                f"{(0.95 * threshold_value):.1f} - {(1.05 * threshold_value):.1f} km/h",
                f"{speed_to_pace(1.05 * threshold_value)} - {speed_to_pace(0.95 * threshold_value)} min/km",
                f"{int(hr_threshold - 5)} - {int(hr_threshold + 8)} bpm"
            ])
            zone_data.append([
                "Z4 - Threshold",
                f"{(1.05 * threshold_value):.1f} - {(1.15 * threshold_value):.1f} km/h",
                f"{speed_to_pace(1.15 * threshold_value)} - {speed_to_pace(1.05 * threshold_value)} min/km",
                f"{int(hr_threshold + 8)} - {int(hr_threshold + 18)} bpm"
            ])
            zone_data.append([
                "Z5 - VO2Max",
                f"above {(1.15 * threshold_value):.1f} km/h",
                f"faster than {speed_to_pace(1.15 * threshold_value)} min/km",
                f"above {int(hr_threshold + 18)} bpm"
            ])
        else:  # Swimming
            # Calculate pace ranges
            def speed_to_pace_100m(speed_ms):
                pace_seconds = 100 / speed_ms
                pace_min = int(pace_seconds // 60)
                pace_sec = int(pace_seconds % 60)
                return f"{pace_min}:{pace_sec:02d}"
            
            zone_header = ["Zone", "Speed Range", "Pace Range", "Heart Rate"]
            
            zone_data = [zone_header]
            zone_data.append([
                "Z1 - Recovery",
                f"below {(0.7 * threshold_value):.2f} m/s",
                f"slower than {speed_to_pace_100m(0.7 * threshold_value)} min/100m",
                f"below {int(hr_threshold - 20)} bpm"
            ])
            zone_data.append([
                "Z2 - Endurance",
                f"{(0.7 * threshold_value):.2f} - {(0.9 * threshold_value):.2f} m/s",
                f"{speed_to_pace_100m(0.9 * threshold_value)} - {speed_to_pace_100m(0.7 * threshold_value)} min/100m",
                f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
            ])
            zone_data.append([
                "Z3 - Tempo/Sweetspot",
                f"{(0.9 * threshold_value):.2f} - {(1.03 * threshold_value):.2f} m/s",
                f"{speed_to_pace_100m(1.03 * threshold_value)} - {speed_to_pace_100m(0.9 * threshold_value)} min/100m",
                f"{int(hr_threshold - 5)} - {int(hr_threshold + 8)} bpm"
            ])
            zone_data.append([
                "Z4 - Threshold",
                f"{(1.03 * threshold_value):.2f} - {(1.12 * threshold_value):.2f} m/s",
                f"{speed_to_pace_100m(1.12 * threshold_value)} - {speed_to_pace_100m(1.03 * threshold_value)} min/100m",
                f"{int(hr_threshold + 8)} - {int(hr_threshold + 18)} bpm"
            ])
            zone_data.append([
                "Z5 - VO2Max",
                f"above {(1.12 * threshold_value):.2f} m/s",
                f"faster than {speed_to_pace_100m(1.12 * threshold_value)} min/100m",
                f"above {int(hr_threshold + 18)} bpm"
            ])
        
        # Create zones table
        col_widths = [120, 120, 120, 120]
        zones_table = Table(zone_data, colWidths=col_widths)
        zones_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Montserrat'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6754E')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONT', (0, 0), (-1, 0), 'Montserrat-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Color zones (background)
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightblue),
            ('BACKGROUND', (0, 2), (-1, 2), colors.lightyellow),
            ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#FFD0A0')),  # Light orange
            ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#FFB0B0')),  # Light red
            ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#FFAAAA'))   # Darker red
        ]))
        
        elements.append(zones_table)
        elements.append(Spacer(1, 12))
    
    # Add lactate curve
    elements.append(Paragraph("Lactate Curve", heading_style))
    
    # Convert matplotlib figure to PNG
    if 'matplotlib_fig' in st.session_state:
        buf = io.BytesIO()
        st.session_state['matplotlib_fig'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Add the plot to the PDF
        img = Image(buf, width=450, height=300)
        elements.append(img)
    
    # Add explanations
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Threshold Explanations", heading_style))
    elements.append(Paragraph("Lactate Threshold (LT): The intensity at which lactate begins to accumulate in the blood at a faster rate than it can be removed.", normal_style))
    elements.append(Paragraph("Individual Anaerobic Threshold (IAT): The highest intensity that can be maintained without continuous lactate accumulation.", normal_style))
    elements.append(Paragraph("Modified Dmax: The point on the curve with the maximum perpendicular distance from the line connecting the first significant lactate rise and the last data point.", normal_style))
    
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Training Zones Explanation", heading_style))
    elements.append(Paragraph("Zone 1 (Recovery): Very light intensity training for active recovery.", normal_style))
    elements.append(Paragraph("Zone 2 (Endurance): Builds aerobic capacity and fat metabolism. Comfortable, conversational pace.", normal_style))
    elements.append(Paragraph("Zone 3 (Tempo/Sweetspot): Moderately hard effort that improves lactate clearance and endurance.", normal_style))
    elements.append(Paragraph("Zone 4 (Threshold): Hard effort at or near your lactate threshold that improves lactate tolerance.", normal_style))
    elements.append(Paragraph("Zone 5 (VO2Max): Very hard, intense efforts that develop maximum aerobic capacity.", normal_style))
    
    # Add footer
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} by Lindblom Coaching Lactate Threshold Analysis Tool", 
                             ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, alignment=TA_CENTER)))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes     

if __name__ == "__main__":
    # Code to initialize app          