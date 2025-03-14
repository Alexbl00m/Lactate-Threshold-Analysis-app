import streamlit as st
import pandas as pd
from datetime import datetime, date

def setup_page():
    """Configure page settings for the Streamlit app"""
    st.set_page_config(
        page_title="Lindblom Coaching - Lactate Threshold Analysis",
        page_icon="🏃‍♂️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_custom_css():
    """Add custom CSS styling to the app"""
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

def create_sidebar():
    """
    Create sidebar with user inputs
    
    Returns:
    --------
    tuple
        (test_type, athlete_info, threshold_methods, test_parameters)
    """
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
    
    test_parameters = {}
    
    if test_type == "Cycling":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=0.8, format="%.1f")
        test_parameters = {
            'stage_duration': stage_duration,
            'rest_hr': rest_hr,
            'rest_lactate': rest_lactate
        }
        
    elif test_type == "Running":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=1.4, format="%.1f")
        treadmill_incline = st.number_input("Treadmill incline (%)", value=0.0, format="%.1f")
        test_parameters = {
            'stage_duration': stage_duration,
            'rest_hr': rest_hr,
            'rest_lactate': rest_lactate,
            'treadmill_incline': treadmill_incline
        }
        
    elif test_type == "Swimming":
        stage_duration = st.slider("Stage duration (minutes)", 3, 10, 4)
        rest_hr = st.number_input("Resting heart rate", value=60)
        rest_lactate = st.number_input("Resting lactate (mmol/L)", value=1.4, format="%.1f")
        test_parameters = {
            'stage_duration': stage_duration,
            'rest_hr': rest_hr,
            'rest_lactate': rest_lactate
        }

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
    
    # Package all athlete info into a dictionary
    athlete_info = {
        'name': athlete_name,
        'birth_date': birth_date,
        'height': height,
        'weight': weight,
        'test_date': test_date
    }
    
    return test_type, athlete_info, threshold_methods, test_parameters

def create_protocol_setup_tab(test_type, test_parameters):
    """
    Create content for protocol setup tab
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    test_parameters : dict
        Dictionary of test parameters
    
    Returns:
    --------
    dict
        Protocol settings
    """
    st.write(f"Indicate the sport that you would like to analyze lactate thresholds on: **{test_type}**")
    
    # Include heart rate data
    include_hr = st.toggle("Include heart rate data?", value=True)
    
    # Number of steps
    col1, col2 = st.columns([1, 4])
    with col1:
        num_steps = st.number_input("Indicate how many steps were done (including rest)", 
                                    min_value=3, max_value=20, value=8, step=1)
    
    # Step length
    col1, col2 = st.columns([1, 4])
    with col1:
        step_length = st.number_input("Indicate the length (in minutes) of each step", 
                                     min_value=1, max_value=10, value=4, step=1)
    
    # Sport-specific inputs
    sport_settings = {}
    
    if test_type == "Cycling":
        sport_settings = cycling_protocol_inputs()
    elif test_type == "Running":
        sport_settings = running_protocol_inputs()
    elif test_type == "Swimming":
        sport_settings = swimming_protocol_inputs()
    
    # Last step completion
    last_step_completed = st.toggle("Was the last step fully completed?", value=False)
    
    last_step_time = None
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
    
    # Combine all settings
    protocol_settings = {
        'include_hr': include_hr,
        'num_steps': num_steps,
        'step_length': step_length,
        'last_step_completed': last_step_completed,
        'last_step_time': last_step_time,
        'fitting_method': fitting_method,
        'include_baseline': include_baseline,
        'log_log_portion': log_log_portion,
        **sport_settings
    }
    
    return protocol_settings

def cycling_protocol_inputs():
    """
    UI for cycling-specific protocol settings
    
    Returns:
    --------
    dict
        Cycling settings
    """
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_load = st.number_input("Indicate the starting load (Watts)", 
                                       min_value=0, max_value=500, value=100, step=10)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_increment = st.number_input("Indicate the step increase (Watts)", 
                                        min_value=5, max_value=100, value=25, step=5)
    
    st.info("For cycling, input the intensity in watts.")
    
    return {
        'starting_load': starting_load,
        'load_increment': load_increment
    }

def running_protocol_inputs():
    """
    UI for running-specific protocol settings
    
    Returns:
    --------
    dict
        Running settings
    """
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_speed = st.number_input("Indicate the starting speed (km/h)", 
                                        min_value=5.0, max_value=20.0, value=8.0, step=0.5, format="%.1f")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        speed_increment = st.number_input("Indicate the step increase (km/h)", 
                                         min_value=0.5, max_value=5.0, value=0.5, step=0.5, format="%.1f")
    
    st.info("For running, input the intensity in km/h.")
    
    return {
        'starting_speed': starting_speed,
        'speed_increment': speed_increment
    }

def swimming_protocol_inputs():
    """
    UI for swimming-specific protocol settings
    
    Returns:
    --------
    dict
        Swimming settings
    """
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_speed = st.number_input("Indicate the starting speed (m/s)", 
                                        min_value=0.5, max_value=2.0, value=0.8, step=0.1, format="%.1f")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        speed_increment = st.number_input("Indicate the step increase (m/s)", 
                                         min_value=0.05, max_value=0.5, value=0.1, step=0.05, format="%.2f")
    
    st.info("For swimming, input the intensity in m/s.")
    
    return {
        'starting_speed': starting_speed,
        'speed_increment': speed_increment
    }

def create_data_input_tab(test_type, protocol_settings, test_parameters):
    """
    Create content for data input tab
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    protocol_settings : dict
        Protocol settings from protocol setup tab
    test_parameters : dict
        Test parameters from sidebar
    
    Returns:
    --------
    pandas.DataFrame or None
        Edited dataframe with test data
    """
    if 'df_template' not in st.session_state:
        # Create empty dataframe based on protocol settings
        df_template = generate_protocol_template(
            test_type,
            protocol_settings.get('num_steps', 8),
            protocol_settings.get('step_length', 4),
            test_parameters.get('rest_hr', 60),
            test_parameters.get('rest_lactate', 0.8),
            protocol_settings
        )
        st.session_state['df_template'] = df_template
    
    st.subheader("Data Input")
    
    df = st.session_state['df_template'].copy()
    
    # Set up column configuration based on test type
    column_config = {
        "step": st.column_config.NumberColumn("Step", help="Step number"),
        "length": st.column_config.NumberColumn("Length (min)", help="Duration of step in minutes"),
        "heart_rate": st.column_config.NumberColumn("Heart Rate", help="Heart rate in bpm"),
        "lactate": st.column_config.NumberColumn("Lactate", help="Blood lactate in mmol/L", format="%.2f"),
        "rpe": st.column_config.NumberColumn("RPE (6-20)", help="Rating of perceived exertion (6-20 scale)", min_value=6, max_value=20)
    }
    
    # Add test-specific columns
    if test_type == "Cycling":
        column_config["load_watts"] = st.column_config.NumberColumn("Watts", help="Power in watts")
    elif test_type == "Running":
        column_config["speed_kmh"] = st.column_config.NumberColumn("Speed (km/h)", help="Running speed in km/h")
    elif test_type == "Swimming":
        column_config["speed_ms"] = st.column_config.NumberColumn("Speed (m/s)", help="Swimming speed in m/s")
    
    # Create input fields for the data
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=["step"],
        column_config=column_config
    )
    
    # Store the edited dataframe in session state
    st.session_state['edited_df'] = edited_df
    
    return edited_df

def generate_protocol_template(test_type, num_steps, step_length, rest_hr, rest_lactate, protocol_settings):
    """
    Generate a protocol template dataframe
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    num_steps : int
        Number of steps in protocol
    step_length : int
        Length of each step in minutes
    rest_hr : int
        Resting heart rate
    rest_lactate : float
        Resting lactate value
    protocol_settings : dict
        Protocol settings from protocol setup
    
    Returns:
    --------
    pandas.DataFrame
        Template dataframe for data input
    """
    if test_type == "Cycling":
        starting_load = protocol_settings.get('starting_load', 100)
        load_increment = protocol_settings.get('load_increment', 25)
        
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
        starting_speed = protocol_settings.get('starting_speed', 8.0)
        speed_increment = protocol_settings.get('speed_increment', 0.5)
        
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
        starting_speed = protocol_settings.get('starting_speed', 0.8)
        speed_increment = protocol_settings.get('speed_increment', 0.1)
        
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
    
    return df_template

def create_results_tabs():
    """
    Create tabs for results display
    
    Returns:
    --------
    tuple
        (summary_tab, graphs_tab, zones_tab)
    """
    return st.tabs(["Summary", "Graphs", "Training Zones"])

def display_summary_results(test_type, athlete_info, threshold_results):
    """
    Display threshold results summary
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    athlete_info : dict
        Athlete information
    threshold_results : dict
        Dictionary of threshold results
    """
    from utils import speed_to_pace, speed_to_pace_100m
    
    st.subheader("Calculated Thresholds")
    
    # Create results dataframe
    results_data = []
    weight = athlete_info.get('weight', 75.0)
    
    for method, result in threshold_results.items():
        # Create different data based on test type
        if test_type == "Cycling":
            results_data.append({
                'Method': method,
                'Power (Watts)': f"{result['value']:.1f}",
                'Power (W/kg)': f"{result['value'] / weight:.2f}",
                'Heart Rate (bpm)': f"{result['hr']}",
                'Lactate (mmol/L)': f"{result['lactate']:.2f}"
            })
        elif test_type == "Running":
            # Calculate pace
            pace = speed_to_pace(result['value'])
            
            results_data.append({
                'Method': method,
                'Speed (km/h)': f"{result['value']:.1f}",
                'Pace (min/km)': pace,
                'Heart Rate (bpm)': f"{result['hr']}",
                'Lactate (mmol/L)': f"{result['lactate']:.2f}"
            })
        elif test_type == "Swimming":
            # Calculate pace
            pace = speed_to_pace_100m(result['value'])
            
            results_data.append({
                'Method': method,
                'Speed (m/s)': f"{result['value']:.2f}",
                'Pace (min/100m)': pace,
                'Heart Rate (bpm)': f"{result['hr']}",
                'Lactate (mmol/L)': f"{result['lactate']:.2f}"
            })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No threshold results available.")

def display_training_zones_table(test_type, athlete_info, threshold_results):
    """
    Display training zones as a table
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    athlete_info : dict
        Athlete information
    threshold_results : dict
        Dictionary of threshold results
    """
    from utils import determine_primary_threshold_method, create_training_zones_data
    
    st.subheader("Training Zones")
    
    # Find primary threshold method
    primary_method = determine_primary_threshold_method(threshold_results)
    
    if primary_method:
        threshold_value = threshold_results[primary_method]['value']
        hr_threshold = threshold_results[primary_method]['hr']
        weight = athlete_info.get('weight', 75.0)
        
        # Generate zones table data based on test type
        zones_data = create_training_zones_data(
            test_type, 
            primary_method, 
            threshold_value, 
            hr_threshold, 
            weight
        )
        
        # Display as dataframe
        if zones_data:
            zones_df = pd.DataFrame(zones_data)
            st.dataframe(zones_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not generate training zones.")
    else:
        st.warning("No threshold results available for training zone calculation.")

def display_export_options(test_type, athlete_info):
    """
    Display export options for results
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    athlete_info : dict
        Athlete information
    """
    from reporting import generate_pdf_report
    
    st.subheader("Export Results")
    
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF Report..."):
            # Get required data for the report
            pdf_bytes = generate_pdf_report(
                test_type=test_type,
                athlete_name=athlete_info.get('name', ''),
                birth_date=athlete_info.get('birth_date', ''),
                test_date=athlete_info.get('test_date', ''),
                weight=athlete_info.get('weight', 75.0),
                height=athlete_info.get('height', 180)
            )
            
            # Get athlete name or default
            athlete_name = athlete_info.get('name', 'athlete')
            if not athlete_name.strip():
                athlete_name = 'athlete'
            
            # Format test date
            test_date = athlete_info.get('test_date', datetime.now().date())
            if isinstance(test_date, date):
                test_date_str = test_date.strftime('%Y-%m-%d')
            else:
                test_date_str = 'report'
            
            # Download button
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"lactate_threshold_report_{athlete_name}_{test_date_str}.pdf",
                mime="application/pdf"
            )

def display_graphs():
    """Display lactate curve graphs"""
    import streamlit as st
    
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
    else:
        st.info("No lactate curve data available. Please calculate thresholds first.")

def display_training_zones():
    """Display training zones visualization"""
    import streamlit as st
    
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
    else:
        st.info("No training zones data available. Please calculate thresholds first.")