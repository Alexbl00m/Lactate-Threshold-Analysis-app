import streamlit as st
import pandas as pd
import os
import sys

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

# Import from modules
from ui import setup_page, setup_custom_css, create_sidebar
from threshold_methods import calculate_thresholds
from plotting import generate_plots, add_training_zones
from reporting import generate_pdf_report
from utils import load_logo, initialize_session_state

def main():
    """Main application entry point"""
    # Set up page configuration
    setup_page()
    
    # Apply custom styling
    setup_custom_css()
    
    # Load logo for the application
    logo = load_logo()
    
    # Initialize session state for persistent storage
    initialize_session_state()
    
    # Create sidebar and get user inputs
    with st.sidebar:
        if logo:
            st.image(logo, width=200)
        else:
            st.title("Lindblom Coaching")
        
        st.header("Lactate Threshold Analysis")
        
        # Get user inputs from sidebar
        test_type, athlete_info, threshold_methods, test_parameters = create_sidebar()
    
    # Main content area
    st.title("Lactate Threshold Analysis Tool")
    st.header(f"{test_type} Protocol Setup")
    
    # Create tabs for the main workflow
    tab1, tab2, tab3 = st.tabs(["Protocol Setup", "Data Input", "Results"])
    
    # Tab 1: Protocol Setup
    with tab1:
        protocol_setup(test_type, test_parameters)
    
    # Tab 2: Data Input
    with tab2:
        data_input(test_type)
    
    # Tab 3: Results
    with tab3:
        display_results(test_type, athlete_info)

def protocol_setup(test_type, test_parameters):
    """Handle the protocol setup tab - MODIFIED to ensure last step values can be entered"""
    import streamlit as st
    
    st.write(f"Indicate the sport that you would like to analyze lactate thresholds on: **{test_type}**")
    
    # Include heart rate data
    include_hr = st.toggle("Include heart rate data?", value=True)
    
    # Protocol details
    col1, col2 = st.columns([1, 4])
    with col1:
        num_steps = st.number_input("Indicate how many steps were done (including rest)", 
                                    min_value=3, max_value=20, value=8, step=1)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        step_length = st.number_input("Indicate the length (in minutes) of each step", 
                                     min_value=1, max_value=10, value=4, step=1)
    
    # Sport-specific inputs
    if test_type == "Cycling":
        cycling_protocol_inputs()
    elif test_type == "Running":
        running_protocol_inputs()
    elif test_type == "Swimming":
        swimming_protocol_inputs()
    
    # Last step completion - Handle it here, not in protocol_additional_settings
    last_step_completed = st.toggle("Was the last step fully completed?", value=False, key="last_step_toggle")
    
    if not last_step_completed:
        col1, col2 = st.columns([1, 4])
        with col1:
            last_step_time = st.text_input("Then, indicate how long it was (in the mm:ss format)", value="02:00")
        st.info("Note: You'll still be able to enter lactate, heart rate, and other values for the last step even though it wasn't completed.")
    else:
        last_step_time = None
    
    # Store the last step info properly
    st.session_state['last_step_completed'] = last_step_completed
    st.session_state['last_step_time'] = last_step_time
    
    # Additional protocol settings - MODIFIED to remove duplicate toggle
    modified_protocol_additional_settings()
    
    # Generate protocol button
    if st.button("Generate Test Protocol", type="primary"):
        generate_protocol_template(test_type, num_steps, step_length, test_parameters)
        st.success("Protocol generated! Please proceed to the Data Input tab.")

def modified_protocol_additional_settings():
    """Additional protocol settings without the last step toggle"""
    import streamlit as st
    
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
    
    # Store values in session state
    st.session_state['fitting_method'] = fitting_method
    st.session_state['include_baseline'] = include_baseline
    st.session_state['log_log_portion'] = log_log_portion

def cycling_protocol_inputs():
    """Cycling-specific protocol inputs"""
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_load = st.number_input("Indicate the starting load (Watts)", 
                                       min_value=0, max_value=500, value=100, step=10)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        load_increment = st.number_input("Indicate the step increase (Watts)", 
                                        min_value=5, max_value=100, value=25, step=5)
    
    st.info("For cycling, input the intensity in watts.")
    
    # Store values in session state
    st.session_state['starting_load'] = starting_load
    st.session_state['load_increment'] = load_increment

def running_protocol_inputs():
    """Running-specific protocol inputs"""
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_speed = st.number_input("Indicate the starting speed (km/h)", 
                                        min_value=5.0, max_value=20.0, value=8.0, step=0.5, format="%.1f")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        speed_increment = st.number_input("Indicate the step increase (km/h)", 
                                         min_value=0.5, max_value=5.0, value=0.5, step=0.5, format="%.1f")
    
    st.info("For running, input the intensity in km/h.")
    
    # Store values in session state
    st.session_state['starting_speed'] = starting_speed
    st.session_state['speed_increment'] = speed_increment

def swimming_protocol_inputs():
    """Swimming-specific protocol inputs"""
    col1, col2 = st.columns([1, 4])
    with col1:
        starting_speed = st.number_input("Indicate the starting speed (m/s)", 
                                        min_value=0.5, max_value=2.0, value=0.8, step=0.1, format="%.1f")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        speed_increment = st.number_input("Indicate the step increase (m/s)", 
                                         min_value=0.05, max_value=0.5, value=0.1, step=0.05, format="%.2f")
    
    st.info("For swimming, input the intensity in m/s.")
    
    # Store values in session state
    st.session_state['starting_speed'] = starting_speed
    st.session_state['speed_increment'] = speed_increment

def protocol_additional_settings():
    """Additional protocol settings"""
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
    
    # Store values in session state
    st.session_state['last_step_completed'] = last_step_completed
    st.session_state['last_step_time'] = last_step_time if not last_step_completed else None
    st.session_state['fitting_method'] = fitting_method
    st.session_state['include_baseline'] = include_baseline
    st.session_state['log_log_portion'] = log_log_portion

def generate_protocol_template(test_type, num_steps, step_length, test_parameters):
    """Generate a protocol template based on user input - MODIFIED to handle incomplete last step"""
    import pandas as pd
    import streamlit as st
    
    rest_hr = test_parameters.get('rest_hr', 60)
    rest_lactate = test_parameters.get('rest_lactate', 0.8)
    
    # Create a list to store the step lengths
    step_lengths = [step_length] * num_steps
    
    # Modify the last step length if it wasn't completed
    if not st.session_state.get('last_step_completed', True) and num_steps > 1:
        last_step_time = st.session_state.get('last_step_time', "00:00")
        
        # Convert time string to minutes (simple conversion for this example)
        try:
            parts = last_step_time.split(':')
            minutes = int(parts[0])
            seconds = int(parts[1]) if len(parts) > 1 else 0
            completed_minutes = minutes + seconds / 60
            
            # Update the last step length
            if completed_minutes > 0:
                step_lengths[num_steps - 1] = completed_minutes
        except:
            # If conversion fails, keep the original length
            pass
    
    if test_type == "Cycling":
        starting_load = st.session_state.get('starting_load', 100)
        load_increment = st.session_state.get('load_increment', 25)
        
        df_template = pd.DataFrame({
            "step": range(num_steps),
            "load_watts": [starting_load + i * load_increment for i in range(num_steps)],
            "length": step_lengths,
            "heart_rate": [None] * num_steps,
            "lactate": [None] * num_steps,
            "rpe": [None] * num_steps
        })
        
        # Set rest values
        df_template.loc[0, "load_watts"] = 0
        df_template.loc[0, "heart_rate"] = rest_hr
        df_template.loc[0, "lactate"] = rest_lactate
        
    elif test_type == "Running":
        starting_speed = st.session_state.get('starting_speed', 8.0)
        speed_increment = st.session_state.get('speed_increment', 0.5)
        
        df_template = pd.DataFrame({
            "step": range(num_steps),
            "speed_kmh": [starting_speed + i * speed_increment for i in range(num_steps)],
            "length": step_lengths,
            "heart_rate": [None] * num_steps,
            "lactate": [None] * num_steps,
            "rpe": [None] * num_steps
        })
        
        # Set rest values
        df_template.loc[0, "speed_kmh"] = 0
        df_template.loc[0, "heart_rate"] = rest_hr
        df_template.loc[0, "lactate"] = rest_lactate
        
    elif test_type == "Swimming":
        starting_speed = st.session_state.get('starting_speed', 0.8)
        speed_increment = st.session_state.get('speed_increment', 0.1)
        
        df_template = pd.DataFrame({
            "step": range(num_steps),
            "speed_ms": [starting_speed + i * speed_increment for i in range(num_steps)],
            "length": step_lengths,
            "heart_rate": [None] * num_steps,
            "lactate": [None] * num_steps,
            "rpe": [None] * num_steps
        })
        
        # Set rest values
        df_template.loc[0, "speed_ms"] = 0
        df_template.loc[0, "heart_rate"] = rest_hr
        df_template.loc[0, "lactate"] = rest_lactate
    
    # Store template in session state
    st.session_state['df_template'] = df_template

def data_input(test_type):
    """Handle the data input tab - MODIFIED to clarify incomplete last step"""
    import streamlit as st
    
    if 'df_template' not in st.session_state:
        st.info("Please set up the protocol in the Protocol Setup tab first.")
    else:
        st.subheader("Data Input")
        
        df = st.session_state['df_template'].copy()
        
        # Display note for incomplete last step
        if not st.session_state.get('last_step_completed', True) and len(df) > 1:
            last_step_num = len(df) - 1
            last_step_time = st.session_state.get('last_step_time', "unknown")
            st.info(f"Step {last_step_num} was not completed fully (duration: {last_step_time}). Please still enter all values for this step.")
        
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
        
        st.session_state['edited_df'] = edited_df
        
        # Add a note below the data editor
        st.write("Make sure to enter all values including for the last step, even if it wasn't completed fully.")
        
        # Calculate button
        if st.button("Calculate Thresholds", type="primary"):
            process_threshold_calculation(test_type)

def process_threshold_calculation(test_type):
    """Process the data and calculate thresholds"""
    import streamlit as st
    
    # Check if all necessary lactate values are present
    if 'edited_df' not in st.session_state or st.session_state['edited_df']['lactate'].isna().any():
        st.error("Please fill in all lactate values before calculating thresholds.")
    else:
        # Get data and parameters
        df = st.session_state['edited_df'].copy()
        
        # Get threshold methods from session state or sidebar
        threshold_methods = st.session_state.get('threshold_methods', [])
        if not threshold_methods:
            # If methods weren't stored, use defaults
            threshold_methods = ["IAT (Individual Anaerobic Threshold)", "Modified Dmax"]
        
        fitting_method = st.session_state.get('fitting_method', "3rd degree polynomial")
        log_log_portion = st.session_state.get('log_log_portion', 0.75)
        include_baseline = st.session_state.get('include_baseline', False)
        
        # Import calculation function
        from threshold_methods import calculate_thresholds
        
        # Calculate thresholds
        threshold_results = calculate_thresholds(
            df, 
            test_type, 
            threshold_methods, 
            fitting_method, 
            log_log_portion,
            include_baseline
        )
        
        # Store results in session state
        st.session_state['threshold_results'] = threshold_results
        
        # Generate plots
        from plotting import generate_plots
        generate_plots(df, test_type, threshold_results)
        
        # Success message
        st.success("Thresholds calculated successfully! See results in the Results tab.")

def display_results(test_type, athlete_info):
    """Handle the results tab"""
    if 'threshold_results' not in st.session_state:
        st.info("Please set up the protocol in the Protocol Setup tab and calculate thresholds in the Data Input tab.")
    else:
        st.header("Results")
        
        # Create tabs for different result views
        results_tab1, results_tab2, results_tab3 = st.tabs(["Summary", "Graphs", "Training Zones"])
        
        with results_tab1:
            display_summary_results(test_type, athlete_info)
        
        with results_tab2:
            display_graphs()
        
        with results_tab3:
            display_training_zones()

def display_summary_results(test_type, athlete_info):
    """Display threshold results summary"""
    st.subheader("Calculated Thresholds")
    
    # Create results dataframe
    results_data = []
    weight = athlete_info.get('weight', 75.0)
    
    for method, result in st.session_state['threshold_results'].items():
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
            from utils import speed_to_pace
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
            from utils import speed_to_pace_100m
            pace = speed_to_pace_100m(result['value'])
            
            results_data.append({
                'Method': method,
                'Speed (m/s)': f"{result['value']:.2f}",
                'Pace (min/100m)': pace,
                'Heart Rate (bpm)': f"{result['hr']}",
                'Lactate (mmol/L)': f"{result['lactate']:.2f}"
            })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Display training zones
    display_training_zones_table(test_type, athlete_info)
    
    # Export button
    st.subheader("Export Results")
    
    if st.button("Generate PDF Report", type="primary"):
        # Get required data for the report
        pdf_bytes = generate_pdf_report(
            test_type=test_type,
            athlete_name=athlete_info.get('name', ''),
            birth_date=athlete_info.get('birth_date', ''),
            test_date=athlete_info.get('test_date', ''),
            weight=athlete_info.get('weight', 75.0),
            height=athlete_info.get('height', 180)
        )
        
        # Download button
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"lactate_threshold_report_{athlete_info.get('name', 'athlete')}_{athlete_info.get('test_date', 'today')}.pdf",
            mime="application/pdf"
        )

def display_graphs():
    """Display lactate curve graphs with error handling"""
    import streamlit as st
    import plotly.graph_objects as go
    
    st.subheader("Lactate Curve and Thresholds")
    
    if 'plotly_fig' in st.session_state and st.session_state['plotly_fig'] is not None:
        try:
            # Try to display the plotly figure
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
        except Exception as e:
            # Handle any errors with the plotly figure
            st.error(f"Could not display the lactate curve. You may need to recalculate thresholds.")
            
            # Create a fallback simple figure
            if 'edited_df' in st.session_state and 'threshold_results' in st.session_state:
                try:
                    # Try to create a simple fallback figure
                    df = st.session_state['edited_df']
                    
                    # Determine intensity column based on test type
                    if 'load_watts' in df.columns:
                        intensity_col = 'load_watts'
                        intensity_label = 'Power (Watts)'
                    elif 'speed_kmh' in df.columns:
                        intensity_col = 'speed_kmh'
                        intensity_label = 'Speed (km/h)'
                    elif 'speed_ms' in df.columns:
                        intensity_col = 'speed_ms'
                        intensity_label = 'Speed (m/s)'
                    else:
                        intensity_col = None
                    
                    if intensity_col is not None:
                        # Create a simple fallback figure
                        fig = go.Figure()
                        
                        # Add lactate curve
                        fig.add_trace(go.Scatter(
                            x=df[intensity_col],
                            y=df['lactate'],
                            mode='lines+markers',
                            name='Lactate',
                            line=dict(color='#E6754E', width=3),
                            marker=dict(size=8)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Lactate-to-performance-curve',
                            xaxis_title=intensity_label,
                            yaxis_title='Lactate (mmol/L)',
                            template="plotly_white"
                        )
                        
                        # Display fallback figure
                        st.plotly_chart(fig, use_container_width=True)
                        st.info("Displaying a simplified lactate curve. For detailed thresholds, try recalculating.")
                except Exception as fallback_error:
                    # If even the fallback fails, just show a message
                    st.info("Could not display lactate curve data. Please calculate thresholds first.")
            else:
                st.info("No lactate curve data available. Please calculate thresholds first.")
    else:
        st.info("No lactate curve data available. Please calculate thresholds first.")

def display_training_zones():
    """Display training zones visualization with error handling"""
    import streamlit as st
    import plotly.graph_objects as go
    
    st.subheader("Training Zones")
    
    if 'plotly_zones' in st.session_state and st.session_state['plotly_zones'] is not None:
        try:
            # Try to display the zones figure
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
        except Exception as e:
            # Handle any errors with the zones figure
            st.error("Could not display the training zones chart. You may need to recalculate thresholds.")
            st.info("Training zones can be viewed in the Summary tab as a table.")
    else:
        st.info("No training zones data available. Please calculate thresholds first.")

def display_training_zones_table(test_type, athlete_info):
    """Display training zones as a table"""
    st.subheader("Training Zones")
    
    # Find primary threshold method
    primary_method = determine_primary_threshold_method()
    
    if primary_method:
        threshold_value = st.session_state['threshold_results'][primary_method]['value']
        hr_threshold = st.session_state['threshold_results'][primary_method]['hr']
        weight = athlete_info.get('weight', 75.0)
        
        # Generate zones table data based on test type
        from utils import create_training_zones_data
        zones_data = create_training_zones_data(
            test_type, 
            primary_method, 
            threshold_value, 
            hr_threshold, 
            weight
        )
        
        # Display as dataframe
        zones_df = pd.DataFrame(zones_data)
        st.dataframe(zones_df, use_container_width=True, hide_index=True)

def determine_primary_threshold_method():
    """Determine the primary threshold method for zones"""
    if 'threshold_results' not in st.session_state:
        return None
    
    if 'IAT' in st.session_state['threshold_results']:
        return 'IAT'
    elif 'Modified Dmax' in st.session_state['threshold_results']:
        return 'Modified Dmax'
    elif 'Fixed 4.0 mmol/L' in st.session_state['threshold_results']:
        return 'Fixed 4.0 mmol/L'
    elif len(st.session_state['threshold_results']) > 0:
        return list(st.session_state['threshold_results'].keys())[0]
    
    return None

def process_threshold_calculation(test_type):
    """Process the data and calculate thresholds with improved error handling"""
    import streamlit as st
    import plotly.graph_objects as go
    
    # Check if all necessary lactate values are present
    if 'edited_df' not in st.session_state or st.session_state['edited_df']['lactate'].isna().any():
        st.error("Please fill in all lactate values before calculating thresholds.")
    else:
        # Get data and parameters
        df = st.session_state['edited_df'].copy()
        
        # Get threshold methods from session state or sidebar
        threshold_methods = st.session_state.get('threshold_methods', [])
        if not threshold_methods:
            # If methods weren't stored, use defaults
            threshold_methods = ["IAT (Individual Anaerobic Threshold)", "Modified Dmax"]
        
        fitting_method = st.session_state.get('fitting_method', "3rd degree polynomial")
        log_log_portion = st.session_state.get('log_log_portion', 0.75)
        include_baseline = st.session_state.get('include_baseline', False)
        
        # Import calculation function
        from threshold_methods import calculate_thresholds
        
        try:
            # Calculate thresholds
            threshold_results = calculate_thresholds(
                df, 
                test_type, 
                threshold_methods, 
                fitting_method, 
                log_log_portion,
                include_baseline
            )
            
            # Store results in session state
            st.session_state['threshold_results'] = threshold_results
            
            try:
                # Try to generate plots with error handling
                from plotting import generate_plots
                generate_plots(df, test_type, threshold_results)
                
                # Verify that the plots were generated correctly
                if 'plotly_fig' not in st.session_state or st.session_state['plotly_fig'] is None:
                    # Create a simple fallback figure
                    create_fallback_plot(df, test_type, threshold_results)
                
                # Success message
                st.success("Thresholds calculated successfully! See results in the Results tab.")
            except Exception as plot_error:
                # Handle plotting errors
                st.warning("Thresholds were calculated, but there was an error generating the plots.")
                st.info("You can still view the threshold values in the Results tab.")
                
                # Create a fallback plot
                create_fallback_plot(df, test_type, threshold_results)
                
                # For debugging
                st.session_state['plot_error'] = str(plot_error)
        except Exception as calc_error:
            # Handle threshold calculation errors
            st.error("There was an error calculating the thresholds. Please check your input data.")
            st.session_state['calc_error'] = str(calc_error)

def create_fallback_plot(df, test_type, threshold_results):
    """Create a simple fallback plot when the main plotting function fails"""
    import streamlit as st
    import plotly.graph_objects as go
    
    # Determine intensity column based on test type
    if test_type == "Cycling":
        intensity_col = "load_watts"
        intensity_label = "Power (Watts)"
    elif test_type == "Running":
        intensity_col = "speed_kmh"
        intensity_label = "Speed (km/h)"
    else:  # Swimming
        intensity_col = "speed_ms"
        intensity_label = "Speed (m/s)"
    
    # Store in session state for reference
    st.session_state['intensity_col'] = intensity_col
    st.session_state['intensity_label'] = intensity_label
    
    # Remove rest values (step 0) for plotting
    df_plot = df[df['step'] > 0].copy()
    
    try:
        # Create a simple figure
        fig = go.Figure()
        
        # Add lactate curve
        fig.add_trace(go.Scatter(
            x=df_plot[intensity_col],
            y=df_plot['lactate'],
            mode='lines+markers',
            name='Lactate',
            line=dict(color='#E6754E', width=3),
            marker=dict(size=8)
        ))
        
        # Add heart rate if available
        if 'heart_rate' in df_plot.columns and not df_plot['heart_rate'].isna().all():
            fig.add_trace(go.Scatter(
                x=df_plot[intensity_col],
                y=df_plot['heart_rate'],
                mode='lines+markers',
                name='Heart Rate',
                line=dict(color='#4EA1E6', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            # Update layout for dual axis
            fig.update_layout(
                yaxis2=dict(
                    title='Heart Rate (bpm)',
                    titlefont=dict(color='#4EA1E6'),
                    tickfont=dict(color='#4EA1E6'),
                    anchor="x",
                    overlaying="y",
                    side="right"
                )
            )
        
        # Add thresholds as vertical lines
        for method, result in threshold_results.items():
            if 'value' in result:
                fig.add_vline(
                    x=result['value'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"{method}: {result['value']:.1f}",
                    annotation_position="top right"
                )
        
        # Update layout
        fig.update_layout(
            title='Lactate-to-performance-curve',
            xaxis_title=intensity_label,
            yaxis=dict(
                title='Lactate (mmol/L)',
                titlefont=dict(color='#E6754E'),
                tickfont=dict(color='#E6754E')
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
        
        # Store figure in session state
        st.session_state['plotly_fig'] = fig
        
        # Also create a simple zones figure
        create_fallback_zones(df_plot, test_type, threshold_results)
        
    except Exception as e:
        # If even the fallback plot fails
        st.session_state['fallback_error'] = str(e)
        st.session_state['plotly_fig'] = None

def create_fallback_zones(df_plot, test_type, threshold_results):
    """Create a simple fallback zones plot"""
    import streamlit as st
    import plotly.graph_objects as go
    import numpy as np
    
    # Get intensity column
    intensity_col = st.session_state.get('intensity_col')
    intensity_label = st.session_state.get('intensity_label')
    
    try:
        # Create a simple zones figure
        fig_zones = go.Figure()
        
        # Determine a primary threshold method
        primary_method = None
        for method in ['IAT', 'Modified Dmax', 'Fixed 4.0 mmol/L']:
            if method in threshold_results:
                primary_method = method
                break
        
        if primary_method is None and threshold_results:
            # Use the first available method
            primary_method = list(threshold_results.keys())[0]
        
        if primary_method and primary_method in threshold_results:
            threshold_value = threshold_results[primary_method]['value']
            
            # Set zone colors
            zone_colors = {
                'Z1': 'rgba(173, 216, 230, 0.3)',  # Light blue
                'Z2': 'rgba(255, 255, 150, 0.3)',  # Light yellow
                'Z3': 'rgba(255, 200, 150, 0.3)',  # Light orange
                'Z4': 'rgba(255, 150, 150, 0.3)',  # Light red
                'Z5': 'rgba(200, 120, 120, 0.3)'   # Darker red
            }
            
            # Get min and max intensity values
            x_min = min(df_plot[intensity_col])
            x_max = max(df_plot[intensity_col]) * 1.1
            y_max = max(df_plot['lactate']) * 1.1
            
            # Create simple zone boundaries
            zones = [
                {'name': 'Z1', 'min': x_min, 'max': 0.7 * threshold_value, 'color': zone_colors['Z1']},
                {'name': 'Z2', 'min': 0.7 * threshold_value, 'max': 0.9 * threshold_value, 'color': zone_colors['Z2']},
                {'name': 'Z3', 'min': 0.9 * threshold_value, 'max': 1.0 * threshold_value, 'color': zone_colors['Z3']},
                {'name': 'Z4', 'min': 1.0 * threshold_value, 'max': 1.1 * threshold_value, 'color': zone_colors['Z4']},
                {'name': 'Z5', 'min': 1.1 * threshold_value, 'max': x_max, 'color': zone_colors['Z5']}
            ]
            
            # Add rectangles for each zone
            for zone in zones:
                fig_zones.add_shape(
                    type="rect",
                    x0=zone['min'],
                    x1=zone['max'],
                    y0=0,
                    y1=y_max,
                    fillcolor=zone['color'],
                    line=dict(width=0)
                )
                
                # Add zone labels
                fig_zones.add_annotation(
                    x=(zone['min'] + zone['max']) / 2,
                    y=y_max * 0.5,
                    text=zone['name'],
                    showarrow=False,
                    font=dict(
                        size=14,
                        color="black"
                    )
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
            
            # Add threshold line
            fig_zones.add_vline(
                x=threshold_value,
                line_dash="dash",
                line_color="black",
                annotation_text=f"{primary_method}: {threshold_value:.1f}",
                annotation_position="top"
            )
            
            # Update layout
            fig_zones.update_layout(
                title='Training Zones',
                xaxis_title=intensity_label,
                yaxis_title='Lactate (mmol/L)',
                template="plotly_white"
            )
            
            # Store in session state
            st.session_state['plotly_zones'] = fig_zones
        else:
            st.session_state['plotly_zones'] = None
            
    except Exception as e:
        # If zones fallback fails
        st.session_state['zones_fallback_error'] = str(e)
        st.session_state['plotly_zones'] = None

# Application entry point
if __name__ == "__main__":
    main()
