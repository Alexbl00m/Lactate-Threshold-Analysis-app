import os
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

def load_logo():
    """
    Load logo from assets folder or create a placeholder
    
    Returns:
    --------
    str or None
        Path to logo file
    """
    # Try to find logo in different locations
    possible_paths = [
        "logo.png",
        os.path.join("assets", "logo.png"),
        os.path.join(os.path.dirname(__file__), "..", "assets", "logo.png")
    ]
    
    # Check if logo exists in any of the paths
    for logo_path in possible_paths:
        if os.path.exists(logo_path):
            return logo_path
    
    # If not found, create a placeholder logo
    try:
        # Create a directory for assets if it doesn't exist
        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Path for the new logo
        new_logo_path = os.path.join(assets_dir, "logo.png")
        
        # Create a simple placeholder logo using PIL
        img = Image.new('RGBA', (400, 200), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font if available, otherwise use default
        try:
            font = ImageFont.truetype("arial.ttf", 50)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw company name using brand color
        draw.text((20, 50), "LINDBLOM", fill=(230, 117, 78), font=font)
        draw.text((20, 120), "COACHING", fill=(230, 117, 78), font=font)
        
        # Save the image
        img.save(new_logo_path)
        return new_logo_path
    
    except Exception as e:
        print(f"Error creating logo: {e}")
        return None

def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    if 'threshold_results' not in st.session_state:
        st.session_state['threshold_results'] = {}
    
    if 'df_template' not in st.session_state:
        st.session_state['df_template'] = None
    
    if 'edited_df' not in st.session_state:
        st.session_state['edited_df'] = None
    
    if 'intensity_col' not in st.session_state:
        st.session_state['intensity_col'] = None
    
    if 'intensity_label' not in st.session_state:
        st.session_state['intensity_label'] = None
    
    if 'matplotlib_fig' not in st.session_state:
        st.session_state['matplotlib_fig'] = None
    
    if 'plotly_fig' not in st.session_state:
        st.session_state['plotly_fig'] = None
    
    if 'plotly_zones' not in st.session_state:
        st.session_state['plotly_zones'] = None

def speed_to_pace(speed_kmh):
    """
    Convert running speed to pace (min:sec per km)
    
    Parameters:
    -----------
    speed_kmh : float
        Running speed in km/h
    
    Returns:
    --------
    str
        Pace in the format 'min:sec'
    """
    if speed_kmh <= 0:
        return "--:--"
    
    # Calculate pace in seconds per km
    pace_seconds = (1000 / speed_kmh) * 3.6
    
    # Convert to minutes and seconds
    pace_min = int(pace_seconds // 60)
    pace_sec = int(pace_seconds % 60)
    
    # Format as min:sec
    return f"{pace_min}:{pace_sec:02d}"

def speed_to_pace_100m(speed_ms):
    """
    Convert swimming speed to pace (min:sec per 100m)
    
    Parameters:
    -----------
    speed_ms : float
        Swimming speed in m/s
    
    Returns:
    --------
    str
        Pace in the format 'min:sec'
    """
    if speed_ms <= 0:
        return "--:--"
    
    # Calculate pace in seconds per 100m
    pace_seconds = 100 / speed_ms
    
    # Convert to minutes and seconds
    pace_min = int(pace_seconds // 60)
    pace_sec = int(pace_seconds % 60)
    
    # Format as min:sec
    return f"{pace_min}:{pace_sec:02d}"

def determine_primary_threshold_method(threshold_results):
    """
    Determine which threshold method to use as primary for zone calculations
    
    Parameters:
    -----------
    threshold_results : dict
        Dictionary of threshold results
    
    Returns:
    --------
    str or None
        Name of primary threshold method
    """
    # Order of preference for zone calculation
    if 'IAT' in threshold_results:
        return 'IAT'
    elif 'Modified Dmax' in threshold_results:
        return 'Modified Dmax'
    elif 'Fixed 4.0 mmol/L' in threshold_results:
        return 'Fixed 4.0 mmol/L'
    elif len(threshold_results) > 0:
        # If none of the preferred methods available, use the first one
        return list(threshold_results.keys())[0]
    
    # No methods available
    return None

def create_training_zones_data(test_type, primary_method, threshold_value, hr_threshold, weight):
    """
    Create training zones data for display in tables
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    primary_method : str
        Primary threshold method used
    threshold_value : float
        Threshold value
    hr_threshold : int
        Heart rate at threshold
    weight : float
        Athlete weight in kg
    
    Returns:
    --------
    list
        List of dictionaries with zone data
    """
    # Check for valid inputs
    if not test_type or not primary_method or not threshold_value or not hr_threshold:
        return []
    
    # Create zones based on test type
    if test_type == "Cycling":
        return create_cycling_zones(primary_method, threshold_value, hr_threshold, weight)
    elif test_type == "Running":
        return create_running_zones(primary_method, threshold_value, hr_threshold)
    elif test_type == "Swimming":
        return create_swimming_zones(primary_method, threshold_value, hr_threshold)
    
    return []

def create_cycling_zones(primary_method, threshold_value, hr_threshold, weight):
    """
    Create training zones for cycling
    
    Parameters:
    -----------
    primary_method : str
        Primary threshold method used
    threshold_value : float
        Threshold value (watts)
    hr_threshold : int
        Heart rate at threshold
    weight : float
        Athlete weight in kg
    
    Returns:
    --------
    list
        List of dictionaries with zone data
    """
    # Different zone calculations based on threshold method
    if primary_method == 'IAT':
        # IAT-based zones
        return [
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
    
    elif primary_method == 'Modified Dmax':
        # Modified Dmax-based zones
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Power Range': f"below {int(0.65 * threshold_value)} W",
                'Relative Power': f"below {(0.65 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"below {int(hr_threshold - 25)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Power Range': f"{int(0.65 * threshold_value)} - {int(0.9 * threshold_value)} W",
                'Relative Power': f"{(0.65 * threshold_value / weight):.2f} - {(0.9 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold - 25)} - {int(hr_threshold - 10)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Power Range': f"{int(0.9 * threshold_value)} - {int(1.0 * threshold_value)} W",
                'Relative Power': f"{(0.9 * threshold_value / weight):.2f} - {(1.0 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold - 10)} - {int(hr_threshold)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Power Range': f"{int(1.0 * threshold_value)} - {int(1.1 * threshold_value)} W",
                'Relative Power': f"{(1.0 * threshold_value / weight):.2f} - {(1.1 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold)} - {int(hr_threshold + 10)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Power Range': f"above {int(1.1 * threshold_value)} W",
                'Relative Power': f"above {(1.1 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"above {int(hr_threshold + 10)} bpm"
            }
        ]
    
    else:
        # Fixed threshold-based zones
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Power Range': f"below {int(0.6 * threshold_value)} W",
                'Relative Power': f"below {(0.6 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"below {int(hr_threshold - 30)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Power Range': f"{int(0.6 * threshold_value)} - {int(0.85 * threshold_value)} W",
                'Relative Power': f"{(0.6 * threshold_value / weight):.2f} - {(0.85 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold - 30)} - {int(hr_threshold - 15)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Power Range': f"{int(0.85 * threshold_value)} - {int(0.95 * threshold_value)} W",
                'Relative Power': f"{(0.85 * threshold_value / weight):.2f} - {(0.95 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold - 15)} - {int(hr_threshold - 5)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Power Range': f"{int(0.95 * threshold_value)} - {int(1.05 * threshold_value)} W",
                'Relative Power': f"{(0.95 * threshold_value / weight):.2f} - {(1.05 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 5)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Power Range': f"above {int(1.05 * threshold_value)} W",
                'Relative Power': f"above {(1.05 * threshold_value / weight):.2f} W/kg",
                'Heart Rate': f"above {int(hr_threshold + 5)} bpm"
            }
        ]

def create_running_zones(primary_method, threshold_value, hr_threshold):
    """
    Create training zones for running
    
    Parameters:
    -----------
    primary_method : str
        Primary threshold method used
    threshold_value : float
        Threshold value (km/h)
    hr_threshold : int
        Heart rate at threshold
    
    Returns:
    --------
    list
        List of dictionaries with zone data
    """
    # Calculate different paces
    if primary_method == 'IAT':
        # IAT-based zones
        return [
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
    
    elif primary_method == 'Modified Dmax':
        # Modified Dmax-based zones
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Speed Range': f"below {(0.7 * threshold_value):.1f} km/h",
                'Pace Range': f"slower than {speed_to_pace(0.7 * threshold_value)} min/km",
                'Heart Rate': f"below {int(hr_threshold - 20)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Speed Range': f"{(0.7 * threshold_value):.1f} - {(0.9 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(0.9 * threshold_value)} - {speed_to_pace(0.7 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Speed Range': f"{(0.9 * threshold_value):.1f} - {(1.0 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(1.0 * threshold_value)} - {speed_to_pace(0.9 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Speed Range': f"{(1.0 * threshold_value):.1f} - {(1.1 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(1.1 * threshold_value)} - {speed_to_pace(1.0 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold)} - {int(hr_threshold + 10)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Speed Range': f"above {(1.1 * threshold_value):.1f} km/h",
                'Pace Range': f"faster than {speed_to_pace(1.1 * threshold_value)} min/km",
                'Heart Rate': f"above {int(hr_threshold + 10)} bpm"
            }
        ]
    
    else:
        # Fixed threshold-based zones
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Speed Range': f"below {(0.6 * threshold_value):.1f} km/h",
                'Pace Range': f"slower than {speed_to_pace(0.6 * threshold_value)} min/km",
                'Heart Rate': f"below {int(hr_threshold - 25)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Speed Range': f"{(0.6 * threshold_value):.1f} - {(0.85 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(0.85 * threshold_value)} - {speed_to_pace(0.6 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold - 25)} - {int(hr_threshold - 10)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Speed Range': f"{(0.85 * threshold_value):.1f} - {(0.95 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(0.95 * threshold_value)} - {speed_to_pace(0.85 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold - 10)} - {int(hr_threshold - 5)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Speed Range': f"{(0.95 * threshold_value):.1f} - {(1.05 * threshold_value):.1f} km/h",
                'Pace Range': f"{speed_to_pace(1.05 * threshold_value)} - {speed_to_pace(0.95 * threshold_value)} min/km",
                'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 5)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Speed Range': f"above {(1.05 * threshold_value):.1f} km/h",
                'Pace Range': f"faster than {speed_to_pace(1.05 * threshold_value)} min/km",
                'Heart Rate': f"above {int(hr_threshold + 5)} bpm"
            }
        ]

def create_swimming_zones(primary_method, threshold_value, hr_threshold):
    """
    Create training zones for swimming
    
    Parameters:
    -----------
    primary_method : str
        Primary threshold method used
    threshold_value : float
        Threshold value (m/s)
    hr_threshold : int
        Heart rate at threshold
    
    Returns:
    --------
    list
        List of dictionaries with zone data
    """
    # Calculate different paces for swimming
    if primary_method in ['IAT', 'Modified Dmax']:
        # IAT or Modified Dmax-based zones
        factor1 = 0.7 if primary_method == 'IAT' else 0.7
        factor2 = 0.9 if primary_method == 'IAT' else 0.9
        factor3 = 1.03 if primary_method == 'IAT' else 1.0
        factor4 = 1.12 if primary_method == 'IAT' else 1.1
        
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Speed Range': f"below {(factor1 * threshold_value):.2f} m/s",
                'Pace Range': f"slower than {speed_to_pace_100m(factor1 * threshold_value)} min/100m",
                'Heart Rate': f"below {int(hr_threshold - 20)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Speed Range': f"{(factor1 * threshold_value):.2f} - {(factor2 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(factor2 * threshold_value)} - {speed_to_pace_100m(factor1 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold - 20)} - {int(hr_threshold - 5)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Speed Range': f"{(factor2 * threshold_value):.2f} - {(factor3 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(factor3 * threshold_value)} - {speed_to_pace_100m(factor2 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 8)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Speed Range': f"{(factor3 * threshold_value):.2f} - {(factor4 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(factor4 * threshold_value)} - {speed_to_pace_100m(factor3 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold + 8)} - {int(hr_threshold + 18)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Speed Range': f"above {(factor4 * threshold_value):.2f} m/s",
                'Pace Range': f"faster than {speed_to_pace_100m(factor4 * threshold_value)} min/100m",
                'Heart Rate': f"above {int(hr_threshold + 18)} bpm"
            }
        ]
    
    else:
        # Fixed threshold-based zones
        return [
            {
                'Zone': 'Z1 - Recovery',
                'Speed Range': f"below {(0.6 * threshold_value):.2f} m/s",
                'Pace Range': f"slower than {speed_to_pace_100m(0.6 * threshold_value)} min/100m",
                'Heart Rate': f"below {int(hr_threshold - 25)} bpm"
            },
            {
                'Zone': 'Z2 - Endurance',
                'Speed Range': f"{(0.6 * threshold_value):.2f} - {(0.85 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(0.85 * threshold_value)} - {speed_to_pace_100m(0.6 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold - 25)} - {int(hr_threshold - 10)} bpm"
            },
            {
                'Zone': 'Z3 - Tempo/Sweetspot',
                'Speed Range': f"{(0.85 * threshold_value):.2f} - {(0.95 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(0.95 * threshold_value)} - {speed_to_pace_100m(0.85 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold - 10)} - {int(hr_threshold - 5)} bpm"
            },
            {
                'Zone': 'Z4 - Threshold',
                'Speed Range': f"{(0.95 * threshold_value):.2f} - {(1.05 * threshold_value):.2f} m/s",
                'Pace Range': f"{speed_to_pace_100m(1.05 * threshold_value)} - {speed_to_pace_100m(0.95 * threshold_value)} min/100m",
                'Heart Rate': f"{int(hr_threshold - 5)} - {int(hr_threshold + 5)} bpm"
            },
            {
                'Zone': 'Z5 - VO2Max',
                'Speed Range': f"above {(1.05 * threshold_value):.2f} m/s",
                'Pace Range': f"faster than {speed_to_pace_100m(1.05 * threshold_value)} min/100m",
                'Heart Rate': f"above {int(hr_threshold + 5)} bpm"
            }
        ]

def format_time_duration(seconds):
    """
    Format seconds to hours:minutes:seconds
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def parse_time_to_seconds(time_str):
    """
    Parse time string in format mm:ss or hh:mm:ss to seconds
    
    Parameters:
    -----------
    time_str : str
        Time string in format mm:ss or hh:mm:ss
    
    Returns:
    --------
    float
        Time in seconds
    """
    parts = time_str.split(':')
    
    if len(parts) == 2:
        # mm:ss format
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        # hh:mm:ss format
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        # Invalid format
        return 0

def time_mm_ss_to_min(time_str):
    """
    Convert mm:ss string to minutes as float
    
    Parameters:
    -----------
    time_str : str
        Time string in format mm:ss
    
    Returns:
    --------
    float
        Time in minutes
    """
    try:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1]) if len(parts) > 1 else 0
        
        return minutes + seconds / 60
    except:
        return 0

def estimate_vo2max(weight, threshold_value, test_type):
    """
    Estimate VO2max based on threshold value
    
    Parameters:
    -----------
    weight : float
        Athlete weight in kg
    threshold_value : float
        Threshold value
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    
    Returns:
    --------
    tuple
        (VO2max in L/min, relative VO2max in ml/min/kg)
    """
    if test_type == "Cycling":
        # Using ACSM formula for cycling
        # VO2 (ml/min/kg) = (10.8 * Watts / Weight) + 7
        rel_vo2max = (10.8 * threshold_value / weight) + 7
        abs_vo2max = rel_vo2max * weight / 1000  # Convert to L/min
        
    elif test_type == "Running":
        # Using ACSM formula for running
        # VO2 (ml/min/kg) = 3.5 + (0.2 * Speed in m/min) + (0.9 * Speed in m/min * Grade)
        # Convert km/h to m/min
        speed_m_min = threshold_value * 1000 / 60
        grade = 0  # Assume flat surface
        rel_vo2max = 3.5 + (0.2 * speed_m_min) + (0.9 * speed_m_min * grade / 100)
        abs_vo2max = rel_vo2max * weight / 1000  # Convert to L/min
        
    elif test_type == "Swimming":
        # Simplified estimation for swimming
        # VO2 (ml/min/kg) = (Speed in m/s * 300) / Weight
        rel_vo2max = threshold_value * 300 / weight
        abs_vo2max = rel_vo2max * weight / 1000  # Convert to L/min
    
    else:
        # Default fallback
        rel_vo2max = 0
        abs_vo2max = 0
    
    return abs_vo2max, rel_vo2max

def calculate_percentiles(threshold_value, weight, test_type, threshold_method):
    """
    Calculate performance percentiles compared to reference populations
    
    Parameters:
    -----------
    threshold_value : float
        Threshold value
    weight : float
        Athlete weight in kg
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    threshold_method : str
        Method used to calculate threshold
    
    Returns:
    --------
    dict
        Dictionary with percentiles for different reference populations
    """
    # Note: These are rough estimates and should be replaced with actual percentile data
    # from sports science literature for more accurate results
    
    # Relative threshold (normalized by weight)
    rel_threshold = threshold_value / weight
    
    # Default percentiles
    percentiles = {
        'general_population': 50,
        'recreational_athletes': 50,
        'competitive_athletes': 50
    }
    
    if test_type == "Cycling":
        # Cycling power-to-weight ratios (W/kg)
        if threshold_method in ['IAT', 'LT']:
            if rel_threshold < 2.0:
                percentiles['general_population'] = 30
                percentiles['recreational_athletes'] = 10
                percentiles['competitive_athletes'] = 5
            elif rel_threshold < 2.5:
                percentiles['general_population'] = 50
                percentiles['recreational_athletes'] = 20
                percentiles['competitive_athletes'] = 10
            elif rel_threshold < 3.0:
                percentiles['general_population'] = 70
                percentiles['recreational_athletes'] = 40
                percentiles['competitive_athletes'] = 20
            elif rel_threshold < 3.5:
                percentiles['general_population'] = 85
                percentiles['recreational_athletes'] = 60
                percentiles['competitive_athletes'] = 30
            elif rel_threshold < 4.0:
                percentiles['general_population'] = 95
                percentiles['recreational_athletes'] = 80
                percentiles['competitive_athletes'] = 50
            else:
                percentiles['general_population'] = 99
                percentiles['recreational_athletes'] = 95
                percentiles['competitive_athletes'] = 70
        
        elif threshold_method in ['Modified Dmax', 'Fixed 4.0 mmol/L', 'OBLA']:
            # Adjust percentiles for other methods
            if rel_threshold < 2.5:
                percentiles['general_population'] = 30
                percentiles['recreational_athletes'] = 10
                percentiles['competitive_athletes'] = 5
            elif rel_threshold < 3.0:
                percentiles['general_population'] = 50
                percentiles['recreational_athletes'] = 20
                percentiles['competitive_athletes'] = 10
            elif rel_threshold < 3.5:
                percentiles['general_population'] = 70
                percentiles['recreational_athletes'] = 40
                percentiles['competitive_athletes'] = 20
            elif rel_threshold < 4.0:
                percentiles['general_population'] = 85
                percentiles['recreational_athletes'] = 60
                percentiles['competitive_athletes'] = 30
            elif rel_threshold < 4.5:
                percentiles['general_population'] = 95
                percentiles['recreational_athletes'] = 80
                percentiles['competitive_athletes'] = 50
            else:
                percentiles['general_population'] = 99
                percentiles['recreational_athletes'] = 95
                percentiles['competitive_athletes'] = 70
    
    elif test_type == "Running":
        # Running speeds (km/h)
        if threshold_value < 10:
            percentiles['general_population'] = 30
            percentiles['recreational_athletes'] = 10
            percentiles['competitive_athletes'] = 5
        elif threshold_value < 12:
            percentiles['general_population'] = 50
            percentiles['recreational_athletes'] = 20
            percentiles['competitive_athletes'] = 10
        elif threshold_value < 14:
            percentiles['general_population'] = 70
            percentiles['recreational_athletes'] = 40
            percentiles['competitive_athletes'] = 20
        elif threshold_value < 16:
            percentiles['general_population'] = 85
            percentiles['recreational_athletes'] = 60
            percentiles['competitive_athletes'] = 30
        elif threshold_value < 18:
            percentiles['general_population'] = 95
            percentiles['recreational_athletes'] = 80
            percentiles['competitive_athletes'] = 50
        else:
            percentiles['general_population'] = 99
            percentiles['recreational_athletes'] = 95
            percentiles['competitive_athletes'] = 70
    
    elif test_type == "Swimming":
        # Swimming speeds (m/s)
        if threshold_value < 0.8:
            percentiles['general_population'] = 30
            percentiles['recreational_athletes'] = 10
            percentiles['competitive_athletes'] = 5
        elif threshold_value < 1.0:
            percentiles['general_population'] = 50
            percentiles['recreational_athletes'] = 20
            percentiles['competitive_athletes'] = 10
        elif threshold_value < 1.2:
            percentiles['general_population'] = 70
            percentiles['recreational_athletes'] = 40
            percentiles['competitive_athletes'] = 20
        elif threshold_value < 1.4:
            percentiles['general_population'] = 85
            percentiles['recreational_athletes'] = 60
            percentiles['competitive_athletes'] = 30
        elif threshold_value < 1.6:
            percentiles['general_population'] = 95
            percentiles['recreational_athletes'] = 80
            percentiles['competitive_athletes'] = 50
        else:
            percentiles['general_population'] = 99
            percentiles['recreational_athletes'] = 95
            percentiles['competitive_athletes'] = 70
    
    return percentiles

def predict_race_performances(threshold_value, test_type):
    """
    Predict race performances based on threshold value
    
    Parameters:
    -----------
    threshold_value : float
        Threshold value
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    
    Returns:
    --------
    dict
        Dictionary with predicted performances for different race distances
    """
    predictions = {}
    
    if test_type == "Cycling":
        # Predict cycling performances (time in minutes for standard distances)
        # These are simplified models and should be replaced with more accurate ones
        
        # 20 km Time Trial
        predictions['20km_tt'] = 60 * 20 / (threshold_value * 0.85 * 0.278)  # km / (W * efficiency * m/s per W)
        
        # 40 km Time Trial
        predictions['40km_tt'] = 60 * 40 / (threshold_value * 0.82 * 0.278)
        
        # 100 km Time Trial
        predictions['100km_tt'] = 60 * 100 / (threshold_value * 0.75 * 0.278)
        
    elif test_type == "Running":
        # Predict running performances (time in minutes for standard distances)
        
        # 5K
        predictions['5k'] = 5 / (threshold_value * 0.90 / 60)
        
        # 10K
        predictions['10k'] = 10 / (threshold_value * 0.85 / 60)
        
        # Half Marathon
        predictions['half_marathon'] = 21.1 / (threshold_value * 0.80 / 60)
        
        # Marathon
        predictions['marathon'] = 42.2 / (threshold_value * 0.75 / 60)
        
    elif test_type == "Swimming":
        # Predict swimming performances (time in minutes for standard distances)
        
        # 100m
        predictions['100m'] = 100 / (threshold_value * 0.95 * 60)
        
        # 400m
        predictions['400m'] = 400 / (threshold_value * 0.90 * 60)
        
        # 1500m
        predictions['1500m'] = 1500 / (threshold_value * 0.85 * 60)
    
    # Format predictions as time strings
    for distance, time_min in predictions.items():
        sec = int(time_min * 60)
        predictions[distance] = format_time_duration(sec)
    
    return predictions

def format_date(date_obj):
    """
    Format date object as string
    
    Parameters:
    -----------
    date_obj : datetime.date
        Date object
    
    Returns:
    --------
    str
        Formatted date string
    """
    if date_obj:
        return date_obj.strftime("%Y-%m-%d")
    return ""

def calculate_bmi(weight, height):
    """
    Calculate Body Mass Index (BMI)
    
    Parameters:
    -----------
    weight : float
        Weight in kg
    height : float
        Height in cm
    
    Returns:
    --------
    float
        BMI value
    """
    if weight <= 0 or height <= 0:
        return 0
    
    # Convert height from cm to m
    height_m = height / 100
    
    # Calculate BMI
    bmi = weight / (height_m * height_m)
    
    return bmi

def get_bmi_category(bmi):
    """
    Get BMI category
    
    Parameters:
    -----------
    bmi : float
        BMI value
    
    Returns:
    --------
    str
        BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def calculate_age(birth_date):
    """
    Calculate age from birth date
    
    Parameters:
    -----------
    birth_date : datetime.date
        Birth date
    
    Returns:
    --------
    int
        Age in years
    """
    if not birth_date:
        return 0
    
    today = datetime.now().date()
    age = today.year - birth_date.year
    
    # Check if birthday has occurred this year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    
    return age

def calculate_max_hr(age, gender="male"):
    """
    Calculate estimated maximum heart rate
    
    Parameters:
    -----------
    age : int
        Age in years
    gender : str
        Gender ('male' or 'female')
    
    Returns:
    --------
    int
        Estimated maximum heart rate
    """
    if gender.lower() == "female":
        # Gulati formula for women
        max_hr = 206 - (0.88 * age)
    else:
        # Tanaka formula for men
        max_hr = 208 - (0.7 * age)
    
    return int(max_hr)

def parse_csv_data(csv_data):
    """
    Parse CSV data into a pandas DataFrame
    
    Parameters:
    -----------
    csv_data : str
        CSV data as string
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the CSV data
    """
    import pandas as pd
    import io
    
    try:
        # Try to parse the CSV data
        df = pd.read_csv(io.StringIO(csv_data))
        return df
    except Exception as e:
        print(f"Error parsing CSV data: {e}")
        return None

def export_data_to_csv(df):
    """
    Export DataFrame to CSV string
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to export
    
    Returns:
    --------
    str
        CSV data as string
    """
    import pandas as pd
    import io
    
    try:
        # Export DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")
        return ""