import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d

def generate_plots(df, test_type, threshold_results):
    """
    Generate matplotlib and plotly figures for lactate threshold data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    threshold_results : dict
        Dictionary of threshold results from threshold_methods.py
    
    Returns:
    --------
    None
        Stores figures in session state
    """
    import streamlit as st
    
    # Remove rest values (step 0) for plotting
    df_plot = df[df['step'] > 0].copy()
    
    # Get intensity column name based on test type
    if test_type == "Cycling":
        intensity_col = "load_watts"
        intensity_label = "Power (Watts)"
    elif test_type == "Running":
        intensity_col = "speed_kmh"
        intensity_label = "Speed (km/h)"
    else:  # Swimming
        intensity_col = "speed_ms"
        intensity_label = "Speed (m/s)"
    
    # Store in session state for reference by other functions
    st.session_state['intensity_col'] = intensity_col
    st.session_state['intensity_label'] = intensity_label
    
    # Generate matplotlib figure for PDF export
    matplotlib_fig = generate_matplotlib_plot(df_plot, intensity_col, intensity_label, threshold_results)
    
    # Generate interactive plotly figure
    plotly_fig = generate_plotly_plot(df_plot, intensity_col, intensity_label, threshold_results)
    
    # Generate training zones figure
    plotly_zones = add_training_zones(df_plot, test_type, threshold_results)
    
    # Store figures in session state
    st.session_state['matplotlib_fig'] = matplotlib_fig
    st.session_state['plotly_fig'] = plotly_fig
    st.session_state['plotly_zones'] = plotly_zones

def generate_matplotlib_plot(df_plot, intensity_col, intensity_label, threshold_results):
    """
    Generate matplotlib figure for PDF export
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    intensity_col : str
        Column name for intensity values
    intensity_label : str
        Label for intensity axis
    threshold_results : dict
        Dictionary of threshold results
    
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure for PDF export
    """
    # Create figure and axes
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
    for method, result in threshold_results.items():
        if method == "Modified Dmax":
            # Get the indices for plotting
            first_idx = result.get('first_idx', 0)
            max_idx = result.get('max_idx', 0)
            
            # Plot the line connecting first and last point
            if first_idx < len(df_plot) and 'first_idx' in result:
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
    
    # Improve figure layout
    fig.tight_layout()
    
    return fig

def generate_plotly_plot(df_plot, intensity_col, intensity_label, threshold_results):
    """
    Generate interactive plotly figure
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    intensity_col : str
        Column name for intensity values
    intensity_label : str
        Label for intensity axis
    threshold_results : dict
        Dictionary of threshold results
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure
    """
    # Create figure
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
    # Add thresholds
    for method, result in threshold_results.items():
        if method == "Modified Dmax":
            # Get the indices for plotting
            first_idx = result.get('first_idx', 0)
            max_idx = result.get('max_idx', 0)
            
            # Add line connecting first point and last point
            if first_idx < len(df_plot) and 'first_idx' in result:
                fig.add_trace(go.Scatter(
                    x=[df_plot.iloc[first_idx][intensity_col], df_plot.iloc[-1][intensity_col]],
                    y=[df_plot.iloc[first_idx]['lactate'], df_plot.iloc[-1]['lactate']],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=False
                ))
            
            # Add threshold point
            fig.add_trace(go.Scatter(
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
    
    return fig

def add_training_zones(df_plot, test_type, threshold_results):
    """
    Generate training zones visualization
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    threshold_results : dict
        Dictionary of threshold results
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive plotly figure with training zones
    """
    import streamlit as st
    
    # Get intensity column
    intensity_col = st.session_state.get('intensity_col')
    intensity_label = st.session_state.get('intensity_label')
    
    if not intensity_col or not intensity_label:
        if test_type == "Cycling":
            intensity_col = "load_watts"
            intensity_label = "Power (Watts)"
        elif test_type == "Running":
            intensity_col = "speed_kmh"
            intensity_label = "Speed (km/h)"
        else:  # Swimming
            intensity_col = "speed_ms"
            intensity_label = "Speed (m/s)"
    
    # Determine training zones based on test type and threshold calculations
    zones = calculate_training_zones(df_plot, test_type, intensity_col, threshold_results)
    
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
    for method, result in threshold_results.items():
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

def calculate_training_zones(df_plot, test_type, intensity_col, threshold_results):
    """
    Calculate training zones based on threshold results
    
    Parameters:
    -----------
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    intensity_col : str
        Column name for intensity values
    threshold_results : dict
        Dictionary of threshold results
    
    Returns:
    --------
    list
        List of zone dictionaries with min, max, name, color
    """
    # Set default zone colors
    zone_colors = {
        'Z1': 'rgba(173, 216, 230, 0.3)',  # Light blue
        'Z2': 'rgba(255, 255, 150, 0.3)',  # Light yellow
        'Z3': 'rgba(255, 200, 150, 0.3)',  # Light orange
        'Z4': 'rgba(255, 150, 150, 0.3)',  # Light red
        'Z5': 'rgba(200, 120, 120, 0.3)'   # Darker red
    }
    
    # Find primary threshold method for zone calculation
    primary_method = determine_primary_threshold_method(threshold_results)
    
    # Set default empty zones list
    zones = []
    
    if primary_method:
        # Get threshold value and heart rate
        threshold_value = threshold_results[primary_method]['value']
        hr_threshold = threshold_results[primary_method]['hr']
        
        # Calculate zones based on test type and threshold method
        if primary_method == 'IAT':
            zones = calculate_iat_zones(test_type, threshold_value, hr_threshold, intensity_col, df_plot, zone_colors)
        elif primary_method == 'Modified Dmax':
            zones = calculate_mdmax_zones(test_type, threshold_value, hr_threshold, intensity_col, df_plot, zone_colors)
        elif 'Fixed 4.0 mmol/L' in primary_method:
            zones = calculate_fixed_zones(test_type, threshold_value, hr_threshold, intensity_col, df_plot, zone_colors)
    
    # Return calculated zones
    return zones

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

def calculate_iat_zones(test_type, iat_value, hr_iat, intensity_col, df_plot, zone_colors):
    """
    Calculate training zones based on IAT threshold
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    iat_value : float
        IAT threshold value
    hr_iat : int
        Heart rate at IAT
    intensity_col : str
        Column name for intensity values
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    zone_colors : dict
        Dictionary of zone colors
    
    Returns:
    --------
    list
        List of zone dictionaries
    """
    if test_type == "Cycling":
        # Z1: Recovery - below 70% of IAT
        # Z2: Endurance - 70-95% of IAT
        # Z3: Tempo/Sweetspot - 95-105% of IAT
        # Z4: Threshold - 105-120% of IAT
        # Z5: VO2Max - above 120% of IAT
        return [
            {'name': 'Z1', 'min': 0, 'max': 0.7 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 25, 'color': zone_colors['Z1']},
            {'name': 'Z2', 'min': 0.7 * iat_value, 'max': 0.95 * iat_value, 'hr_min': hr_iat - 25, 'hr_max': hr_iat - 10, 'color': zone_colors['Z2']},
            {'name': 'Z3', 'min': 0.95 * iat_value, 'max': 1.05 * iat_value, 'hr_min': hr_iat - 10, 'hr_max': hr_iat + 5, 'color': zone_colors['Z3']},
            {'name': 'Z4', 'min': 1.05 * iat_value, 'max': 1.2 * iat_value, 'hr_min': hr_iat + 5, 'hr_max': hr_iat + 15, 'color': zone_colors['Z4']},
            {'name': 'Z5', 'min': 1.2 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 15, 'hr_max': 220, 'color': zone_colors['Z5']}
        ]
    
    elif test_type == "Running":
        # Similar zones adjusted for running
        return [
            {'name': 'Z1', 'min': 0, 'max': 0.75 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 20, 'color': zone_colors['Z1']},
            {'name': 'Z2', 'min': 0.75 * iat_value, 'max': 0.95 * iat_value, 'hr_min': hr_iat - 20, 'hr_max': hr_iat - 5, 'color': zone_colors['Z2']},
            {'name': 'Z3', 'min': 0.95 * iat_value, 'max': 1.05 * iat_value, 'hr_min': hr_iat - 5, 'hr_max': hr_iat + 8, 'color': zone_colors['Z3']},
            {'name': 'Z4', 'min': 1.05 * iat_value, 'max': 1.15 * iat_value, 'hr_min': hr_iat + 8, 'hr_max': hr_iat + 18, 'color': zone_colors['Z4']},
            {'name': 'Z5', 'min': 1.15 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 18, 'hr_max': 220, 'color': zone_colors['Z5']}
        ]
    
    elif test_type == "Swimming":
        # Similar zones adjusted for swimming
        return [
            {'name': 'Z1', 'min': 0, 'max': 0.7 * iat_value, 'hr_min': 0, 'hr_max': hr_iat - 20, 'color': zone_colors['Z1']},
            {'name': 'Z2', 'min': 0.7 * iat_value, 'max': 0.9 * iat_value, 'hr_min': hr_iat - 20, 'hr_max': hr_iat - 5, 'color': zone_colors['Z2']},
            {'name': 'Z3', 'min': 0.9 * iat_value, 'max': 1.03 * iat_value, 'hr_min': hr_iat - 5, 'hr_max': hr_iat + 8, 'color': zone_colors['Z3']},
            {'name': 'Z4', 'min': 1.03 * iat_value, 'max': 1.12 * iat_value, 'hr_min': hr_iat + 8, 'hr_max': hr_iat + 18, 'color': zone_colors['Z4']},
            {'name': 'Z5', 'min': 1.12 * iat_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_iat + 18, 'hr_max': 220, 'color': zone_colors['Z5']}
        ]

def calculate_mdmax_zones(test_type, mdmax_value, hr_mdmax, intensity_col, df_plot, zone_colors):
    """
    Calculate training zones based on Modified Dmax threshold
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    mdmax_value : float
        Modified Dmax threshold value
    hr_mdmax : int
        Heart rate at Modified Dmax
    intensity_col : str
        Column name for intensity values
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    zone_colors : dict
        Dictionary of zone colors
    
    Returns:
    --------
    list
        List of zone dictionaries
    """
    if test_type == "Cycling":
        # Similar to IAT zones but with different percentages
        return [
            {'name': 'Z1', 'min': 0, 'max': 0.65 * mdmax_value, 'hr_min': 0, 'hr_max': hr_mdmax - 25, 'color': zone_colors['Z1']},
            {'name': 'Z2', 'min': 0.65 * mdmax_value, 'max': 0.9 * mdmax_value, 'hr_min': hr_mdmax - 25, 'hr_max': hr_mdmax - 10, 'color': zone_colors['Z2']},
            {'name': 'Z3', 'min': 0.9 * mdmax_value, 'max': 1.0 * mdmax_value, 'hr_min': hr_mdmax - 10, 'hr_max': hr_mdmax, 'color': zone_colors['Z3']},
            {'name': 'Z4', 'min': 1.0 * mdmax_value, 'max': 1.1 * mdmax_value, 'hr_min': hr_mdmax, 'hr_max': hr_mdmax + 10, 'color': zone_colors['Z4']},
            {'name': 'Z5', 'min': 1.1 * mdmax_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_mdmax + 10, 'hr_max': 220, 'color': zone_colors['Z5']}
        ]
    else:
        # Similar for running and swimming
        return [
            {'name': 'Z1', 'min': 0, 'max': 0.7 * mdmax_value, 'hr_min': 0, 'hr_max': hr_mdmax - 20, 'color': zone_colors['Z1']},
            {'name': 'Z2', 'min': 0.7 * mdmax_value, 'max': 0.9 * mdmax_value, 'hr_min': hr_mdmax - 20, 'hr_max': hr_mdmax - 5, 'color': zone_colors['Z2']},
            {'name': 'Z3', 'min': 0.9 * mdmax_value, 'max': 1.0 * mdmax_value, 'hr_min': hr_mdmax - 5, 'hr_max': hr_mdmax, 'color': zone_colors['Z3']},
            {'name': 'Z4', 'min': 1.0 * mdmax_value, 'max': 1.1 * mdmax_value, 'hr_min': hr_mdmax, 'hr_max': hr_mdmax + 10, 'color': zone_colors['Z4']},
            {'name': 'Z5', 'min': 1.1 * mdmax_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_mdmax + 10, 'hr_max': 220, 'color': zone_colors['Z5']}
        ]

def calculate_fixed_zones(test_type, threshold_value, hr_threshold, intensity_col, df_plot, zone_colors):
    """
    Calculate training zones based on fixed threshold (e.g., 4.0 mmol/L)
    
    Parameters:
    -----------
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    threshold_value : float
        Fixed threshold value
    hr_threshold : int
        Heart rate at threshold
    intensity_col : str
        Column name for intensity values
    df_plot : pandas.DataFrame
        Dataframe containing test data for plotting
    zone_colors : dict
        Dictionary of zone colors
    
    Returns:
    --------
    list
        List of zone dictionaries
    """
    # Zones based on fixed threshold value
    return [
        {'name': 'Z1', 'min': 0, 'max': 0.6 * threshold_value, 'hr_min': 0, 'hr_max': hr_threshold - 30, 'color': zone_colors['Z1']},
        {'name': 'Z2', 'min': 0.6 * threshold_value, 'max': 0.85 * threshold_value, 'hr_min': hr_threshold - 30, 'hr_max': hr_threshold - 15, 'color': zone_colors['Z2']},
        {'name': 'Z3', 'min': 0.85 * threshold_value, 'max': 0.95 * threshold_value, 'hr_min': hr_threshold - 15, 'hr_max': hr_threshold - 5, 'color': zone_colors['Z3']},
        {'name': 'Z4', 'min': 0.95 * threshold_value, 'max': 1.05 * threshold_value, 'hr_min': hr_threshold - 5, 'hr_max': hr_threshold + 5, 'color': zone_colors['Z4']},
        {'name': 'Z5', 'min': 1.05 * threshold_value, 'max': max(df_plot[intensity_col]) * 1.1, 'hr_min': hr_threshold + 5, 'hr_max': 220, 'color': zone_colors['Z5']}
    ]

def highlight_threshold_region(fig, threshold_value, lactate_value, label, color='rgba(255,200,200,0.3)'):
    """
    Add a highlighted region around threshold point
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add highlight to
    threshold_value : float
        Threshold value on x-axis
    lactate_value : float
        Lactate value at threshold
    label : str
        Label for annotation
    color : str
        Color for highlight region
    
    Returns:
    --------
    None
        Modifies fig in place
    """
    # Add a rectangular highlight
    span = threshold_value * 0.05  # 5% of threshold value
    
    fig.add_shape(
        type="rect",
        x0=threshold_value - span,
        x1=threshold_value + span,
        y0=0,
        y1=lactate_value * 1.5,
        fillcolor=color,
        line=dict(width=0),
        layer="below"
    )
    
    # Add annotation
    fig.add_annotation(
        x=threshold_value,
        y=lactate_value * 1.2,
        text=label,
        showarrow=True,
        arrowhead=2,
        arrowcolor="#636363",
        arrowwidth=2,
        ax=0,
        ay=-30
    )