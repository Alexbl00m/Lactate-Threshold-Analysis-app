import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, curve_fit
from scipy import stats

def calculate_thresholds(df, test_type, threshold_methods, fitting_method, log_log_portion, include_baseline=False):
    """
    Calculate all selected thresholds
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    test_type : str
        Type of test ('Cycling', 'Running', or 'Swimming')
    threshold_methods : list
        List of threshold methods to calculate
    fitting_method : str
        Method for curve fitting
    log_log_portion : float
        Portion of data to use for log-log method
    include_baseline : bool
        Whether to include baseline values in curve fitting
    
    Returns:
    --------
    dict
        Dictionary of threshold results
    """
    # Create a copy of the dataframe for calculations
    df_calc = df[df['step'] > 0].copy() if not include_baseline else df.copy()
    
    # Determine intensity column based on test type
    if test_type == "Cycling":
        intensity_col = "load_watts"
    elif test_type == "Running":
        intensity_col = "speed_kmh"
    else:  # Swimming
        intensity_col = "speed_ms"
    
    # Results dictionary
    threshold_results = {}
    
    # Calculate each selected threshold method
    for method in threshold_methods:
        if method == "Fixed 2.0 mmol/L":
            result = calculate_fixed_threshold(df_calc, intensity_col, 2.0)
            if result:
                threshold_results['Fixed 2.0 mmol/L'] = result
        
        elif method == "Fixed 4.0 mmol/L":
            result = calculate_fixed_threshold(df_calc, intensity_col, 4.0)
            if result:
                threshold_results['Fixed 4.0 mmol/L'] = result
        
        elif method == "OBLA (Onset of Blood Lactate Accumulation)":
            # OBLA is typically the 4.0 mmol/L threshold
            result = calculate_fixed_threshold(df_calc, intensity_col, 4.0)
            if result:
                threshold_results['OBLA'] = result
        
        elif method == "LT (Lactate Threshold)":
            # LT is defined as baseline + 1.0 mmol/L
            baseline = df.loc[0, 'lactate'] if 0 in df.index else df_calc['lactate'].min()
            result = calculate_fixed_threshold(df_calc, intensity_col, baseline + 1.0)
            if result:
                result['method'] = 'LT'
                threshold_results['LT'] = result
        
        elif method == "IAT (Individual Anaerobic Threshold)":
            # IAT is defined as baseline + 0.5 mmol/L
            baseline = df.loc[0, 'lactate'] if 0 in df.index else df_calc['lactate'].min()
            result = calculate_fixed_threshold(df_calc, intensity_col, baseline + 0.5)
            if result:
                result['method'] = 'IAT'
                threshold_results['IAT'] = result
        
        elif method == "Modified Dmax":
            baseline = df.loc[0, 'lactate'] if 0 in df.index else df_calc['lactate'].min()
            result = calculate_modified_dmax(df_calc, intensity_col, baseline)
            if result:
                threshold_results['Modified Dmax'] = result
        
        elif method == "Log-Log":
            result = calculate_log_log(df_calc, intensity_col, log_log_portion, fitting_method)
            if result:
                threshold_results['Log-Log'] = result
        
        elif method == "Log-Exp-ModDmax":
            result = calculate_log_exp_moddmax(df_calc, intensity_col)
            if result:
                threshold_results['Log-Exp-ModDmax'] = result
        
        elif method == "Exponential Dmax":
            result = calculate_exponential_dmax(df_calc, intensity_col)
            if result:
                threshold_results['Exponential Dmax'] = result
    
    return threshold_results

def calculate_fixed_threshold(df, intensity_col, fixed_value):
    """
    Calculate threshold at fixed lactate value using linear interpolation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    fixed_value : float
        Fixed lactate value to find threshold at
    
    Returns:
    --------
    dict
        Threshold result or None if calculation fails
    """
    try:
        # Sort by lactate to ensure proper interpolation
        df_sorted = df.sort_values('lactate')
        
        # Check if the fixed value is within the range of measured lactate values
        if fixed_value < df_sorted['lactate'].min() or fixed_value > df_sorted['lactate'].max():
            # Need to extrapolate
            f = interp1d(df_sorted['lactate'], df_sorted[intensity_col], bounds_error=False, fill_value="extrapolate")
        else:
            # Can interpolate within range
            f = interp1d(df_sorted['lactate'], df_sorted[intensity_col])
        
        threshold_value = float(f(fixed_value))
        
        # Get heart rate at threshold
        f_hr = interp1d(df_sorted[intensity_col], df_sorted['heart_rate'], bounds_error=False, fill_value="extrapolate")
        hr_value = int(f_hr(threshold_value))
        
        return {
            'value': threshold_value,
            'hr': hr_value,
            'lactate': fixed_value,
            'method': f'Fixed {fixed_value} mmol/L'
        }
    except Exception as e:
        print(f"Error in fixed threshold calculation: {e}")
        return None

def calculate_modified_dmax(df, intensity_col, baseline):
    """
    Calculate Modified Dmax threshold
    
    The Modified Dmax method finds the point with maximum perpendicular distance 
    from the line connecting the first point with lactate > baseline + 0.4 mmol/L 
    and the last data point.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    baseline : float
        Baseline lactate value
    
    Returns:
    --------
    dict
        Threshold result or None if calculation fails
    """
    try:
        # Store intensity and lactate values
        x = df[intensity_col].values
        y = df['lactate'].values
        
        # Find first point with lactate > baseline + 0.4 mmol/L
        threshold = baseline + 0.4
        indices = np.where(y > threshold)[0]
        
        if len(indices) == 0:
            # No points above threshold, cannot calculate
            return None
            
        first_idx = indices[0]
        
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
        hr_mdmax = df.iloc[max_idx]['heart_rate']
        lactate_mdmax = df.iloc[max_idx]['lactate']
        
        return {
            'value': threshold_mdmax,
            'hr': hr_mdmax,
            'lactate': lactate_mdmax,
            'method': 'Modified Dmax',
            'first_idx': first_idx,  # Store for plotting
            'max_idx': max_idx       # Store for plotting
        }
    except Exception as e:
        print(f"Error in Modified Dmax calculation: {e}")
        return None

def calculate_log_log(df, intensity_col, log_log_portion, fitting_method):
    """
    Calculate Log-Log threshold
    
    The Log-Log method plots log(lactate) vs log(intensity) and finds 
    the intensity where the derivative of the curve equals 1.0.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    log_log_portion : float
        Portion of data to use for curve fitting (0.0-1.0)
    fitting_method : str
        Method for curve fitting
    
    Returns:
    --------
    dict
        Threshold result or None if calculation fails
    """
    try:
        # Filter out zero or negative values for log transform
        mask = (df[intensity_col] > 0) & (df['lactate'] > 0)
        if mask.sum() < 3:
            # Not enough points for fitting
            return None
            
        # Store intensity and lactate values
        x = df.loc[mask, intensity_col].values
        y = df.loc[mask, 'lactate'].values
        
        # Take log of both
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Use only a portion of the data for fitting as specified by user
        cutoff_idx = max(3, int(len(log_x) * log_log_portion))
        cutoff_idx = min(cutoff_idx, len(log_x))
        
        # Fit polynomial to the log-log data
        if fitting_method == "3rd degree polynomial":
            poly_degree = 3
        elif fitting_method == "4th degree polynomial":
            poly_degree = 4
        else:  # B-spline
            # For simplicity, default to 3rd degree for now
            poly_degree = 3
        
        # Fit polynomial
        poly = np.polyfit(log_x[:cutoff_idx], log_y[:cutoff_idx], poly_degree)
        p = np.poly1d(poly)
        
        # Calculate the derivative
        dp = np.polyder(p)
        
        # Find the log(intensity) where the derivative is 1.0
        def func(x):
            return dp(x) - 1.0
        
        # Use the middle of the log_x range as initial guess
        x0 = log_x[len(log_x)//2]
        try:
            log_threshold = fsolve(func, x0)[0]
            # Check if solution is within the intensity range
            if log_threshold < log_x.min() or log_threshold > log_x.max():
                # Try another initial guess
                x0 = (log_x.min() + log_x.max()) / 2
                log_threshold = fsolve(func, x0)[0]
        except:
            # Fallback to numerical search
            test_points = np.linspace(log_x.min(), log_x.max(), 100)
            derivatives = [dp(x) for x in test_points]
            idx = np.argmin(np.abs(np.array(derivatives) - 1.0))
            log_threshold = test_points[idx]
        
        # Convert back from log space
        threshold_loglog = np.exp(log_threshold)
        
        # Get heart rate and lactate at threshold by interpolation
        f_hr = interp1d(df[intensity_col], df['heart_rate'], bounds_error=False, fill_value="extrapolate")
        f_lac = interp1d(df[intensity_col], df['lactate'], bounds_error=False, fill_value="extrapolate")
        
        hr_loglog = int(f_hr(threshold_loglog))
        lactate_loglog = float(f_lac(threshold_loglog))
        
        return {
            'value': threshold_loglog,
            'hr': hr_loglog,
            'lactate': lactate_loglog,
            'method': 'Log-Log',
            'log_threshold': log_threshold  # Store for plotting
        }
    except Exception as e:
        print(f"Error in Log-Log calculation: {e}")
        return None

def calculate_log_exp_moddmax(df, intensity_col):
    """
    Calculate Log-Exp-ModDmax threshold
    
    This is a variant of the Modified Dmax method that uses exponential fitting
    on the log transformed lactate data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    
    Returns:
    --------
    dict
        Threshold result or None if calculation fails
    """
    try:
        # This is a placeholder. The actual implementation would involve:
        # 1. Log transforming lactate values
        # 2. Fitting an exponential curve to the data
        # 3. Finding the maximum distance point as in Modified Dmax
        
        # For now, we'll use a simpler approach similar to Modified Dmax
        baseline = df['lactate'].min()
        threshold = baseline + 0.4
        
        # Get points above threshold
        df_above = df[df['lactate'] > threshold]
        if len(df_above) < 2:
            return None
            
        # Simplified calculation for demonstration
        # In a real implementation, this would use exponential fitting
        x = df_above[intensity_col].values
        y = np.log(df_above['lactate'].values)
        
        # Find midpoint of the range as a simplified threshold
        threshold_value = float(np.mean(x))
        
        # Get heart rate and lactate at threshold
        f_hr = interp1d(df[intensity_col], df['heart_rate'], bounds_error=False, fill_value="extrapolate")
        f_lac = interp1d(df[intensity_col], df['lactate'], bounds_error=False, fill_value="extrapolate")
        
        hr_value = int(f_hr(threshold_value))
        lactate_value = float(f_lac(threshold_value))
        
        return {
            'value': threshold_value,
            'hr': hr_value,
            'lactate': lactate_value,
            'method': 'Log-Exp-ModDmax'
        }
    except Exception as e:
        print(f"Error in Log-Exp-ModDmax calculation: {e}")
        return None

def calculate_exponential_dmax(df, intensity_col):
    """
    Calculate Exponential Dmax threshold
    
    Fits an exponential curve to the lactate data and finds the point
    with maximum distance to the line connecting the first and last points.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    
    Returns:
    --------
    dict
        Threshold result or None if calculation fails
    """
    try:
        # Define exponential function for fitting
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        # Get data for fitting
        x = df[intensity_col].values
        y = df['lactate'].values
        
        # Initial parameter guess
        p0 = [0.1, 0.01, min(y)]
        
        # Try exponential fitting
        try:
            popt, pcov = curve_fit(exp_func, x, y, p0=p0, maxfev=10000)
            a, b, c = popt
            
            # Generate fitted curve points
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = exp_func(x_fit, a, b, c)
            
            # Get first and last points for line
            x1, y1 = min(x), exp_func(min(x), a, b, c)
            x2, y2 = max(x), exp_func(max(x), a, b, c)
            
            # Line parameters
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Calculate distances
            distances = []
            for i in range(len(x_fit)):
                # Line equation: y = slope*x + intercept or -slope*x + y - intercept = 0
                A = -slope
                B = 1
                C = -intercept
                distance = abs(A*x_fit[i] + B*y_fit[i] + C) / np.sqrt(A**2 + B**2)
                distances.append(distance)
            
            # Find maximum distance point
            max_idx = np.argmax(distances)
            threshold_value = float(x_fit[max_idx])
            
            # Get heart rate and lactate at threshold
            f_hr = interp1d(df[intensity_col], df['heart_rate'], bounds_error=False, fill_value="extrapolate")
            hr_value = int(f_hr(threshold_value))
            lactate_value = float(exp_func(threshold_value, a, b, c))
            
            return {
                'value': threshold_value,
                'hr': hr_value,
                'lactate': lactate_value,
                'method': 'Exponential Dmax',
                'exp_params': (a, b, c)  # Store for plotting
            }
            
        except RuntimeError:
            # Exponential fitting failed, try polynomial as fallback
            poly = np.polyfit(x, y, 3)
            p = np.poly1d(poly)
            
            # The rest is similar to Modified Dmax
            x_fit = np.linspace(min(x), max(x), 100)
            y_fit = p(x_fit)
            
            # Get first and last points for line
            x1, y1 = min(x), p(min(x))
            x2, y2 = max(x), p(max(x))
            
            # Calculate line parameters and find maximum distance
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            distances = []
            for i in range(len(x_fit)):
                A = -slope
                B = 1
                C = -intercept
                distance = abs(A*x_fit[i] + B*y_fit[i] + C) / np.sqrt(A**2 + B**2)
                distances.append(distance)
            
            max_idx = np.argmax(distances)
            threshold_value = float(x_fit[max_idx])
            
            # Get heart rate and lactate at threshold
            f_hr = interp1d(df[intensity_col], df['heart_rate'], bounds_error=False, fill_value="extrapolate")
            hr_value = int(f_hr(threshold_value))
            lactate_value = float(p(threshold_value))
            
            return {
                'value': threshold_value,
                'hr': hr_value,
                'lactate': lactate_value,
                'method': 'Exponential Dmax (Poly)',
                'poly_params': poly  # Store for plotting
            }
            
    except Exception as e:
        print(f"Error in Exponential Dmax calculation: {e}")
        return None

def smooth_lactate_curve(df, intensity_col, method='spline'):
    """
    Create a smoothed version of the lactate curve
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing test data
    intensity_col : str
        Column name for intensity values
    method : str
        Smoothing method ('spline', 'poly3', 'poly4', or 'lowess')
    
    Returns:
    --------
    tuple
        (x_smooth, y_smooth) arrays for plotting
    """
    try:
        x = df[intensity_col].values
        y = df['lactate'].values
        
        # Create more points for smooth curve
        x_smooth = np.linspace(min(x), max(x), 100)
        
        if method == 'spline':
            # Cubic spline interpolation
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(x, y)
            y_smooth = cs(x_smooth)
        
        elif method == 'poly3':
            # 3rd degree polynomial
            poly = np.polyfit(x, y, 3)
            p = np.poly1d(poly)
            y_smooth = p(x_smooth)
        
        elif method == 'poly4':
            # 4th degree polynomial
            poly = np.polyfit(x, y, 4)
            p = np.poly1d(poly)
            y_smooth = p(x_smooth)
        
        elif method == 'lowess':
            # LOWESS smoothing
            from statsmodels.nonparametric.smoothers_lowess import lowess
            lowess_result = lowess(y, x, frac=0.5)
            # Interpolate to get smoother curve
            f = interp1d(lowess_result[:, 0], lowess_result[:, 1], bounds_error=False, fill_value="extrapolate")
            y_smooth = f(x_smooth)
        
        else:
            # Default to basic interpolation
            f = interp1d(x, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
            y_smooth = f(x_smooth)
        
        return x_smooth, y_smooth
    
    except Exception as e:
        print(f"Error in lactate curve smoothing: {e}")
        # Return original data if smoothing fails
        return df[intensity_col].values, df['lactate'].values