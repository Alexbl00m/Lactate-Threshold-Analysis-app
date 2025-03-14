# Lactate Threshold Analysis Tool

A comprehensive web application for analyzing lactate threshold tests for cycling, running, and swimming. The tool uses various methods to determine physiological thresholds and generates training zones based on the results.

## Features

- Support for cycling (power-based), running (pace-based), and swimming (speed-based) tests
- Multiple threshold calculation methods:
  - Fixed 2.0 mmol/L
  - Fixed 4.0 mmol/L
  - OBLA (Onset of Blood Lactate Accumulation)
  - LT (Lactate Threshold)
  - IAT (Individual Anaerobic Threshold)
  - Modified Dmax
  - Log-Log
  - Log-Exp-ModDmax
  - Exponential Dmax
- Interactive data input interface
- Visualization of lactate curves and thresholds
- Training zone calculations specific to each sport
- PDF report generation with test results
- Responsive design with custom branding

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lactate-threshold-analysis.git
   cd lactate-threshold-analysis
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Select the sport type (Cycling, Running, or Swimming)
2. Enter athlete information
3. Set up the test protocol parameters
4. Input the measured data (intensity, heart rate, lactate)
5. Calculate thresholds
6. View results, visualize data, and export PDF reports

## Requirements

- Python 3.8+
- Libraries listed in requirements.txt

## Customization

- Place your logo in the root directory as "logo.png"
- The app uses Montserrat font for PDF reports. Place font files in the root directory:
  - Montserrat-Regular.ttf
  - Montserrat-Bold.ttf

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Developed for Lindblom Coaching
- Uses scientific threshold detection methods based on sports science literature