import streamlit as st
import base64
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import pickle
import time

# --- PAGE CONFIG: must be first Streamlit command ---
st.set_page_config(
    page_title="Erdrutsch-Simulation / Landslide Simulation",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Place all language-related code (functions and dictionaries) at the very top of the file, before any usage
# 1. get_parameter_explanation, get_educational_content, get_parameter_explanation_en, get_educational_content_en
# 2. LANGUAGES dictionary
# 3. Language selector and labels assignment

# --- LANGUAGE FUNCTIONS ---
def get_parameter_explanation():
    """Gibt Erklärungen zu den physikalischen Parametern zurück."""
    erklaerung = """
    ### Verständnis der physikalischen Parameter
    
    #### Coulomb-Reibung (μ)
    Dieser Parameter beschreibt die grundlegende Reibung zwischen dem Erdrutschmaterial und der Hangoberfläche.
    - **Höhere Werte** (näher an 0,5) bedeuten mehr Reibung, was zu kürzeren Auslaufweiten und langsameren Erdrutschen führt
    - **Niedrigere Werte** (näher an 0,01) stehen für rutschigere Bedingungen, wodurch der Erdrutsch schneller und weiter läuft
    
    #### Turbulente Reibung (ξ)
    Dieser Parameter beschreibt den Widerstand durch Turbulenzen im fließenden Erdrutschmaterial.
    - **Höhere Werte** (näher an 2200) bedeuten mehr innere Turbulenz und beeinflussen das Fließverhalten
    - **Niedrigere Werte** (näher an 100) stehen für laminare Fließbedingungen
    
    Die Kombination dieser Parameter bestimmt das Verhalten des Erdrutsches in den Simulationen.
    """
    return erklaerung

def get_educational_content():
    """Gibt Lerninhalte zu Erdrutschen zurück."""
    inhalt = """
    ### Erdrutsche: Mächtige Naturkräfte
    
    Erdrutsche gehören zu den mächtigsten und zerstörerischsten Naturereignissen. Sie entstehen, wenn Massen aus Gestein, Erde oder Schutt aufgrund der Schwerkraft einen Hang hinabgleiten.
    
    #### Wichtige Einflussfaktoren:
    1. **Hangneigung** – Steilere Hänge sind instabiler
    2. **Materialeigenschaften** – Reibung und Kohäsion bestimmen die Stabilität
    3. **Wassergehalt** – Erhöht das Gewicht und verringert die Reibung
    4. **Auslöser** – Erdbeben, starke Regenfälle, menschliche Aktivitäten
    
    Mit dieser Simulation kannst du erforschen, wie verschiedene Reibungsbedingungen das Verhalten von Erdrutschen beeinflussen – wichtig für Gefahrenabschätzung und Risikovermeidung in gefährdeten Gebieten.
    """
    return inhalt

def get_parameter_explanation_en():
    explanation = """
    ### Understanding the Physics Parameters
    
    #### Coulomb Friction (μ)
    This parameter represents the basic friction between the landslide material and the slope surface.
    - **Higher values** (closer to 0.5) create more friction, resulting in shorter runout distances and slower landslides
    - **Lower values** (closer to 0.01) represent more slippery conditions, leading to faster, farther-traveling landslides
    
    #### Turbulent Friction (ξ)
    This parameter represents resistance due to turbulence within the flowing landslide material.
    - **Higher values** (closer to 2200) indicate more internal turbulence, affecting how the material flows
    - **Lower values** (closer to 100) represent more laminar flow conditions
    
    The combination of these parameters determines the landslide behavior you observe in the simulations.
    """
    return explanation

def get_educational_content_en():
    content = """
    ### Landslides: Nature's Powerful Forces
    
    Landslides are one of nature's most powerful and destructive forces. They occur when masses of rock, earth, or debris move down a slope due to gravity.
    
    #### Key Factors Affecting Landslides:
    1. **Slope Angle** - Steeper slopes are more likely to fail
    2. **Material Properties** - Friction and cohesion determine stability
    3. **Water Content** - Increases weight and reduces friction
    4. **Triggers** - Earthquakes, rainfall, human activities
    
    This simulation allows you to explore how different friction conditions affect landslide behavior, which is crucial for hazard assessment and risk mitigation in vulnerable areas.
    """
    return content

# --- LANGUAGE DICTIONARIES ---
LANGUAGES = {
    "de": {
        "page_title": "Erdrutsch-Simulation",
        "tab_sim": "Simulationsmodus",
        "tab_game": "Spielmodus",
        "tab_learn": "Lerne über Erdrutsche",
        "main_header": "Erdrutsch-Simulation",
        "phys_params": "Physikalische Parameter",
        "start_sim": "Simulation starten",
        "show_metrics": "Metriken anzeigen",
        "sim_metrics": "Simulationsmetriken",
        "landslide_anim": "Erdrutsch-Animation",
        "param_expander": "Parameter verstehen",
        "game_title": "Kalibrierungs-Challenge: Erdrutsch",
        "game_desc": "Stelle die Parameter so ein, dass der Felsbrocken möglichst nah am Sensor zum Stehen kommt.",
        "sensor_info": "Der Sensor befindet sich bei **{observation_x} Metern**. Passe die Parameter an, damit der Felsbrocken möglichst nah dort stoppt.",
        "your_guess": "### Deine Vorhersage",
        "mu_slider": "Coulomb-Reibung (μ)",
        "xi_slider": "Turbulente Reibung (ξ)",
        "submit_guess": "Vorhersage einreichen",
        "no_data": "Für die gewählten Parameter sind keine Simulationsdaten vorhanden. Bitte wähle eine andere Kombination.",
        "no_gif": "Keine Animation gefunden für μ = {mu}, ξ = {xi}. Nächste verfügbare Optionen werden angezeigt.",
        "try_again": "Nochmal versuchen",
        "suggestion_header": "#### Probiere diese verfügbaren Parameter:",
        "learn_header": "# Die Wissenschaft der Erdrutsche",
        "landslide_types": {
            "Murgang": "Schnell fließende Mischung aus Wasser, Gestein, Erde und organischem Material. Tritt oft nach starkem Regen auf.",
            "Felssturz": "Plötzlicher Absturz von Felsen von einer Klippe oder einem steilen Hang durch die Schwerkraft.",
            "Rutschung": "Massenbewegung, bei der Material entlang einer gekrümmten Fläche nach unten und außen gleitet.",
            "Erdfall": "Zähflüssiges Fließen von feinkörnigem, wassergesättigtem Material."
        },
        "types_header": "### Arten von Erdrutschen",
        "practice_header": "### Anwendungen in der Praxis",
        "practice_content": """
        Das Verständnis der Erdrutsch-Physik hilft bei:
        1. **Gefahrenabschätzung**: Risikogebiete erkennen
        2. **Frühwarnsystemen**: Vorhersage von Erdrutsch-Ereignissen
        3. **Infrastrukturplanung**: Sicherer Bau von Straßen und Gebäuden
        4. **Notfallmanagement**: Planung von Evakuierungsrouten und Ressourcen
        """,
        "download": "Animation herunterladen",
        "param_explanation": get_parameter_explanation(),
        "educational_content": get_educational_content(),
        "metrics": {
            "runout_distance": {
                "name": "Auslaufweite",
                "description": "Maximale Strecke, die der Erdrutsch zurücklegt",
                "unit": "Meter"
            },
            "max_velocity": {
                "name": "Maximale Geschwindigkeit",
                "description": "Höchstgeschwindigkeit während des Erdrutsch-Ereignisses",
                "unit": "m/s"
            },
            "affected_area": {
                "name": "Betroffene Fläche",
                "description": "Gesamte vom Erdrutsch betroffene Fläche",
                "unit": "m²"
            },
            "duration": {
                "name": "Ereignisdauer",
                "description": "Gesamtdauer bis der Erdrutsch stoppt",
                "unit": "Sekunden"
            }
        }
    },
    "en": {
        "page_title": "Landslide Simulation",
        "tab_sim": "Simulation Mode",
        "tab_game": "Game Mode",
        "tab_learn": "Learn About Landslides",
        "main_header": "Landslide Simulation Explorer",
        "phys_params": "Physics Parameters",
        "start_sim": "Start Simulation",
        "show_metrics": "Display Metrics",
        "sim_metrics": "Simulation Metrics",
        "landslide_anim": "Landslide Animation",
        "param_expander": "Understanding Parameters",
        "game_title": "Landslide Calibration Challenge",
        "game_desc": "Adjust the parameters so that the boulder stops as close as possible to the sensor location.",
        "sensor_info": "Sensor is located at **{observation_x} meters**. Tune parameters so the boulder stops as close as possible to it.",
        "your_guess": "### Your Prediction",
        "mu_slider": "Coulomb Friction (μ)",
        "xi_slider": "Turbulent Friction (ξ)",
        "submit_guess": "Submit Prediction",
        "no_data": "No simulation data for selected parameters. Try a different combination.",
        "no_gif": "No animation found for μ = {mu}, ξ = {xi}. Showing closest available options.",
        "try_again": "Try Again",
        "suggestion_header": "#### Try these available parameters:",
        "learn_header": "# The Science of Landslides",
        "landslide_types": {
            "Debris Flow": "Fast-moving mixture of water, rock, soil, and organic matter. Often follows heavy rainfall.",
            "Rockfall": "Sudden falling of rock from a cliff or steep slope due to gravity.",
            "Slump": "Mass movement where material moves downward and outward along a curved surface.",
            "Earthflow": "Viscous flow of fine-grained materials that have been saturated with water."
        },
        "types_header": "### Types of Landslides",
        "practice_header": "### Real-World Applications",
        "practice_content": """
        Understanding landslide physics helps in:
        1. **Hazard Assessment**: Identifying areas at risk
        2. **Early Warning Systems**: Predicting when landslides might occur
        3. **Infrastructure Planning**: Building safer roads and structures
        4. **Emergency Response**: Planning evacuation routes and resources
        """,
        "download": "Download Animation",
        "param_explanation": get_parameter_explanation_en(),
        "educational_content": get_educational_content_en(),
        "metrics": {
            "runout_distance": {
                "name": "Runout Distance",
                "description": "Maximum distance traveled by the landslide",
                "unit": "meters"
            },
            "max_velocity": {
                "name": "Maximum Velocity",
                "description": "Peak velocity reached during the landslide event",
                "unit": "m/s"
            },
            "affected_area": {
                "name": "Affected Area",
                "description": "Total area impacted by the landslide",
                "unit": "m²"
            },
            "duration": {
                "name": "Event Duration",
                "description": "Total time until landslide stops",
                "unit": "seconds"
            }
        }
    }
}

# --- LANGUAGE SELECTOR ---
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'de'
lang = st.sidebar.radio('Sprache / Language', options=[('de', 'Deutsch'), ('en', 'English')], format_func=lambda x: x[1])[0]
st.session_state['lang'] = lang
labels = LANGUAGES[lang]

# Constants
GIF_DIRECTORY = "animations_score_mit_Improvements"
MU_VALUES = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48]
XI_VALUES = [100, 200, 400, 600, 800, 1200, 1600, 1800, 2000, 2200]

# Initial values
INITIAL_MU = 0.48
INITIAL_XI = 1200

# Add simulation data for physics insights
# These could be replaced with actual data from your simulations
SIMULATION_METRICS = labels["metrics"]

# Helper functions
@st.cache_data
def load_gif(filepath):
    """Load and encode a GIF file to base64."""
    try:
        file_path = Path(filepath)
        if not file_path.exists():
            return None
            
        with open(file_path, "rb") as f:
            gif_bytes = f.read()
        gif_base64 = base64.b64encode(gif_bytes).decode("utf-8")
        return gif_base64
    except Exception as e:
        st.error(f"Error loading animation: {str(e)}")
        return None

# Function to find exact match in a list of values
def find_exact_value_index(value, options):
    """Find the exact index of a value in a list, or the closest index if not found."""
    try:
        return options.index(value)
    except ValueError:
        # If not found, find the closest value
        closest = min(options, key=lambda x: abs(x - value))
        return options.index(closest)

@st.cache_data
def generate_parameter_heatmap():
    """Generate a heatmap to visualize available parameter combinations."""
    # Create a placeholder matrix for available simulations
    availability_matrix = np.zeros((len(MU_VALUES), len(XI_VALUES)))
    
    # Check which combinations have GIF files
    for i, mu in enumerate(MU_VALUES):
        for j, xi in enumerate(XI_VALUES):
            filename = f'speed_up_30_animation_mu_{mu}_xi_{xi}.gif'
            filepath = os.path.join(GIF_DIRECTORY, filename)
            if os.path.exists(filepath):
                availability_matrix[i, j] = 1
    
    return availability_matrix

def create_download_link(gif_base64, filename="landslide_animation.gif"):
    """Create a download link for the current animation."""
    href = f'<a href="data:image/gif;base64,{gif_base64}" download="{filename}" class="download-button">{labels["download"]}</a>'
    return href

@st.cache_data
def generate_simulation_data(mu, xi):
    """Generate synthetic data for simulation metrics based on parameters.
    In a real application, this would retrieve actual simulation results."""
    
    # These are synthetic relationships - replace with actual physics relationships
    with open(r"linked_data.pkl", "rb") as f:
        linked_data = pickle.load(f)
        
    x_final = linked_data[mu, xi]
        
    runout = x_final[-1]
    
    def compute_maximum_velocity(x_final, time_step=0.1):
        """
        Compute the maximum velocity from a position time series.
        
        Parameters:
        -----------
        x_final : array_like
            Array of position values
        time_step : float, optional
            Time step between consecutive position measurements (default=1.0)
            
        Returns:
        --------
        max_velocity : float
            Maximum velocity value
        max_velocity_index : int
            Index where the maximum velocity occurs
        velocities : numpy.ndarray
            Array of all calculated velocities
        total_time : float
            Total time span of the position data
        """
        # Convert to numpy array if not already
        x = np.array(x_final)
        
        # Calculate velocities using central difference method
        # For internal points: v(i) = (x(i+1) - x(i-1)) / (2*dt)
        # For endpoints, use forward/backward difference
        velocities = np.zeros(len(x))
        
        # Forward difference for first point
        velocities[0] = (x[1] - x[0]) / time_step
        
        # Central difference for interior points
        for i in range(1, len(x) - 1):
            velocities[i] = (x[i+1] - x[i-1]) / (2 * time_step)
        
        # Backward difference for last point
        velocities[-1] = (x[-1] - x[-2]) / time_step
        
        # Calculate the magnitude of velocity (in case x contains vectors)
        if velocities.ndim > 1:
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        else:
            velocity_magnitudes = np.abs(velocities)
        
        # Find the maximum velocity and its index
        max_velocity = np.max(velocity_magnitudes)
        max_velocity_index = np.argmax(velocity_magnitudes)
        
        # Calculate total time
        total_time = (len(x) - 1) * time_step
        
        return max_velocity, total_time
    
    velocity, duration = compute_maximum_velocity(x_final)
    
    
    
    return {
        "runout_distance": round(runout, 1),
        "max_velocity": round(velocity, 1),
        "duration": round(duration, 1)
    }

# Interactive mode functions


def game_mode():
    st.markdown(f"## {labels['game_title']}")
    st.write(labels['game_desc'])

    observation_x = 2600  # Sensorposition
    st.info(labels['sensor_info'].format(observation_x=observation_x))

    # Load simulation results
    with open(r"linked_data.pkl", "rb") as f:
        linked_data = pickle.load(f)

    # Prediction sliders
    st.markdown(labels['your_guess'])
    pred_mu_slider = st.slider(labels['mu_slider'], min_value=min(MU_VALUES), max_value=max(MU_VALUES), step=0.01, format="%.2f")
    pred_xi_slider = st.slider(labels['xi_slider'], min_value=min(XI_VALUES), max_value=max(XI_VALUES), step=100)

    # Snap to closest available parameter
    def find_closest(value, options):
        return min(options, key=lambda x: abs(x - value))

    pred_mu = find_closest(pred_mu_slider, MU_VALUES)
    pred_xi = find_closest(pred_xi_slider, XI_VALUES)

    def calculate_score(x_final, observation_x):
        boulder_final_x = x_final[-1]
        if boulder_final_x <= observation_x:
            distance = abs(boulder_final_x - observation_x)
            return max(0, 10 - distance / 150)
        else:
            distance = abs(boulder_final_x - observation_x)
            return max(0, 10 - distance / 50)

    if "score" not in st.session_state:
        st.session_state.score = 0
    if "attempts" not in st.session_state:
        st.session_state.attempts = 0

    if st.button(labels['submit_guess']):
        st.session_state.attempts += 1
        x_final = linked_data.get((pred_mu, pred_xi))

        if x_final is None:
            st.error(labels['no_data'])
        else:
            score = calculate_score(x_final, observation_x)
            st.session_state.score += score

            filename = f'speed_up_30_animation_mu_{pred_mu}_xi_{pred_xi}.gif'
            filepath = os.path.join(GIF_DIRECTORY, filename)
            gif_base64 = load_gif(filepath)

            if gif_base64:
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: center; margin: 24px 0;'>
                        <img src='data:image/gif;base64,{gif_base64}' alt='Vorhersage-Animation' style='max-width: 100%; max-height: 600px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(labels['no_gif'].format(mu=pred_mu, xi=pred_xi))
                suggestions = []
                for mu in MU_VALUES:
                    for xi in XI_VALUES:
                        test_file = f'speed_up_30_animation_mu_{mu}_xi_{xi}.gif'
                        test_path = os.path.join(GIF_DIRECTORY, test_file)
                        if os.path.exists(test_path):
                            dist = ((mu - pred_mu)/max(MU_VALUES))**2 + ((xi - pred_xi)/max(XI_VALUES))**2
                            suggestions.append((mu, xi, dist))
                if suggestions:
                    st.markdown(labels['suggestion_header'])
                    suggestions.sort(key=lambda x: x[2])
                    cols = st.columns(min(3, len(suggestions)))
                    for i, (sugg_mu, sugg_xi, _) in enumerate(suggestions[:3]):
                        with cols[i]:
                            if st.button(f"μ = {sugg_mu}, ξ = {sugg_xi}", key=f"sugg_{i}"):
                                st.session_state["sim_mu"] = sugg_mu
                                st.session_state["sim_xi"] = sugg_xi
                                st.session_state.simulation_started = False
                                st.session_state.last_sim_mu = sugg_mu
                                st.session_state.last_sim_xi = sugg_xi
                                st.session_state.show_metrics = False
                                st.rerun()

        if st.button(labels['try_again'], type="primary"):
            st.experimental_rerun()



# Main application
def main():
    # Custom CSS for better visual design
    st.markdown("""
    <style>
        .stApp {
            background-color: #f5f7f9;
        }
        .main-header {
            text-align: center;
            color: #1E3A8A;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .download-button {
            display: inline-block;
            background-color: #1E3A8A;
            color: white;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 4px;
            margin-top: 10px;
        }
        .parameter-map {
            margin-top: 20px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .simulation-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .tab-content {
            padding: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Seitenüberschrift
    st.markdown(f"<h1 class='main-header'>{labels['main_header']}</h1>", unsafe_allow_html=True)
    
    # App-Navigation (Tabs)
    tab1, tab2, tab3 = st.tabs([labels["tab_sim"], labels["tab_game"], labels["tab_learn"]])
    
    # Tab 1: Simulationsmodus
    with tab1:
        col1, col2 = st.columns([1, 1], gap="small")
        
        with col1:
            st.markdown(f"### {labels['phys_params']}")
            # Track parameter changes to reset simulation state
            if "sim_mu" not in st.session_state:
                st.session_state.sim_mu = INITIAL_MU
            if "sim_xi" not in st.session_state:
                st.session_state.sim_xi = INITIAL_XI
            if "simulation_started" not in st.session_state:
                st.session_state.simulation_started = False
            if "last_sim_mu" not in st.session_state:
                st.session_state.last_sim_mu = None
            if "last_sim_xi" not in st.session_state:
                st.session_state.last_sim_xi = None
            if "show_metrics" not in st.session_state:
                st.session_state.show_metrics = False

            mu_slider = st.slider(
                labels["mu_slider"],
                min_value=min(MU_VALUES),
                max_value=max(MU_VALUES),
                value=st.session_state.sim_mu,
                step=0.01,
                format="%.2f",
                key="sim_mu_slider"
            )
            mu_index = find_exact_value_index(mu_slider, MU_VALUES)
            selected_mu = MU_VALUES[mu_index]

            xi_slider = st.slider(
                labels["xi_slider"],
                min_value=min(XI_VALUES),
                max_value=max(XI_VALUES),
                value=st.session_state.sim_xi,
                step=100,
                key="sim_xi_slider"
            )
            xi_index = find_exact_value_index(xi_slider, XI_VALUES)
            selected_xi = XI_VALUES[xi_index]

            # If sliders change, reset simulation state and hide metrics
            if (
                st.session_state.last_sim_mu != selected_mu or
                st.session_state.last_sim_xi != selected_xi
            ):
                st.session_state.simulation_started = False
                st.session_state.show_metrics = False  # Reset metrics display on parameter change

            # Start Simulation button
            if st.button(labels["start_sim"], key="start_sim_btn"):
                st.session_state.simulation_started = True
                st.session_state.last_sim_mu = selected_mu
                st.session_state.last_sim_xi = selected_xi
                st.session_state.sim_mu = selected_mu
                st.session_state.sim_xi = selected_xi
                st.session_state.show_metrics = False  # Also reset metrics on simulation start

            # Button to display metrics
            if st.button(labels["show_metrics"], key="display_metrics_btn") or st.session_state.get("show_metrics", False):
                st.session_state.show_metrics = True
                st.markdown(f"### {labels['sim_metrics']}")
                metrics = generate_simulation_data(selected_mu, selected_xi)
                for key, value in metrics.items():
                    metric_info = SIMULATION_METRICS[key]
                    st.metric(
                        label=f"{metric_info['name']} ({metric_info['unit']})",
                        value=value,
                        help=metric_info['description']
                    )

        with col2:
            st.markdown(f"### {labels['landslide_anim']}")
            if st.session_state.get("simulation_started", False):
                filename = f'speed_up_30_animation_mu_{selected_mu}_xi_{selected_xi}.gif'
                filepath = os.path.join(GIF_DIRECTORY, filename)
                gif_base64 = load_gif(filepath)
                if gif_base64:
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: center; margin: 0 0 20px 0;'>
                            <img src='data:image/gif;base64,{gif_base64}' alt='Erdrutsch-Animation' style='max-width: 100%; max-height: 500px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state.show_metrics = False
                    st.session_state.metrics_ready_time = None
                    st.error(f"Keine Animation gefunden für μ = {selected_mu}, ξ = {selected_xi}")
                    suggestions = []
                    for mu in MU_VALUES:
                        for xi in XI_VALUES:
                            test_file = f'speed_up_30_animation_mu_{mu}_xi_{xi}.gif'
                            test_path = os.path.join(GIF_DIRECTORY, test_file)
                            if os.path.exists(test_path):
                                distance = ((mu - selected_mu) / max(MU_VALUES))**2 + ((xi - selected_xi) / max(XI_VALUES))**2
                                suggestions.append((mu, xi, distance))
                    if suggestions:
                        st.markdown("#### Probiere diese verfügbaren Parameter:")
                        suggestions.sort(key=lambda x: x[2])
                        cols = st.columns(min(3, len(suggestions)))
                        for i, (sugg_mu, sugg_xi, _) in enumerate(suggestions[:3]):
                            with cols[i]:
                                if st.button(f"μ = {sugg_mu}, ξ = {sugg_xi}", key=f"sugg_{i}"):
                                    st.session_state["sim_mu"] = sugg_mu
                                    st.session_state["sim_xi"] = sugg_xi
                                    st.session_state.simulation_started = False
                                    st.session_state.last_sim_mu = sugg_mu
                                    st.session_state.last_sim_xi = sugg_xi
                                    st.session_state.show_metrics = False
                                    st.rerun()
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            with st.expander(labels["param_expander"]):
                st.markdown(labels["param_explanation"])
    
    # Tab 2: Spielmodus
    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        game_mode()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Lerninhalte
    with tab3:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown(labels['learn_header'])
        st.markdown(labels['educational_content'])
        st.markdown(labels['types_header'])
        landslide_types = labels['landslide_types']
        col1, col2 = st.columns(2)
        for i, (ls_type, description) in enumerate(landslide_types.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"#### {ls_type}")
                st.markdown(description)
        st.markdown(labels['practice_header'])
        st.markdown(labels['practice_content'])
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Load saved state from session if available
    if "sim_mu" in st.session_state:
        INITIAL_MU = st.session_state["sim_mu"]
    if "sim_xi" in st.session_state:
        INITIAL_XI = st.session_state["sim_xi"]
        
    main()