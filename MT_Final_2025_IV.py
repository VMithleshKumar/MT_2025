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

# Page configuration
st.set_page_config(
    page_title="Landslide Simulation",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Constants
GIF_DIRECTORY = "animations_score_mit_Improvements"
MU_VALUES = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48]
XI_VALUES = [100, 200, 400, 600, 800, 1200, 1600, 1800, 2000, 2200]

# Initial values
INITIAL_MU = 0.48
INITIAL_XI = 1200

# Add simulation data for physics insights
# These could be replaced with actual data from your simulations
SIMULATION_METRICS = {
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
    href = f'<a href="data:image/gif;base64,{gif_base64}" download="{filename}" class="download-button">Download Animation</a>'
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

def get_parameter_explanation():
    """Return explanations about the physical parameters."""
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

def get_educational_content():
    """Return educational content about landslides."""
    content = """
    ### Landslide Science
    
    Landslides are one of nature's most powerful and destructive forces. They occur when masses of rock, earth, or debris move down a slope due to gravity.
    
    #### Key Factors Affecting Landslides:
    1. **Slope Angle** - Steeper slopes are more likely to fail
    2. **Material Properties** - Friction and cohesion determine stability
    3. **Water Content** - Increases weight and reduces friction
    4. **Triggers** - Earthquakes, rainfall, human activities
    
    This simulation allows you to explore how different friction conditions affect landslide behavior, which is crucial for hazard assessment and risk mitigation in vulnerable areas.
    """
    return content

# Interactive mode functions


def game_mode():
    """Inverse game mode where users tune parameters so the boulder stops at the sensor location."""
    st.markdown("## Landslide Calibration Challenge")
    st.write("Adjust the parameters so that the boulder stops as close as possible to the sensor location.")

    observation_x = 3000  # Sensor location
    st.info(f"Sensor is located at **{observation_x} meters**. Tune parameters so the boulder stops as close as possible to it.")

    # Load simulation results
    with open(r"linked_data.pkl", "rb") as f:
        linked_data = pickle.load(f)

    # Prediction sliders
    st.markdown("### Your Prediction")
    pred_mu_slider = st.slider("Coulomb Friction (μ)", min_value=min(MU_VALUES), max_value=max(MU_VALUES), step=0.01, format="%.2f")
    pred_xi_slider = st.slider("Turbulent Friction (ξ)", min_value=min(XI_VALUES), max_value=max(XI_VALUES), step=100)

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

    if st.button("Submit Prediction"):
        st.session_state.attempts += 1
        x_final = linked_data.get((pred_mu, pred_xi))

        if x_final is None:
            st.error("No simulation data for selected parameters. Try a different combination.")
        else:
            score = calculate_score(x_final, observation_x)
            st.session_state.score += score

            st.markdown("### Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Coulomb Friction (μ)", pred_mu)
                st.metric("Turbulent Friction (ξ)", pred_xi)
                st.metric("Score this round", round(score, 2))
                st.metric("Total Score", round(st.session_state.score, 2))
                st.metric("Attempts", st.session_state.attempts)
                st.metric("Final Boulder Position", round(x_final[-1], 2))

            with col2:
                filename = f'speed_up_30_animation_mu_{pred_mu}_xi_{pred_xi}.gif'
                filepath = os.path.join(GIF_DIRECTORY, filename)
                gif_base64 = load_gif(filepath)

                if gif_base64:
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: center; margin: 10px 0;">
                            <img src="data:image/gif;base64,{gif_base64}" alt="Predicted Animation" style="max-height: 400px;">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.warning(f"No animation found for μ = {pred_mu}, ξ = {pred_xi}. Showing closest available options.")
                    # Suggest nearest available gifs
                    suggestions = []
                    for mu in MU_VALUES:
                        for xi in XI_VALUES:
                            test_file = f'speed_up_30_animation_mu_{mu}_xi_{xi}.gif'
                            test_path = os.path.join(GIF_DIRECTORY, test_file)
                            if os.path.exists(test_path):
                                dist = ((mu - pred_mu)/max(MU_VALUES))**2 + ((xi - pred_xi)/max(XI_VALUES))**2
                                suggestions.append((mu, xi, dist))
                    if suggestions:
                        st.markdown("#### Try these available parameters:")
                        suggestions.sort(key=lambda x: x[2])
                        cols = st.columns(min(3, len(suggestions)))
                        for i, (sugg_mu, sugg_xi, _) in enumerate(suggestions[:3]):
                            with cols[i]:
                                if st.button(f"μ = {sugg_mu}, ξ = {sugg_xi}", key=f"sugg_{i}"):
                                    st.session_state["sim_mu"] = sugg_mu
                                    st.session_state["sim_xi"] = sugg_xi
                                    st.experimental_rerun()

        if st.button("Try Again", type="primary"):
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
    
    # Page header
    st.markdown("<h1 class='main-header'>Landslide Simulation Explorer</h1>", unsafe_allow_html=True)
    
    # App navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Simulation Mode", "Game Mode", "Parameter Map", "Learn About Landslides"])
    
    # Tab 1: Simulation Mode (Enhanced original functionality)
    with tab1:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("<div class='simulation-container'>", unsafe_allow_html=True)
            st.markdown("### Physics Parameters")
            
            # Create a slider for μ that snaps to available values
            mu_slider = st.slider(
                "Coulomb Friction (μ)", 
                min_value=min(MU_VALUES), 
                max_value=max(MU_VALUES), 
                value=INITIAL_MU,
                step=0.01,
                format="%.2f",
                key="sim_mu"
            )
            
            # Find the exact value from MU_VALUES
            mu_index = find_exact_value_index(mu_slider, MU_VALUES)
            selected_mu = MU_VALUES[mu_index]
            
            # Create a slider for ξ that snaps to available values
            xi_slider = st.slider(
                "Turbulent Friction (ξ)", 
                min_value=min(XI_VALUES), 
                max_value=max(XI_VALUES), 
                value=INITIAL_XI,
                step=100,
                key="sim_xi"
            )
            
            # Find the exact value from XI_VALUES
            xi_index = find_exact_value_index(xi_slider, XI_VALUES)
            selected_xi = XI_VALUES[xi_index]
            
            # Construct filename based on selected values
            filename = f'speed_up_30_animation_mu_{selected_mu}_xi_{selected_xi}.gif'
            filepath = os.path.join(GIF_DIRECTORY, filename)
            
            # Display simulation metrics
            st.markdown("### Simulation Metrics")
            metrics = generate_simulation_data(selected_mu, selected_xi)
            
            for key, value in metrics.items():
                metric_info = SIMULATION_METRICS[key]
                st.metric(
                    label=f"{metric_info['name']} ({metric_info['unit']})",
                    value=value,
                    help=metric_info['description']
                )
            
            # Parameter explanation
            with st.expander("Understanding Parameters"):
                st.markdown(get_parameter_explanation())
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            # Animation display
            st.markdown("<div class='simulation-container'>", unsafe_allow_html=True)
            st.markdown(f"### Landslide Animation")
            
            gif_base64 = load_gif(filepath)
            
            if gif_base64:
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; margin: 10px 0;">
                        <img src="data:image/gif;base64,{gif_base64}" alt="Landslide Animation" style="max-width: 100%; max-height: 500px;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Download button
                st.markdown(create_download_link(gif_base64, filename), unsafe_allow_html=True)
            else:
                # Error message with suggestions
                st.error(f"No animation found for μ = {selected_mu}, ξ = {selected_xi}")
                
                # Find available alternatives
                suggestions = []
                for mu in MU_VALUES:
                    for xi in XI_VALUES:
                        test_file = f'speed_up_30_animation_mu_{mu}_xi_{xi}.gif'
                        test_path = os.path.join(GIF_DIRECTORY, test_file)
                        if os.path.exists(test_path):
                            # Calculate "distance" from current parameters
                            distance = ((mu - selected_mu) / max(MU_VALUES))**2 + ((xi - selected_xi) / max(XI_VALUES))**2
                            suggestions.append((mu, xi, distance))
                
                # Display the 3 closest suggestions
                if suggestions:
                    st.markdown("#### Try these available parameters:")
                    suggestions.sort(key=lambda x: x[2])
                    cols = st.columns(min(3, len(suggestions)))
                    for i, (sugg_mu, sugg_xi, _) in enumerate(suggestions[:3]):
                        with cols[i]:
                            if st.button(f"μ = {sugg_mu}, ξ = {sugg_xi}", key=f"sugg_{i}"):
                                st.session_state["sim_mu"] = sugg_mu
                                st.session_state["sim_xi"] = sugg_xi
                                st.experimental_rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Game Mode
    with tab2:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        # Run game mode functions
        game_mode()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Parameter Map
    with tab3:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("<div class='parameter-map'>", unsafe_allow_html=True)
        st.markdown("### Parameter Space Map")
        st.markdown("This map shows which parameter combinations have available simulations.")
        
        # Generate availability matrix
        availability_matrix = generate_parameter_heatmap()
        
        # Create a figure for the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap
        im = ax.imshow(availability_matrix, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xlabel('Turbulent Friction (ξ)')
        ax.set_ylabel('Coulomb Friction (μ)')
        
        # Set ticks
        ax.set_xticks(np.arange(len(XI_VALUES)))
        ax.set_yticks(np.arange(len(MU_VALUES)))
        
        # Set tick labels
        ax.set_xticklabels(XI_VALUES)
        ax.set_yticklabels(MU_VALUES)
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Not Available', 'Available'])
        
        # Loop over data dimensions and create text annotations
        for i in range(len(MU_VALUES)):
            for j in range(len(XI_VALUES)):
                text = ax.text(j, i, "✓" if availability_matrix[i, j] else "✗",
                              ha="center", va="center", color="w")
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # Additional explanation
        st.markdown("""
        ### Understanding the Parameter Space
        
        This heatmap shows which combinations of Coulomb Friction (μ) and Turbulent Friction (ξ) have simulations available.
        
        - **Green cells with ✓**: Simulation available
        - **Dark cells with ✗**: No simulation available
        
        The pattern of available simulations helps visualize how these two parameters interact to create different landslide behaviors.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 4: Educational Content
    with tab4:
        st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
        st.markdown("<div class='simulation-container'>", unsafe_allow_html=True)
        
        st.markdown("# The Science of Landslides")
        
        st.markdown(get_educational_content())
        
        # Example landslide types
        st.markdown("### Types of Landslides")
        
        landslide_types = {
            "Debris Flow": "Fast-moving mixture of water, rock, soil, and organic matter. Often follows heavy rainfall.",
            "Rockfall": "Sudden falling of rock from a cliff or steep slope due to gravity.",
            "Slump": "Mass movement where material moves downward and outward along a curved surface.",
            "Earthflow": "Viscous flow of fine-grained materials that have been saturated with water."
        }
        
        col1, col2 = st.columns(2)
        
        for i, (ls_type, description) in enumerate(landslide_types.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"#### {ls_type}")
                st.markdown(description)
        
        # Real-world implications
        st.markdown("### Real-World Applications")
        st.markdown("""
        Understanding landslide physics helps in:
        
        1. **Hazard Assessment**: Identifying areas at risk
        2. **Early Warning Systems**: Predicting when landslides might occur
        3. **Infrastructure Planning**: Building safer roads and structures
        4. **Emergency Response**: Planning evacuation routes and resources
        
        The parameters in our simulation (μ and ξ) are simplified versions of the complex factors that affect real landslides.
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Load saved state from session if available
    if "sim_mu" in st.session_state:
        INITIAL_MU = st.session_state["sim_mu"]
    if "sim_xi" in st.session_state:
        INITIAL_XI = st.session_state["sim_xi"]
        
    main()