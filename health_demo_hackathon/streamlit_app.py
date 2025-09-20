import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="ASHA-doot Community Health Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0.5rem 0 0 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        color: #d32f2f !important;
    }
    .risk-high h3 {
        color: #d32f2f !important;
    }
    .risk-high p {
        color: #333 !important;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        color: #f57c00 !important;
    }
    .risk-medium h3 {
        color: #f57c00 !important;
    }
    .risk-medium p {
        color: #333 !important;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        color: #388e3c !important;
    }
    .risk-low h3 {
        color: #388e3c !important;
    }
    .risk-low p {
        color: #333 !important;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    /* Force dark text for better visibility */
    .stApp {
        color: #333 !important;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div[data-testid="metric-container"] > label {
        color: #333 !important;
    }
    div[data-testid="metric-container"] > div {
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load the health dataset and prepare it for machine learning."""
    try:
        # Check if the dataset exists
        if not os.path.exists('custom_health_data.csv'):
            st.error("❌ Dataset file 'custom_health_data.csv' not found!")
            st.error("Please run 'create_dataset.py' first to generate the dataset.")
            st.stop()
        
        # Load the dataset
        df = pd.read_csv('custom_health_data.csv')
        
        # Verify required columns exist
        required_columns = ['Turbidity (NTU)', 'Diarrhea Cases', 'Risk Level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.stop()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

@st.cache_resource
def train_model(df):
    """Train the Decision Tree classifier."""
    # Prepare features and target
    X = df[['Turbidity (NTU)', 'Diarrhea Cases']]
    y = df['Risk Level']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the model
    model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

def get_risk_color(risk_level):
    """Return appropriate color for risk level."""
    colors = {
        'Low': '#4caf50',
        'Medium': '#ff9800',
        'High': '#f44336'
    }
    return colors.get(risk_level, '#666666')

def get_risk_emoji(risk_level):
    """Return appropriate emoji for risk level."""
    emojis = {
        'Low': '✅',
        'Medium': '⚠️',
        'High': '🚨'
    }
    return emojis.get(risk_level, '❓')

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 ASHA-doot Community Health Monitor</h1>
        <p>Water-borne Disease Outbreak Risk Prediction for Rural Assam</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("📊 Loading health data and training AI model..."):
        df = load_and_prepare_data()
        model, accuracy, X_test, y_test, y_pred = train_model(df)
    
    # Display model performance
    st.success(f"✅ AI Model trained successfully! Accuracy: {accuracy:.2%}")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Input Data")
        st.markdown("Enter the current water and health conditions:")
        
        # Input widgets
        turbidity = st.slider(
            "Water Turbidity (NTU)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
            help="Turbidity measures water cloudiness. Higher values indicate more contaminated water."
        )
        
        diarrhea_cases = st.number_input(
            "Number of Diarrhea Cases Today",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
            help="Number of reported diarrhea cases in your community today."
        )
        
        # Prediction button
        predict_button = st.button("🔍 Assess Health Risk", type="primary")
        
        # Add information about Assam's seasonal patterns
        with st.expander("ℹ️ Seasonal Information for Assam"):
            st.markdown("""
            **Monsoon Season (June-September):**
            - Higher water turbidity expected
            - Increased risk of water-borne diseases
            - Extra vigilance recommended
            
            **Post-Monsoon (October-November):**
            - Gradually improving water quality
            - Monitor for delayed outbreak patterns
            
            **Winter/Pre-Monsoon (December-May):**
            - Generally lower turbidity levels
            - Baseline disease surveillance
            """)
    
    with col2:
        st.markdown("### 🎯 Risk Assessment Results")
        
        if predict_button or st.session_state.get('auto_predict', False):
            # Make prediction
            input_data = np.array([[turbidity, diarrhea_cases]])
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Get confidence (probability of predicted class)
            class_names = model.classes_
            confidence = prediction_proba[list(class_names).index(prediction)]
            
            # Display prediction with styling
            risk_color = get_risk_color(prediction)
            risk_emoji = get_risk_emoji(prediction)
            
            if prediction == "High":
                st.markdown(f"""
                <div class="risk-high">
                    <h3>{risk_emoji} HIGH RISK ALERT</h3>
                    <p><strong>Immediate action required!</strong></p>
                    <p>• Alert health authorities</p>
                    <p>• Distribute water purification tablets</p>
                    <p>• Increase disease surveillance</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "Medium":
                st.markdown(f"""
                <div class="risk-medium">
                    <h3>{risk_emoji} MEDIUM RISK</h3>
                    <p><strong>Enhanced monitoring needed</strong></p>
                    <p>• Advise boiling drinking water</p>
                    <p>• Monitor for additional cases</p>
                    <p>• Prepare for possible escalation</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>{risk_emoji} LOW RISK</h3>
                    <p><strong>Continue routine monitoring</strong></p>
                    <p>• Maintain standard precautions</p>
                    <p>• Regular water quality checks</p>
                    <p>• Health education activities</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model confidence
            st.metric("AI Model Confidence", f"{confidence:.1%}")
            
            # Feature importance visualization
            st.markdown("### 🤖 AI Decision Explanation")
            
            # Get feature importances
            feature_names = ['Water Turbidity', 'Diarrhea Cases']
            importances = model.feature_importances_
            
            # Create feature importance chart
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=importances,
                    marker_color=['#2E8B57', '#3CB371'],
                    text=[f'{imp:.1%}' for imp in importances],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Which factors influenced this decision?",
                xaxis_title="Input Factors",
                yaxis_title="Importance Score",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show all risk probabilities
            st.markdown("### 📈 Risk Probability Breakdown")
            prob_df = pd.DataFrame({
                'Risk Level': class_names,
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)
            
            fig_prob = px.bar(
                prob_df, 
                x='Risk Level', 
                y='Probability',
                color='Risk Level',
                color_discrete_map={
                    'Low': '#4caf50',
                    'Medium': '#ff9800',
                    'High': '#f44336'
                },
                text='Probability'
            )
            fig_prob.update_traces(texttemplate='%{text:.1%}', textposition='auto')
            fig_prob.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_prob, use_container_width=True)
        
        else:
            st.info("👆 Enter values and click 'Assess Health Risk' to get AI prediction")
    
    # Dataset overview section
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Total Records", f"{len(df):,}")
    with col4:
        st.metric("Training Accuracy", f"{accuracy:.1%}")
    with col5:
        risk_distribution = df['Risk Level'].value_counts()
        most_common_risk = risk_distribution.index[0]
        st.metric("Most Common Risk", most_common_risk)
    
    # Show data distribution
    with st.expander("📈 View Data Patterns"):
        col6, col7 = st.columns(2)
        
        with col6:
            # Turbidity distribution by risk level
            fig_turb = px.box(df, x='Risk Level', y='Turbidity (NTU)', 
                            color='Risk Level',
                            color_discrete_map={
                                'Low': '#4caf50',
                                'Medium': '#ff9800',
                                'High': '#f44336'
                            },
                            title="Water Turbidity by Risk Level")
            st.plotly_chart(fig_turb, use_container_width=True)
        
        with col7:
            # Cases distribution by risk level
            fig_cases = px.box(df, x='Risk Level', y='Diarrhea Cases',
                             color='Risk Level',
                             color_discrete_map={
                                 'Low': '#4caf50',
                                 'Medium': '#ff9800',
                                 'High': '#f44336'
                             },
                             title="Diarrhea Cases by Risk Level")
            st.plotly_chart(fig_cases, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🏥 ASHA-doot Community Health Monitor | Empowering Frontline Health Workers</p>
        <p>Built for rural Assam healthcare workers | AI-powered disease outbreak prediction</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    