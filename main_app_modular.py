"""
ASHA-doot Community Health Monitor - Main Application
Modular version with separate service modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta

# Import custom service modules
try:
    from sms_service import SMSService
    from government_api import GovernmentAPIService
    from emergency_service import EmergencyResponse
except ImportError as e:
    st.error(f"❌ Error importing service modules: {e}")
    st.error("Please ensure all service modules are in the same directory:")
    st.error("• sms_service.py")
    st.error("• government_api.py") 
    st.error("• emergency_service.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="ASHA-doot Community Health Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: white;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    .emergency-panel {
        background: linear-gradient(90deg, #dc3545, #c82333);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .sms-panel {
        background: linear-gradient(90deg, #17a2b8, #138496);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .govt-api-panel {
        background: linear-gradient(90deg, #6f42c1, #5a32a3);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        color: #d32f2f !important;
    }
    .risk-high h3, .risk-high p {
        color: #d32f2f !important;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        color: #f57c00 !important;
    }
    .risk-medium h3, .risk-medium p {
        color: #f57c00 !important;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        color: #388e3c !important;
    }
    .risk-low h3, .risk-low p {
        color: #388e3c !important;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .status-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load the health dataset and prepare it for machine learning."""
    try:
        if not os.path.exists('custom_health_data.csv'):
            st.error("❌ Dataset file 'custom_health_data.csv' not found!")
            st.error("Please run 'python create_dataset.py' first to generate the dataset.")
            st.stop()
        
        df = pd.read_csv('custom_health_data.csv')
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
    X = df[['Turbidity (NTU)', 'Diarrhea Cases']]
    y = df['Risk Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X_test, y_test, y_pred

def initialize_services():
    """Initialize all service modules"""
    try:
        sms_service = SMSService()
        govt_api_service = GovernmentAPIService()
        emergency_service = EmergencyResponse()
        
        return sms_service, govt_api_service, emergency_service
    except Exception as e:
        st.error(f"❌ Error initializing services: {str(e)}")
        return None, None, None

def render_sidebar():
    """Render the sidebar with emergency features and configurations"""
    with st.sidebar:
        st.markdown("""
        <div class="emergency-panel">
            <h3>🚨 Emergency Hub</h3>
            <p>Quick access to emergency services</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Emergency contacts display
        st.markdown("### 📞 Emergency Contacts")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ambulance:** 108")
            st.markdown("**Health:** 104") 
            st.markdown("**Police:** 100")
        with col2:
            st.markdown("**Fire:** 101")
            st.markdown("**Disaster:** 1077")
            st.markdown("**Women:** 1091")
        
        st.markdown("---")
        
        # SMS Configuration
        st.markdown("""
        <div class="sms-panel">
            <h4>📱 SMS Alert System</h4>
        </div>
        """, unsafe_allow_html=True)
        
        enable_sms = st.checkbox("Enable SMS Alerts", key="enable_sms")
        
        if enable_sms:
            supervisor_phone = st.text_input("ASHA Supervisor Phone", "+91-XXXXXXXXXX", key="supervisor_phone")
            health_officer_phone = st.text_input("Health Officer Phone", "+91-XXXXXXXXXX", key="health_officer_phone")
            additional_contacts = st.text_area("Additional Contacts (one per line)", 
                                             placeholder="+91-XXXXXXXXXX\n+91-YYYYYYYYYY", key="additional_contacts")
            
            # Store SMS settings in session state
            st.session_state['sms_enabled'] = True
            st.session_state['emergency_contacts'] = {
                'supervisor': supervisor_phone,
                'health_officer': health_officer_phone,
                'additional': additional_contacts.split('\n') if additional_contacts else []
            }
        else:
            st.session_state['sms_enabled'] = False
        
        st.markdown("---")
        
        # Government API Configuration
        st.markdown("""
        <div class="govt-api-panel">
            <h4>🏛️ Government Integration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        enable_govt_api = st.checkbox("Enable Government Reporting", key="enable_govt_api")
        
        if enable_govt_api:
            district_code = st.selectbox("Select District", 
                                       ["Kamrup", "Guwahati", "Jorhat", "Dibrugarh", "Silchar", "Tezpur", "Nagaon"],
                                       key="district_select")
            block_name = st.text_input("Block/Tehsil Name", "Enter block name", key="block_name")
            village_code = st.text_input("Village Code (if any)", "VIL-001", key="village_code")
            
            st.session_state['govt_api_enabled'] = True
            st.session_state['location_details'] = {
                'district': district_code,
                'block': block_name,
                'village_code': village_code
            }
        else:
            st.session_state['govt_api_enabled'] = False
        
        st.markdown("---")
        
        # Service status indicators
        st.markdown("### 🔧 Service Status")
        
        # Check if services are initialized
        sms_status = "🟢 Ready" if st.session_state.get('sms_enabled', False) else "🔴 Disabled"
        govt_status = "🟢 Ready" if st.session_state.get('govt_api_enabled', False) else "🔴 Disabled"
        
        st.markdown(f"**SMS Service:** {sms_status}")
        st.markdown(f"**Government API:** {govt_status}")
        st.markdown("**AI Model:** 🟢 Trained")

def get_risk_color(risk_level):
    """Return appropriate color for risk level."""
    colors = {'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'}
    return colors.get(risk_level, '#666666')

def get_risk_emoji(risk_level):
    """Return appropriate emoji for risk level."""
    emojis = {'Low': '✅', 'Medium': '⚠️', 'High': '🚨'}
    return emojis.get(risk_level, '❓')

def handle_emergency_response(emergency_service, prediction, village_name, diarrhea_cases, turbidity):
    """Handle emergency response activation"""
    emergency_data = {
        'type': 'disease_outbreak',
        'location': village_name,
        'severity': prediction,
        'cases': diarrhea_cases,
        'turbidity': turbidity,
        'start_time': datetime.now().isoformat()
    }
    
    response = emergency_service.trigger_emergency_response(
        'disease_outbreak', village_name, emergency_data
    )
    
    st.markdown(f"""
    <div class="status-success">
        <strong>🚨 Emergency Response Activated!</strong><br>
        <strong>Emergency ID:</strong> {response['emergency_id']}<br>
        <strong>Priority:</strong> {response['priority'].upper()}<br>
        <strong>Response Time:</strong> {response['estimated_response_time']}<br>
        <strong>Coordinator:</strong> {response['coordinator']}
    </div>
    """, unsafe_allow_html=True)
    
    # Show immediate actions
    st.markdown("**Immediate Actions Required:**")
    for action in response['immediate_actions'][:4]:
        st.markdown(f"• {action}")
    
    return response

def handle_sms_alert(sms_service, prediction, village_name, diarrhea_cases, turbidity):
    """Handle SMS alert sending"""
    # Gather all contacts
    contacts = []
    emergency_contacts = st.session_state.get('emergency_contacts', {})
    
    if emergency_contacts.get('supervisor') and emergency_contacts['supervisor'] != "+91-XXXXXXXXXX":
        contacts.append(emergency_contacts['supervisor'])
    if emergency_contacts.get('health_officer') and emergency_contacts['health_officer'] != "+91-XXXXXXXXXX":
        contacts.append(emergency_contacts['health_officer'])
    
    # Add additional contacts
    for contact in emergency_contacts.get('additional', []):
        if contact.strip() and contact.strip() != "+91-XXXXXXXXXX":
            contacts.append(contact.strip())
    
    if not contacts:
        st.markdown(f"""
        <div class="status-error">
            No valid phone numbers configured. Please add contacts in the sidebar.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Validate phone numbers
    validation_result = sms_service.validate_phone_numbers(contacts)
    
    if validation_result['total_valid'] == 0:
        st.markdown(f"""
        <div class="status-error">
            No valid phone numbers found. Please check phone number format.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Send emergency alert
    sms_results = sms_service.send_emergency_alert(
        validation_result['valid'], 
        prediction, 
        village_name, 
        diarrhea_cases, 
        turbidity
    )
    
    # Show results
    for result in sms_results:
        if result['status'] == 'success':
            st.markdown(f"""
            <div class="status-success">
                <strong>📱 SMS Alert Sent Successfully!</strong><br>
                <strong>Provider:</strong> {result['provider']}<br>
                <strong>Recipients:</strong> {len(result['recipients'])} contacts<br>
                <strong>Time:</strong> {datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
            break
    else:
        st.markdown(f"""
        <div class="status-error">
            SMS sending failed. Please check your configuration.
        </div>
        """, unsafe_allow_html=True)

def handle_government_reporting(govt_api_service, prediction, village_name, diarrhea_cases, turbidity):
    """Handle government API reporting"""
    location_details = st.session_state.get('location_details', {})
    
    outbreak_data = {
        'location': village_name,
        'district': location_details.get('district', 'Kamrup'),
        'block': location_details.get('block', 'Unknown'),
        'village_code': location_details.get('village_code', 'VIL-001'),
        'risk_level': prediction,
        'cases': diarrhea_cases,
        'turbidity': turbidity,
        'timestamp': datetime.now().isoformat(),
        'asha_worker': 'Current User',
        'contact': '+91-XXXXXXXXXX'
    }
    
    govt_response = govt_api_service.report_outbreak_to_government(outbreak_data)
    
    if govt_response['status'] == 'success':
        st.markdown(f"""
        <div class="status-success">
            <strong>🏛️ Government Report Submitted!</strong><br>
            <strong>Reference:</strong> {govt_response['reference_number']}<br>
            <strong>Assigned Officer:</strong> {govt_response.get('assigned_officer', 'District Health Officer')}<br>
            <strong>Tracking URL:</strong> <a href="{govt_response.get('tracking_url', '#')}" target="_blank">Track Progress</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Next Steps from Government:**")
        for step in govt_response.get('next_steps', [])[:3]:
            st.markdown(f"• {step}")
    else:
        st.markdown(f"""
        <div class="status-error">
            Government reporting failed: {govt_response.get('message', 'Unknown error')}
        </div>
        """, unsafe_allow_html=True)

def render_quick_emergency_actions(emergency_service, govt_api_service):
    """Render quick emergency action buttons"""
    st.markdown("### 🚨 Quick Emergency Actions")
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        if st.button("🚑 Call Ambulance (108)", type="secondary"):
            st.markdown("""
            <div class="status-info">
                <strong>🚑 Emergency Call Protocol:</strong><br>
                • Dial 108 immediately<br>
                • Provide exact location<br>
                • Describe medical emergency<br>
                • Stay on line for instructions
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if st.button("🏥 Find Nearest Hospital", type="secondary"):
            location_details = st.session_state.get('location_details', {})
            district = location_details.get('district', 'Kamrup')
            
            resources = govt_api_service.get_health_resources(district, 'hospitals')
            if resources['status'] == 'success':
                hospitals = resources['resources']
                st.markdown("**Nearest Medical Facilities:**")
                for hospital in hospitals[:2]:
                    st.markdown(f"• {hospital['name']} ({hospital['distance_km']} km)")
    
    with col5:
        if st.button("💧 Request Water Test", type="secondary"):
            test_request = emergency_service.trigger_emergency_response(
                'water_contamination',
                'Current Location',
                {'turbidity': 'high', 'source': 'primary_well'}
            )
            st.markdown(f"""
            <div class="status-info">
                <strong>💧 Water Testing Requested</strong><br>
                Request ID: {test_request['emergency_id']}<br>
                Team arrival: {test_request['estimated_response_time']}
            </div>
            """, unsafe_allow_html=True)
    
    with col6:
        if st.button("📋 Get Health Guidelines", type="secondary"):
            guidelines = govt_api_service.get_health_guidelines('waterborne')
            if guidelines['status'] == 'success':
                st.markdown("**Official Health Guidelines:**")
                for action in guidelines['guidelines']['immediate_actions'][:3]:
                    st.markdown(f"• {action}")

def main():
    """Main application function"""
    
    # Initialize services
    sms_service, govt_api_service, emergency_service = initialize_services()
    
    if not all([sms_service, govt_api_service, emergency_service]):
        st.error("❌ Failed to initialize services. Please check service modules.")
        return
    
    # Render sidebar
    render_sidebar()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 ASHA-doot Community Health Monitor</h1>
        <p>Advanced Disease Outbreak Prediction with Emergency Response System</p>
        <p><em>Modular Architecture • SMS Alerts • Government Integration • Emergency Coordination</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("📊 Loading health data and training AI model..."):
        df = load_and_prepare_data()
        model, accuracy, X_test, y_test, y_pred = train_model(df)
    
    # Display model performance
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    with col_perf1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>🎯 AI Model Accuracy</h3>
            <h2 style="color: #2E8B57;">{accuracy:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_perf2:
        total_records = len(df)
        st.markdown(f"""
        <div class="metric-box">
            <h3>📊 Training Data</h3>
            <h2 style="color: #17a2b8;">{total_records:,}</h2>
            <p>Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_perf3:
        services_active = sum([
            st.session_state.get('sms_enabled', False),
            st.session_state.get('govt_api_enabled', False),
            True  # AI model always active
        ])
        st.markdown(f"""
        <div class="metric-box">
            <h3>⚡ Active Services</h3>
            <h2 style="color: #6f42c1;">{services_active}/3</h2>
            <p>Systems Online</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Health Data Input")
        
        # Location information
        village_name = st.text_input("🏘️ Village/Area Name", "Enter location name", key="village_input")
        
        # Health worker information
        with st.expander("👩‍⚕️ ASHA Worker Information"):
            asha_name = st.text_input("ASHA Worker Name", "Enter your name")
            asha_id = st.text_input("ASHA ID", "ASHA-001")
            contact_number = st.text_input("Contact Number", "+91-XXXXXXXXXX")
        
        # Main health inputs
        turbidity = st.slider(
            "💧 Water Turbidity (NTU)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=0.5,
            help="Turbidity measures water cloudiness. Higher values indicate contaminated water."
        )
        
        diarrhea_cases = st.number_input(
            "🤒 Diarrhea Cases Today",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
            help="Number of reported diarrhea cases in your area today."
        )
        
        # Additional health parameters
        with st.expander("📋 Additional Health Data (Optional)"):
            col_extra1, col_extra2 = st.columns(2)
            with col_extra1:
                fever_cases = st.number_input("🌡️ Fever Cases", 0, 30, 0)
                vomiting_cases = st.number_input("🤮 Vomiting Cases", 0, 30, 0)
            with col_extra2:
                dehydration_cases = st.number_input("💦 Dehydration Cases", 0, 20, 0)
                hospitalized = st.number_input("🏥 Hospitalized", 0, 10, 0)
        
        # Prediction button
        predict_button = st.button("🔍 Assess Health Risk", type="primary", use_container_width=True)
        
        # Seasonal information
        with st.expander("🌧️ Seasonal Information for Assam"):
            current_month = datetime.now().month
            if current_month in [6, 7, 8, 9]:
                st.warning("**Monsoon Season (June-September)**\n- Higher disease risk expected\n- Increased water contamination\n- Enhanced surveillance needed")
            elif current_month in [10, 11]:
                st.info("**Post-Monsoon (October-November)**\n- Monitor for outbreak patterns\n- Water quality improving")
            else:
                st.success("**Dry Season (December-May)**\n- Lower baseline risk\n- Routine monitoring sufficient")
    
    with col2:
        st.markdown("### 🎯 AI Risk Assessment")
        
        if predict_button:
            # Make prediction
            input_data = np.array([[turbidity, diarrhea_cases]])
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            class_names = model.classes_
            confidence = prediction_proba[list(class_names).index(prediction)]
            
            # Display prediction with appropriate styling
            risk_emoji = get_risk_emoji(prediction)
            
            if prediction == "High":
                st.markdown(f"""
                <div class="risk-high">
                    <h3>{risk_emoji} HIGH RISK ALERT</h3>
                    <p><strong>Immediate action required!</strong></p>
                    <p>• Alert health authorities immediately</p>
                    <p>• Distribute water purification tablets</p>
                    <p>• Set up temporary medical camps</p>
                    <p>• Begin contact tracing</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Emergency action buttons for high risk
                st.markdown("#### 🚨 Emergency Actions")
                
                col_emrg1, col_emrg2 = st.columns(2)
                
                with col_emrg1:
                    if st.button("🚨 Activate Emergency Response"):
                        handle_emergency_response(emergency_service, prediction, village_name, diarrhea_cases, turbidity)
                
                with col_emrg2:
                    if st.button("📱 Send SMS Alert") and st.session_state.get('sms_enabled'):
                        handle_sms_alert(sms_service, prediction, village_name, diarrhea_cases, turbidity)
                
                # Government reporting for high risk
                if st.session_state.get('govt_api_enabled') and st.button("🏛️ Report to Government"):
                    handle_government_reporting(govt_api_service, prediction, village_name, diarrhea_cases, turbidity)
                
            elif prediction == "Medium":
                st.markdown(f"""
                <div class="risk-medium">
                    <h3>{risk_emoji} MEDIUM RISK</h3>
                    <p><strong>Enhanced monitoring needed</strong></p>
                    <p>• Advise boiling all drinking water</p>
                    <p>• Monitor for additional cases closely</p>
                    <p>• Prepare for possible escalation</p>
                    <p>• Distribute health education materials</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Optional actions for medium risk
                if st.button("📱 Send Monitoring Alert") and st.session_state.get('sms_enabled'):
                    handle_sms_alert(sms_service, prediction, village_name, diarrhea_cases, turbidity)
            
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>{risk_emoji} LOW RISK</h3>
                    <p><strong>Continue routine monitoring</strong></p>
                    <p>• Maintain standard health precautions</p>
                    <p>• Regular water quality checks</p>
                    <p>• Continue health education activities</p>
                    <p>• Monitor for any changes</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display model confidence
            st.metric("🎯 AI Model Confidence", f"{confidence:.1%}")
            
            # Feature importance visualization
            st.markdown("### 🤖 AI Decision Explanation")
            
            feature_names = ['Water Turbidity', 'Diarrhea Cases']
            importances = model.feature_importances_
            
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
            
            # Risk probability breakdown
            st.markdown("### 📈 Risk Probability Distribution")
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
            st.info("👆 Enter health data and click 'Assess Health Risk' to get AI prediction")
            
            # Show sample predictions
            st.markdown("#### 📝 Sample Risk Scenarios")
            st.markdown("**High Risk Example:** Turbidity > 30 NTU, Cases > 10")
            st.markdown("**Medium Risk Example:** Turbidity 15-30 NTU, Cases 5-10")
            st.markdown("**Low Risk Example:** Turbidity < 15 NTU, Cases < 5")
    
    st.markdown("---")
    
    # Quick emergency actions
    render_quick_emergency_actions(emergency_service, govt_api_service)
    
    st.markdown("---")
    
    # Data insights and patterns
    st.markdown("### 📊 Health Data Insights")
    
    col_insights1, col_insights2 = st.columns(2)
    
    with col_insights1:
        # Monthly pattern analysis
        if 'Month' in df.columns:
            monthly_cases = df.groupby('Month')['Diarrhea Cases'].mean().round(1)
            fig_monthly = px.line(
                x=monthly_cases.index,
                y=monthly_cases.values,
                title="Average Monthly Diarrhea Cases",
                labels={'x': 'Month', 'y': 'Average Cases'}
            )
            fig_monthly.update_layout(height=300)
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col_insights2:
        # Risk level distribution
        risk_counts = df['Risk Level'].value_counts()
        fig_risk_dist = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Overall Risk Level Distribution",
            color_discrete_map={
                'Low': '#4caf50',
                'Medium': '#ff9800',
                'High': '#f44336'
            }
        )
        fig_risk_dist.update_layout(height=300)
        st.plotly_chart(fig_risk_dist, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🏥 ASHA-doot Community Health Monitor v3.0</h4>
        <p><strong>Modular Architecture Edition</strong></p>
        <p>✨ AI-Powered Disease Prediction • 📱 SMS Emergency Alerts • 🏛️ Government Integration • 🚨 Emergency Response</p>
        <p><em>Empowering ASHA workers with advanced technology for better community health outcomes</em></p>
        <p>Built with ❤️ for rural healthcare workers in Assam, India</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()