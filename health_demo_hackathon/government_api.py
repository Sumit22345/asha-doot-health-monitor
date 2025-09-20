"""
Government API Service Module for ASHA-doot Health Monitor
Handles integration with government health authorities and databases
"""

import requests
import json
from datetime import datetime, timedelta
import hashlib

class GovernmentAPIService:
    """Handle Government Health API integrations"""
    
    def __init__(self):
        self.apis = {
            'assam_health': {
                'base_url': 'https://api.assam.gov.in/health',
                'api_key': 'YOUR_ASSAM_HEALTH_API_KEY',
                'endpoints': {
                    'report_outbreak': '/outbreak/report',
                    'get_guidelines': '/guidelines',
                    'register_case': '/case/register',
                    'get_resources': '/resources',
                    'emergency_response': '/emergency/activate'
                }
            },
            'india_health': {
                'base_url': 'https://api.india.gov.in/health',
                'api_key': 'YOUR_INDIA_HEALTH_API_KEY',
                'endpoints': {
                    'disease_surveillance': '/surveillance',
                    'emergency_response': '/emergency',
                    'idsp_report': '/idsp/report'  # Integrated Disease Surveillance Program
                }
            },
            'who_india': {
                'base_url': 'https://api.who.int/india',
                'endpoints': {
                    'disease_alerts': '/alerts',
                    'guidelines': '/guidelines'
                }
            }
        }
        
        # Assam district codes for proper reporting
        self.assam_districts = {
            'Kamrup': 'AS001',
            'Guwahati': 'AS002', 
            'Jorhat': 'AS003',
            'Dibrugarh': 'AS004',
            'Silchar': 'AS005',
            'Tezpur': 'AS006',
            'Nagaon': 'AS007',
            'Barpeta': 'AS008',
            'Dhubri': 'AS009',
            'Golaghat': 'AS010'
        }
    
    def report_outbreak_to_government(self, outbreak_data):
        """Report disease outbreak to government health authorities"""
        try:
            # Generate unique reference number
            ref_number = self._generate_reference_number(outbreak_data)
            
            # Format data according to government standards
            formatted_data = {
                'reference_number': ref_number,
                'district_code': self.assam_districts.get(outbreak_data.get('district'), 'AS000'),
                'location': outbreak_data.get('location'),
                'disease_type': 'waterborne_diarrhea',
                'risk_level': outbreak_data.get('risk_level'),
                'affected_population': outbreak_data.get('cases', 0),
                'water_quality': {
                    'turbidity_ntu': outbreak_data.get('turbidity'),
                    'contamination_level': self._assess_contamination_level(outbreak_data.get('turbidity', 0))
                },
                'timestamp': outbreak_data.get('timestamp', datetime.now().isoformat()),
                'reporting_officer': outbreak_data.get('asha_worker', 'ASHA Worker'),
                'contact_details': outbreak_data.get('contact', '+91-XXXXXXXXXX'),
                'immediate_actions_taken': self._get_immediate_actions(outbreak_data.get('risk_level')),
                'resources_needed': self._assess_resources_needed(outbreak_data)
            }
            
            # Demo mode response (replace with actual API call in production)
            if self.apis['assam_health']['api_key'] == 'YOUR_ASSAM_HEALTH_API_KEY':
                return self._simulate_government_response(formatted_data, 'outbreak_report')
            
            # Actual API call (uncomment when API keys are available)
            # response = requests.post(
            #     self.apis['assam_health']['base_url'] + self.apis['assam_health']['endpoints']['report_outbreak'],
            #     json=formatted_data,
            #     headers={'Authorization': f"Bearer {self.apis['assam_health']['api_key']}"}
            # )
            # return response.json()
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def register_health_case(self, case_data):
        """Register individual health case with government system"""
        try:
            case_id = f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            formatted_case = {
                'case_id': case_id,
                'patient_age_group': case_data.get('age_group', 'adult'),
                'symptoms': case_data.get('symptoms', ['diarrhea']),
                'severity': case_data.get('severity', 'moderate'),
                'location': case_data.get('location'),
                'date_of_onset': case_data.get('onset_date', datetime.now().date().isoformat()),
                'treatment_given': case_data.get('treatment', 'ORS'),
                'outcome': case_data.get('outcome', 'under_treatment')
            }
            
            # Demo response
            return {
                'status': 'success',
                'case_id': case_id,
                'message': 'Health case registered successfully',
                'follow_up_required': True,
                'next_checkup': (datetime.now() + timedelta(days=3)).isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_health_guidelines(self, disease_type='waterborne'):
        """Get official health guidelines from government"""
        
        guidelines_database = {
            'waterborne': {
                'immediate_actions': [
                    'Isolate suspected contaminated water sources immediately',
                    'Distribute ORS packets to affected families',
                    'Set up temporary medical camps in affected areas',
                    'Initiate contact tracing for all confirmed cases',
                    'Notify District Health Officer within 2 hours',
                    'Collect water samples for laboratory testing'
                ],
                'prevention_measures': [
                    'Boil all drinking water for at least 10 minutes',
                    'Use water purification tablets (1 tablet per liter)',
                    'Maintain proper sanitation and hygiene practices',
                    'Report any new cases immediately to health authorities',
                    'Avoid consumption of raw vegetables and fruits',
                    'Ensure proper waste disposal and sewage management'
                ],
                'treatment_protocol': [
                    'Assess dehydration level using WHO guidelines',
                    'Administer ORS for mild to moderate dehydration',
                    'Provide IV fluids for severe dehydration cases',
                    'Monitor for danger signs every 4-6 hours',
                    'Refer complicated cases to PHC/CHC immediately',
                    'Follow up with patients after 24-48 hours'
                ],
                'reporting_requirements': [
                    'Report to IDSP within 24 hours',
                    'Maintain daily case logs',
                    'Submit weekly surveillance reports',
                    'Coordinate with District Surveillance Officer'
                ]
            },
            'cholera': {
                'immediate_actions': [
                    'Immediate isolation of suspected cases',
                    'Alert District Health Emergency Response Team',
                    'Set up cholera treatment units',
                    'Implement water and sanitation measures',
                    'Mass health education campaigns'
                ],
                'treatment_protocol': [
                    'Rapid rehydration with ORS or IV fluids',
                    'Antibiotic therapy for severe cases',
                    'Zinc supplementation for children',
                    'Strict monitoring of fluid balance'
                ]
            }
        }
        
        guidelines = guidelines_database.get(disease_type, guidelines_database['waterborne'])
        
        return {
            'status': 'success',
            'disease_type': disease_type,
            'guidelines': guidelines,
            'last_updated': '2024-01-15',
            'authority': 'Ministry of Health & Family Welfare, Government of Assam',
            'reference_document': f'MOHFW-AS-GL-{disease_type.upper()}-2024',
            'emergency_contacts': {
                'state_control_room': '104',
                'district_health_officer': '+91-361-2234567',
                'ambulance': '108',
                'emergency_response_team': '+91-361-2345678'
            }
        }
    
    def get_health_resources(self, district, resource_type='all'):
        """Get available health resources in district"""
        
        # Sample resource data (replace with actual government database)
        resources_db = {
            'Kamrup': {
                'hospitals': [
                    {'name': 'Gauhati Medical College', 'beds': 850, 'distance_km': 5.2},
                    {'name': 'Dispur Hospital', 'beds': 200, 'distance_km': 3.8}
                ],
                'phc': [
                    {'name': 'Jalukbari PHC', 'doctors': 2, 'distance_km': 2.1},
                    {'name': 'Basistha PHC', 'doctors': 1, 'distance_km': 4.5}
                ],
                'medical_supplies': {
                    'ors_packets': 2500,
                    'iv_fluids': 150,
                    'antibiotics': 80,
                    'water_purification_tablets': 5000
                },
                'ambulances': {
                    'available': 12,
                    'response_time_avg': '15-20 minutes'
                }
            }
        }
        
        district_resources = resources_db.get(district, resources_db['Kamrup'])
        
        if resource_type == 'all':
            return {
                'status': 'success',
                'district': district,
                'resources': district_resources,
                'last_updated': datetime.now().isoformat()
            }
        else:
            return {
                'status': 'success',
                'district': district,
                'resource_type': resource_type,
                'resources': district_resources.get(resource_type, {}),
                'last_updated': datetime.now().isoformat()
            }
    
    def activate_emergency_response(self, emergency_data):
        """Activate government emergency response system"""
        
        response_teams = {
            'disease_outbreak': {
                'team_type': 'Rapid Response Team (RRT)',
                'composition': ['Epidemiologist', 'Lab Technician', 'ASHA Supervisor', 'Pharmacist'],
                'response_time': '2-4 hours',
                'equipment': ['Water testing kit', 'Sample collection kit', 'Medical supplies']
            },
            'water_contamination': {
                'team_type': 'Water Quality Assessment Team',
                'composition': ['Water Engineer', 'Lab Technician', 'Health Inspector'],
                'response_time': '1-2 hours',
                'equipment': ['Water testing equipment', 'Sampling tools', 'Purification chemicals']
            },
            'medical_emergency': {
                'team_type': 'Emergency Medical Services',
                'composition': ['Paramedic', 'Doctor', 'Ambulance crew'],
                'response_time': '15-30 minutes',
                'equipment': ['Ambulance', 'Emergency medicines', 'Life support equipment']
            }
        }
        
        emergency_type = emergency_data.get('type', 'disease_outbreak')
        team_info = response_teams.get(emergency_type, response_teams['disease_outbreak'])
        
        emergency_id = f"EMRG-AS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return {
            'status': 'activated',
            'emergency_id': emergency_id,
            'emergency_type': emergency_type,
            'location': emergency_data.get('location'),
            'response_team': team_info,
            'estimated_arrival': self._calculate_arrival_time(team_info['response_time']),
            'contact_person': 'District Emergency Coordinator',
            'tracking_number': emergency_id,
            'next_update': (datetime.now() + timedelta(minutes=30)).isoformat(),
            'instructions': [
                'Maintain current safety measures',
                'Prepare area for response team arrival',
                'Keep affected persons in designated area',
                'Document all new cases until team arrives'
            ]
        }
    
    def submit_idsp_report(self, surveillance_data):
        """Submit report to Integrated Disease Surveillance Program (IDSP)"""
        
        idsp_report = {
            'report_id': f"IDSP-AS-{datetime.now().strftime('%Y%W')}",  # Year-Week format
            'reporting_unit': surveillance_data.get('health_facility', 'ASHA Center'),
            'reporting_period': surveillance_data.get('week', datetime.now().strftime('%Y-W%U')),
            'district': surveillance_data.get('district'),
            'cases_reported': {
                'acute_diarrheal_disease': surveillance_data.get('diarrhea_cases', 0),
                'fever': surveillance_data.get('fever_cases', 0),
                'respiratory_illness': surveillance_data.get('respiratory_cases', 0)
            },
            'deaths_reported': surveillance_data.get('deaths', 0),
            'outbreak_alerts': surveillance_data.get('outbreaks', []),
            'water_quality_issues': surveillance_data.get('water_issues', [])
        }
        
        return {
            'status': 'submitted',
            'report_id': idsp_report['report_id'],
            'submission_time': datetime.now().isoformat(),
            'acknowledgment': 'Report received by IDSP-Assam',
            'next_report_due': (datetime.now() + timedelta(days=7)).isoformat()
        }
    
    def _generate_reference_number(self, data):
        """Generate unique reference number for government reports"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        location_hash = hashlib.md5(data.get('location', '').encode()).hexdigest()[:6]
        return f"ASSAM-HEALTH-{timestamp}-{location_hash.upper()}"
    
    def _assess_contamination_level(self, turbidity):
        """Assess water contamination level based on turbidity"""
        if turbidity >= 30:
            return 'severe'
        elif turbidity >= 15:
            return 'moderate'
        elif turbidity >= 5:
            return 'mild'
        else:
            return 'acceptable'
    
    def _get_immediate_actions(self, risk_level):
        """Get immediate actions based on risk level"""
        actions = {
            'High': [
                'Emergency response team deployed',
                'Water source isolation initiated',
                'Medical camps established',
                'Contact tracing started'
            ],
            'Medium': [
                'Health surveillance activated',
                'Water quality testing ordered',
                'Public health advisory issued'
            ],
            'Low': [
                'Routine monitoring continued',
                'Health education activities initiated'
            ]
        }
        return actions.get(risk_level, actions['Low'])
    
    def _assess_resources_needed(self, outbreak_data):
        """Assess resources needed based on outbreak data"""
        cases = outbreak_data.get('cases', 0)
        
        if cases >= 15:
            return {
                'medical_team': 'Full RRT deployment',
                'supplies': 'Emergency medical supplies',
                'logistics': 'Ambulance and transport',
                'duration': '72 hours minimum'
            }
        elif cases >= 5:
            return {
                'medical_team': 'Health supervisor and ASHA',
                'supplies': 'ORS and basic medicines',
                'logistics': 'Local transport',
                'duration': '24-48 hours'
            }
        else:
            return {
                'medical_team': 'ASHA worker monitoring',
                'supplies': 'Basic health supplies',
                'logistics': 'Routine follow-up',
                'duration': 'As needed'
            }
    
    def _simulate_government_response(self, data, report_type):
        """Simulate government API response for demo purposes"""
        
        responses = {
            'outbreak_report': {
                'status': 'success',
                'reference_number': data.get('reference_number'),
                'message': 'Outbreak reported successfully to Assam Health Department',
                'acknowledgment_time': datetime.now().isoformat(),
                'assigned_officer': 'Dr. Rajesh Sharma, District Health Officer',
                'next_steps': [
                    'Rapid Response Team will be dispatched within 4 hours',
                    'Water testing team has been notified and will arrive within 2 hours',
                    'Emergency medical supplies are being arranged',
                    'Surveillance team has been activated for contact tracing',
                    'District Collector has been informed for administrative support'
                ],
                'contact_info': {
                    'emergency_hotline': '108',
                    'district_health_officer': '+91-361-2234567',
                    'state_control_room': '+91-361-2345678',
                    'rrt_coordinator': '+91-361-2456789'
                },
                'tracking_url': f"https://health.assam.gov.in/track/{data.get('reference_number')}",
                'expected_resolution': '24-48 hours'
            }
        }
        
        return responses.get(report_type, {'status': 'error', 'message': 'Unknown report type'})
    
    def _calculate_arrival_time(self, response_time_range):
        """Calculate estimated arrival time"""
        # Extract average time from range like "2-4 hours"
        try:
            if 'hour' in response_time_range:
                hours = int(response_time_range.split('-')[0])
                return (datetime.now() + timedelta(hours=hours)).isoformat()
            elif 'minute' in response_time_range:
                minutes = int(response_time_range.split('-')[0])
                return (datetime.now() + timedelta(minutes=minutes)).isoformat()
        except:
            pass
        
        return (datetime.now() + timedelta(hours=2)).isoformat()  # Default 2 hours

# Example usage and testing
if __name__ == "__main__":
    # Test the Government API service
    govt_api = GovernmentAPIService()
    
    # Test outbreak reporting
    test_outbreak = {
        'location': 'Guwahati Test Village',
        'district': 'Kamrup',
        'risk_level': 'High',
        'cases': 12,
        'turbidity': 35.5,
        'timestamp': datetime.now().isoformat(),
        'asha_worker': 'Test ASHA Worker'
    }
    
    outbreak_response = govt_api.report_outbreak_to_government(test_outbreak)
    print("Outbreak Report Response:", json.dumps(outbreak_response, indent=2))
    
    # Test health guidelines
    guidelines = govt_api.get_health_guidelines('waterborne')
    print("\nHealth Guidelines:", json.dumps(guidelines, indent=2))
    
    # Test resource availability
    resources = govt_api.get_health_resources('Kamrup', 'medical_supplies')
    print("\nResource Availability:", json.dumps(resources, indent=2))
