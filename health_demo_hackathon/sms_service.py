"""
SMS Service Module for ASHA-doot Health Monitor
Handles SMS notifications using various SMS APIs
"""

import requests
from datetime import datetime
import json

class SMSService:
    """Handle SMS notifications using various SMS APIs"""
    
    def __init__(self):
        self.api_configs = {
            'textlocal': {
                'url': 'https://api.textlocal.in/send/',
                'api_key': 'YOUR_TEXTLOCAL_API_KEY'
            },
            'msg91': {
                'url': 'https://api.msg91.com/api/sendhttp.php',
                'auth_key': 'YOUR_MSG91_AUTH_KEY'
            },
            'twilio': {
                'url': 'https://api.twilio.com/2010-04-01/Accounts',
                'account_sid': 'YOUR_TWILIO_ACCOUNT_SID',
                'auth_token': 'YOUR_TWILIO_AUTH_TOKEN'
            }
        }
    
    def send_sms_textlocal(self, phone_numbers, message):
        """Send SMS using TextLocal API (Popular in India)"""
        try:
            if self.api_configs['textlocal']['api_key'] == 'YOUR_TEXTLOCAL_API_KEY':
                return self._simulate_sms_response(phone_numbers, message, 'TextLocal')
            
            data = {
                'apikey': self.api_configs['textlocal']['api_key'],
                'numbers': ','.join(phone_numbers),
                'message': message,
                'sender': 'ASHA-DOOT'
            }
            response = requests.post(self.api_configs['textlocal']['url'], data=data)
            result = response.json()
            return {
                'status': 'success' if result.get('status') == 'success' else 'error',
                'message': 'SMS sent successfully' if result.get('status') == 'success' else result.get('errors'),
                'provider': 'TextLocal',
                'recipients': phone_numbers,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'provider': 'TextLocal'}
    
    def send_sms_msg91(self, phone_numbers, message):
        """Send SMS using MSG91 API (Indian SMS provider)"""
        try:
            if self.api_configs['msg91']['auth_key'] == 'YOUR_MSG91_AUTH_KEY':
                return self._simulate_sms_response(phone_numbers, message, 'MSG91')
            
            url = (f"{self.api_configs['msg91']['url']}?"
                  f"authkey={self.api_configs['msg91']['auth_key']}&"
                  f"mobiles={','.join(phone_numbers)}&"
                  f"message={message}&"
                  f"sender=ASHADOOT&route=4")
            response = requests.get(url)
            return {
                'status': 'success' if response.status_code == 200 else 'error',
                'message': 'SMS sent successfully' if response.status_code == 200 else response.text,
                'provider': 'MSG91',
                'recipients': phone_numbers,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'provider': 'MSG91'}
    
    def send_emergency_alert(self, phone_numbers, risk_level, location, cases, turbidity):
        """Send emergency health alert SMS"""
        if risk_level == "High":
            message = (f"🚨 URGENT HEALTH ALERT 🚨\nLocation: {location}\nRisk Level: HIGH\nDiarrhea Cases: {cases}\nWater Turbidity: {turbidity} NTU\nIMMEDIATE ACTION REQUIRED!\nContact: 108 for emergency\n-ASHA-doot System")
        elif risk_level == "Medium":
            message = (f"⚠️ HEALTH ALERT ⚠️\nLocation: {location}\nRisk Level: MEDIUM\nCases: {cases} | Turbidity: {turbidity} NTU\nEnhanced monitoring needed\n-ASHA-doot System")
        else:
            message = (f"ℹ️ Health Update\nLocation: {location}\nStatus: Under control\nCases: {cases} | Turbidity: {turbidity} NTU\n-ASHA-doot System")
        
        results = []
        result1 = self.send_sms_textlocal(phone_numbers, message)
        results.append(result1)
        if result1['status'] == 'error':
            result2 = self.send_sms_msg91(phone_numbers, message)
            results.append(result2)
        return results
    
    def send_daily_report(self, phone_numbers, report_data):
        """Send daily health summary report"""
        message = (f"📊 Daily Health Report\nDate: {report_data.get('date', 'Today')}\nTotal Cases: {report_data.get('total_cases', 0)}\nHigh Risk Areas: {report_data.get('high_risk_areas', 0)}\nWater Quality Issues: {report_data.get('water_issues', 0)}\nAction Required: {report_data.get('action_needed', 'None')}\n-ASHA-doot Daily Summary")
        return self.send_sms_textlocal(phone_numbers, message)
    
    def _simulate_sms_response(self, phone_numbers, message, provider):
        """Simulate SMS response for demo purposes"""
        return {
            'status': 'success',
            'message': f'SMS sent successfully via {provider} (Demo Mode)',
            'provider': provider,
            'recipients': phone_numbers,
            'message_text': message[:100] + "..." if len(message) > 100 else message,
            'timestamp': datetime.now().isoformat(),
            'demo_mode': True
        }
    
    def validate_phone_numbers(self, phone_numbers):
        """Validate Indian phone numbers"""
        valid_numbers = []
        invalid_numbers = []
        for number in phone_numbers:
            clean_number = ''.join(filter(str.isdigit, number))
            if len(clean_number) == 10 and clean_number.startswith(('6', '7', '8', '9')):
                valid_numbers.append('+91' + clean_number)
            elif len(clean_number) == 12 and clean_number.startswith('91'):
                valid_numbers.append('+' + clean_number)
            elif clean_number.startswith('+91') and len(clean_number) == 13:
                valid_numbers.append(clean_number)
            else:
                invalid_numbers.append(number)
        return {'valid': valid_numbers, 'invalid': invalid_numbers, 'total_valid': len(valid_numbers)}
    
    def get_sms_status(self):
        """Get SMS service status"""
        return {
            'textlocal_configured': self.api_configs['textlocal']['api_key'] != 'YOUR_TEXTLOCAL_API_KEY',
            'msg91_configured': self.api_configs['msg91']['auth_key'] != 'YOUR_MSG91_AUTH_KEY',
            'demo_mode': (self.api_configs['textlocal']['api_key'] == 'YOUR_TEXTLOCAL_API_KEY' and 
                         self.api_configs['msg91']['auth_key'] == 'YOUR_MSG91_AUTH_KEY')
        }

if __name__ == "__main__":
    sms = SMSService()
    test_numbers = ['+91-9876543210', '9876543210', '91-9876543210', 'invalid-number']
    validation_result = sms.validate_phone_numbers(test_numbers)
    print("Phone Validation Result:", validation_result)
    valid_numbers = validation_result['valid'][:1]
    if valid_numbers:
        alert_result = sms.send_emergency_alert(
            phone_numbers=valid_numbers,
            risk_level="High",
            location="Guwahati Village",
            cases=15,
            turbidity=45.2
        )
        print("Emergency Alert Result:", alert_result)
    status = sms.get_sms_status()
    print("SMS Service Status:", status)
