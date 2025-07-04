#!/usr/bin/env python3
"""
Application State Monitor
Monitors the Streamlit application state and logs to simulate user interactions.
"""

import time
import requests
import json
import os
from datetime import datetime

class AppStateMonitor:
    def __init__(self, base_url="http://localhost:8503"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_log = []
        
    def log_state(self, action, status, details=None):
        """Log application state."""
        entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'status': status,
            'details': details or {}
        }
        self.test_log.append(entry)
        
        status_emoji = {"SUCCESS": "✅", "ERROR": "❌", "INFO": "ℹ️", "WARNING": "⚠️"}
        print(f"{status_emoji.get(status, '•')} {entry['timestamp']} | {action}: {status}")
        if details:
            for key, value in details.items():
                print(f"   📋 {key}: {value}")
    
    def check_app_responsiveness(self):
        """Check if app is responding."""
        try:
            response = self.session.get(f"{self.base_url}/healthz", timeout=3)
            if response.status_code == 200:
                self.log_state("Health Check", "SUCCESS", {"status_code": 200})
                return True
            else:
                self.log_state("Health Check", "ERROR", {"status_code": response.status_code})
                return False
        except Exception as e:
            self.log_state("Health Check", "ERROR", {"error": str(e)})
            return False
    
    def analyze_page_content(self):
        """Analyze current page content."""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text.lower()
            
            # Detect current page state
            page_indicators = {
                'multipage_mode': 'multi-page' in content,
                'navigation_sidebar': 'navigation' in content,
                'data_management': 'data management' in content,
                'analysis_workspace': 'analysis' in content and 'workspace' in content,
                'dashboard_gallery': 'dashboard' in content,
                'export_reporting': 'export' in content,
                'file_uploader': 'file_uploader' in content or 'upload' in content,
                'plotting_active': any(plot in content for plot in ['matplotlib', 'plotly', 'chart', 'plot']),
                'data_loaded': 'dataframe' in content or 'data_loaded' in content,
                'error_present': 'error' in content and ('traceback' in content or 'exception' in content)
            }
            
            detected_features = sum(page_indicators.values())
            
            self.log_state("Page Analysis", "SUCCESS", {
                "detected_features": f"{detected_features}/10",
                "multipage_active": page_indicators['multipage_mode'],
                "navigation_present": page_indicators['navigation_sidebar'],
                "upload_available": page_indicators['file_uploader'],
                "errors_detected": page_indicators['error_present']
            })
            
            return page_indicators
            
        except Exception as e:
            self.log_state("Page Analysis", "ERROR", {"error": str(e)})
            return {}
    
    def check_streamlit_logs(self):
        """Check Streamlit logs for errors."""
        try:
            log_file = "streamlit.log"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_logs = f.readlines()[-50:]  # Last 50 lines
                
                error_indicators = []
                warning_indicators = []
                
                for line in recent_logs:
                    line_lower = line.lower()
                    if 'error' in line_lower or 'exception' in line_lower or 'traceback' in line_lower:
                        error_indicators.append(line.strip())
                    elif 'warning' in line_lower:
                        warning_indicators.append(line.strip())
                
                status = "ERROR" if error_indicators else "WARNING" if warning_indicators else "SUCCESS"
                
                self.log_state("Log Analysis", status, {
                    "errors_found": len(error_indicators),
                    "warnings_found": len(warning_indicators),
                    "recent_errors": error_indicators[:3] if error_indicators else [],
                    "log_file_size": len(recent_logs)
                })
                
                return len(error_indicators) == 0
            else:
                self.log_state("Log Analysis", "WARNING", {"message": "No log file found"})
                return True
                
        except Exception as e:
            self.log_state("Log Analysis", "ERROR", {"error": str(e)})
            return False
    
    def simulate_user_workflow(self):
        """Simulate a typical user workflow."""
        print("🚀 Starting Application State Monitoring...")
        print("=" * 60)
        
        # Step 1: Check app health
        if not self.check_app_responsiveness():
            print("❌ App not responding. Cannot continue testing.")
            return False
        
        # Step 2: Analyze initial state
        page_state = self.analyze_page_content()
        
        # Step 3: Check logs for errors
        logs_clean = self.check_streamlit_logs()
        
        # Step 4: Monitor for a period to see if app is stable
        print("\n🔄 Monitoring application stability for 30 seconds...")
        stable_checks = 0
        for i in range(6):  # 6 checks over 30 seconds
            time.sleep(5)
            if self.check_app_responsiveness():
                stable_checks += 1
            print(f"   Stability check {i+1}/6: {'✅ OK' if stable_checks == i+1 else '❌ ISSUE'}")
        
        stability_rate = (stable_checks / 6) * 100
        self.log_state("Stability Test", "SUCCESS" if stability_rate >= 90 else "WARNING", {
            "stability_rate": f"{stability_rate:.1f}%",
            "successful_checks": f"{stable_checks}/6"
        })
        
        return stability_rate >= 90
    
    def save_monitoring_results(self):
        """Save monitoring results."""
        try:
            results_file = "testing_screenshots/app_monitoring_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'monitoring_time': datetime.now().isoformat(),
                    'app_url': self.base_url,
                    'total_checks': len(self.test_log),
                    'log_entries': self.test_log
                }, f, indent=2)
            
            self.log_state("Results Saved", "SUCCESS", {"file": results_file})
            
        except Exception as e:
            self.log_state("Save Results", "ERROR", {"error": str(e)})

if __name__ == "__main__":
    monitor = AppStateMonitor()
    
    try:
        success = monitor.simulate_user_workflow()
        monitor.save_monitoring_results()
        
        print("\n" + "="*60)
        print("📊 MONITORING SUMMARY")
        print("="*60)
        
        total_actions = len(monitor.test_log)
        successful_actions = len([log for log in monitor.test_log if log['status'] == 'SUCCESS'])
        
        print(f"📈 Success Rate: {(successful_actions/total_actions)*100:.1f}%")
        print(f"✅ Successful Actions: {successful_actions}")
        print(f"📊 Total Actions: {total_actions}")
        
        if success:
            print("\n🎉 APPLICATION IS STABLE AND RESPONSIVE")
            print("✅ Ready for detailed functional testing")
        else:
            print("\n⚠️ APPLICATION STABILITY ISSUES DETECTED")
            print("❗ May need troubleshooting before proceeding")
            
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted by user")
    except Exception as e:
        print(f"\n💥 Monitoring failed: {e}")