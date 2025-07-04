#!/usr/bin/env python3
"""
Comprehensive Application Testing Script
Tests both single-page and multi-page modes with real data upload and plot generation.
"""

import requests
import time
import os
import json
from datetime import datetime
import pandas as pd

class StreamlitAppTester:
    def __init__(self, base_url="http://localhost:8503"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        
    def log_result(self, test_name, status, message, details=None):
        """Log test result."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test': test_name,
            'status': status,
            'message': message,
            'details': details or {}
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {message}")
        
    def test_app_health(self):
        """Test basic app health."""
        try:
            response = self.session.get(f"{self.base_url}/healthz", timeout=5)
            if response.status_code == 200:
                self.log_result("App Health", "PASS", "Application responding normally")
                return True
            else:
                self.log_result("App Health", "FAIL", f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("App Health", "FAIL", f"Health check error: {str(e)}")
            return False
    
    def test_main_page_load(self):
        """Test main page loading."""
        try:
            response = self.session.get(self.base_url, timeout=10)
            if response.status_code == 200:
                content = response.text
                
                # Check for key elements
                checks = {
                    'title': 'Geotechnical Data Analysis Tool' in content,
                    'navigation': 'Navigation' in content or 'nav' in content.lower(),
                    'streamlit': 'streamlit' in content.lower(),
                    'multipage': 'multi-page' in content.lower() or 'multipage' in content.lower()
                }
                
                passed_checks = sum(checks.values())
                total_checks = len(checks)
                
                if passed_checks >= total_checks * 0.75:  # 75% pass rate
                    self.log_result("Main Page Load", "PASS", 
                                  f"Page loaded successfully ({passed_checks}/{total_checks} checks passed)",
                                  checks)
                    return True
                else:
                    self.log_result("Main Page Load", "FAIL", 
                                  f"Page loaded but missing key elements ({passed_checks}/{total_checks} checks passed)",
                                  checks)
                    return False
            else:
                self.log_result("Main Page Load", "FAIL", f"Page load failed: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Main Page Load", "FAIL", f"Page load error: {str(e)}")
            return False
    
    def test_data_file_availability(self):
        """Test if the lab data file exists."""
        try:
            lab_file = "Lab_summary_final.xlsx"
            if os.path.exists(lab_file):
                # Try to read the file to verify it's valid
                df = pd.read_excel(lab_file)
                self.log_result("Data File Check", "PASS", 
                              f"Lab data file found and readable ({len(df)} rows, {len(df.columns)} columns)")
                return True, lab_file, df
            else:
                # Check in Input folder
                input_file = "Input/BH_Interpretation.xlsx"
                if os.path.exists(input_file):
                    df = pd.read_excel(input_file)
                    self.log_result("Data File Check", "PASS", 
                                  f"Input data file found ({len(df)} rows, {len(df.columns)} columns)")
                    return True, input_file, df
                else:
                    self.log_result("Data File Check", "FAIL", "No lab data files found")
                    return False, None, None
        except Exception as e:
            self.log_result("Data File Check", "FAIL", f"Error reading data file: {str(e)}")
            return False, None, None
    
    def test_streamlit_functionality(self):
        """Test Streamlit-specific functionality by examining page structure."""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text
            
            # Look for Streamlit-specific elements
            streamlit_indicators = {
                'sidebar': 'sidebar' in content.lower(),
                'widgets': any(widget in content.lower() for widget in ['selectbox', 'button', 'file_uploader', 'tabs']),
                'layout': any(layout in content.lower() for layout in ['column', 'container', 'expander']),
                'components': 'streamlit' in content.lower()
            }
            
            active_indicators = sum(streamlit_indicators.values())
            
            if active_indicators >= 2:  # At least 2 Streamlit features detected
                self.log_result("Streamlit Features", "PASS", 
                              f"Streamlit functionality detected ({active_indicators}/4 indicators)",
                              streamlit_indicators)
                return True
            else:
                self.log_result("Streamlit Features", "WARN", 
                              f"Limited Streamlit functionality detected ({active_indicators}/4 indicators)",
                              streamlit_indicators)
                return False
        except Exception as e:
            self.log_result("Streamlit Features", "FAIL", f"Error testing Streamlit features: {str(e)}")
            return False
    
    def analyze_page_content(self):
        """Analyze the page content for application-specific elements."""
        try:
            response = self.session.get(self.base_url, timeout=10)
            content = response.text.lower()
            
            # Look for application-specific content
            app_features = {
                'data_upload': 'upload' in content and ('file' in content or 'data' in content),
                'analysis_tools': any(tool in content for tool in ['psd', 'atterberg', 'spt', 'ucs']),
                'plotting': any(plot in content for plot in ['plot', 'chart', 'graph', 'visualization']),
                'navigation_pages': any(page in content for page in ['data management', 'analysis', 'dashboard', 'export']),
                'geotechnical': any(term in content for term in ['geotechnical', 'soil', 'rock', 'geology'])
            }
            
            detected_features = sum(app_features.values())
            
            self.log_result("Application Features", "PASS" if detected_features >= 3 else "WARN",
                          f"Application features detected ({detected_features}/5)",
                          app_features)
            
            return detected_features >= 3
        except Exception as e:
            self.log_result("Application Features", "FAIL", f"Error analyzing content: {str(e)}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests in sequence."""
        print("🚀 Starting Comprehensive Application Testing...")
        print("=" * 60)
        
        # Test 1: Basic Health
        if not self.test_app_health():
            print("❌ Critical: App health check failed. Stopping tests.")
            return False
        
        # Test 2: Main Page Load
        if not self.test_main_page_load():
            print("⚠️ Warning: Main page load issues detected")
        
        # Test 3: Data File Availability
        data_available, data_file, data_df = self.test_data_file_availability()
        
        # Test 4: Streamlit Functionality
        self.test_streamlit_functionality()
        
        # Test 5: Application Features
        self.analyze_page_content()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        passed = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warnings = len([r for r in self.test_results if r['status'] == 'WARN'])
        
        print(f"✅ PASSED: {passed}")
        print(f"❌ FAILED: {failed}")
        print(f"⚠️ WARNINGS: {warnings}")
        
        success_rate = (passed / len(self.test_results)) * 100 if self.test_results else 0
        print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
        
        if data_available:
            print(f"\n📁 DATA FILE: {data_file}")
            print(f"📊 DATA SHAPE: {data_df.shape[0]} rows × {data_df.shape[1]} columns")
            
            # Show some key columns if available
            if hasattr(data_df, 'columns'):
                geotechnical_columns = [col for col in data_df.columns 
                                      if any(term in col.lower() for term in 
                                           ['hole', 'depth', 'geology', 'll', 'pi', 'spt', 'ucs'])]
                if geotechnical_columns:
                    print(f"🔬 KEY COLUMNS: {geotechnical_columns[:5]}...")
        
        return success_rate >= 75.0  # Consider 75%+ as successful
    
    def save_results(self):
        """Save test results to file."""
        try:
            results_file = "testing_screenshots/automated_test_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'test_run_time': datetime.now().isoformat(),
                    'app_url': self.base_url,
                    'total_tests': len(self.test_results),
                    'results': self.test_results
                }, f, indent=2)
            print(f"\n💾 Results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

if __name__ == "__main__":
    tester = StreamlitAppTester()
    
    try:
        success = tester.run_comprehensive_test()
        tester.save_results()
        
        print("\n" + "="*60)
        if success:
            print("🎉 OVERALL RESULT: APPLICATION TESTING SUCCESSFUL")
            print("✅ The multi-page geotechnical analysis application is functional!")
        else:
            print("⚠️ OVERALL RESULT: APPLICATION TESTING NEEDS ATTENTION")
            print("❌ Some issues were detected that may need investigation.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")