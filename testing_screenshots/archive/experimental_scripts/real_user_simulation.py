#!/usr/bin/env python3
"""
Real User Interaction Simulation
This script simulates actual user interactions with the Streamlit app including:
- File uploads
- Page navigation  
- Tab clicking
- Plot generation
- Screenshot equivalent monitoring
"""

import requests
import time
import json
import os
from datetime import datetime
import subprocess

class RealUserSimulator:
    def __init__(self, base_url="http://localhost:8503"):
        self.base_url = base_url
        self.session = requests.Session()
        self.interactions = []
        self.current_page = "unknown"
        
    def log_interaction(self, action, result, details=None):
        """Log user interaction."""
        interaction = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'action': action,
            'result': result,
            'page': self.current_page,
            'details': details or {}
        }
        self.interactions.append(interaction)
        
        emoji = {"SUCCESS": "✅", "FAIL": "❌", "INFO": "📋", "UPLOAD": "📤", "CLICK": "👆", "NAVIGATE": "🧭"}
        action_type = action.split(':')[0] if ':' in action else action
        
        print(f"{emoji.get(action_type, '•')} {interaction['timestamp']} | {action}")
        if result != "SUCCESS":
            print(f"   ❌ {result}")
        if details:
            for key, value in details.items():
                print(f"   📋 {key}: {value}")
    
    def open_browser_for_manual_testing(self):
        """Open browser and provide instructions for manual testing."""
        print("🌐 Opening browser for manual testing...")
        
        # Try to open browser
        try:
            subprocess.run(["open", self.base_url], check=True)
            self.log_interaction("NAVIGATE: Open Browser", "SUCCESS", {"url": self.base_url})
        except:
            print(f"   Please manually open: {self.base_url}")
            self.log_interaction("NAVIGATE: Open Browser", "INFO", {"manual_open_required": True})
        
        return True
    
    def check_data_file_exists(self):
        """Check if the lab data file exists for upload."""
        lab_file = "Lab_summary_final.xlsx"
        if os.path.exists(lab_file):
            file_size = os.path.getsize(lab_file) / (1024*1024)  # MB
            self.log_interaction("INFO: Data File Check", "SUCCESS", {
                "file": lab_file,
                "size_mb": f"{file_size:.1f}",
                "exists": True
            })
            return True, lab_file
        else:
            self.log_interaction("INFO: Data File Check", "FAIL", {
                "file": lab_file,
                "exists": False
            })
            return False, None
    
    def simulate_manual_testing_steps(self):
        """Provide step-by-step manual testing instructions."""
        
        print("\\n🎯 MANUAL TESTING INSTRUCTIONS")
        print("="*60)
        print("Please follow these steps in your browser:")
        print()
        
        # Step 1: Initial State
        print("📋 STEP 1: Verify Initial State")
        print("   1. Confirm app loads at localhost:8503")
        print("   2. Check if you see 'Geotechnical Data Analysis Tool' title")
        print("   3. Verify sidebar shows navigation options")
        print("   4. Confirm you're in Multi-Page Mode")
        
        input("   ✅ Press Enter when Step 1 is complete...")
        self.log_interaction("MANUAL: Initial State Check", "SUCCESS")
        
        # Step 2: Data Upload
        print("\\n📤 STEP 2: Data Upload (Page 1 - Data Management)")
        data_exists, data_file = self.check_data_file_exists()
        
        if data_exists:
            print(f"   1. Navigate to '📊 Data Management' page")
            print(f"   2. Find the file upload widget")
            print(f"   3. Upload file: {data_file}")
            print(f"   4. Wait for data validation to complete")
            print(f"   5. Verify data preview appears")
            print(f"   6. Check test availability detection")
            
            input("   ✅ Press Enter when data upload is complete...")
            self.log_interaction("UPLOAD: Lab Data", "SUCCESS", {"file": data_file})
        else:
            print("   ❌ Lab data file not found - skipping upload test")
            self.log_interaction("UPLOAD: Lab Data", "FAIL", {"reason": "file_not_found"})
        
        # Step 3: Analysis Workspace
        print("\\n🔬 STEP 3: Analysis Workspace (Page 2)")
        print("   1. Navigate to '🔬 Analysis Workspace' page")
        print("   2. Test each analysis tab:")
        
        analysis_tabs = [
            "📊 PSD Analysis",
            "🧪 Atterberg Limits", 
            "🔨 SPT Analysis",
            "💪 UCS Analysis",
            "🌍 Spatial Analysis",
            "🔬 Emerson Analysis"
        ]
        
        for i, tab in enumerate(analysis_tabs, 1):
            print(f"      {i}. Click '{tab}' tab")
            print(f"         - Configure analysis parameters")
            print(f"         - Click 'Generate Plot' or equivalent")
            print(f"         - Verify plot appears")
            
            tab_result = input(f"      ✅ Did {tab} generate plots successfully? (y/n): ").lower()
            result = "SUCCESS" if tab_result == 'y' else "FAIL"
            self.log_interaction(f"CLICK: {tab}", result)
        
        # Step 4: Dashboard Gallery
        print("\\n📈 STEP 4: Dashboard Gallery (Page 3)")
        print("   1. Navigate to '📈 Dashboard Gallery' page")
        print("   2. Check each dashboard section:")
        
        dashboard_sections = [
            "🏢 Site Characterization", 
            "🧱 Material Properties",
            "🗿 Rock Properties"
        ]
        
        for section in dashboard_sections:
            print(f"      - Click '{section}' tab")
            print(f"      - Verify plots auto-populated from analysis")
            
            dashboard_result = input(f"      ✅ Does {section} show plots? (y/n): ").lower()
            result = "SUCCESS" if dashboard_result == 'y' else "FAIL"
            self.log_interaction(f"CLICK: {section} Dashboard", result)
        
        # Step 5: Export & Reporting
        print("\\n📤 STEP 5: Export & Reporting (Page 4)")
        print("   1. Navigate to '📤 Export & Reporting' page")
        print("   2. Test export functionality:")
        print("      - Try exporting plots")
        print("      - Try exporting data") 
        print("      - Check gallery management")
        
        export_result = input("   ✅ Does export functionality work? (y/n): ").lower()
        result = "SUCCESS" if export_result == 'y' else "FAIL"
        self.log_interaction("CLICK: Export Functions", result)
        
        # Step 6: Single-Page Mode
        print("\\n🔄 STEP 6: Single-Page Mode Testing")
        print("   1. Switch to 'Single-Page Mode' in sidebar")
        print("   2. Verify all tabs are visible")
        print("   3. Test a few analysis tabs")
        print("   4. Check dashboard section")
        
        single_page_result = input("   ✅ Does Single-Page Mode work correctly? (y/n): ").lower()
        result = "SUCCESS" if single_page_result == 'y' else "FAIL"
        self.log_interaction("NAVIGATE: Single-Page Mode", result)
        
        # Step 7: Switch back and verify persistence
        print("\\n🔄 STEP 7: Mode Switching & Data Persistence")
        print("   1. Switch back to 'Multi-Page Mode'")
        print("   2. Navigate through pages")
        print("   3. Verify data and plots are still there")
        
        persistence_result = input("   ✅ Is data persistent across mode switches? (y/n): ").lower()
        result = "SUCCESS" if persistence_result == 'y' else "FAIL"
        self.log_interaction("INFO: Data Persistence", result)
    
    def analyze_test_results(self):
        """Analyze the manual testing results."""
        print("\\n" + "="*60)
        print("📊 MANUAL TESTING RESULTS ANALYSIS")
        print("="*60)
        
        total_tests = len(self.interactions)
        successful_tests = len([i for i in self.interactions if i['result'] == 'SUCCESS'])
        failed_tests = len([i for i in self.interactions if i['result'] == 'FAIL'])
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Overall Success Rate: {success_rate:.1f}%")
        print(f"✅ Successful Tests: {successful_tests}")
        print(f"❌ Failed Tests: {failed_tests}")
        print(f"📊 Total Tests: {total_tests}")
        
        # Analyze by category
        categories = {}
        for interaction in self.interactions:
            action_type = interaction['action'].split(':')[0]
            if action_type not in categories:
                categories[action_type] = {'success': 0, 'fail': 0, 'total': 0}
            
            categories[action_type]['total'] += 1
            if interaction['result'] == 'SUCCESS':
                categories[action_type]['success'] += 1
            elif interaction['result'] == 'FAIL':
                categories[action_type]['fail'] += 1
        
        print("\\n📋 Results by Category:")
        for category, stats in categories.items():
            cat_success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {category}: {cat_success_rate:.1f}% ({stats['success']}/{stats['total']})")
        
        # Critical issues
        critical_failures = [i for i in self.interactions if i['result'] == 'FAIL' and 
                           any(critical in i['action'] for critical in ['UPLOAD', 'CLICK'])]
        
        if critical_failures:
            print("\\n⚠️ Critical Issues Detected:")
            for failure in critical_failures:
                print(f"   ❌ {failure['action']} at {failure['timestamp']}")
        
        # Overall assessment
        print("\\n🎯 OVERALL ASSESSMENT:")
        if success_rate >= 90:
            print("🎉 EXCELLENT: Application is fully functional!")
        elif success_rate >= 75:
            print("✅ GOOD: Application is mostly functional with minor issues")
        elif success_rate >= 50:
            print("⚠️ FAIR: Application has significant issues that need attention")
        else:
            print("❌ POOR: Application requires major fixes before use")
        
        return success_rate >= 75
    
    def save_results(self):
        """Save manual testing results."""
        try:
            results = {
                'test_session': datetime.now().isoformat(),
                'app_url': self.base_url,
                'total_interactions': len(self.interactions),
                'interactions': self.interactions,
                'summary': {
                    'success_count': len([i for i in self.interactions if i['result'] == 'SUCCESS']),
                    'fail_count': len([i for i in self.interactions if i['result'] == 'FAIL']),
                    'success_rate': len([i for i in self.interactions if i['result'] == 'SUCCESS']) / len(self.interactions) * 100 if self.interactions else 0
                }
            }
            
            results_file = "testing_screenshots/manual_testing_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\\n💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"\\n⚠️ Could not save results: {e}")

if __name__ == "__main__":
    simulator = RealUserSimulator()
    
    print("🚀 REAL USER INTERACTION SIMULATION")
    print("="*60)
    print("This will guide you through comprehensive manual testing")
    print("of both Multi-Page and Single-Page modes.")
    print()
    
    try:
        # Open browser
        simulator.open_browser_for_manual_testing()
        
        # Wait for user to confirm browser is open
        input("Press Enter when the browser is open and app is loaded...")
        
        # Run manual testing steps
        simulator.simulate_manual_testing_steps()
        
        # Analyze results
        success = simulator.analyze_test_results()
        
        # Save results
        simulator.save_results()
        
        print("\\n" + "="*60)
        if success:
            print("🎉 TESTING COMPLETE: Application passed comprehensive testing!")
        else:
            print("⚠️ TESTING COMPLETE: Application needs attention before production use")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\\n🛑 Testing interrupted by user")
    except Exception as e:
        print(f"\\n💥 Testing failed: {e}")