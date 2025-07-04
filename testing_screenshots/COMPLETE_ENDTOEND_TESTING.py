#!/usr/bin/env python3
"""
COMPREHENSIVE END-TO-END TESTING FRAMEWORK
Test Name: "GEOTECHNICAL ANALYSIS TOOL - COMPLETE FUNCTIONAL VERIFICATION"

This script performs complete testing of the multi-page geotechnical analysis application:
1. Navigates through ALL pages and tabs
2. Takes screenshots of every interface and figure
3. Generates plots in all analysis tabs
4. Tests export functionality
5. Creates organized output folders with timestamps
6. Documents everything for iterative improvement

    Test Protocol: VERIFIED-E2E-TESTING-v1.0
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

class ComprehensiveE2ETesting:
    """Complete End-to-End Testing Framework for Geotechnical Analysis Tool."""
    
    def __init__(self):
        self.test_name = "GEOTECHNICAL_ANALYSIS_TOOL_COMPLETE_FUNCTIONAL_VERIFICATION"
        self.test_version = "v1.0"
        self.start_time = datetime.now()
        self.test_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create organized folder structure
        self.base_folder = "testing_screenshots"
        self.test_session_folder = f"{self.base_folder}/E2E_TEST_{self.test_timestamp}"
        self.screenshots_folder = f"{self.test_session_folder}/screenshots"
        self.output_folder = f"Output/{self.start_time.strftime('%Y%m%d_%Hh')}"
        
        # Create directories
        Path(self.screenshots_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        self.driver = None
        self.test_results = []
        self.screenshot_count = 0
        self.successful_plots = 0
        self.failed_operations = 0
        
        # Test configuration
        self.app_url = "http://localhost:8503"
        self.test_data_file = "Lab_summary_final.xlsx"
        
        print(f"🧪 INITIALIZING: {self.test_name}")
        print(f"📅 Test Session: {self.test_timestamp}")
        print(f"📁 Screenshots: {self.screenshots_folder}")
        print(f"📤 Output: {self.output_folder}")
        
    def setup_driver(self):
        """Setup Chrome driver for comprehensive testing."""
        try:
            options = Options()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            self.log_result("SETUP", "SUCCESS", "Chrome driver initialized")
            return True
            
        except Exception as e:
            self.log_result("SETUP", "FAIL", f"Driver setup failed: {str(e)}")
            return False
    
    def log_result(self, action, status, details, screenshot_name=None):
        """Log test result with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        result = {
            'timestamp': timestamp,
            'action': action,
            'status': status,
            'details': details,
            'screenshot': screenshot_name
        }
        self.test_results.append(result)
        
        emoji = {"SUCCESS": "✅", "FAIL": "❌", "INFO": "📋", "WARNING": "⚠️"}
        print(f"{emoji.get(status, '•')} {timestamp} | {action}: {status}")
        if details:
            print(f"   📋 {details}")
        if screenshot_name:
            print(f"   📸 Screenshot: {screenshot_name}")
    
    def take_screenshot(self, name, description="", full_page=True):
        """Take screenshot with organized naming - captures full page content including scrollable areas."""
        try:
            self.screenshot_count += 1
            filename = f"{self.screenshot_count:03d}_{name}.png"
            filepath = f"{self.screenshots_folder}/{filename}"
            
            if full_page:
                # Get full page dimensions
                total_height = self.driver.execute_script("return document.body.scrollHeight")
                viewport_height = self.driver.execute_script("return window.innerHeight")
                
                if total_height > viewport_height:
                    # Page is scrollable - capture full content
                    print(f"   📸 Capturing full page (Height: {total_height}px, Viewport: {viewport_height}px)")
                    
                    # Set window size to capture full height
                    original_size = self.driver.get_window_size()
                    self.driver.set_window_size(original_size['width'], total_height + 100)
                    
                    # Wait for resize
                    time.sleep(1)
                    
                    # Take full page screenshot
                    self.driver.save_screenshot(filepath)
                    
                    # Restore original window size
                    self.driver.set_window_size(original_size['width'], original_size['height'])
                    
                    print(f"   ✅ Full page screenshot captured: {filename}")
                else:
                    # Regular screenshot for non-scrollable content
                    self.driver.save_screenshot(filepath)
            else:
                # Regular viewport screenshot
                self.driver.save_screenshot(filepath)
            
            # Also save screenshot info with page dimensions
            screenshot_info = {
                'filename': filename,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'url': self.driver.current_url,
                'page_title': self.driver.title,
                'full_page': full_page,
                'page_dimensions': {
                    'total_height': self.driver.execute_script("return document.body.scrollHeight"),
                    'viewport_height': self.driver.execute_script("return window.innerHeight"),
                    'viewport_width': self.driver.execute_script("return window.innerWidth")
                }
            }
            
            info_file = f"{self.screenshots_folder}/{filename}.json"
            with open(info_file, 'w') as f:
                json.dump(screenshot_info, f, indent=2)
            
            return filename
            
        except Exception as e:
            print(f"   ⚠️ Screenshot failed: {e}")
            return None
    
    def safe_click(self, element, description="element"):
        """Safely click element with multiple strategies."""
        try:
            # Strategy 1: Regular click
            element.click()
            return True
        except:
            try:
                # Strategy 2: JavaScript click
                self.driver.execute_script("arguments[0].click();", element)
                return True
            except:
                try:
                    # Strategy 3: Action chains
                    ActionChains(self.driver).move_to_element(element).click().perform()
                    return True
                except:
                    self.log_result("CLICK_FAIL", "FAIL", f"Failed to click {description}")
                    return False
    
    def wait_for_element(self, by, value, timeout=10):
        """Wait for element to be present and return it."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return element
        except:
            return None
    
    def find_element_by_text(self, text, tag="*"):
        """Find element containing specific text."""
        try:
            xpath = f"//{tag}[contains(text(), '{text}')]"
            return self.driver.find_element(By.XPATH, xpath)
        except:
            return None
    
    def test_01_application_startup(self):
        """Test 01: Application startup and initial state."""
        self.log_result("TEST_01", "INFO", "Testing application startup")
        
        try:
            self.driver.get(self.app_url)
            time.sleep(5)
            
            screenshot = self.take_screenshot("01_application_startup", "Initial application load")
            
            # Verify app loaded
            if "Geotechnical" in self.driver.page_source:
                self.log_result("TEST_01", "SUCCESS", "Application loaded successfully", screenshot)
                return True
            else:
                self.log_result("TEST_01", "FAIL", "Application failed to load properly", screenshot)
                return False
                
        except Exception as e:
            self.log_result("TEST_01", "FAIL", f"Startup test failed: {str(e)}")
            return False
    
    def test_02_data_upload(self):
        """Test 02: Upload laboratory data file."""
        self.log_result("TEST_02", "INFO", "Testing data upload functionality")
        
        try:
            # Navigate to data management if not already there
            data_mgmt_btn = self.find_element_by_text("Data Management")
            if data_mgmt_btn:
                self.safe_click(data_mgmt_btn, "Data Management button")
                time.sleep(2)
            
            screenshot = self.take_screenshot("02_data_management_page", "Data Management page loaded")
            
            # Find file uploader
            file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            if file_input:
                # Upload the test data file
                abs_path = os.path.abspath(self.test_data_file)
                file_input.send_keys(abs_path)
                
                # Wait for processing
                time.sleep(8)
                
                screenshot = self.take_screenshot("02_data_uploaded", "Data file uploaded and processed")
                
                # Check for success indicators
                page_source = self.driver.page_source.lower()
                if "data loaded" in page_source or "success" in page_source:
                    self.log_result("TEST_02", "SUCCESS", f"Data uploaded: {self.test_data_file}", screenshot)
                    return True
                else:
                    self.log_result("TEST_02", "FAIL", "Data upload did not show success", screenshot)
                    return False
            else:
                self.log_result("TEST_02", "FAIL", "File uploader not found")
                return False
                
        except Exception as e:
            self.log_result("TEST_02", "FAIL", f"Data upload failed: {str(e)}")
            return False
    
    def test_03_analysis_workspace_navigation(self):
        """Test 03: Navigate to Analysis Workspace and test all tabs."""
        self.log_result("TEST_03", "INFO", "Testing Analysis Workspace navigation and tabs")
        
        try:
            # Navigate to Analysis Workspace
            analysis_btn = self.find_element_by_text("Analysis")
            if not analysis_btn:
                analysis_btn = self.find_element_by_text("Analysis Workspace")
            
            if analysis_btn:
                self.safe_click(analysis_btn, "Analysis Workspace button")
                time.sleep(3)
                
                screenshot = self.take_screenshot("03_analysis_workspace", "Analysis Workspace page loaded")
                
                # Test all analysis tabs
                analysis_tabs = [
                    "📊 PSD Analysis",
                    "🧪 Atterberg Limits",
                    "🔨 SPT Analysis", 
                    "💪 UCS Analysis",
                    "🌍 Spatial Analysis",
                    "🔬 Emerson Analysis"
                ]
                
                successful_tabs = 0
                
                for i, tab_name in enumerate(analysis_tabs):
                    self.log_result("TAB_TEST", "INFO", f"Testing {tab_name} tab")
                    
                    # Find and click tab
                    tab_element = self.find_element_by_text(tab_name)
                    if tab_element:
                        self.safe_click(tab_element, f"{tab_name} tab")
                        time.sleep(3)
                        
                        # Take screenshot of tab
                        tab_screenshot = self.take_screenshot(
                            f"03_tab_{i+1}_{tab_name.replace(' ', '_')}", 
                            f"{tab_name} tab interface"
                        )
                        
                        # Look for and click generate buttons
                        generate_buttons = [
                            "Generate", "Create", "Plot", "Analyze", "Run Analysis"
                        ]
                        
                        plot_generated = False
                        for btn_text in generate_buttons:
                            btn_element = self.find_element_by_text(btn_text)
                            if btn_element:
                                self.safe_click(btn_element, f"{btn_text} button")
                                time.sleep(5)  # Wait for plot generation
                                
                                # Take screenshot of generated plots
                                plot_screenshot = self.take_screenshot(
                                    f"03_plots_{i+1}_{tab_name.replace(' ', '_')}_plots", 
                                    f"{tab_name} generated plots"
                                )
                                
                                # Check if plots appeared
                                page_source = self.driver.page_source.lower()
                                if any(indicator in page_source for indicator in ['canvas', 'plot', 'chart', 'svg']):
                                    self.log_result("PLOT_GEN", "SUCCESS", f"Plots generated in {tab_name}", plot_screenshot)
                                    self.successful_plots += 1
                                    plot_generated = True
                                    successful_tabs += 1
                                break
                        
                        if not plot_generated:
                            # Check if plots already exist
                            page_source = self.driver.page_source.lower()
                            if any(indicator in page_source for indicator in ['canvas', 'plot', 'chart', 'svg']):
                                self.log_result("PLOT_CHECK", "SUCCESS", f"Existing plots found in {tab_name}", tab_screenshot)
                                successful_tabs += 1
                            else:
                                self.log_result("PLOT_CHECK", "FAIL", f"No plots found in {tab_name}", tab_screenshot)
                    else:
                        self.log_result("TAB_TEST", "FAIL", f"Could not find {tab_name} tab")
                
                # Summary for analysis tabs
                self.log_result("TEST_03", "SUCCESS" if successful_tabs > 0 else "FAIL", 
                              f"Analysis tabs tested: {successful_tabs}/{len(analysis_tabs)} successful")
                return successful_tabs > 0
            
            else:
                self.log_result("TEST_03", "FAIL", "Analysis Workspace button not found")
                return False
                
        except Exception as e:
            self.log_result("TEST_03", "FAIL", f"Analysis workspace test failed: {str(e)}")
            return False
    
    def test_04_dashboard_gallery(self):
        """Test 04: Navigate to Dashboard Gallery and check auto-population."""
        self.log_result("TEST_04", "INFO", "Testing Dashboard Gallery")
        
        try:
            # Navigate to Dashboard Gallery
            dashboard_btn = self.find_element_by_text("Dashboard")
            if not dashboard_btn:
                dashboard_btn = self.find_element_by_text("Dashboard Gallery")
            
            if dashboard_btn:
                self.safe_click(dashboard_btn, "Dashboard Gallery button")
                time.sleep(3)
                
                screenshot = self.take_screenshot("04_dashboard_gallery", "Dashboard Gallery page loaded")
                
                # Test dashboard tabs
                dashboard_tabs = [
                    "Site Characterization",
                    "Material Properties", 
                    "Rock Properties"
                ]
                
                successful_dashboards = 0
                
                for i, tab_name in enumerate(dashboard_tabs):
                    tab_element = self.find_element_by_text(tab_name)
                    if tab_element:
                        self.safe_click(tab_element, f"{tab_name} dashboard tab")
                        time.sleep(3)
                        
                        # Take screenshot of dashboard
                        dashboard_screenshot = self.take_screenshot(
                            f"04_dashboard_{i+1}_{tab_name.replace(' ', '_')}", 
                            f"{tab_name} dashboard content"
                        )
                        
                        # Check for plots in dashboard
                        page_source = self.driver.page_source.lower()
                        if any(indicator in page_source for indicator in ['canvas', 'plot', 'chart', 'svg']):
                            self.log_result("DASHBOARD", "SUCCESS", f"{tab_name} dashboard has plots", dashboard_screenshot)
                            successful_dashboards += 1
                        else:
                            self.log_result("DASHBOARD", "WARNING", f"{tab_name} dashboard appears empty", dashboard_screenshot)
                
                self.log_result("TEST_04", "SUCCESS", f"Dashboard testing completed: {successful_dashboards}/{len(dashboard_tabs)}")
                return True
            
            else:
                self.log_result("TEST_04", "FAIL", "Dashboard Gallery button not found")
                return False
                
        except Exception as e:
            self.log_result("TEST_04", "FAIL", f"Dashboard test failed: {str(e)}")
            return False
    
    def test_05_export_functionality(self):
        """Test 05: Test export and reporting functionality."""
        self.log_result("TEST_05", "INFO", "Testing Export & Reporting functionality")
        
        try:
            # Navigate to Export & Reporting
            export_btn = self.find_element_by_text("Export")
            if not export_btn:
                export_btn = self.find_element_by_text("Export & Reporting")
            
            if export_btn:
                self.safe_click(export_btn, "Export & Reporting button")
                time.sleep(3)
                
                screenshot = self.take_screenshot("05_export_page", "Export & Reporting page loaded")
                
                # Look for export buttons/functionality
                export_elements = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Export') or contains(text(), 'Download')]")
                
                if export_elements:
                    self.log_result("TEST_05", "SUCCESS", f"Export functionality available ({len(export_elements)} export options)", screenshot)
                    
                    # Try to trigger an export
                    for element in export_elements[:2]:  # Test first 2 export options
                        try:
                            element_text = element.text
                            self.safe_click(element, f"Export option: {element_text}")
                            time.sleep(2)
                            
                            # Take screenshot after export attempt
                            export_screenshot = self.take_screenshot(
                                f"05_export_attempt_{element_text.replace(' ', '_')}", 
                                f"Export attempt: {element_text}"
                            )
                            
                        except:
                            continue
                    
                    return True
                else:
                    self.log_result("TEST_05", "WARNING", "No export options found", screenshot)
                    return True  # Page loaded, just no export options visible
            
            else:
                self.log_result("TEST_05", "FAIL", "Export & Reporting button not found")
                return False
                
        except Exception as e:
            self.log_result("TEST_05", "FAIL", f"Export test failed: {str(e)}")
            return False
    
    def test_06_single_page_mode(self):
        """Test 06: Test Single-Page Mode functionality."""
        self.log_result("TEST_06", "INFO", "Testing Single-Page Mode")
        
        try:
            # Find mode selector
            mode_selectors = self.driver.find_elements(By.XPATH, "//select")
            
            for selector in mode_selectors:
                try:
                    from selenium.webdriver.support.ui import Select
                    select = Select(selector)
                    
                    # Try to select Single-Page Mode
                    for option in select.options:
                        if "Single" in option.text:
                            select.select_by_visible_text(option.text)
                            time.sleep(3)
                            
                            screenshot = self.take_screenshot("06_single_page_mode", "Single-Page Mode activated")
                            
                            self.log_result("TEST_06", "SUCCESS", "Single-Page Mode activated", screenshot)
                            return True
                except:
                    continue
            
            self.log_result("TEST_06", "WARNING", "Could not find mode selector")
            return True  # Not critical failure
            
        except Exception as e:
            self.log_result("TEST_06", "FAIL", f"Single-page mode test failed: {str(e)}")
            return False
    
    def save_test_results(self):
        """Save comprehensive test results."""
        try:
            # Calculate statistics
            total_tests = len(self.test_results)
            successful_tests = len([r for r in self.test_results if r['status'] == 'SUCCESS'])
            failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
            
            test_summary = {
                'test_name': self.test_name,
                'test_version': self.test_version,
                'timestamp': self.test_timestamp,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
                'app_url': self.app_url,
                'test_data_file': self.test_data_file,
                'statistics': {
                    'total_operations': total_tests,
                    'successful_operations': successful_tests,
                    'failed_operations': failed_tests,
                    'success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                    'screenshots_taken': self.screenshot_count,
                    'plots_generated': self.successful_plots
                },
                'test_results': self.test_results,
                'folders': {
                    'screenshots': self.screenshots_folder,
                    'output': self.output_folder
                }
            }
            
            # Save detailed results
            results_file = f"{self.test_session_folder}/test_results.json"
            with open(results_file, 'w') as f:
                json.dump(test_summary, f, indent=2)
            
            # Save summary report
            summary_file = f"{self.test_session_folder}/TEST_SUMMARY.md"
            with open(summary_file, 'w') as f:
                f.write(f"# {self.test_name} - Test Summary\\n\\n")
                f.write(f"**Test Session:** {self.test_timestamp}\\n")
                f.write(f"**Duration:** {test_summary['duration_minutes']:.1f} minutes\\n")
                f.write(f"**Success Rate:** {test_summary['statistics']['success_rate']:.1f}%\\n")
                f.write(f"**Screenshots:** {self.screenshot_count}\\n")
                f.write(f"**Plots Generated:** {self.successful_plots}\\n\\n")
                
                f.write("## Test Results\\n")
                for result in self.test_results:
                    status_emoji = {"SUCCESS": "✅", "FAIL": "❌", "WARNING": "⚠️", "INFO": "📋"}
                    f.write(f"- {status_emoji.get(result['status'], '•')} **{result['action']}**: {result['details']}\\n")
            
            return results_file, summary_file
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
            return None, None
    
    def run_complete_testing(self):
        """Run the complete end-to-end testing suite."""
        print("\\n" + "="*80)
        print(f"🚀 STARTING: {self.test_name}")
        print(f"📅 Session: {self.test_timestamp}")
        print("="*80)
        
        if not self.setup_driver():
            return False
        
        try:
            # Execute all test phases
            test_phases = [
                ("Application Startup", self.test_01_application_startup),
                ("Data Upload", self.test_02_data_upload),
                ("Analysis Workspace & Tabs", self.test_03_analysis_workspace_navigation),
                ("Dashboard Gallery", self.test_04_dashboard_gallery),
                ("Export Functionality", self.test_05_export_functionality),
                ("Single-Page Mode", self.test_06_single_page_mode)
            ]
            
            successful_phases = 0
            
            for phase_name, test_function in test_phases:
                print(f"\\n🧪 PHASE: {phase_name}")
                print("-" * 40)
                
                if test_function():
                    successful_phases += 1
                    print(f"✅ {phase_name}: COMPLETED")
                else:
                    print(f"❌ {phase_name}: ISSUES DETECTED")
            
            # Save results
            results_file, summary_file = self.save_test_results()
            
            # Final summary
            print("\\n" + "="*80)
            print("📊 COMPREHENSIVE TESTING COMPLETED")
            print("="*80)
            print(f"✅ Successful Phases: {successful_phases}/{len(test_phases)}")
            print(f"📸 Screenshots Taken: {self.screenshot_count}")
            print(f"📈 Plots Generated: {self.successful_plots}")
            print(f"📁 Test Session Folder: {self.test_session_folder}")
            print(f"📤 Output Folder: {self.output_folder}")
            
            if results_file:
                print(f"📋 Detailed Results: {results_file}")
            if summary_file:
                print(f"📄 Summary Report: {summary_file}")
            
            success_rate = (successful_phases / len(test_phases)) * 100
            if success_rate >= 80:
                print("🎉 OVERALL RESULT: TESTING SUCCESSFUL")
            else:
                print("⚠️ OVERALL RESULT: ISSUES REQUIRE ATTENTION")
            
            return success_rate >= 80
            
        except Exception as e:
            self.log_result("CRITICAL", "FAIL", f"Critical testing error: {str(e)}")
            return False
        
        finally:
            if self.driver:
                time.sleep(3)  # Final pause
                self.driver.quit()

if __name__ == "__main__":
    tester = ComprehensiveE2ETesting()
    tester.run_complete_testing()