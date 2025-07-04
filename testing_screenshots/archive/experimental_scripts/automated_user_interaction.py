#!/usr/bin/env python3
"""
Automated User Interaction Simulator
This script will actually interact with the Streamlit app by:
1. Uploading the Excel file
2. Clicking through pages and tabs
3. Capturing screenshots/state
4. Identifying issues and testing again
"""

import requests
import time
import json
import os
import subprocess
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd

class AutomatedUserInteraction:
    def __init__(self, base_url="http://localhost:8503"):
        self.base_url = base_url
        self.driver = None
        self.test_results = []
        self.screenshots_taken = 0
        
    def setup_browser(self):
        """Setup Chrome browser for automation."""
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            # Don't run headless so we can see what's happening
            # chrome_options.add_argument("--headless")
            
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            
            self.log_action("Setup Browser", "SUCCESS", "Chrome browser initialized")
            return True
            
        except Exception as e:
            self.log_action("Setup Browser", "FAIL", f"Browser setup failed: {str(e)}")
            return False
    
    def log_action(self, action, status, details="", screenshot=False):
        """Log an action with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        entry = {
            'timestamp': timestamp,
            'action': action,
            'status': status,
            'details': details
        }
        self.test_results.append(entry)
        
        emoji = {"SUCCESS": "✅", "FAIL": "❌", "INFO": "📋", "WARNING": "⚠️"}
        print(f"{emoji.get(status, '•')} {timestamp} | {action}: {status}")
        if details:
            print(f"   📋 {details}")
            
        if screenshot and self.driver:
            self.take_screenshot(f"{action.replace(' ', '_')}")
    
    def take_screenshot(self, name):
        """Take a screenshot of current state."""
        try:
            if self.driver:
                self.screenshots_taken += 1
                filename = f"testing_screenshots/screenshot_{self.screenshots_taken:02d}_{name}.png"
                self.driver.save_screenshot(filename)
                print(f"   📸 Screenshot saved: {filename}")
                return filename
        except Exception as e:
            print(f"   ⚠️ Screenshot failed: {e}")
        return None
    
    def open_application(self):
        """Open the Streamlit application."""
        try:
            print(f"🌐 Opening application at {self.base_url}")
            self.driver.get(self.base_url)
            
            # Wait for page to load
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(3)  # Additional wait for Streamlit to fully load
            
            # Check if app loaded correctly
            page_title = self.driver.title
            page_source = self.driver.page_source.lower()
            
            if "streamlit" in page_source or "geotechnical" in page_source:
                self.log_action("Open Application", "SUCCESS", 
                              f"App loaded successfully. Title: {page_title}", screenshot=True)
                return True
            else:
                self.log_action("Open Application", "FAIL", 
                              f"App may not have loaded correctly. Title: {page_title}", screenshot=True)
                return False
                
        except Exception as e:
            self.log_action("Open Application", "FAIL", f"Failed to open app: {str(e)}", screenshot=True)
            return False
    
    def check_excel_file(self):
        """Check if the Excel file exists."""
        excel_file = "Lab_summary_final.xlsx"
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
                self.log_action("Check Excel File", "SUCCESS", 
                              f"File found: {excel_file} ({len(df)} rows, {len(df.columns)} columns)")
                return True, excel_file
            except Exception as e:
                self.log_action("Check Excel File", "FAIL", f"File exists but cannot read: {e}")
                return False, excel_file
        else:
            self.log_action("Check Excel File", "FAIL", f"File not found: {excel_file}")
            return False, excel_file
    
    def navigate_to_page(self, page_name):
        """Navigate to a specific page."""
        try:
            # Look for navigation buttons in sidebar
            page_buttons = self.driver.find_elements(By.XPATH, f"//button[contains(text(), '{page_name}')]")
            
            if not page_buttons:
                # Try different selectors
                page_buttons = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{page_name}')]")
            
            if page_buttons:
                button = page_buttons[0]
                self.driver.execute_script("arguments[0].click();", button)
                time.sleep(2)
                
                self.log_action(f"Navigate to {page_name}", "SUCCESS", 
                              "Page navigation successful", screenshot=True)
                return True
            else:
                self.log_action(f"Navigate to {page_name}", "FAIL", 
                              f"Could not find navigation button for {page_name}", screenshot=True)
                return False
                
        except Exception as e:
            self.log_action(f"Navigate to {page_name}", "FAIL", 
                          f"Navigation failed: {str(e)}", screenshot=True)
            return False
    
    def upload_excel_file(self, file_path):
        """Upload the Excel file."""
        try:
            # Look for file uploader
            file_uploaders = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            
            if not file_uploaders:
                # Try different selectors
                file_uploaders = self.driver.find_elements(By.XPATH, "//input[@type='file']")
            
            if file_uploaders:
                uploader = file_uploaders[0]
                
                # Get absolute path
                abs_path = os.path.abspath(file_path)
                uploader.send_keys(abs_path)
                
                # Wait for upload to process
                time.sleep(5)
                
                # Check for success indicators
                page_source = self.driver.page_source.lower()
                if "success" in page_source or "uploaded" in page_source or "processed" in page_source:
                    self.log_action("Upload Excel File", "SUCCESS", 
                                  f"File uploaded successfully: {file_path}", screenshot=True)
                    return True
                else:
                    self.log_action("Upload Excel File", "WARNING", 
                                  "File upload completed but no clear success indicator", screenshot=True)
                    return True
            else:
                self.log_action("Upload Excel File", "FAIL", 
                              "Could not find file uploader widget", screenshot=True)
                return False
                
        except Exception as e:
            self.log_action("Upload Excel File", "FAIL", 
                          f"File upload failed: {str(e)}", screenshot=True)
            return False
    
    def click_tab(self, tab_name):
        """Click on a specific tab."""
        try:
            # Look for tabs with various selectors
            tab_selectors = [
                f"//div[contains(@class, 'stTabs')]//button[contains(text(), '{tab_name}')]",
                f"//button[contains(text(), '{tab_name}')]",
                f"//*[contains(text(), '{tab_name}') and (name()='button' or @role='tab')]"
            ]
            
            tab_element = None
            for selector in tab_selectors:
                tabs = self.driver.find_elements(By.XPATH, selector)
                if tabs:
                    tab_element = tabs[0]
                    break
            
            if tab_element:
                self.driver.execute_script("arguments[0].click();", tab_element)
                time.sleep(3)  # Wait for tab content to load
                
                self.log_action(f"Click Tab: {tab_name}", "SUCCESS", 
                              "Tab clicked successfully", screenshot=True)
                return True
            else:
                self.log_action(f"Click Tab: {tab_name}", "FAIL", 
                              f"Could not find tab: {tab_name}", screenshot=True)
                return False
                
        except Exception as e:
            self.log_action(f"Click Tab: {tab_name}", "FAIL", 
                          f"Tab click failed: {str(e)}", screenshot=True)
            return False
    
    def check_for_plots(self):
        """Check if plots are visible on the current page."""
        try:
            # Look for plot indicators
            plot_indicators = [
                "canvas",  # Matplotlib/Plotly canvas
                "[data-testid='stPlotlyChart']",  # Streamlit Plotly charts
                ".js-plotly-plot",  # Plotly plots
                "img[src*='data:image']",  # Base64 images
                ".stImage"  # Streamlit images
            ]
            
            plots_found = 0
            for indicator in plot_indicators:
                elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                plots_found += len(elements)
            
            if plots_found > 0:
                self.log_action("Check for Plots", "SUCCESS", 
                              f"Found {plots_found} plot elements", screenshot=True)
                return True, plots_found
            else:
                self.log_action("Check for Plots", "FAIL", 
                              "No plot elements found", screenshot=True)
                return False, 0
                
        except Exception as e:
            self.log_action("Check for Plots", "FAIL", 
                          f"Plot check failed: {str(e)}", screenshot=True)
            return False, 0
    
    def generate_plots_in_tab(self, tab_name):
        """Try to generate plots in the current tab."""
        try:
            # Look for generate/create/plot buttons
            button_texts = ["Generate", "Create", "Plot", "Analyze", "Run", "Execute"]
            
            for button_text in button_texts:
                buttons = self.driver.find_elements(By.XPATH, 
                    f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{button_text.lower()}')]")
                
                if buttons:
                    button = buttons[0]
                    self.driver.execute_script("arguments[0].click();", button)
                    time.sleep(5)  # Wait for plot generation
                    
                    # Check if plots appeared
                    has_plots, plot_count = self.check_for_plots()
                    
                    if has_plots:
                        self.log_action(f"Generate Plots in {tab_name}", "SUCCESS", 
                                      f"Generated {plot_count} plots", screenshot=True)
                        return True
                    else:
                        self.log_action(f"Generate Plots in {tab_name}", "WARNING", 
                                      f"Clicked {button_text} but no plots appeared", screenshot=True)
            
            # If no generate button found, just check if plots are already there
            has_plots, plot_count = self.check_for_plots()
            if has_plots:
                self.log_action(f"Check Existing Plots in {tab_name}", "SUCCESS", 
                              f"Found {plot_count} existing plots", screenshot=True)
                return True
            else:
                self.log_action(f"Generate Plots in {tab_name}", "FAIL", 
                              "No generate button found and no existing plots", screenshot=True)
                return False
                
        except Exception as e:
            self.log_action(f"Generate Plots in {tab_name}", "FAIL", 
                          f"Plot generation failed: {str(e)}", screenshot=True)
            return False
    
    def run_comprehensive_test(self):
        """Run the complete automated test."""
        print("🚀 Starting Automated User Interaction Test")
        print("=" * 60)
        
        # Setup browser
        if not self.setup_browser():
            return False
        
        try:
            # Step 1: Check Excel file
            file_exists, excel_file = self.check_excel_file()
            if not file_exists:
                return False
            
            # Step 2: Open application
            if not self.open_application():
                return False
            
            # Step 3: Navigate to Data Management page and upload file
            self.log_action("START", "INFO", "Beginning Multi-Page Mode Testing")
            
            if self.navigate_to_page("Data Management"):
                if self.upload_excel_file(excel_file):
                    time.sleep(5)  # Wait for processing
                else:
                    self.log_action("File Upload", "FAIL", "Cannot proceed without data upload")
                    return False
            
            # Step 4: Test Analysis Workspace
            if self.navigate_to_page("Analysis"):
                analysis_tabs = [
                    "PSD Analysis",
                    "Atterberg", 
                    "SPT Analysis",
                    "UCS Analysis",
                    "Spatial",
                    "Emerson"
                ]
                
                successful_tabs = 0
                for tab in analysis_tabs:
                    if self.click_tab(tab):
                        if self.generate_plots_in_tab(tab):
                            successful_tabs += 1
                        time.sleep(2)
                
                self.log_action("Analysis Tabs Summary", "INFO", 
                              f"Successfully generated plots in {successful_tabs}/{len(analysis_tabs)} tabs")
            
            # Step 5: Test Dashboard Gallery
            if self.navigate_to_page("Dashboard"):
                dashboard_tabs = ["Site Characterization", "Material Properties", "Rock Properties"]
                
                for tab in dashboard_tabs:
                    if self.click_tab(tab):
                        has_plots, plot_count = self.check_for_plots()
                        if has_plots:
                            self.log_action(f"Dashboard {tab}", "SUCCESS", 
                                          f"Dashboard shows {plot_count} plots")
                        else:
                            self.log_action(f"Dashboard {tab}", "FAIL", 
                                          "Dashboard empty - plots not transferring")
            
            # Step 6: Test Export page
            if self.navigate_to_page("Export"):
                has_plots, plot_count = self.check_for_plots()
                self.log_action("Export Page", "SUCCESS" if has_plots else "WARNING", 
                              f"Export page shows {plot_count} plots")
            
            # Step 7: Test Single-Page Mode
            self.log_action("MODE SWITCH", "INFO", "Testing Single-Page Mode")
            
            # Try to switch to single-page mode
            try:
                # Look for mode selector
                mode_selectors = self.driver.find_elements(By.XPATH, 
                    "//select | //button[contains(text(), 'Single')]")
                
                if mode_selectors:
                    selector = mode_selectors[0]
                    if selector.tag_name == "select":
                        # It's a dropdown
                        from selenium.webdriver.support.ui import Select
                        select = Select(selector)
                        select.select_by_visible_text("Single-Page Mode")
                    else:
                        # It's a button
                        selector.click()
                    
                    time.sleep(3)
                    self.log_action("Switch to Single-Page", "SUCCESS", 
                                  "Mode switched successfully", screenshot=True)
                else:
                    self.log_action("Switch to Single-Page", "FAIL", 
                                  "Could not find mode selector", screenshot=True)
                    
            except Exception as e:
                self.log_action("Switch to Single-Page", "FAIL", 
                              f"Mode switch failed: {str(e)}", screenshot=True)
            
            return True
            
        except Exception as e:
            self.log_action("CRITICAL ERROR", "FAIL", f"Test failed: {str(e)}", screenshot=True)
            return False
        
        finally:
            if self.driver:
                time.sleep(2)  # Final pause before closing
                self.driver.quit()
    
    def analyze_results(self):
        """Analyze test results and provide recommendations."""
        print("\n" + "=" * 60)
        print("📊 AUTOMATED TEST RESULTS ANALYSIS")
        print("=" * 60)
        
        total_actions = len(self.test_results)
        successful_actions = len([r for r in self.test_results if r['status'] == 'SUCCESS'])
        failed_actions = len([r for r in self.test_results if r['status'] == 'FAIL'])
        
        success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0
        
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"✅ Successful Actions: {successful_actions}")
        print(f"❌ Failed Actions: {failed_actions}")
        print(f"📸 Screenshots Taken: {self.screenshots_taken}")
        
        # Critical issues
        critical_failures = [r for r in self.test_results if r['status'] == 'FAIL']
        if critical_failures:
            print("\n🚨 CRITICAL ISSUES FOUND:")
            for failure in critical_failures[-5:]:  # Show last 5 failures
                print(f"   ❌ {failure['timestamp']} | {failure['action']}: {failure['details']}")
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS:")
        if success_rate >= 80:
            print("🎉 Application is working well! Minor issues may need attention.")
        elif success_rate >= 60:
            print("⚠️ Application has significant issues that need fixing.")
        else:
            print("❌ Application has major problems requiring immediate attention.")
        
        return success_rate >= 70
    
    def save_results(self):
        """Save detailed test results."""
        try:
            results = {
                'test_run_time': datetime.now().isoformat(),
                'app_url': self.base_url,
                'total_actions': len(self.test_results),
                'screenshots_taken': self.screenshots_taken,
                'results': self.test_results
            }
            
            results_file = "testing_screenshots/automated_interaction_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
            
        except Exception as e:
            print(f"\n⚠️ Could not save results: {e}")

if __name__ == "__main__":
    # Check if Chrome/Selenium is available
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("❌ Selenium not installed. Installing...")
        subprocess.run(["pip", "install", "selenium"])
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    
    tester = AutomatedUserInteraction()
    
    try:
        success = tester.run_comprehensive_test()
        success = tester.analyze_results()
        tester.save_results()
        
        print("\n" + "="*60)
        if success:
            print("🎉 AUTOMATED TESTING COMPLETED SUCCESSFULLY")
        else:
            print("⚠️ AUTOMATED TESTING FOUND ISSUES REQUIRING ATTENTION")
        print("="*60)
        
    except Exception as e:
        print(f"\n💥 Automated testing failed: {e}")
        if tester.driver:
            tester.driver.quit()