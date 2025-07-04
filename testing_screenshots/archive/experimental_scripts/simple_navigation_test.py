#!/usr/bin/env python3
"""
Simple Navigation Test
Uses a different approach to navigate and test the application.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def setup_driver():
    """Setup Chrome driver with better options."""
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def click_element_with_retry(driver, element, max_retries=3):
    """Click element with multiple retry strategies."""
    for i in range(max_retries):
        try:
            # Method 1: Regular click
            element.click()
            return True
        except:
            try:
                # Method 2: JavaScript click
                driver.execute_script("arguments[0].click();", element)
                return True
            except:
                try:
                    # Method 3: Action chains
                    from selenium.webdriver.common.action_chains import ActionChains
                    ActionChains(driver).move_to_element(element).click().perform()
                    return True
                except:
                    if i == max_retries - 1:
                        return False
                    time.sleep(1)
    return False

def test_navigation_to_analysis():
    """Test navigation to Analysis Workspace page."""
    driver = None
    try:
        driver = setup_driver()
        
        print("🌐 Opening application...")
        driver.get("http://localhost:8503")
        time.sleep(5)
        
        print("📸 Taking initial screenshot...")
        driver.save_screenshot("testing_screenshots/manual_01_initial_load.png")
        
        print("🔍 Looking for Analysis Workspace navigation...")
        
        # Try multiple selectors for the Analysis Workspace button
        selectors = [
            "//button[contains(text(), 'Analysis Workspace')]",
            "//div[contains(text(), 'Analysis Workspace')]",
            "//span[contains(text(), 'Analysis Workspace')]",
            "//*[contains(text(), 'Analysis') and contains(text(), 'Workspace')]",
            "//button[contains(text(), 'page 2 analysis')]",
            "//div[contains(text(), 'page 2 analysis')]"
        ]
        
        analysis_button = None
        for selector in selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    analysis_button = elements[0]
                    print(f"✅ Found Analysis button with selector: {selector}")
                    break
            except:
                continue
        
        if analysis_button:
            print("👆 Clicking Analysis Workspace...")
            
            # Scroll to element first
            driver.execute_script("arguments[0].scrollIntoView(true);", analysis_button)
            time.sleep(1)
            
            if click_element_with_retry(driver, analysis_button):
                print("✅ Successfully clicked Analysis Workspace")
                time.sleep(3)
                
                driver.save_screenshot("testing_screenshots/manual_02_analysis_page.png")
                
                # Check if we're on the analysis page
                page_source = driver.page_source.lower()
                if "analysis" in page_source and ("psd" in page_source or "atterberg" in page_source):
                    print("✅ Successfully navigated to Analysis page!")
                    return driver, True
                else:
                    print("⚠️ Click succeeded but page didn't change")
                    return driver, False
            else:
                print("❌ Failed to click Analysis Workspace button")
                return driver, False
        else:
            print("❌ Could not find Analysis Workspace button")
            
            # Take screenshot of what we can see
            driver.save_screenshot("testing_screenshots/manual_02_navigation_failed.png")
            
            # Print page source to debug
            print("\n🔍 DEBUGGING - Available text on page:")
            page_text = driver.page_source.lower()
            relevant_keywords = ["analysis", "workspace", "page", "navigation", "sidebar"]
            for keyword in relevant_keywords:
                if keyword in page_text:
                    print(f"   Found: {keyword}")
            
            return driver, False
    
    except Exception as e:
        print(f"❌ Error in navigation test: {e}")
        if driver:
            driver.save_screenshot("testing_screenshots/manual_02_error.png")
        return driver, False

def test_analysis_tabs(driver):
    """Test clicking through analysis tabs."""
    
    if not driver:
        return False
    
    print("\n🧪 Testing Analysis Tabs...")
    
    analysis_tabs = [
        "PSD Analysis",
        "Atterberg", 
        "SPT",
        "UCS",
        "Spatial",
        "Emerson"
    ]
    
    successful_tabs = 0
    
    for i, tab_name in enumerate(analysis_tabs):
        print(f"\n📊 Testing {tab_name} tab...")
        
        # Look for tab
        tab_selectors = [
            f"//button[contains(text(), '{tab_name}')]",
            f"//div[contains(text(), '{tab_name}')]",
            f"//*[contains(text(), '{tab_name}') and (@role='tab' or contains(@class, 'tab'))]"
        ]
        
        tab_element = None
        for selector in tab_selectors:
            try:
                elements = driver.find_elements(By.XPATH, selector)
                if elements:
                    tab_element = elements[0]
                    break
            except:
                continue
        
        if tab_element:
            if click_element_with_retry(driver, tab_element):
                print(f"✅ Successfully clicked {tab_name} tab")
                time.sleep(2)
                
                # Take screenshot
                driver.save_screenshot(f"testing_screenshots/manual_03_tab_{i+1}_{tab_name.replace(' ', '_')}.png")
                
                # Check for plots or generate buttons
                page_source = driver.page_source.lower()
                if "plot" in page_source or "chart" in page_source or "canvas" in page_source:
                    print(f"   📈 Found plot elements in {tab_name}")
                    successful_tabs += 1
                else:
                    print(f"   ⚠️ No obvious plots found in {tab_name}")
                
            else:
                print(f"❌ Failed to click {tab_name} tab")
        else:
            print(f"❌ Could not find {tab_name} tab")
    
    print(f"\n📊 Tab Testing Summary: {successful_tabs}/{len(analysis_tabs)} tabs accessible")
    return successful_tabs > 0

def main():
    """Main testing function."""
    
    print("🚀 SIMPLE NAVIGATION TESTING")
    print("=" * 50)
    
    # Test navigation to Analysis page
    driver, navigation_success = test_navigation_to_analysis()
    
    if navigation_success:
        # Test analysis tabs
        tab_success = test_analysis_tabs(driver)
        
        print("\n✅ Navigation test completed successfully!")
    else:
        print("\n❌ Navigation test failed - manual intervention needed")
    
    # Clean up
    if driver:
        time.sleep(2)
        driver.quit()
    
    print("\n📸 Screenshots saved in testing_screenshots/ folder")
    print("🌐 Application remains running at: http://localhost:8503")

if __name__ == "__main__":
    main()