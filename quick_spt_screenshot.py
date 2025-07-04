#!/usr/bin/env python3
"""
Quick SPT Screenshot Tool
Directly captures the current state of SPT tab for debugging
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import os
from datetime import datetime

def take_spt_screenshot():
    print("Starting SPT screenshot capture...")
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = None
    try:
        # Initialize driver
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, 20)
        
        # Navigate to app
        print("Navigating to http://localhost:8503...")
        driver.get("http://localhost:8503")
        
        # Handle password if present
        try:
            password_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
            print("Password screen detected, entering password...")
            password_input.send_keys("123456")
            password_input.submit()
            time.sleep(3)
        except:
            print("No password screen found, continuing...")
        
        # Wait for main app to load
        print("Waiting for main app...")
        time.sleep(5)
        
        # Take initial screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshots_dir = f"spt_debug_screenshots_{timestamp}"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Get page height and resize for full capture
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        original_size = driver.get_window_size()
        
        if total_height > viewport_height:
            driver.set_window_size(original_size['width'], total_height + 200)
        
        # Take full page screenshot
        screenshot_path = f"{screenshots_dir}/01_main_page.png"
        driver.save_screenshot(screenshot_path)
        print(f"Main page screenshot saved: {screenshot_path}")
        
        # Look for SPT or Analysis tab
        try:
            # Try to find Analysis Workspace button/tab
            analysis_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Analysis') or contains(text(), 'SPT') or contains(text(), 'Workspace')]")
            print(f"Found {len(analysis_elements)} analysis-related elements")
            
            for i, element in enumerate(analysis_elements):
                print(f"  Element {i}: {element.text} (tag: {element.tag_name})")
                if element.is_displayed() and element.is_enabled():
                    try:
                        element.click()
                        print(f"Clicked on: {element.text}")
                        time.sleep(3)
                        break
                    except Exception as e:
                        print(f"Could not click element: {e}")
                        
        except Exception as e:
            print(f"Could not find analysis elements: {e}")
        
        # Take screenshot after potential navigation
        screenshot_path = f"{screenshots_dir}/02_after_navigation.png"
        driver.save_screenshot(screenshot_path)
        print(f"After navigation screenshot saved: {screenshot_path}")
        
        # Look for SPT-specific elements
        try:
            spt_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'SPT') or contains(text(), 'Standard Penetration')]")
            print(f"Found {len(spt_elements)} SPT-related elements")
            
            for i, element in enumerate(spt_elements):
                print(f"  SPT Element {i}: {element.text} (tag: {element.tag_name})")
                if "SPT" in element.text and element.is_displayed():
                    try:
                        element.click()
                        print(f"Clicked on SPT: {element.text}")
                        time.sleep(3)
                        break
                    except Exception as e:
                        print(f"Could not click SPT element: {e}")
                        
        except Exception as e:
            print(f"Could not find SPT elements: {e}")
        
        # Final screenshot
        screenshot_path = f"{screenshots_dir}/03_final_state.png"
        driver.save_screenshot(screenshot_path)
        print(f"Final state screenshot saved: {screenshot_path}")
        
        # Get page source for debugging
        with open(f"{screenshots_dir}/page_source.html", "w") as f:
            f.write(driver.page_source)
        print(f"Page source saved: {screenshots_dir}/page_source.html")
        
        print(f"\nScreenshots saved in: {screenshots_dir}/")
        return screenshots_dir
        
    except Exception as e:
        print(f"Error during screenshot capture: {e}")
        return None
        
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    take_spt_screenshot()