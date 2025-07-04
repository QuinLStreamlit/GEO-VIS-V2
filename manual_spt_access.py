#!/usr/bin/env python3
"""
Manual SPT Access Tool
Properly handles password and navigates to SPT tab
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import os
from datetime import datetime

def access_spt_tab():
    print("Starting manual SPT access...")
    
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
        wait = WebDriverWait(driver, 30)
        
        # Navigate to app
        print("Navigating to http://localhost:8503...")
        driver.get("http://localhost:8503")
        
        # Handle password properly
        try:
            print("Looking for password field...")
            password_input = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']")))
            print("Found password field, entering password...")
            password_input.clear()
            password_input.send_keys("123456")
            
            # Press Enter to submit
            password_input.send_keys(Keys.RETURN)
            print("Password submitted via Enter key")
            
            # Wait for login to process
            time.sleep(5)
            
        except Exception as e:
            print(f"Password handling failed: {e}")
            return None
        
        # Setup screenshot directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshots_dir = f"manual_spt_access_{timestamp}"
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # Take screenshot after login
        driver.save_screenshot(f"{screenshots_dir}/01_after_login.png")
        print("Screenshot taken after login")
        
        # Wait for main app interface to load
        print("Waiting for main interface to load...")
        time.sleep(5)
        
        # Look for file upload or main interface
        try:
            # Check if we're at the main app interface
            page_text = driver.page_source
            print("Current page contains:")
            if "Upload" in page_text:
                print("  - Upload functionality")
            if "Analysis" in page_text:
                print("  - Analysis options")
            if "SPT" in page_text:
                print("  - SPT content")
            if "Dashboard" in page_text:
                print("  - Dashboard content")
                
        except Exception as e:
            print(f"Error checking page content: {e}")
        
        # Take screenshot of current state
        total_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
        original_size = driver.get_window_size()
        driver.set_window_size(original_size['width'], total_height + 200)
        
        driver.save_screenshot(f"{screenshots_dir}/02_main_interface.png")
        print("Main interface screenshot captured")
        
        # Look for navigation elements
        try:
            # Find all clickable elements that might be navigation
            clickable_elements = driver.find_elements(By.XPATH, "//button | //a | //*[@role='button'] | //*[@role='tab']")
            
            print(f"Found {len(clickable_elements)} clickable elements")
            for i, elem in enumerate(clickable_elements[:10]):  # Show first 10
                try:
                    text = elem.text.strip()
                    if text:
                        print(f"  {i}: '{text}' (tag: {elem.tag_name})")
                except:
                    pass
                    
            # Look specifically for SPT or Analysis related elements
            spt_related = driver.find_elements(By.XPATH, "//*[contains(text(), 'SPT') or contains(text(), 'Analysis') or contains(text(), 'Workspace')]")
            print(f"Found {len(spt_related)} SPT/Analysis related elements")
            
            for elem in spt_related:
                try:
                    print(f"  SPT/Analysis: '{elem.text}' (tag: {elem.tag_name}, visible: {elem.is_displayed()})")
                except:
                    pass
                    
        except Exception as e:
            print(f"Error finding navigation elements: {e}")
        
        # Save page source for debugging
        with open(f"{screenshots_dir}/page_source.html", "w") as f:
            f.write(driver.page_source)
        
        print(f"\nDebug files saved in: {screenshots_dir}/")
        return screenshots_dir
        
    except Exception as e:
        print(f"Error during access: {e}")
        return None
        
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    access_spt_tab()