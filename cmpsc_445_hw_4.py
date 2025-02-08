from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

session = requests.Session()  # Create a session object
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Referer': 'https://socialblade.com/',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'DNT': '1',  # Do Not Track header
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
}

def clean_numeric_value(value):
    return float(value.replace('$', '').replace(',', ''))

def scrape_listings(zipcode):
    base_url = f'https://www.redfin.com/zipcode/{zipcode}'

    session.headers.update(headers)  # Apply headers to all requests
    options = uc.ChromeOptions()
    options.headless = False  # Keep visible for testing
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = uc.Chrome(options=options)
    driver.get(base_url)

    # Wait for a specific element to be present
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.bp-Homecard__Stats'))  # Adjust the selector as needed
        )
        # Additional sleep to ensure all JavaScript has executed
        time.sleep(1)
    except Exception as e:
        print("Error waiting for page to load:", e)
    
    data = []
    while True:
        listings = driver.find_elements(By.CLASS_NAME, 'HomeCardContainer')
        
        for listing in listings:
            try:
                beds = clean_numeric_value(listing.find_element(By.CLASS_NAME, 'bp-Homecard__Stats--beds').text.strip().split()[0])
                baths = clean_numeric_value(listing.find_element(By.CLASS_NAME, 'bp-Homecard__Stats--baths').text.strip().split()[0])
                size = clean_numeric_value(listing.find_element(By.CLASS_NAME, 'bp-Homecard__Stats--sqft .bp-Homecard__LockedStat--value').text.strip().split()[0])
                price = clean_numeric_value(listing.find_element(By.CLASS_NAME, 'bp-Homecard__Price--value').text.strip())
                data.append([zipcode, beds, baths, size, price])
            except:
                print(f'Cannot find: {listing}')
                continue
        
        try:
            next_button = driver.find_element(By.XPATH, "//button[@aria-label='next']")
            next_button.click()
            time.sleep(3)
        except:
            break  # No more pages
    
    driver.quit()
    df = pd.DataFrame(data, columns=['Zipcode', 'Beds', 'Baths', 'Size', 'Price'])
    df = df.dropna()  # Drop rows w/ null values
    return df

def train_model(file_name):
    data = pd.read_csv(file_name)
    X = data[['Beds', 'Baths', 'Size', 'Price']]
    y = data['Zipcode']

    # Split into training/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Make prediction
    y_pred = model.predict(X_test_scaled)

    # Evaluate performance
    print(f"Accuracy on test set: {accuracy_score(y_test, y_pred)}")

file_name = 'redfin_listings.csv'
rescrape_listings = False  # Change to regenerate data

if rescrape_listings:
	zipcodes = ['20001', '10001', '15001', '95001']
	all_dfs = [scrape_listings(zipcode) for zipcode in zipcodes]
	final_df = pd.concat(all_dfs, ignore_index=True)  # 95001 is actually null here (only 1 entry w/ null values)
	print(final_df.head())
	final_df.to_csv(file_name, index=False)

train_model(file_name)