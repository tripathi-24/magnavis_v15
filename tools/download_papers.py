import os
import requests
import urllib.parse
import re

titles = [
    "Multihead Self-attention and LSTM for Spacecraft Telemetry Anomaly Detection",
    "Geomagnetic Anomaly Detection and Signal Processing using Tri-axial Magnetometers",
    "Data-driven visualization and correlation analysis of multi-sensor networks",
    "Deep Learning for Magnetic Anomaly Detection: A GRU-based Adaptive Thresholding Approach",
    "Cyclical Feature Encoding for Diurnal Time-Series Forecasting using Recurrent Neural Networks",
    "Attention-Bi-LSTM Network for Real-Time Magnetic Anomaly Detection",
    "Adaptive Residual Thresholding for Non-Stationary Sensor Data Anomaly Detection",
    "Source Localization and Direction Finding using Machine Learning Regressors",
    "Distance and Bearing Estimation of Magnetic Anomalies using K-Nearest Neighbors and Extra Trees",
    "Circular Variable Encoding for Neural Network Regression in Geospatial Applications",
    "Supervised Learning Models for 3D Target Localization using Tri-axis Magnetic Sensor Arrays"
]

OUT_DIR = "downloaded_papers"
os.makedirs(OUT_DIR, exist_ok=True)
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for idx, title in enumerate(titles, 2):
    print(f"Searching for: {title}")
    try:
        url = f"https://api.crossref.org/works?query.title={urllib.parse.quote(title)}&select=DOI,title&rows=1"
        res = requests.get(url, headers=headers).json()
        items = res.get('message', {}).get('items', [])
        if not items:
            print("  -> DOI not found.")
            continue
            
        doi = items[0]['DOI']
        print(f"  -> Found DOI: {doi}")
        
        pdf_url = None
        # 1. Try Unpaywall
        up_url = f"https://api.unpaywall.org/v2/{doi}?email=test@example.com"
        up_res = requests.get(up_url, headers=headers).json()
        if up_res.get('is_oa', False):
            pdf_url = up_res.get('best_oa_location', {}).get('url_for_pdf')
            if pdf_url:
                print(f"  -> Found Open Access PDF: {pdf_url}")
        
        # 2. Try Sci-Hub if needed
        if not pdf_url:
            print(f"  -> Trying Sci-Hub...")
            sh_url = f"https://sci-hub.se/{doi}"
            sh_html = requests.get(sh_url, headers=headers).text
            match = re.search(r'<button\s+onclick="location\.href=\'(//.*?\.pdf.*?)\'">', sh_html)
            if not match:
                # Alternate pattern
                match = re.search(r'<embed\s+type="application/pdf"\s+src="(.*?#view=FitH)"', sh_html)
                if not match:
                    match = re.search(r'iframe src="(//.*?\.pdf.*?)"', sh_html)
            if match:
                pdf_url = match.group(1)
                if pdf_url.startswith('//'):
                    pdf_url = 'https:' + pdf_url
                print(f"  -> Found Sci-Hub PDF: {pdf_url}")

        if pdf_url:
            print(f"  -> Downloading...")
            pdf_data = requests.get(pdf_url, headers=headers, timeout=20).content
            safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip()
            filename = os.path.join(OUT_DIR, f"Paper_{idx}_{safe_title}.pdf")
            with open(filename, 'wb') as f:
                f.write(pdf_data)
            print(f"  -> Saved as {filename}")
        else:
            print("  -> Could not find a downloadable PDF link.")
            
    except Exception as e:
        print(f"  -> Error: {e}")
