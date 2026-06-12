import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_system():
    print("Starting full system integration test...\n")
    
    # 0. Test GET /api/crops
    print("Testing GET /api/crops...")
    try:
        res = requests.get(f"{BASE_URL}/api/crops")
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                crops = data.get("crops", [])
                print(f"SUCCESS: Retrieved {len(crops)} crops from external dataset.")
                print(f"         Sample crops: {crops[:5]}")
            else:
                print(f"FAILED: Expected success status, got: {data}")
        else:
            print(f"FAILED: HTTP {res.status_code} - {res.text}")
    except Exception as e:
        print(f"FAILED: Request exception - {e}")
        
    print("\n-------------------------------------------------\n")

    # 1. Test GET /api/similarity
    print("Testing GET /api/similarity with crop='wheat', lat=28.6139, lon=77.2090...")
    try:
        res = requests.get(f"{BASE_URL}/api/similarity", params={"lat": 28.6139, "lon": 77.2090, "crop": "wheat"})
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                print(f"SUCCESS: Similarity matrix retrieved. Found {len(data.get('crop_comparisons', []))} comparisons.")
                print(f"         Schedule generated with {len(data.get('schedule', []))} days.")
            else:
                print(f"FAILED: Expected success status, got: {data}")
        else:
            print(f"FAILED: HTTP {res.status_code} - {res.text}")
    except Exception as e:
        print(f"FAILED: Request exception - {e}")
        
    print("\n-------------------------------------------------\n")
    
    # 2. Test POST /api/run-pipeline
    print("Testing POST /api/run-pipeline with crop='rice'...")
    try:
        res = requests.post(f"{BASE_URL}/api/run-pipeline", json={"lat": 28.6139, "lon": 77.2090, "crop": "rice"})
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                print("SUCCESS: Pipeline executed successfully.")
            else:
                print(f"FAILED: Expected success status, got: {data}")
        else:
            print(f"FAILED: HTTP {res.status_code} - {res.text}")
    except Exception as e:
        print(f"FAILED: Request exception - {e}")

    print("\n-------------------------------------------------\n")

    # 3. Test GET /api/similarity again for 'rice'
    print("Testing GET /api/similarity with crop='rice'...")
    try:
        res = requests.get(f"{BASE_URL}/api/similarity", params={"lat": 28.6139, "lon": 77.2090, "crop": "rice"})
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                print(f"SUCCESS: Similarity matrix retrieved for rice. Active crop in response: {data.get('crop')}")
            else:
                print(f"FAILED: Expected success status, got: {data}")
        else:
            print(f"FAILED: HTTP {res.status_code} - {res.text}")
    except Exception as e:
        print(f"FAILED: Request exception - {e}")

if __name__ == "__main__":
    test_system()
