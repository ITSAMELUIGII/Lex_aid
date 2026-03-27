import requests
import json

try:
    resp = requests.post(
        'http://127.0.0.1:8000/api/chat', 
        json={"prompt": "Tenant deposit issue with landlord", "language": "English"},
        headers={"Content-Type": "application/json"}
    )
    print("STATUS:", resp.status_code)
    print("RESPONSE:", resp.text)
except Exception as e:
    print("CONNECTION EXCEPTION:", str(e))
