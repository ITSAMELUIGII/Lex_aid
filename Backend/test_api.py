import json
import urllib.request
import urllib.error

req = urllib.request.Request(
    'http://127.0.0.1:8000/api/chat',
    data=json.dumps({'prompt':'test', 'language':'English'}).encode('utf-8'),
    headers={'Content-Type':'application/json'}
)

try:
    res = urllib.request.urlopen(req)
    print("SUCCESS STATUS:", res.status)
    print("SUCCESS BODY:", res.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print("ERROR STATUS:", e.code)
    print("ERROR BODY:", e.read().decode('utf-8'))
except Exception as e:
    print("OTHER ERROR:", e)
