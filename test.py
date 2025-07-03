import requests

url = "http://127.0.0.1:5000/start_detection"

json_data = {
    "task_id": "202507020001",
    "standalone": "P12-PK11_X01",
    "img_paths": {
        "state_1": "C:/Users/13557/Desktop/DSdatabase/PK11_X01_1.jpg",
        "state_2": "C:/Users/13557/Desktop/DSdatabase/PK11_X01_4.jpg",  # 未使用
        "state_3": "C:/Users/13557/Desktop/DSdatabase/PK11_X01_2.jpg",
        "state_4": "C:/Users/13557/Desktop/DSdatabase/PK11_X01_3.jpg"
    }
}

response = requests.post(url, json=json_data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
