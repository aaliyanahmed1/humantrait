import requests

url = "http://localhost:8000/upload/"
files = {"file": open("F://Human-Trait-main//new-test//test.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
