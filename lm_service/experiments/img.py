import requests
import base64

# Sample image from the internet
image_url = "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d?auto=format&fit=crop&w=400&q=80"

# Download the image
response = requests.get(image_url)
if response.status_code != 200:
	raise Exception(f"Failed to download the image: {response.status_code}")

# Encode the image in Base64
img_base64 = base64.b64encode(response.content).decode('utf-8')

# Show the start of the encoded string
print(img_base64[:100] + "...")

# Prepare the payload for Ollama
url = "http://localhost:11434/api/generate"
payload = {
	"model": "gemma3:4b",
	"prompt": "Opisz obrazek w 2 zdaniach",
	"images": [img_base64],
	"stream": False
}

# Send the request
resp = requests.post(url, json=payload)
data = resp.json()
print(data)
