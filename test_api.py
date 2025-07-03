import requests, base64

# Replace this with a real 28x28 grayscale image file
with open('sample_image.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://127.0.0.1:5000/predict',
    json={'image': image_data}
)

print(response.json())
