import requests

# API URL for the model
API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-base-handwritten"

# Your API token
headers = {"Authorization": "Bearer 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"}

def query(filename):
    """Send an image file to the Hugging Face API for OCR."""
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

# Example usage
output = query("01.png")  # Replace 'example.jpg' with your image file
print(output)
