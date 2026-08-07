import requests
from bs4 import BeautifulSoup

# Step 1: Specify the URL
url = "https://wikipedia.com"

# Step 2: Send a GET request to the website
response = requests.get(url)

# Step 3: Parse the website content
soup = BeautifulSoup(response.text, "html.parser")

# Step 4: Extract all text from the page
text_data = soup.get_text()

# Step 5: Print the first 500 characters of the text
print(text_data[:500])