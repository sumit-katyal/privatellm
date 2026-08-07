import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "My name is Abdul Rauf jatoi"

# Step 1: Tokenize text into words
tokens = word_tokenize(text)

# Step 2: Convert to lowercase and remove non-alphanumeric tokens
tokens = [word.lower() for word in tokens if word.isalnum()]

# Step 3: Remove stop words
filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

# Display the preprocessed tokens
print(filtered_tokens)