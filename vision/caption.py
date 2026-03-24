from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

stop_words = set(stopwords.words('english'))

def generate_tags_from_caption(caption):
    words = word_tokenize(caption.lower())
    
    tags = []
    for word in words:
        if word.isalnum() and word not in stop_words and len(word) > 2:
            tags.append(word)

    return list(set(tags))[:5]


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    tags = generate_tags_from_caption(caption)

    return caption, tags