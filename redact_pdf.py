# This file is used to redact sensitive information from a (scanned) PDF file.
# Note: The redaction is not perfect :)

# Import the required libraries

import cv2 # OpenCV, to read and manipulate images
import numpy as np # Numpy, for numerical operations
import easyocr # EasyOCR, for OCR
from pdf2image import convert_from_path # pdf2image, to convert PDF to images
import img2pdf # img2pdf, to convert images to PDF
import torch # PyTorch, for deep learning   
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline # Hugging Face Transformers, for NER
import os # OS, for file operations
from multiprocessing import cpu_count # Multiprocessing, to get the number of CPU cores
from time import time # Time, to measure the time taken

# Track the time taken
start_time = time()

# PDF file name
input_pdf = "sample.pdf"

# Load the model

# OCR model
print("Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=True)

# NER model
print("Loading NER model...")
# model_name = "dslim/bert-large-NER" # 
# model_name = "dslim/distilbert-NER"
model_name = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Move the model to the GPU for speed
device = torch.device("cuda")
model.to(device)

# Create a pipeline for NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

## Main code

# Create a directory to store pdf images
pdf_images_dir = f'{input_pdf}_images'
os.makedirs(pdf_images_dir, exist_ok=True)

# Convert the PDF to images
print("Converting PDF to images...")
pdf_images = convert_from_path(input_pdf, dpi=300, thread_count=cpu_count())


# List to hold the file paths of the saved images
redacted_image_files = []

# Assuming `images` is a list of PIL images
# Assuming `reader` is an instance of easyocr.Reader
# Assuming `nlp` is the Named Entity Recognition (NER) model

# Loop through the images
print("Redacting sensitive information...")
for i, pil_image in enumerate(pdf_images):
    
    print(f"Processing image {i + 1}...")
    # Convert PIL Image to a numpy array
    img_array = np.array(pil_image)

    # Ensure the image is in BGR format if using OpenCV functions later
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Read the text from the image
    result = reader.readtext(img_array, height_ths=0, width_ths=0, x_ths=0, y_ths=0)

    # Draw bounding boxes
    for (bbox, text, prob) in result:

        # Get the coordinates of the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Calculate the centers of the top and bottom of the bounding box
        # center_top = (int((top_left[0] + top_right[0]) / 2), int((top_left[1] + top_right[1]) / 2))
        # center_bottom = (int((bottom_left[0] + bottom_right[0]) / 2), int((bottom_left[1] + bottom_right[1]) / 2))

        # Perform NER on the text
        ner_result = nlp(text)

        # If the NER result is not empty, and the score is high
        if len(ner_result) > 0 and ner_result[0]['score'] > 0.9:

            # Get the entity and score
            # entity = ner_result[0]['entity']
            # score = str(ner_result[0]['score'])

            # Apply a irreversible redaction
            cv2.rectangle(img_array, top_left, bottom_right, (0, 0, 0), -1)
        # else:
            # entity = 'O'
            # score = '0'
        
        # # Draw the bounding box
        # cv2.rectangle(img_array, top_left, bottom_right, (0, 255, 0), 1)
        # # Draw the entity and score
        # cv2.putText(img_array, entity, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(img_array, score, center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the redacted image
    print(f"Saving redacted image {i + 1}...")
    redacted_image_path = f'{pdf_images_dir}/redacted_image_{i}.png'
    cv2.imwrite(redacted_image_path, img_array)
    redacted_image_files.append(redacted_image_path)

# Convert the redacted images to a single PDF
print("Converting redacted images to PDF...")
pdf_path = f'{input_pdf}_redacted.pdf'
with open(pdf_path, 'wb') as f:
    f.write(img2pdf.convert(redacted_image_files))

print(f"PDF saved as {pdf_path}")

# Remove the directory with the images
print("Cleaning up...")
for file in redacted_image_files:
    os.remove(file)

os.rmdir(pdf_images_dir)

# Print the time taken
print(f"Time taken: {time() - start_time:.2f} seconds")
