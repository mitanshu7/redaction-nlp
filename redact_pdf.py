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
import multiprocessing as mp # Multiprocessing, to speed up the process
from time import time # Time, to measure the time taken
from glob import glob # Glob, to get file paths



##########################################################################################################

# Load the model

# OCR model
print("Loading OCR model...")
reader = easyocr.Reader(['en'], gpu=True)

# NER model
print("Loading NER model...")
# model_name = "dslim/bert-large-NER" # 334M parameters
# model_name = "dslim/distilbert-NER" # 65.2M parameters
model_name = "dslim/bert-base-NER" # 108M parameters

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Move the model to the GPU for speed
device = torch.device("cuda")
model.to(device)

# Create a pipeline for NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

##########################################################################################################
## Functions

def convert_to_images(pdf_file_path):

    # Create a directory to store pdf images
    pdf_images_dir = f'{pdf_file_path}_images'
    os.makedirs(pdf_images_dir, exist_ok=True)

    # Convert the PDF to images
    print("Converting PDF to images...")
    convert_from_path(pdf_file_path, dpi=300, thread_count=mp.cpu_count(), output_folder=pdf_images_dir)

    # Fix the file names
    for file in os.listdir(pdf_images_dir):
        os.rename(os.path.join(pdf_images_dir, file), os.path.join(pdf_images_dir, file.split('-')[-1]))

    # Return the directory with the images
    return pdf_images_dir

def redact_image(pdf_image_path):

    # Loop through the images
    print("Redacting sensitive information...")

    print(f"Processing {pdf_image_path}...")
    # Read the image
    cv_image = cv2.imread(pdf_image_path)

    # Read the text from the image
    result = reader.readtext(cv_image, height_ths=0, width_ths=0, x_ths=0, y_ths=0)

    # Get the text from the result
    text = ' '.join([text for (bbox, text, prob) in result])

    # Perform NER on the text
    ner_results = nlp(text)

    # Draw bounding boxes
    for ((bbox, text, prob),ner_result) in zip(result, ner_results):

        # Get the coordinates of the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Calculate the centers of the top and bottom of the bounding box
        # center_top = (int((top_left[0] + top_right[0]) / 2), int((top_left[1] + top_right[1]) / 2))
        # center_bottom = (int((bottom_left[0] + bottom_right[0]) / 2), int((bottom_left[1] + bottom_right[1]) / 2))


        # If the NER result is not empty, and the score is high
        if len(ner_result) > 0 and ner_result['score'] > 0.9:

            # Get the entity and score
            # entity = ner_result[0]['entity']
            # score = str(ner_result[0]['score'])

            # Apply a irreversible redaction
            cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 0), -1)
        # else:
            # entity = 'O'
            # score = '0'
            
        # # Draw the bounding box
        # cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 1)
        # # Draw the entity and score
        # cv2.putText(cv_image, entity, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(cv_image, score, center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the redacted image
    print(f"Saving redacted {pdf_image_path}...")
    redacted_image_path = pdf_image_path.replace('.ppm', '_redacted.png')
    # Save the redacted image in png format
    cv2.imwrite(redacted_image_path, cv_image)

    return redacted_image_path

def stich_images_to_pdf(redacted_image_files, input_pdf_name):

    # Sort the redacted images
    redacted_image_files.sort()

    # Convert the redacted images to a single PDF
    print("Converting redacted images to PDF...")
    redacted_pdf_path = f'{input_pdf_name}_redacted.pdf'
    with open(redacted_pdf_path, 'wb') as f:
        f.write(img2pdf.convert(redacted_image_files))

    print(f"PDF saved as {redacted_pdf_path}")

    return redacted_pdf_path

def cleanup(redacted_image_files, pdf_images, pdf_images_dir):

    # Remove the directory with the images
    print("Cleaning up...")

    # Remove the redacted images
    for file in redacted_image_files:
        os.remove(file)

    # Remove the pdf images
    for file in pdf_images:
        os.remove(file)

    # Remove the pdf images directory
    os.rmdir(pdf_images_dir)

    return None

##########################################################################################################

if __name__ == '__main__':

    # Get the input PDF file
    input_pdf_path = 'sample.pdf'
    input_pdf_name = input_pdf_path.split('.')[-2]

    # Start the timer
    start_time = time()

    # Convert the PDF to images
    pdf_images_dir = convert_to_images(input_pdf_path)

    # Get the file paths of the images
    pdf_images = glob(f'{pdf_images_dir}/*.ppm', recursive=True)
    # pdf_images.sort()


    # Redact the sensitive information in parallel
    mp.set_start_method('forkserver')
    with mp.Pool(2) as pool:
        redacted_image_files = pool.map(redact_image, pdf_images)

    # Convert the redacted images to a single PDF
    redacted_pdf_path = stich_images_to_pdf(redacted_image_files, input_pdf_name)

    # Cleanup
    cleanup(redacted_image_files, pdf_images, pdf_images_dir)
    

    # Print the time taken
    print(f"Time taken: {time() - start_time:.2f} seconds")
