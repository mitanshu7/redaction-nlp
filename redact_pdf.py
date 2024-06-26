# This file is used to redact sensitive information from a (scanned) PDF file.
# Note: The redaction is not perfect :)

# Import the required libraries

import cv2 # OpenCV, to read and manipulate images
import numpy as np # Numpy, for numerical operations
import pytesseract # PyTesseract, for OCR
from pytesseract import Output 
from pdf2image import convert_from_path # pdf2image, to convert PDF to images
import img2pdf # img2pdf, to convert images to PDF
import torch # PyTorch, for deep learning   
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline # Hugging Face Transformers, for NER
import os # OS, for file operations
import multiprocessing as mp # Multiprocessing, to speed up the process
from time import time # Time, to measure the time taken
from glob import glob # Glob, to get file paths




##########################################################################################################

# Set the parameters
# For tessaract to use all cores
os.environ['OMP_THREAD_LIMIT'] = str(mp.cpu_count())

# Load the model

# NER model
print("Loading NER model...")
# model_name = "dslim/bert-large-NER" # 334M parameters
# model_name = "dslim/distilbert-NER" # 65.2M parameters
model_name = "dslim/bert-base-NER" # 108M parameters
# model_name = "Clinical-AI-Apollo/Medical-NER" # 184M parameters

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Move the model to the GPU for speed
device = torch.device("cuda")
model.to(device)

# Create a pipeline for NER
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

# Image ocr options:
# Image format
img_format = 'ppm'

# DPI
dpi = 150

# Redaction score threshold
redaction_score_threshold = 0.0

##########################################################################################################
## Functions

def convert_to_images(pdf_file_path):
    """
    This function converts a PDF file to images.
    
    Args:
    pdf_file_path (str): The path to the PDF file.

    Returns:
    pdf_images_dir (str): The directory with the images.
    """

    # Create a directory to store pdf images
    pdf_images_dir = f'{pdf_file_path}_images'
    os.makedirs(pdf_images_dir, exist_ok=True)

    # Convert the PDF to images
    print("Converting PDF to images...")
    convert_from_path(pdf_file_path, dpi=dpi, thread_count=mp.cpu_count(), output_folder=pdf_images_dir, fmt=img_format)

    # Fix the file names
    for file in os.listdir(pdf_images_dir):
        os.rename(os.path.join(pdf_images_dir, file), os.path.join(pdf_images_dir, file.split('-')[-1]))

    # Return the directory with the images
    return pdf_images_dir

def ocr_image(image_path):
    """
    This function performs OCR on an image.

    Args:
    image_path (str): The path to the image.

    Returns:
    ocr_result (dict): The OCR results.
    """

    # Read the image
    cv_image = cv2.imread(image_path)

    # Perform OCR on the image
    ocr_result = pytesseract.image_to_data(cv_image, output_type=Output.DICT)

    return ocr_result

def ner_text(ocr_results):
    """
    This function performs NER on the text.

    Args:
    ocr_results (dict): The OCR results.

    Returns:
    ner_results (list): The NER results.
    """

    # Get the text from the OCR results
    text = ' '.join([tmp_text for tmp_text in ocr_results['text']])

    # Perform NER on the text
    ner_results = nlp(text)

    return ner_results


def redact_image(pdf_image_path):

    # Read the image
    cv_image = cv2.imread(pdf_image_path)

    # Perform OCR on the image
    result = ocr_image(pdf_image_path)

    # Perform NER on the text
    ner_results = ner_text(result)


    # Draw bounding boxes over recognized text (words)
    for i, (word, ner_result) in enumerate(zip(result['text'], ner_results)):

        # Get the coordinates of the bounding box
        (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        center_top = (x, y - 10)
        center_bottom = (x, y + h + 10)

        # If the NER result is not empty, and the score is higher than the threshold
        if len(ner_result) > 0 and ner_result['score'] > redaction_score_threshold and result['level'][i] == 5:

            # Get the entity and score
            # entity = ner_result[0]['entity']
            # score = str(ner_result[0]['score'])

            # Apply a irreversible redaction
            # cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 0), -1)
            pass
        # else:
            # entity = 'O'
            # score = '0'
            
        # # Draw the bounding box
        cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 1)
        cv2.putText(cv_image, word, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(cv_image, str(result['conf'][i]), center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # # Draw the entity and score
        # cv2.putText(cv_image, entity, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # cv2.putText(cv_image, score, center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save the redacted image
    print(f"Saving redacted {pdf_image_path}...")
    redacted_image_path = pdf_image_path.replace(f'.{img_format}', f'_redacted.{img_format}')
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
    input_pdf_path = 'sample3.pdf'
    input_pdf_name = input_pdf_path.split('.')[-2]

    # Get the number of processes
    num_processes = 2

    # Start the timer
    start_time = time()

    # Convert the PDF to images
    pdf_images_dir = convert_to_images(input_pdf_path)

    # Get the file paths of the images
    pdf_images = glob(f'{pdf_images_dir}/*.{img_format}', recursive=True)
    pdf_images.sort()


    # Redact the sensitive information in parallel
    # mp.set_start_method('spawn')
    mp.set_start_method('forkserver')
    with mp.Pool(num_processes) as pool:
        redacted_image_files = pool.map(redact_image, pdf_images)

    # Convert the redacted images to a single PDF
    redacted_pdf_path = stich_images_to_pdf(redacted_image_files, input_pdf_name)

    # Cleanup
    cleanup(redacted_image_files, pdf_images, pdf_images_dir)
    

    # Print the time taken
    print(f"Time taken: {time() - start_time:.2f} seconds")
