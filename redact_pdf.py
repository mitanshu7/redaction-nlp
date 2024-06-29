# This file is used to redact sensitive information from a (scanned) PDF file.
# Note: The redaction is not perfect :)

# Import the required libraries

import cv2 # OpenCV, to read and manipulate images
import pytesseract as pt # PyTesseract, for OCR
from pdf2image import convert_from_path # pdf2image, to convert PDF to images
import img2pdf # img2pdf, to convert images to PDF
import torch # PyTorch, for deep learning   
from torch import multiprocessing as mpt # Torch Multiprocessing, to speed up the process
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline # Hugging Face Transformers, for NER
import os # OS, for file operations
import multiprocessing as mp # Multiprocessing, to speed up the process
from time import time # Time, to measure the time taken
from glob import glob # Glob, to get file paths
import copy # Copy, to copy objects




##########################################################################################################

# Set the parameters
# For tessaract to use all cores
# os.environ['OMP_THREAD_LIMIT'] = str(mp.cpu_count())



# Image ocr options:
# Image format
img_format = 'ppm'

# DPI
dpi = 300

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
    pdf_file_name = os.path.basename(pdf_file_path).split('.')[0]
    pdf_images_dir = f'{pdf_file_name}_images'
    os.makedirs(pdf_images_dir, exist_ok=True)

    # Convert the PDF to images
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
    image_ocr_result (dict): The OCR results as value to the image path as key.
    """

    # Dictionary to store the OCR results wrt the image
    image_ocr_result = {}

    # Read the image
    cv_image = cv2.imread(image_path)

    # Perform OCR on the image
    ocr_result = pt.image_to_data(cv_image, output_type=pt.Output.DICT)

    # Store the OCR results
    image_ocr_result[f'{image_path}'] = ocr_result  

    # Return the OCR results
    return image_ocr_result

def ner_text(image_ocr_result):
    """
    This function performs NER on the text.

    Args:
    ocr_results (dict): The OCR results.

    Returns:
    image_ocr_ner_result (dict): The NER and OCR results as value to the image path as key.
    
    """

    # Load the model

    # NER model
    # model_name = "dslim/bert-large-NER" # 334M parameters
    # model_name = "dslim/distilbert-NER" # 65.2M parameters
    # model_name = "dslim/bert-base-NER" # 108M parameters
    model_name = "Clinical-AI-Apollo/Medical-NER" # 184M parameters

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Move the model to the GPU for speed
    device = torch.device("cuda")
    model.to(device)

    # Create a pipeline for NER
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0)

    # Dictionary to store the NER and OCR results wrt the image
    image_ocr_ner_result = copy.deepcopy(image_ocr_result)


    # Get key from the OCR results dictionary
    image_path = list(image_ocr_ner_result.keys())[0]

    # Get the text from the OCR results
    text = ' '.join([tmp_text for tmp_text in image_ocr_ner_result[image_path]['text']])

    # Perform NER on the text
    ner_result = nlp(text)

    # Store the NER results in the OCR results dictionary
    image_ocr_ner_result[image_path]['ner'] = ner_result

    # Return the NER results along with the OCR results
    return image_ocr_ner_result


def redact_image(image_ocr_ner_result):

    # Get the image path
    pdf_image_path = list(image_ocr_ner_result.keys())[0]

    # Read the image
    cv_image = cv2.imread(pdf_image_path)

    # Get the OCR results
    ocr_result = image_ocr_ner_result[pdf_image_path]

    # Get the NER results
    ner_result_list = ocr_result['ner']


    # Draw bounding boxes over recognized text (words)
    for i, (word, ner_result) in enumerate(zip(ocr_result['text'], ner_result_list)):

        # Get the coordinates of the bounding box
        (x, y, w, h) = (ocr_result['left'][i], ocr_result['top'][i], ocr_result['width'][i], ocr_result['height'][i])
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        center_top = (x, y - 10)
        center_bottom = (x, y + h + 10)

        # If the NER result is not empty, and the score is higher than the threshold
        if (len(ner_result) > 0) and (ner_result['score'] > redaction_score_threshold) and (ocr_result['level'][i] == 5):

            # Get the entity and score
            # entity = ner_result[0]['entity']
            # score = str(ner_result[0]['score'])

            # Apply a irreversible redaction
            cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 0), -1)
            # pass
        # else:
            # entity = 'O'
            # score = '0'
            
        # # Draw the bounding box and text
        # cv2.rectangle(cv_image, top_left, bottom_right, (0, 255, 0), 1)
        # cv2.putText(cv_image, word, center_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.putText(cv_image, str(result['conf'][i]), center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(cv_image, ner_result['entity'], center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        # cv2.putText(cv_image, ner_result['score'], center_bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

       

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

    # Start the timer
    start_time = time()

    # Set the Multiprocessing start method
    mp.set_start_method('forkserver')

    # Set the number of processes for gpu
    num_gpu_processes = 5

    # Set the number of processes for cpu
    num_cpu_processes = mp.cpu_count()

    # Get the input PDF file
    input_pdf_path = 'sample3.pdf'
    input_pdf_name = os.path.basename(input_pdf_path).split('.')[0]
    print(f"Input PDF: {input_pdf_path}")

    # Convert PDF to images
    print("Converting PDF to images...")
    pdf_images_dir = convert_to_images(input_pdf_path)

    # Do OCR on the images 
    print("Performing OCR on the images...")

    # Get the image files
    pdf_images = glob(f'{pdf_images_dir}/*.{img_format}', recursive=True)
    pdf_images.sort()
    
    # Perform OCR in parallel (CPU)
    with mp.Pool(num_cpu_processes) as pool:
        image_ocr_results = pool.map(ocr_image, pdf_images)

    # Perform NER in parallel (GPU)
    with mpt.Pool(num_gpu_processes) as pool:
        image_ocr_ner_results = pool.map(ner_text, image_ocr_results)
    
    # Perform Redaction in parallel (CPU)
    with mp.Pool(num_cpu_processes) as pool:
        redacted_image_files = pool.map(redact_image, image_ocr_ner_results)

    # Convert the redacted images to a single PDF
    redacted_pdf_path = stich_images_to_pdf(redacted_image_files, input_pdf_name)

    # Cleanup
    cleanup(redacted_image_files, pdf_images, pdf_images_dir)
    

    # Print the time taken
    print(f"Time taken: {time() - start_time:.2f} seconds")
