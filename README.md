# Redaction-NLP

## Overview

Redaction-NLP is a Python-based project designed to redact Personally Identifiable Information (PII) from various text sources using Natural Language Processing (NLP). The project employs OCR to extract text from images and Named Entity Recognition (NER) models to identify PII. It offers these primary functionalities:

1. **PDF Redaction**:
   - Convert a PDF to images for OCR
   - Use OCR to identify text and bounding boxes.
   - Apply mask on boxes identified as PII using Named-Entity-Recognition Model.
   - Output the same PDF with PII masked.


## Features

- **OCR Integration**: Extracts text from images.
- **NER Models**: Identifies PII in text and images.
- **Redaction Techniques**: Masks PII in images and text.
- **PDF Handling**: Processes and redacts text in PDFs.

## Usage

This project can be useful for organizations needing to ensure privacy by redacting sensitive information from documents and images before sharing or publishing.

## Technologies Used

- Python
- OCR for text extraction (EasyOCR in Main, Tesseract in another branch)
- Named Entity Recognition for PII identification with models from huggingface.
- PDF and image processing libraries.
