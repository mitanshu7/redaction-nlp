## This file helps create a dataset for the redaction NLP model.
from redaction_nlp.utils import xml_to_txt, unique_text, sanitize_unique_text

def create_dataset(xml_file):
    """
    Creates a dataset for the redaction NLP model by extracting unique words from an XML file.
    
    Args:
        xml_file: Path to the XML file.
    """
    # Convert the XML file to a text file
    txt_file = xml_to_txt(xml_file)
    
    # Extract unique words from the text file
    unique_txt_file = unique_text(txt_file)
    
    # Sanitize the unique words
    sanitized_txt_file = sanitize_unique_text(unique_txt_file)
    
    print(f"Sanitized unique words saved to {sanitized_txt_file}")
    print(f"Dataset created for {xml_file}")

# Test the create_dataset function
xml_file = 'sample.xml'
create_dataset(xml_file)
