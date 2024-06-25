# Description: This file contains utility functions for extracting text content from XML files and sanitizing unique text files.

# Import the ElementTree module from the xml package
import xml.etree.ElementTree as ET

# This function converts an XML file to a text file.
def xml_to_txt(xml_file):

    """
    Extracts text content from an XML file and saves it to a TXT file.
    
    Args:
        xml_file: Path to the XML file.

    Returns:
        txt_file: Path to the output TXT file.
    """

    print(f"Extracting text content from {xml_file}...")
    # Get the root element of the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # rename the txt file to the same name as the xml file
    txt_file = xml_file.replace('.xml', '.txt')
    
    # Open the TXT file in write mode
    with open(txt_file, 'w') as f:
        # Iterate through all elements and write text content
        for element in root.iter():
            # Write text content to the file if it exists
            if element.text:
                f.write(element.text + '\n')

    print(f"Text content extracted from {xml_file} and saved to {txt_file}")

    return txt_file

## This function converts a text file to a text file which contains only one unique word on each new line.
def unique_text(txt_file):
    """
    Converts a text file to a text file with only unique words on each new line.

    Args:
        txt_file: Path to the input text file.

    Returns:
        unique_txt_file: Path to the output unique text file.
    """
    # Print the input file
    print(f"Extracting unique words from {txt_file}...")

    # Rename input file to output file
    unique_txt_file = txt_file.replace('.txt', '_unique.txt')

    # Create a set to store unique words
    unique_words = set()

    # Read the input file and extract unique words
    with open(txt_file, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                unique_words.add(word)

    # Write the unique words to the output file
    with open(unique_txt_file, 'w') as f:
        for word in unique_words:
            f.write(word + '\n')

    print(f"Unique words extracted from {txt_file} and saved to {unique_txt_file}")

    return unique_txt_file

# This function sanitizes the unique text file by remove any non-alphabetic characters and rmeove numbers.
def sanitize_unique_text(txt_file):
    """
    Sanitizes a unique text file by removing non-alphabetic characters and numbers.

    Args:
        txt_file: Path to the input unique text file.

    Returns:
        sanitized_txt_file: Path to the output sanitized unique text file.
    """
    # Print the input file
    print(f"Sanitizing unique words in {txt_file}...")

    # Rename input file to output file
    sanitized_txt_file = txt_file.replace('_unique.txt', '_unique_sanitized.txt')

    # Create a set to store unique words
    unique_words = set()

    # Read the input file and extract unique words
    with open(txt_file, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                # Remove non-alphabetic characters and numbers
                word = ''.join(e for e in word if e.isalpha())
                # Add word to the set if it is not empty
                if word:
                    unique_words.add(word)

    # Write the sanitized unique words to the output file
    with open(sanitized_txt_file, 'w') as f:
        for word in unique_words:
            f.write(word + '\n')

    print(f"Unique words sanitized and saved to {sanitized_txt_file}")

    return sanitized_txt_file

