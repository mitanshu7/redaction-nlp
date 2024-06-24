import xml.etree.ElementTree as ET

def xml_to_txt(xml_file):

    """
    Extracts text content from an XML file and saves it to a TXT file.
    
    Args:
        xml_file: Path to the XML file.
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




# Main code
xml_file = "sample-0001.xml"
xml_to_txt(xml_file)