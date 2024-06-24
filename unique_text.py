## This file converts a text file to a text file which contains only one unique word on each new line.

import os

def unique_text(input_file):
    """
    Converts a text file to a text file with only unique words on each new line.

    Args:
        input_file: Path to the input text file.
    """

    # Rename input file to output file
    output_file = input_file.replace('.txt', '_unique.txt')

    # Create a set to store unique words
    unique_words = set()

    # Read the input file and extract unique words
    with open(input_file, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                unique_words.add(word)

    # Write the unique words to the output file
    with open(output_file, 'w') as f:
        for word in unique_words:
            f.write(word + '\n')

    print(f"Unique words extracted from {input_file} and saved to {output_file}")

# Main code
input_file = "sample-0001.txt"
unique_text(input_file)