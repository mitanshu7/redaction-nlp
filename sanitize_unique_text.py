# This file sanitizes the unique text file by remove any non-alphabetic characters and rmeove numbers.

import os

def sanitize_unique_text(input_file):
    """
    Sanitizes a unique text file by removing non-alphabetic characters and numbers.

    Args:
        input_file: Path to the input unique text file.
    """

    # Rename input file to output file
    output_file = input_file.replace('_unique.txt', '_unique_sanitized.txt')

    # Create a set to store unique words
    unique_words = set()

    # Read the input file and extract unique words
    with open(input_file, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                # Remove non-alphabetic characters and numbers
                word = ''.join(e for e in word if e.isalpha())
                unique_words.add(word)

    # Write the sanitized unique words to the output file
    with open(output_file, 'w') as f:
        for word in unique_words:
            f.write(word + '\n')

    print(f"Unique words sanitized and saved to {output_file}")

# Main code
input_file = "sample-0001_unique.txt"

sanitize_unique_text(input_file)