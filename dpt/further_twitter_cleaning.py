import csv
import re

def clean_text(text):
    # Remove Unicode characters (emojis)
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)
    # Remove special characters and punctuation except for @ if you decide to keep it
    text = re.sub(r'[!\"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)
    # Optionally, remove Twitter handles
    text = re.sub(r'@\w+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_csv(input_filename, output_filename):
    with open(input_filename, mode='r', encoding='utf-8') as infile, \
         open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Assuming first row is the header
        header = next(reader)
        writer.writerow(header)

        # Process each row
        for row in reader:
            # Assuming the tweet text is in the first column, and label in the second
            cleaned_text = clean_text(row[0])
            # Write the cleaned text with other columns unchanged
            row[0] = cleaned_text
            writer.writerow(row)


# Example usage
input_csv = '../data/cleaned_twitter.csv'  # Make sure to replace this with your actual input file path
output_csv = '../data/david_cleaned_twitter.csv'  # The output file where the cleaned data will be written
process_csv(input_csv, output_csv)
