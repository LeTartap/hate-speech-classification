import pandas as pd

# Load the CSV file
file_path = '../data/cleaned_reddit.csv'  # Update with the actual path to your CSV file
data = pd.read_csv(file_path)

# Convert float to int in the 'label' column
data['label'] = data['label'].astype(int)

# Save the updated DataFrame back to CSV
data.to_csv('cleaned_reddit_int.csv', index=False)

