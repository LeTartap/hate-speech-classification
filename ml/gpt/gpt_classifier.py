from utilities.generator import Generator
from utilities.parameters import Parameters
import pandas as pd
import os 
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


folder_path_test = "hate-speech-classification\data\cleaned\\test"
names = ["Facebook", "Reddit", "Twitter", "Youtube"]

# Load datasets
dfs_test = {n: pd.read_csv(os.path.join(folder_path_test, f"{n.lower()}_test.csv")) for n in names}

# get api key
key_path = "hate-speech-classification\ml\gpt\gpt-api-key.txt"
with open(key_path, "r") as f:
    api_key = f.read().strip()

# system message - consistent over all chats.
system_message = (
    "You are a hate-speech classifier, which is only allowed to otput two numbers: 1 if the provided text is hate speech, "
    "0 if the provided text does not contain hate speech. You will receive a piece of text (a comment/post by a user posted on the internet), "
    "to which you will reply with a classification number (1 if hate speech, 0 if not hate-speech). You can only reply with either 1, or 0 (numerical). "
    "Anything else is strictly forbidden."
    )


# a wrapper dictionary, to hold all of the api messages, for the purpose of parallel processing.
content = {} 
for name, df in dfs_test.items():
    for row in df.itertuples(index=True, name='Pandas'):
        id = f"{name}_{row.Index}"
        user_message = row.text
        # array passed to the api.
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}]
        parameters = Parameters(max_tokens=1)
        content[id] = {"messages": messages, "parameters": parameters}

# initialize gpt api wrapper class with gpt 3.5 turbo 0125 model.
generator = Generator(api_key=api_key, model="gpt-3.5-turbo-0125")
# generate result
result = generator.generate_batch(content, threads=7)

result_dfs = {}
for name in names:
    result_dfs[name] = dfs_test[name].copy()
    result_dfs[name]["predicted"] = None


for id, label in result.items():
    df_name, row_index = id.split("_")[0], int(id.split("_")[1])
    result_dfs[df_name].at[row_index, 'predicted'] = label

for name, df in result_dfs.items():
    df.to_csv(f"{name}.csv")

results = []
for name in names:  
    y_test = result_dfs[name]["label"]
    y_pred = result_dfs[name]["predicted"].astype("int64")
    # Evaluate classifier
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Collect results for each class
    for cls in ['0', '1']:
        results.append({
            'Test Dataset': name,
            'Class': cls,
            'Precision': report[cls]['precision'],
            'Recall': report[cls]['recall'],
            'F1-Score': report[cls]['f1-score'],
            'Accuracy': np.nan  # Accuracy is not class-specific
        })
    
    # Include the overall metrics for the test dataset
    results.append({
        'Test Dataset': name,
        'Class': 'overall',
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Accuracy': accuracy
    })

results_df = pd.DataFrame(results)

# Save the results DataFrame to a CSV file
results_df.to_csv('hate-speech-classification/ml/results/gpt_baseline_classification_results.csv', index=False)

# Display the results DataFrame
print(results_df)