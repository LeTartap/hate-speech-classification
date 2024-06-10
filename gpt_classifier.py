from sklearn.metrics import classification_report, accuracy_score
from utilities.generator import Generator
from utilities.parameters import Parameters
import pandas as pd
import os 
from sklearn.model_selection import train_test_split


folder_path = "hate-speech-classification\data"
names = ["Facebook", "Reddit", "Twitter", "Youtube"]
dfs = {}

# load data
for n in names:
    dfs[n] = pd.read_csv(os.path.join(folder_path, f"cleaned_{n.lower()}.csv"))

print(dfs["Facebook"]["label"].unique())
# get api key
key_path = "hate-speech-classification\gpt-api-key.txt"
with open(key_path, "r") as f:
    api_key = f.read().strip()

# system message - consistent over all chats.
system_message = (
    "You are a hate-speech classifier, which is only allowed to otput two numbers: 1 if the provided text is hate speech, "
    "0 if the provided text does not contain hate speech. You will receive a piece of text (a comment/post by a user posted on the internet), "
    "to which you will replace with a classification number (1 if hate speech, 0 if not hate-speech). "
    )

# Split data
test_dfs = {}
train_dfs = {}
for name in names:
    # Separate features and target
    X = dfs[name].drop(columns='label')
    y = dfs[name]['label']

    # Split the DataFrame
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42, stratify=y)

    # Combine X and y back into DataFrames
    train_dfs[name] = pd.concat([X_train, y_train], axis=1)
    test_dfs[name] = pd.concat([X_test, y_test], axis=1)

# a wrapper dictionary, to hold all of the api messages, for the purpose of parallel processing.
content = {} 
for name, df in test_dfs.items():
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
result = generator.generate_batch(content)

result_dfs = {}
for name in names:
    result_dfs[name] = test_dfs[name].copy()
    result_dfs[name]["predicted"] = None

for id, label in result.items():
    df_name, row_index = id.split("_")[0], int(id.split("_")[1])
    result_dfs[df_name].at[row_index, 'predicted'] = label
   
results = {}
for name in names:  
    y_test = result_dfs[name]["label"]
    y_pred = result_dfs[name]["predicted"].astype("int64")
    print(y_test)
    print(y_pred)
    # Evaluate classifier
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'classification_report': report,
        'accuracy': accuracy
    }

# Convert results to DataFrame for display
results_df = pd.DataFrame({
    name: {
        'accuracy': results[name]['accuracy'],
        'precision': results[name]['classification_report']['weighted avg']['precision'],
        'recall': results[name]['classification_report']['weighted avg']['recall'],
        'f1-score': results[name]['classification_report']['weighted avg']['f1-score']
    }
    for name in names
}).T

# Display the results DataFrame
print(results_df)