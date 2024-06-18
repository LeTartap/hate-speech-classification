from sklearn.metrics import classification_report, accuracy_score
from utilities.generator import Generator
from utilities.parameters import Parameters
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import json
import time
from openai import OpenAI
import sys

folder_path = "hate-speech-classification\data"
names = ["Facebook", "Reddit", "Twitter", "Youtube"]
dfs = {}

# load data
for n in names:
    df = pd.read_csv(os.path.join(folder_path, f"cleaned_{n.lower()}.csv"))
    df_label_1 = df[df["label"] == 1].head(10)
    df_label_0 = df[df["label"] == 0].head(10)
    dfs[n] = pd.concat([df_label_1, df_label_0], axis=0).reset_index(drop=True)
print(dfs["Facebook"]["label"].unique())
# get api key
key_path = "hate-speech-classification\ml\gpt\gpt-api-key.txt"
with open(key_path, "r") as f:
    api_key = f.read().strip()

# system message - consistent over all chats.
system_message = (
    "You are a hate-speech classifier, which is only allowed to otput two numbers: 1 if the provided text is hate speech, "
    "0 if the provided text does not contain hate speech. You will receive a piece of text (a comment/post by a user posted on the internet), "
    "to which you will reply with a classification number (1 if hate speech, 0 if not hate-speech). "
    )

# split data into training and test sets while stratyfying labels
def split_data(dfs):
    test_dfs = {}
    train_dfs = {}
    for name in names:
        # Separate features and target
        X = dfs[name].drop(columns='label')
        y = dfs[name]['label']
        # Split the DataFrame
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Combine X and y back into DataFrames
        train_dfs[name] = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_dfs[name] = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    return train_dfs, test_dfs


# train_content = {"id": {"messages": messages in openai format, "parameters": parameters for the model}}
def get_train_content(train_df, sys_message):
    train_content = {}
    for row in train_df.itertuples(index=True, name='Pandas'):
        id = f"{row.Index}"
        user_message = row.text
        label = row.label
        # array passed to the api.
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": str(label)}]
        parameters = Parameters(max_tokens=1)
        train_content[id] = {"messages": messages, "parameters": parameters}
    return train_content


def get_test_content(test_df, sys_message):
    test_content = {}
    for row in test_df.itertuples(index=True, name='Pandas'):
        id = f"{row.Index}"
        user_message = row.text
        # array passed to the api.
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}]
        parameters = Parameters(max_tokens=1)
        test_content[id] = {"messages": messages, "parameters": parameters}
    return test_content


def fine_tune(train_content, dataset_name):
    # save fine-tuning data into a file
    data_path = os.path.join(folder_path, f"{dataset_name}_fine_tuning.jsonl")
    data = []
    for key, val in train_content.items():
        data.append({"messages": val["messages"]})
    with open(data_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    # upload the file
    client = OpenAI(api_key=api_key)
    file_object = client.files.create(
        file=open(data_path, "rb"),
        purpose="fine-tune"
    )
    id = file_object.id
    # start fine-tuning job. If upload is not yet done, the job will wait.
    client.fine_tuning.jobs.create(
        training_file=id, 
        model="gpt-3.5-turbo-0125",
        hyperparameters= {
            "n_epochs": 2,
            "learning_rate_multiplier": 0.01
        }
        )


train_dfs, test_dfs = split_data(dfs)
# populate manually, way easier than automation, as fine-tuning job takes a while.
model_ids = {
    "Facebook": "ft:gpt-3.5-turbo-0125:personal::9ZfeiSZQ", 
    "Reddit": "ft:gpt-3.5-turbo-0125:personal::9ZfeJ1kZ", 
    "Twitter": "ft:gpt-3.5-turbo-0125:personal::9ZfeiSZQ", 
    "Youtube": "ft:gpt-3.5-turbo-0125:personal::9ZfjLj3E"}
finetuning_jobs_num = 0
for name, train_df in train_dfs.items():
    if model_ids[name] == None:
        if finetuning_jobs_num < 3:
            train_content = get_train_content(train_df, system_message)
            fine_tune(train_content, name)
            finetuning_jobs_num += 1

print("fine-tuning jobs started if needed")
# comment when all of the models are fine-tuned
# sys.exit(1)

# test
result_dfs = {}

for name in names:
    result_dfs[name] = test_dfs[name].copy()
    result_dfs[name]["predicted"] = None

for name, test_df in test_dfs.items():
    test_content = get_test_content(test_df, system_message)
    generator = Generator(api_key=api_key, model=model_ids[name])
    result = generator.generate_batch(test_content)
    for id, label in result.items():
        row_index = int(id)
        result_dfs[name].at[row_index, 'predicted'] = label
results = {}
for name in names:  
    y_test = result_dfs[name]["label"].astype("int64")
    y_pred = result_dfs[name]["predicted"].astype("int64")
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