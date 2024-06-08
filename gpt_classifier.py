from utilities.generator import Generator
from utilities.parameters import Parameters

key_path = "hate-speech-classification\gpt-api-key.txt"
with open(key_path, "r") as f:
    api_key = f.read().strip()

system_message = (
    "You are a hate-speech classifier, which is only allowed to otput two numbers: 1 if the provided text is hate speech, "
    "0 if the provided text does not contain hate speech. You will receive a piece of text (a comment/post by a user posted on the internet), "
    "to which you will replace with a classification number (1 if hate speech, 0 if not hate-speech). The text might include offensive language "
    "and slurs, without being hate speech, consider this carefully."
    )

user_message = "These niggas were asking for this, finally someone kicked their ass."

messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": user_message}]
parameters = Parameters(max_tokens=1)
generator = Generator(api_key=api_key, model="gpt-3.5-turbo-0125")

content = {"test-id-1": {"messages":messages, "parameters":parameters}}
result = generator.generate_batch(content)
print(result)