import pandas as pd
import pygwalker as pyg

reddit = pd.read_json('../data/reddit_comments.json')
pyg.walk(reddit)
print(reddit.head())
