import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pprint import pprint
url = 'https://api.thecatapi.com/v1/breeds'

response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)
df.head()

print(df.columns)

weight = df['weight']
type(weight)


# average weight of the cats
avg_weight = []
for i in weight:
    lowest, highest = i['metric'].split(' - ')
    avg = (int(lowest) + int(highest)) / 2
    avg_weight.append(avg)

sum(avg_weight) / len(avg_weight)

# average life span of the cats
df['life_span'].head()
df['life_span'].dtype
df['life_span'] = df['life_span'].str.extract('(\d+)').astype(int)
df['life_span'].mean()

# create a frequency table of the cats origin
df.groupby('origin')['origin'].count()

countries = list(df.groupby('origin')['origin'].count().index)
print(countries)
values = list(df.groupby('origin')['origin'].count().values)
print(values)
type(values)

cats = pd.DataFrame({'countries': countries, 'values': values})
print(cats)

# create a bar graph of the cat origin and number of breeds
plt.bar(countries, values)
plt.xticks(rotation=75)
plt.show()

# create a word cloud of the cat description
from wordcloud import WordCloud
text = ' '.join(df['description'])
wordcloud = WordCloud(width=100, height=20).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

df['description'].head()