from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('Dataset_Start-Right - Sheet1.csv')
df = df.dropna()

# Preprocess your data
tfidf_vectorizer = TfidfVectorizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_skill = request.form['input_skill']
        input_domain = request.form['input_domain']
        input_experience = int(request.form['input_experience'])

        # Filter the DataFrame based on input criteria
        filtered_df = df[df['experience'] > input_experience]
        filtered_df['cosine_similarity1'] = filtered_df.apply(lambda x: cosine_similarity_strings(x['SKILLS'], input_skill), axis=1)
        filtered_df['cosine_similarity2'] = filtered_df.apply(lambda x: cosine_similarity_strings(x['DOMAIN'], input_domain), axis=1)
        filtered_df['cosine_similarity'] = filtered_df['cosine_similarity1'] + filtered_df['cosine_similarity2']

        # Sort and select the top results
        sorted_data_descending = filtered_df.sort_values(by='cosine_similarity', ascending=False)

        p_name = sorted_data_descending.drop_duplicates().head(20)
        p_name=p_name.drop('cosine_similarity',axis=1)

        return render_template('result.html', p_name=p_name)

    return render_template('index.html')

def cosine_similarity_strings(str1, str2):
    tfidf = tfidf_vectorizer.fit_transform([str1, str2])
    cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()
    return cosine_similarities[0]

if __name__ == '__main__':
    app.run(debug=True)

