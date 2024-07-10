from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment-analysis pipeline with explicit model and revision
sentiment_analyzer = pipeline(
    'sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', revision='af0f99b')


@app.route('/')
def game_page():
    return render_template('index.html')


@app.route('/analyse', methods=['POST'])
def analyze_sentiment():
    text = request.form['text']
    result = sentiment_analyzer(text)
    return render_template('index.html', text=text, result=result[0]['label'])


if __name__ == '__main__':
    app.run(debug=True)
