from unittest import result

from flask import Flask, render_template, request
import pickle

from matplotlib.pyplot import title
import time
import requests
from gnews import GNews

google_news = GNews(language='en',country='India',max_results=10)

app = Flask(__name__)
NEWS_API_KEY = 'f98deb82035b4540bdda971bdc586d28'
TRUSTED_SOURCES = [
    "BBC News", "Reuters","Associated Press","The Hindu","NDTV","CNN","The Indian Express",
    "Times of India","Hindustan Times","Al Jazeera English","ABC News","The Washington Post",
    "ESPN","Cricbuzz","ICC","Sky Sports","The Guardian","NPR","Bloomberg","CNBC","Fox News","CBS News"
]
def search_google_news(query):
    try:
        return google_news.get_news(query)
    except Exception as e:
        print(e)
        return []

def search_news(query):
    print("search news function called")
    url = "https://newsapi.org/v2/everything"
    parameters = {
        "q": query,
        "pageSize": 20,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=parameters)
    data = response.json()
    trusted_articles = []
    if data.get("status") == "ok":
        for article in data.get("articles", []):
            source = article["source"]["name"]

            if source in TRUSTED_SOURCES:
                trusted_articles.append(article)

    data["trusted_articles"] = trusted_articles
    return data

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    title = request.form['title']
    article = request.form['article']
    search_query = title + " " + article[:150]
    news_data = search_news(search_query)
    google_articles = search_google_news(search_query)
    newsapi_articles = news_data.get("trusted_articles", [])
    all_articles = []
    all_articles.extend(newsapi_articles)
    all_articles.extend(google_articles)
    print("NewsAPI articles:",len(newsapi_articles))
    print("Google News articles:",len(google_articles))
    print("Total Combined Articles:", len(all_articles))
    articles = news_data.get("trusted_articles", [])
    trusted_count = 0
    for news_item in all_articles:
        if "source" in news_item:
            source = news_item["source"]["name"]
        else:
            source = news_item["publisher"]["title"]
        for trusted_score in TRUSTED_SOURCES:
            if trusted_score.lower() in source.lower():
                trusted_count += 1
                break
    print("Trusted Sources Found:", trusted_count)
    if len(all_articles) > 0:
        first_article = all_articles[0]
        if "source" in first_article:
            source_name = first_article["source"]["name"]
            article_url = first_article["url"]
            published_at = first_article["publishedAt"]
        else:
            source_name = first_article["publisher"]["title"]
            article_url = first_article["url"]
            published_at = first_article.get("published date", "Unknown")
    else:
        source_name = "No trusted source found"
        article_url = "#"
        published_at = "Not Available"
    trusted = trusted_count >= 3
    news = title + ' ' + article
    news_vector = vectorizer.transform([news])
    start = time.time()
    prediction = model.predict(news_vector)
    probablity = model.predict_proba(news_vector)
    real_probability = round(probablity[0][1] * 100, 2)
    fake_probability = round(probablity[0][0] * 100, 2)
    confidence = round(max(probablity[0]) * 100, 2)
    google_count = len(google_articles)
    google_score = min(google_count * 3, 30)
    ml_score = 60 if prediction[0] == 1 else 0
    newsapi_score = min(trusted_count * 5, 10)
    hybrid_score = google_score+ml_score+newsapi_score
    final_verdict = "NEEDS MANUAL VERIFICATION"
    if google_count >= 8:
        result = "VERIFIED REAL NEWS"
        final_verdict = "✅ Verified by Hybrid AI"
        card_class = "real"
    elif hybrid_score >= 80:
        result = "VERIFIED REAL NEWS"
        final_verdict = "✅ Verified by Hybrid AI"
        card_class = "real"
    elif hybrid_score >= 60:
        result = "LIKELY REAL NEWS"
        final_verdict = "🟢 Supported by Trusted Sources"
        card_class = "real"
    elif hybrid_score >= 40:
        result = "NEEDS MANUAL VERIFICATION"
        final_verdict = "🟡 ML and News Sources Disagree"
        card_class = "warning"
    else:
        result = "LIKELY FAKE NEWS"
        final_verdict = "🔴 No Trusted Verification"
        card_class = "fake"
    end = time.time()
    predict_time = round(end - start, 3)
    return render_template('result.html', prediction=result, confidence=confidence,
                           real_probability=real_probability, fake_probability=fake_probability,
                           card_class=card_class,title=title,article=article,predict_time=predict_time,
                           source_name=source_name,article_url=article_url,published_at=published_at,
                           trusted=trusted,final_verdict=final_verdict,trusted_count=trusted_count,
                           total_articles=trusted_count+len(google_articles),google_count=len(google_articles),
                           hybrid_score=hybrid_score)

if __name__ == '__main__':
    app.run(debug=True)

