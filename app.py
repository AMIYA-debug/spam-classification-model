from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load BoW model + vectorizer
bow_model = pickle.load(open("bow_model_1.pkl", "rb"))
bow_vectorizer = pickle.load(open("bow_vectorizer_1.pkl", "rb"))

# Load TF-IDF model + vectorizer
tfidf_model = pickle.load(open("tfidf_model_2.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer_2.pkl", "rb"))

# Example accuracies from training (update with real scores)
bow_accuracy = 97.8
tfidf_accuracy = 98.2

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        # --- BoW Prediction ---
        bow_vector = bow_vectorizer.transform([message])
        bow_pred = bow_model.predict(bow_vector)[0]  # 0 or 1
        bow_label = "Not Spam" if bow_pred == 1 else "Spam"

        # --- TF-IDF Prediction ---
        tfidf_vector = tfidf_vectorizer.transform([message])
        tfidf_pred = tfidf_model.predict(tfidf_vector)[0]  # 0 or 1
        tfidf_label = "Not Spam" if tfidf_pred == 1 else "Spam"

        # Decide which model is better (based on accuracy)
        better_model = "TF-IDF" if tfidf_accuracy > bow_accuracy else "BoW"

        return render_template("index.html",
                               message=message,
                               bow_result=bow_label,
                               tfidf_result=tfidf_label,
                               bow_accuracy=bow_accuracy,
                               tfidf_accuracy=tfidf_accuracy,
                               better_model=better_model)

if __name__ == "__main__":
    app.run(debug=True)
