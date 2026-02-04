from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

history = []   # store last predictions


@app.route("/")
def home():
    return render_template("index.html", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    msg_vec = vectorizer.transform([message])

    prediction = model.predict(msg_vec)[0]
    prob = model.predict_proba(msg_vec)[0].max() * 100   # confidence %

    result = {
        "text": message,
        "label": prediction,
        "confidence": round(prob, 2)
    }

    history.insert(0, result)

    if len(history) > 5:   # keep last 5 only
        history.pop()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=result["confidence"],
        history=history
    )


if __name__ == "__main__":
    app.run(debug=True)
