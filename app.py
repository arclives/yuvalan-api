from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("yuvalan_model.pkl")
vectorizer = joblib.load("yuvalan_vectorizer.pkl")

@app.route('/tahmin', methods=['POST'])
def tahmin():
    veri = request.json
    semptomlar = veri["semptomlar"]
    vektor = vectorizer.transform([semptomlar])
    tahmin = model.predict(vektor)
    return jsonify({"hastalik": tahmin[0]})

if __name__ == '__main__':
    app.run(debug=True)