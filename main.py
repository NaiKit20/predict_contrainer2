from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

lr = pickle.load(open('LogisticRegression_model.pkl','rb'))
X_train = pickle.load(open('X_train.pkl','rb'))

cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
cvec.fit_transform(X_train)
cvec.vocabulary_

app = Flask(__name__)

# CORS Middleware
# CORS(app, supports_credentials=True)

# Routes
@app.route("/")
def index():
    return jsonify({"message": "Hello world"})

@app.route('/api/predict', methods=['POST'])
def predict():
    text = request.get_json()['text']
    my_bow = cvec.transform(pd.Series([text]))
    my_predictions = lr.predict(my_bow)                 

    # แมปคลาสตัวเลขเป็น string
    if my_predictions == "neg":
        result_string = "ลบ"
    else:
        result_string = "บวก"
    return jsonify({"predict":result_string})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)