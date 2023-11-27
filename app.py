from flask import Flask, request, render_template
import pickle
import dataprocessing as dp


app = Flask(__name__)
model = pickle.load(open('model.pkl', "rb"))
vector = pickle.load(open('vectorizer.pkl', "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    trans_txt = dp.transform_text(features[0])
    trans_txt = [trans_txt]
    vect_text = vector.transform(trans_txt)
    prediction = model.predict(vect_text)
    if prediction == 0:
        msg = 'ham'
    else:
        msg = 'spam'
    return render_template('index.html', prediction_text = 'Hi, this is a {} message'.format(msg))


if __name__ == '__main__':
    app.run(debug=True)