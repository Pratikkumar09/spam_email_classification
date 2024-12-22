from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/',methods=["GET"])
def home():
    return render_template('index.html', email_content='', result='')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get email content from the form
        email_text = request.form['email']

        # Vectorize the email content
        email_vectorized = vectorizer.transform([email_text])

        # Predict the label (0 for ham, 1 for spam)
        prediction = model.predict(email_vectorized)

        # Map prediction to string ("spam" or "ham")
        result = 'spam' if prediction[0] == 1 else 'not spam'

        # Return the result and email content to the template
        return render_template('index.html', email_content=email_text, result=result)

    except Exception as e:
        return render_template('index.html', email_content=email_text, result=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
