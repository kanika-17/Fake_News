from flask import Flask,render_template,url_for,request
import pickle
import joblib

filename = 'pickle.pkl'
clf = joblib.load(open(filename, 'rb'))
cv=joblib.load(open('transform.pkl','rb'))
app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():   
	return render_template('homee.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('resultt.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True,port=8000)