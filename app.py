from flask import Flask, render_template, redirect, url_for, request
from repo_ai_analyzer import main_handler, generate_response
app = Flask(__name__)
question_context = None
@app.route('/', methods=['POST', 'GET'])
def index():
	if request.method == 'GET':
		return render_template("index.html")
			

@app.route('/user_input/', methods=['GET','POST'])
def user_input():
	if request.method == 'POST':
		url = request.form.get('url')
		global question_context
		question_context = main_handler(url)
		return render_template('user_input.html')

@app.route('/answer/',methods=['GET','POST'])
def generate_answer():
	if request.method == 'POST':
		question = request.form.get('question')
		global question_context
		answer = generate_response(question, question_context)
		return render_template('answer.html', answer=answer)

if __name__ == '__main__':
	app.run(debug=True)
