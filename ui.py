from flask import Flask, render_template, request
import query_data_url
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer', methods=['POST'])
def answer():
    question = request.form['question']

    response_text, references, compare_text, response_url = query_data_url.query_rag(question)

    return render_template('index.html', answer=response_text, references=references, compare=compare_text, urlanswer=response_url, question=question)  # Replace with actual references


if __name__ == '__main__':
    app.run()

