from flask import Flask, render_template, request
from flask_predict import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')  # 改为 index.html，不是 index1.html


@app.route('/ask', methods=['POST'])
def ask():
    # 从表单获取 user_input（和HTML一致）
    user_input = request.form['user_input']  # HTML里用的是 user_input

    # 使用 GPT-2 模型进行问答处理
    response = model_predict(user_input)

    # 渲染时用 index.html
    return render_template('index.html', user_input=user_input, answer=response)


if __name__ == '__main__':
    app.run(debug=True)