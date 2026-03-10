# app.py
from flask import Flask, render_template, request, jsonify
from flask_predict import model_predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


# 原有的表单提交端点（可选保留）
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = model_predict(user_input)
    return render_template('index.html', user_input=user_input, answer=response)


# 新增的 API 端点供 Vue 调用
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')

    if not user_input:
        return jsonify({'error': '请输入问题'}), 400

    response = model_predict(user_input)
    return jsonify({
        'user_input': user_input,
        'answer': response
    })


if __name__ == '__main__':
    app.run(debug=True)