from flask import Flask, render_template, request, jsonify
import RetreivalAugmentedGeneration
import ModelInference

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chatbot.html')


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')
    model_no = int(data.get('model_no'))

    if model_no == 1:
        answer = ModelInference.llm_chain.invoke(input=f"{user_query}")
        answer = answer['text']
    else:
        answer = RetreivalAugmentedGeneration.LLM_Run(str(user_query))

    return jsonify({'response': answer})


if __name__ == "__main__":
    app.run(debug=True)
