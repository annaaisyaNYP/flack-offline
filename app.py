from flask import Flask, render_template, request, url_for, jsonify
import chatbot, form

app = Flask(__name__, static_folder='D:/flack-offline/Static')

# Frontend ##############################################################################################
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/SHSS')
def SHSS():
    return render_template('SHSS.html')

# Image Detection #######################################################################################

@app.route('/SIT', methods=["GET","POST"])
def SIT():
    img = form.Image(request.form)

    if request.method == 'POST' and input.validate():
        return render_template('test.html')

    return render_template('SIT.html', form=img)

# Chatbot App ###########################################################################################
textList = []
botList = []
count = 0

@app.route('/FAQ', methods=["GET","POST"])
def FAQ():
    input = form.Input(request.form)
    global count

    if request.method == 'POST' and input.validate():
        text = input.text.data
        response = chatbot.result(text)
        text = "You: " + text
        textList.extend([text])
        botList.extend([response])
        count += 1
        return render_template('FAQ.html', botList=botList, textList=textList, count=count , form=input)

    return render_template('FAQ.html', form=input, botList=botList, textList=textList, count=count )

@app.route('/clearChat')
def clearChat():
    input = form.Input(request.form)
    global count
    count = 0
    textList = []
    botList = []
    return render_template('FAQ.html', form=input, botList=botList, textList=textList, count=count )

@app.route("/chatbot", methods=["GET","POST"])
def chatAPI():

    if request.method == 'POST':
        input = request.get_json(force=True)
        print(input)

        response = chatbot.result(input)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
