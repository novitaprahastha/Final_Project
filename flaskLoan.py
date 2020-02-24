from flask import request, redirect, Flask, send_from_directory, abort, jsonify, render_template
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sb
# from sklearn.linear_model import LogisticRegression
import io
import base64


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('loanpredict.html')

@app.route('/predict', methods = ['GET','POST'])
def result():
    if request.method=='GET':
        return render_template('loanpredict.html')
    elif request.method == 'POST':
        Name = str(request.form['Name'])
        Credit = int(request.form['Credit_History'])
        Income = int(request.form['Income'])
        Loan_Amount = int(request.form['Loan_Amount'])
        prediksi = model.predict([[Credit, Income, Loan_Amount]])[0]

        if prediksi == 1:
            prediksi = "Loan Status : Approved!"
        else:
            prediksi = "Loan Status: Dissaproved"

        dataHasil = {
            'Name' : Name,
            'Credit_History': Credit,
            'Income':Income,
            'Loan_Amount':Loan_Amount,
            'Prediction':prediksi
        }
        probaloan = model.predict_proba([[Credit,Income,Loan_Amount]])
        probaloan = probaloan[0]
        labels = ['Dissaproved', 'Approved']

        plt.close()
        plt.figure(figsize=(5,5))
        plt.title('Loan Approval Probability')
        plt.pie(x=probaloan, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.3)
        plt.legend(labels)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        graph = 'data:image/png;base64,{}'.format(graph_url)

        return render_template('result.html', Hasil= dataHasil, graph=graph)
    else:
        abort(404)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=='__main__':
    model = joblib.load('modelJoblib')
    app.run(debug=True)

