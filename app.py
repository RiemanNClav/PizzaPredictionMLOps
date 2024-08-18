
from flask import Flask,request,render_template
import numpy as np
import pandas as pd


from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            diameter=float(request.form.get('diameter')),
            company=request.form.get('company'),
            topping=request.form.get('topping'),
            variant=request.form.get('variant'),
            size=request.form.get('size'),
            extra_sauce=request.form.get('extra_sauce'),
            extra_cheese=request.form.get('extra_cheese'),
            extra_mushrooms=request.form.get('extra_mushrooms')


        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")