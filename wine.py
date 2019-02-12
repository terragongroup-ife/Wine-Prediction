import json
from flask import Flask, abort,request, render_template,jsonify
from flask_restful import reqparse, abort, Api, Resource
import numpy
from sklearn.externals import joblib
from model import the_model
import traceback
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)
#tmodel= the_model()

lr = joblib.load('model.pkl')
class Base(Resource):
    def get(self):
        return jsonify({'message': 'home'})

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Winequality(Resource):
    def post(self):
        file=request.json
        query_df=pd.DataFrame(file)
        query = pd.get_dummies(query_df)
        total=lr.predict(query)
        return json.dumps({'prediction': total}, cls=NumpyEncoder)

api.add_resource(Winequality, '/api')    
api.add_resource(Base, '/')


if __name__ == '__main__':
    app.run(debug=True)