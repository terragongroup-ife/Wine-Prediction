from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score


#parameters = {"n_estimators": [10, 20, 30], 'max_features': [3,4,5,None], 'max_depth': [5,6,7,None]}
#scorer = make_scorer(fbeta_score, beta=0.5, average= "micro")




class the_model(object):
    
    def __init__(self):
        self.clf = RandomForestClassifier(max_depth=None, random_state=None)
        #self.grid_obj = GridSearchCV(clf, parameters, scoring=scorer)


    def predict_proba(self, X):
        y_pred = self.clf.predict(X)
        return y_pred

   