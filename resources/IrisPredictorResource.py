import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import json
import numpy as np
import falcon



def predict_knn(features, model):

    """
    This function gets the features and models and predicts the output
    Args:
        features(list): list of features. It must include 4 floating numbers
        model(sklearn model): knn sklearn model, loaded from models folder
    output:    
        prediceted_class(str)
    """
    
    classes = ['setosa', 'versicolor', 'virginica']

    predicted_class = classes[model.predict(features)]
    
    return predicted_class

### resource
class IrisPredictorResource():
    """
    Class to predict the iris class on POST
    """
    def __init__(self, model_path, logger):
        """
        Constructor of the predictor resourse object. 
        Initializes logging and load the predictor model files
        @params:
            model_path: math to model files
            logger : logger associated with the class
        """
        self.logger = logger
        self.model = pickle.load(open(model_path, 'rb'))
        self.logger.info("Starting: IrisPredictor")
    
    def on_post(self, req, resp):
        """
        Function to be called on POST
        @params:
            req : request json sent of POST
            resp: response json to be filled and sent back after POST
        """
        try:
            self.logger.info("IrisPredictor: reading file")
            request_bytes = req.stream.read()

            try:
                request = json.loads(request_bytes.decode("utf-8"))
            
            except Exception as e:
                self.logger.error(e, exc_info=True)
                resp.status = falcon.HTTP_400
                resp.body = "Invalid JSON\n"
                return
            
            # Check for the validity of json document
            if 'features' not in request.keys():
                resp.status = falcon.HTTP_400
                resp.body = "Invalid input\n"
                return
            if not isinstance(request['features'], list):
                resp.status = falcon.HTTP_400
                resp.body = "Invalid input\n"
                return
            if  not len(request['features']) == 4:
                resp.status = falcon.HTTP_400
                resp.body = "Invalid input\n"
                return
            if not all(isinstance(request['features'], float):
                resp.status = falcon.HTTP_400
                resp.body = "Invalid input\n"
                return
                
            features = request['features']
            
            ## In this part, you consider the input is correct and 
            ## just need to return the result
            prediction = predict_knn(features, self.model)

            self.logger.info('IrisPredictor: the prediction is %s' % prediction)
            response = {"predicted_class": prediction}

            self.logger.info('IrisPredictor: Sending the results \n')
            
            resp.status = falcon.HTTP_200
            resp.body = json.dumps(response) + '\n'

        except Exception as e:
            self.logger.error(e, exc_info=True)
            resp.status = falcon.HTTP_500
