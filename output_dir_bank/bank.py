import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction import DictVectorizer
import json
import os

class BankModel:

    
    def datapalio_interface(self, **kwargs):
        
        """
        This is the method used by DataPal.io to interact with the model.
        
        Inputs:
                
        - pipe_id (integer): id of the pipe that has to be used.
        
        - input_data (dictionary): dictionary that contains the input data. The keys of the dictionary 
        correspond to the names of the inputs specified in models_definition.json for the selected pipe.
        Each key has an associated value. For the input variables the associated value is the value
        of the variable, whereas for the input files the associated value is its filename. 
        
        - input_files_dir (string): Relative path of the directory where the input files are stored
        (the algorithm has to read the input files from there).

        - output_files_dir (string): Relative path of the directory where the output files must be stored
        (the algorithm must store the output files in there).
        
        Outputs:
        
        - output_data (dictionary): dictionary that contains the output data. The keys of the dictionary 
        correspond to the names of the outputs specified in models_definition.json for the selected pipe. 
        Each key has an associated value. For the output variables the associated value is the value
        of the variable, whereas for the output files the associated value is its filename.  

        """
            
        pipe_id = kwargs['pipe_id']
        input_data = kwargs['input_data']
        input_files_dir = kwargs['input_files_dir']
        output_files_dir = kwargs['output_files_dir']
        
        output_data = self.train_or_predict(pipe_id, input_data, input_files_dir, output_files_dir)
        
        return output_data
        

    def train_or_predict(self, pipe_id, input_data, input_files_dir, output_files_dir):
        
        """
        Handles user requests.
        
        """
            
        if pipe_id in [0,1]:
            # load data
            # if one by one predictions
            if pipe_id == 0:

                # put all the input variables into a list to be able to easly construct the DataFrame
                for key in input_data.keys():
                    input_data[key] = [input_data[key]]

                # transform the input data to a DataFrame
                input_data_df = pd.DataFrame.from_dict(input_data)

                # encode input data
                one_hot_encoded_df, vecData = self.one_hot_dataframe_given_dv(input_data_df, self.categorical_features, self.featureEncoder, replace=True) 
                data = one_hot_encoded_df[self.post_processing_feature_names].values

            elif pipe_id == 1:
                # if bulk predictions
                data, data_args = self.load_input_files(input_files_dir=input_files_dir, input_data=input_data, training_data=False)
                data = data["features"]

            # make prediction
            prediction, out_args = self.predict(features=data)

            # decode prediction
            prediction = self.inverse_binarize_array(prediction)

            # return answer
            if pipe_id == 0:
                output_data = {}
                output_data[self.target_name] = prediction[0]
                return output_data

            elif pipe_id == 1:
                # save output in csv file and return it
                filename = 'predictions.csv'
                filepath = os.path.join(output_files_dir, filename)
                df = data_args['original_data_df']
                df[self.target_name] = prediction
                df.to_csv(filepath, index=False)
                output_data = {}
                output_data["file with predictions"] = filename
                return output_data
            else:
                return
        
        if pipe_id == 2:

            # load data
            data, data_args = self.load_input_files(input_files_dir=input_files_dir, input_data=input_data, training_data=True)

            # find best hyperparameters and fit model
            predictor, out_args = self.find_best_parameters_and_get_fitted_model(data=data, set_predictor_after_training=True)
            # set fitted predictor as the default for the DPModel
            self.predictor = predictor

            # save the label encoder
            self.featureEncoder = data_args['featureEncoder']

            # get unbiased predictions on training data
            y_pred, unbiased_prediction_args = self.get_unbiased_predictions_on_training_data(data=data)

            output_data = {}

            # save model_configuration.json
            model_definition, out_args = self.get_model_definition(data_args=data_args)
            path = os.path.join(output_files_dir, "model_definition.json")
            self.save_json_file(dict_to_save=model_definition, path=path)
            output_data["model_definition"] = 'model_definition.json'

            # save scores.json
            scores, out_args = self.get_scores(data=data, predicted_values=y_pred, data_args=data_args)
            path = os.path.join(output_files_dir, "scores.json")
            self.save_json_file(dict_to_save=scores, path=path)
            output_data["scores"] = 'scores.json'

            return output_data

        else:
            return


    def load_input_files(self, **kwargs):
        
        """
        Loads both files containing training data and data for prediction. 
        
        Encodes the target labels to integers. 
        
        In case the it is training data, it will return in the output args the 
        LabelEncoder used to encode the target labels to integers. We return it 
        instead of directly storing it, because it will be saved in case the training
        ends without errors.
        
        """
        
        input_data = kwargs['input_data']
        input_files_dir = kwargs['input_files_dir']
        
        input_file_path = input_files_dir + input_data['database']
        df = pd.read_csv(input_file_path, sep=";")
        
        training_data = kwargs.pop('training_data', False)
        
        # if we are loading training data, we have to assign an integer to each possible
        # categorical variable in the dataset. We do it by fitting a LabelEncoder for each
        # one of them.
        if training_data:
            out_args = {}
            out_args['original_data_df'] = df
            
            target_name = "success"
            feature_names = list(df.columns)
            
            feature_names.remove(target_name)
            self.feature_names = feature_names
            self.target_name = target_name
            
            self.categorical_features = ['job',
                         'marital',
                         'education',
                         'default',
                         'housing',
                         'loan',
                         'contact',
                         'previous outcome']
            
            # make one hot encoding in features
            one_hot_encoded_df, vecData, vec = self.one_hot_dataframe(df, self.categorical_features, replace=True)
            
            # encode target variable to integers
            df[self.target_name] = self.binarize_array(np.array(df[self.target_name]), "yes", "no")
               
            # return encoded features and targets and both encoders
            post_processing_feature_names = list(one_hot_encoded_df.columns)
            post_processing_feature_names.remove(self.target_name)
            self.post_processing_feature_names = post_processing_feature_names
            
            data = {}
            data['features'] = one_hot_encoded_df[post_processing_feature_names].values
            data['targets'] = np.array(df[self.target_name])

            out_args['featureEncoder'] = vec
            
            return data, out_args

        # if the data is for making predictions, we have to transform the categorical
        # features to integers by using the stored LabelEncoder.
        else:

            one_hot_encoded_df, vecData = self.one_hot_dataframe_given_dv(df, self.categorical_features, self.featureEncoder, replace=True) 
            data = {}
            # ensure that the columns are in the correct order
            data['features'] = one_hot_encoded_df[self.post_processing_feature_names].values
            
            out_args = {}
            out_args['original_data_df'] = df
        
            return data, out_args
    
    
    def find_best_parameters_and_get_fitted_model(self, **kwargs):
        
        """
        Finds the best set of hyperparameters for a Random Forest for the provided data. 
        The best hyperparameters are found by repeatedly drawing random samples from a distribution 
        of parameters and evaluating them by using cross validation.        
        
        """
        
        # load data
        data = kwargs['data']
        X = data['features']
        y = data['targets']
        out_args = {}
        
        # we choose Random Fores Classifier as the Machine Learning algorithm for
        # this DPModel.
        rc = RandomForestClassifier()
        
        # here we define the space of parameters over which we want to perform the random search
        param_distributions = {}
        param_distributions["n_estimators"] = [50, 100, 150]

        # do random search
        random_search_outer = RandomizedSearchCV(rc, param_distributions=param_distributions,
            cv=5, n_iter=3)
        random_search_outer.fit(X, y)
            
        predictor = random_search_outer.best_estimator_

        return predictor, out_args
        

    def predict(self, **kwargs):
        
        """
        Makes predictions using the stored predictor of the DPModel.
        
        """
    
        features = kwargs['features']
        predictor = kwargs.pop('predictor', self.predictor)
        
        X = features
        prediction = predictor.predict(X)
        
        out_args = {}
        
        return prediction, out_args


    def get_unbiased_predictions_on_training_data(self, **kwargs):
        
        """
        This method provides unbiased predictions for all our training samples.
        We accomplish that by performing a nested cross validation:
        We leave a hold out set out, and we past the rest of the data to the 
        find_best_parameters_and_get_fitted_model method, which contains a cross validation itself. 
        Then we make predictions on the hold out set with the resulted predictor. This way, we found
        the best hyperparameters without using the hold out data. We repeat this process leaving out 
        different training samples each time by performing a cross validation.
        
        """
        
        data = kwargs['data']
        
        y_true = None
        y_pred = None
        out_args = {}
        
        X = np.array(data['features'])
        y = np.array(data['targets'])
        out_args = {}
        
        # make unbiased predictions using nested CV
        # We will use this unbiased predictions in order to calculate the performance of the
        # algorithm using multiple scores.
        cv = StratifiedKFold(y, n_folds=5)
        for i, (train, test) in enumerate(cv):
            
            data_fold = {}
            data_fold['features'] = X[train]
            data_fold['targets'] = y[train]
                        
            predictor, out_args = self.find_best_parameters_and_get_fitted_model(data=data_fold, set_predictor_after_training=False)
            y_test_pred, out_args = self.predict(predictor=predictor, features=X[test])
            
            if y_true == None:
                y_true = y[test]
                y_pred = y_test_pred
            else:
                y_true = np.hstack((y_true, y[test]))
                y_pred = np.hstack((y_pred, y_test_pred))

        return y_pred, out_args
    
    
    def get_model_definition(self, **kwargs):
        
        """
        Returns model_definition.json dictionary.
        
        """

        model_definition = {}
        model_definition["name"] = "Sales predictor"
        model_definition["schema_version"] = "0.02"
        model_definition["environment_name"] = "python2.7.9_June14th2015"
        model_definition["description"] = "Based on historical sales data, this predictor " \
                                          "predicts whether a new potential customer will " \
                                          "buy the product or not."
        model_definition["retraining_allowed"] = True
        model_definition["base_algorithm"] = "Random Forest Classifier"     
        model_definition["score_minimized"] = "gini"        

        pipes, out_args = self.get_pipes(**kwargs)
        model_definition["pipes"] = pipes
        
        out_args = {}
        
        return model_definition, out_args
    
    
    def get_pipes(self, **kwargs):
        
        """
        Returns pipes dictionary.
        
        """
        
        df = kwargs['data_args']['original_data_df']
        
        # One by one prediction pipe

        dicts_predicting_inputs = []
        for col in self.feature_names:
            d = {}
            d["name"] = col
            d["type"] = "variable"
            d["required"] = True
            if df[col].dtype == "O":
                d["values"] = list(set(df[col]))
                d["variable_type"] = "string"
            elif df[col].dtype == "int":
                d["variable_type"] = "integer"
            elif df[col].dtype == "float":
                d["variable_type"] = "float"

            dicts_predicting_inputs.append(d)

        dicts_predicting_outputs = []
        for col in [self.target_name]:
            d = {}
            d["name"] = col
            d["type"] = "variable"
            d["required"] = True
            if df[col].dtype == "O":
                d["values"] = list(set(df[col]))
                d["variable_type"] = "string"
            elif df[col].dtype == "int":
                d["variable_type"] = "integer"
            elif df[col].dtype == "float":
                d["variable_type"] = "float"

            dicts_predicting_outputs.append(d)

        pipes = []

        pipe = {}
        pipe["id"] = 0
        pipe["action"] = "predict"
        pipe["name"] = "One by one prediction"
        pipe["description"] = "Make predictions one by one."
        pipe["inputs"] = dicts_predicting_inputs
        pipe["outputs"] = dicts_predicting_outputs
        pipes.append(pipe)


        # Bulk prediction pipe

        pipe = {
            "id": 1,
            "action": "predict",
            "name":"Bulk prediction",
            "description": "Upload csv file.",
            "inputs": [
                {
                    "name": "database",
                    "type": "file",
                    "extensions": ["csv"],
                    "required": True
                }
            ],
            "outputs": [
                {
                    "name": "file with predictions",
                    "type": "file",
                    "extensions": ["csv"]
                }
            ]
        }

        pipes.append(pipe)


        # Training pipe

        pipe = {
            "id": 2,
            "action": "train",
            "name":"Training pipe",
            "description": "Upload database with target labels.",
            "inputs": [
                {
                    "name": "database",
                    "type": "file",
                    "extensions": ["csv"],
                    "required": True
                }
            ],
            "outputs": [
                {
                    "name": "model_definition",
                    "type": "file",
                    "filenames": ["model_definition.json"]
                },
                {
                    "name": "scores",
                    "type": "file",
                    "filenames": ["scores.json"]
                }
            ]
        }
            
        pipes.append(pipe)
        out_args = {}
        
        return pipes, out_args

    
    def get_scores(self, **kwargs):
        
        """
        Calculate scores.
        
        """
        
        data = kwargs['data']
        true_values = np.array(data['targets'])
        predicted_values = kwargs['predicted_values']

        out_args = {}
        scores = []

        sc = accuracy_score (true_values, predicted_values)
        score = {}
        score['name'] = 'Accuracy'
        score['value'] = sc
        scores.append(score)        
        
        sc = f1_score(true_values, predicted_values)
        score = {}
        score['name'] = 'F1 score'
        score['value'] = sc
        scores.append(score)
        
        sc = precision_score(true_values, predicted_values)
        score = {}
        score['name'] = 'Precision'
        score['value'] = sc
        scores.append(score)
        
        sc = recall_score(true_values, predicted_values)
        score = {}
        score['name'] = 'Recall'
        score['value'] = sc
        scores.append(score)
        
        scores_out = {}
        scores_out["scores"] = scores
        scores_out["schema_version"] = "0.02"
        
        return scores_out, out_args
    
    
    def save_json_file(self, **kwargs):
        

        """
        Saves dictionary in path.
        
        """
        
        dict_to_save = kwargs["dict_to_save"]
        path = kwargs["path"]
        with open(path,'wb') as fp:
            json.dump(dict_to_save, fp)
            
        return
    

    def one_hot_dataframe(self, data, cols, replace=False):
        """ 
        Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.

        """
        vec = DictVectorizer()
        mkdict = lambda row: dict((col, row[col]) for col in cols)
        vecData = pd.DataFrame(vec.fit_transform(data[cols].to_dict(outtype='records')).toarray())
        vecData.columns = vec.get_feature_names()
        vecData.index = data.index
        if replace is True:
            data = data.drop(cols, axis=1)
            data = data.join(vecData)
        return (data, vecData, vec)


    def one_hot_dataframe_given_dv(self, data, cols, dv, replace=False):
        """ 
        Takes a dataframe, a list of columns and a DictVectorizer and
        encodes those columns according to the info in the DictVectorizer.
        Returns a 2-tuple comprising the data, and the vectorized data.

        """
        vecData = pd.DataFrame(dv.transform(data[cols].to_dict(outtype='records')).toarray())
        vecData.columns = dv.get_feature_names()
        vecData.index = data.index
        if replace is True:
            data = data.drop(cols, axis=1)
            data = data.join(vecData)
        return (data, vecData)
    
    
    def binarize_array(self, array_to_transform, positive_label, negative_label):
        """
        Converts the given column to 0s and 1s.
        Specially when the converted column is the target one, it is important
        to controll which label becomes 0 and which one becomes 1, in order to
        provide the correct values of precision, recall, specificity, and so on.
        
        """
        
        array_to_transform[array_to_transform == positive_label] = 1
        array_to_transform[array_to_transform == negative_label] = 0
        array_to_transform = array_to_transform.astype(int)
        self.positive_label = positive_label
        self.negative_label = negative_label
        
        return array_to_transform
    
    
    def inverse_binarize_array(self, array_to_transform):
        """
        Converts the given column to 0s and 1s.
        Specially when the converted column is the target one, it is important
        to controll which label becomes 0 and which one becomes 1, in order to
        provide the correct values of precision, recall, specificity, and so on.
        
        """
        
        # important to do the comparisson before changing the data type of the array.
        pos_with_1 = (array_to_transform == 1)
        pos_with_0 = (array_to_transform == 0)
        array_to_transform = array_to_transform.astype(str)
        array_to_transform[pos_with_1] = self.positive_label
        array_to_transform[pos_with_0] = self.negative_label
        
        return array_to_transform