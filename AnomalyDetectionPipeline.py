# Databricks notebook source
!pip install torch
!pip install deap


# COMMAND ----------

# Imports
import pandas as pd  
import pyspark
import pickle
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

from sklearn.ensemble import IsolationForest  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from sklearn.tree import plot_tree, _tree
import matplotlib.pyplot as plt

import random
from deap import base, creator, tools, algorithms  
import functools  
  

# COMMAND ----------

# Standard Variables
TABLE_LOCATION = 'hackathon.were_just_here_for_the_snacks.synthetic_features_data_small_lowfraud'
MODEL_FEATURES = [
       'TX_FRAUD_SCENARIO', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS',
       'TX_FRAUD_SCENARIO', 'TX_DURING_WEEKEND',
       'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']
TARGET = 'TX_FRAUD'
CUSTOMER_ID_COL = 'CUSTOMER_ID'

AUTO_ENCODER_FEATURES=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
       'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
       'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
       'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
       'TERMINAL_ID_RISK_30DAY_WINDOW']

EXISTING_RULE_SET = [
    "(transactionChannel == 'CNP') and (TX_AMOUNT > 200)",
    "(transactionChannel == 'CNP') and (TX_AMOUNT > 500)",
    "(transactionChannel == 'CNP') and (TX_AMOUNT > 500) and ((merchantName == 'Green Shop') or (merchantName == 'Blue Corner'))",
    "(transactionChannel == 'CP') and (TX_AMOUNT > 500) and (merchantCategory == 'Department Stores')",
    "(transactionChannel == 'CNP') and (TX_AMOUNT > 500) and (CUSTOMER_ID_NB_TX_7DAY_WINDOW > 5)"
]


# COMMAND ----------

def read_data(dataset_name):
    feature_df = spark.read.table(dataset_name)
    return feature_df.toPandas()

def custom_train_test_split(data, customer_id_col, test_size=0.2, random_state=42):  
    # Ensure same customers are not in both train and test sets  
    unique_customers = data[customer_id_col].unique()  
    train_customers, test_customers = train_test_split(unique_customers, test_size=test_size, random_state=random_state)  
      
    train_data = data[data[customer_id_col].isin(train_customers)]  
    test_data = data[data[customer_id_col].isin(test_customers)]  
      
    return train_data, test_data

def train_logistic_model(train_data, features, target):
    """
    Train a logistic regression model.
    
    Parameters:
    - train_data: pandas DataFrame, the training data
    - features: list of str, the feature column names
    - target: str, the target column name
    
    Returns:
    - model: trained logistic regression model
    """
    X_train = train_data[features]
    y_train = train_data[target]
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    return model

def get_standard_metrics(y_test, y_pred):
    # Evaluate  
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return precision, recall, f1, accuracy

def evaluate_logistic_model_performance(model, test_data, features, target, threshold=0.5):
    """
    Evaluate the model's performance in terms of precision and recall.
    
    Parameters:
    - model: trained logistic regression model
    - test_data: pandas DataFrame, the test data
    - features: list of str, the feature column names
    - target: str, the target column name
    - threshold: float, the decision threshold for classification
    
    Returns:
    - precision: float, the precision score
    - recall: float, the recall score
    - f1: float, the f1 score
    - accuracy: float, the accuracy score
    """
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Get the predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Apply the threshold to get the predicted classes
    y_pred = (y_prob >= threshold).astype(int)
    
    return get_standard_metrics(y_test, y_pred)


def find_threshold_for_recall(model, val_df, features, target, desired_recall=0.2, step=0.01):
    """
    Find the threshold that provides the desired recall.
    
    Parameters:
    - model: trained logistic regression model
    - val_df: pandas DataFrame, the validation data
    - features: list of str, the feature column names
    - target: str, the target column name
    - desired_recall: float, the desired recall value
    - step: float, the step size for threshold increment
    
    Returns:
    - best_threshold: float, the threshold that provides the desired recall
    - precision: float, the precision score at the best threshold
    - recall: float, the recall score at the best threshold
    """
    best_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    
    for threshold in [1 - i * step for i in range(int(1/step) + 1)]:
        precision, recall, f1, accuracy = evaluate_logistic_model_performance(
            model, val_df, features, target, threshold=threshold
        )
        if recall >= desired_recall:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f1 = f1_score
            best_accuracy = accuracy
            break
    
    return best_threshold, best_precision, best_recall, best_f1, best_accuracy


def score_isolation_forest(data_set, features, model):  
    # Predict on the test set  
    data_set['scores'] = model.decision_function(data_set[features])  
    data_set['predictions'] = model.predict(data_set[features])  
    data_set['predictions'] = data_set['predictions'].map({1: 0, -1: 1})  # Convert to 0 (normal) and (anomalies)  
    return data_set


def train_and_evaluate_isolation_forest(data, features, target, customer_id_col, contamination = 0.01, num_est=10):  
    # Split the data  
    train_data, test_data = custom_train_test_split(data, customer_id_col)
      
    # Train Isolation Forest  
    model = IsolationForest(random_state=42, contamination=contamination, n_estimators=num_est)  
    model.fit(train_data[features])  
    
    # Predict on the test set  
    train_data = score_isolation_forest(train_data, features, model)
    test_data = score_isolation_forest(test_data, features, model)
      
    # Evaluate
    precision, recall, f1, accuracy = get_standard_metrics(test_data[target], test_data['predictions'])
    print(f"Precision: {precision:.2f}")  
    print(f"Recall: {recall:.2f}")  
    print(f"F1 Score: {f1:.2f}")  
    print(f"Accuracy: {accuracy:.2f}")  
    
    return model

def get_anomaly_tags(data_set, features, model):  
    
    data_set['anomaly_scores'] = model.decision_function(data_set[features])  
    data_set['anomaly_tag'] = model.predict(data_set[features])  
    data_set['anomaly_tag'] = data_set['anomaly_tag'].map({1: 0, -1: 1})  # Convert to 0 (normal) and (anomalies)  
    return data_set

def save_artefacts(artefact_to_save, output_file_name):
    """
    Saves artefacts, data in parquet and trained model as pickle
    
    Parameters:
    - artefact_to_save: artefact to be saved
    - output_file_name: str, name of the output file

    Returns:
    - None
    """
    if isinstance(artefact_to_save, pd.DataFrame):
        spark.createDataFrame(artefact_to_save).write \
            .mode("overwrite") \
            .saveAsTable(f"hackathon.were_just_here_for_the_snacks.{output_file_name}")
    elif isinstance(artefact_to_save, pyspark.sql.DataFrame):
        artefact_to_save.write \
            .mode("overwrite") \
            .saveAsTable(f"hackathon.were_just_here_for_the_snacks.{output_file_name}")
    else:
        with open(f"/Workspace/Shared/snacks_home/artefacts/{output_file_name}", 'wb') as file:
            pickle.dump(artefact_to_save, file)
    return None


# COMMAND ----------

class FraudDatasetUnsupervised(torch.utils.data.Dataset):
    
    def __init__(self, x,output=True, device="cpu"):
        'Initialization'
        self.x = x
        self.output = output
        self.device = device

    def __len__(self):
        'Returns the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample index
        item = self.x[index].to(self.device)
        if self.output:
            return item, item
        else:
            return item
        

class Autoencoder(torch.nn.Module):
    
        def __init__(self, input_size, intermediate_size, code_size):
            super(Autoencoder, self).__init__()
            # parameters
            self.input_size = input_size
            self.intermediate_size = intermediate_size           
            self.code_size  = code_size
            
            self.relu = torch.nn.ReLU()   
            
            #encoder
            self.fc1 = torch.nn.Linear(self.input_size, self.intermediate_size)
            self.fc2 = torch.nn.Linear(self.intermediate_size, self.code_size)
            
            #decoder 
            self.fc3 = torch.nn.Linear(self.code_size, self.intermediate_size)            
            self.fc4 = torch.nn.Linear(self.intermediate_size, self.input_size)
            
            
        def forward(self, x):
            
            hidden = self.fc1(x)
            hidden = self.relu(hidden)
            
            code = self.fc2(hidden)
            code = self.relu(code)
 
            hidden = self.fc3(code)
            hidden = self.relu(hidden)
            
            output = self.fc4(hidden)
            #linear activation in final layer)            
            
            return output
    
class EarlyStopping:

    def __init__(self, patience=2, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
    
    def continue_training(self,current_score):
        if self.best_score > current_score:
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print("New best score:", current_score)
        else:
            self.counter+=1
            if self.verbose:
                print(self.counter, " iterations since best score.")
                
        return self.counter <= self.patience

class AutoEncoderObj:

    def __init__(
        self, 
        feature_list,
        output_feature, 
        model=None,
        scaler=None,
        intermediate_size=100,
        encoder_dim=3,
        learning_rate=0.0001,
        max_epochs=100,
        save_path="/Workspace/Shared/snacks_home/artefacts/", 
        device="cpu",
    ):

        self.model = model
        self.feature_list = feature_list
        self.scaler = scaler
        self.save_path = save_path
        self.device = device
        self.output_feature = output_feature
        self.intermediate_size = intermediate_size
        self.encoder_dim = encoder_dim
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def prepare_generators(self, data_set, batch_size, shuffle, num_workers):
    
        dataset_loader_params = {'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers}
        
        dataset_generator = torch.utils.data.DataLoader(data_set, **dataset_loader_params)
        
        return dataset_generator


    def scaleData(self, train, valid, features):
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train[features])

        train[features]=scaler.transform(train[features])
        valid[features]=scaler.transform(valid[features])
        
        return train, valid , scaler
    
    def evaluate_model(self, model, generator, criterion):
        model.eval()
        batch_losses = []
        for x_batch, y_batch in generator:
            # Forward pass
            y_pred = model(x_batch)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_batch)
            batch_losses.append(loss.item())
        mean_loss = np.mean(batch_losses)    
        return mean_loss
    

    def training_loop(
        self,
        model,
        training_generator,
        valid_generator,
        optimizer,
        criterion,
        max_epochs=100,
        apply_early_stopping=True,
        patience=2,
        verbose=False
    ):
        #Setting the model in training mode
        model.train()

        if apply_early_stopping:
            early_stopping = EarlyStopping(verbose=verbose, patience=patience)
        
        all_train_losses = []
        all_valid_losses = []
        
        #Training loop
        start_time=time.time()
        for epoch in range(self.max_epochs):
            model.train()
            train_loss=[]
            for x_batch, y_batch in training_generator:
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(x_batch)
                # Compute Loss
                loss = criterion(y_pred.squeeze(), y_batch)
                # Backward pass
                loss.backward()
                optimizer.step()   
                train_loss.append(loss.item())
            
            #showing last training loss after each epoch
            all_train_losses.append(np.mean(train_loss))
            if verbose:
                print('')
                print('Epoch {}: train loss: {}'.format(epoch, np.mean(train_loss)))
            #evaluating the model on the test set after each epoch    
            valid_loss = self.evaluate_model(model,valid_generator,criterion)
            all_valid_losses.append(valid_loss)
            if verbose:
                print('valid loss: {}'.format(valid_loss))
            if apply_early_stopping:
                if not early_stopping.continue_training(valid_loss):
                    if verbose:
                        print("Early stopping")
                    break
            
        training_execution_time=time.time()-start_time

        return model, training_execution_time, all_train_losses, all_valid_losses


    def fit(self, mode, train_df, valid_df):

        if mode != "train":
            raise ValueError("Mode must be 'train' when you call fit function")
        
        train_df, valid_df, scaler = self.scaleData(train_df, valid_df, self.feature_list)
        # Define the path to save the scaler
        scaler_path = self.save_path + "scaler.pkl"

        # Save the scaler object
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Scaler saved to {scaler_path}")

        x_train = torch.tensor(train_df[self.feature_list].values, dtype=torch.float32)
        x_valid = torch.tensor(valid_df[self.feature_list].values, dtype=torch.float32)

        y_train = torch.tensor(train_df[self.output_feature].values, dtype=torch.float32)
        y_valid =  torch.tensor(valid_df[self.output_feature].values, dtype=torch.float32)

        training_set = FraudDatasetUnsupervised(x_train)
        valid_set = FraudDatasetUnsupervised(x_valid)

        training_generator = self.prepare_generators(training_set, batch_size = 64, shuffle = True, num_workers = 0)
        valid_generator = self.prepare_generators(valid_set, batch_size = 64, shuffle = False, num_workers = 0)

        model = Autoencoder(input_size = len(self.feature_list), intermediate_size = self.intermediate_size, code_size = self.encoder_dim)
        model.to(self.device)

        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)

        model, training_execution_time, train_losses, valid_losses = self.training_loop(
            model,
            training_generator,
            valid_generator,
            optimizer,
            criterion,
            max_epochs=self.max_epochs,
            apply_early_stopping=True,
            verbose=True
        )

        model_path = self.save_path + "autoencoder.pth"
        torch.save(model.state_dict(), model_path)

        print("Model saved successfully")

    def use_encoder_only(self, x, model):
        #  manually set encoder inference
        x_encoder = model.fc1(x)
        x_encoder = model.relu(x_encoder)
        x_encoder = model.fc2(x_encoder)
        x_encoder = model.relu(x_encoder)

        return x_encoder
    
    def generate_embeddings(self, x, mode):

        if mode != "inference":
            raise ValueError("Mode must be 'inference' when you call generate_embeddings function")

        scaler_path = self.save_path + "scaler.pkl"
        if os.path.exists(scaler_path) == False:
            raise ValueError("Scaler path does not exist, run fit first")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("Scaler loaded successfully")

        # Load the model's state dictionary
        model_path = self.save_path + "autoencoder.pth"
        if os.path.exists(model_path) == False:
            raise ValueError("Model path does not exist, run fit first")
        
        model = Autoencoder(input_size = len(self.feature_list), intermediate_size = self.intermediate_size, code_size = self.encoder_dim)
        model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        print("Autoencoder loaded successfully")

        # standardize the input
        x_feature_in_list = x[self.feature_list]
        x_standardized = self.scaler.transform(x_feature_in_list)

        # convert to tensor
        x_tensor = torch.tensor(x_standardized, dtype=torch.float32)

        with torch.no_grad():
            # Encoder is only part used for inference
            x_embedded = self.use_encoder_only(x_tensor, model)
            x_embedded = x_embedded.detach().numpy()
        
        return x_embedded

# COMMAND ----------

# Rule Identification 

def train_tree_pipeline(df, label_col, numeric_cols, categorical_cols, cols_to_drop):

    preprocesser = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    tree = DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
        max_leaf_nodes = 7
    )

    tree_pipeline = Pipeline(
        steps=[
            ("one_hot_encoder", preprocesser),
            ("decision_tree", tree)
        ]
    )

    tree_pipeline.fit(
        df.drop(cols_to_drop, axis=1), 
        df[label_col]
    )

    OHE_COLS = tree_pipeline['one_hot_encoder'].named_transformers_['ohe'].get_feature_names_out(categorical_cols)  
    final_feature_names = list(OHE_COLS) + numeric_cols 

    return tree_pipeline, final_feature_names

# Visualize the decision tree
def visualise_trained_tree(pipeline, feature_names):
    plt.figure(figsize=(16, 10))
    plot_tree(pipeline['decision_tree'], feature_names=feature_names, filled=True,)
    plt.show()

def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = {}
    
    def recurse(node, path, paths, rule_num):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = round(tree_.threshold[node],1)
            path_left = f"({name} <= {threshold})"
            path_right = f"({name} > {threshold})"
            recurse(tree_.children_left[node], path + [path_left], paths, rule_num)
            recurse(tree_.children_right[node], path + [path_right], paths, rule_num)
        else:
            paths[f"hit_{rule_num[0]}"] = " and ".join(path)
            rule_num[0] += 1
    
    recurse(0, [], paths, [0])
    return paths


def generate_rule_stats(dfp, rules_list):

    dfp_hit = dfp.copy()

    rule_num = []
    rule_def = []
    gen_vol = []
    frd_vol = []
    frd_val = []
    FPR = []

    for i, rule in enumerate(rules_list):
        # print(i, rule)
        dfp_hit[f'rule_{i}'] = dfp_hit.eval(rule)
        rule_def.append(rule)
        rule_num.append(f'rule_{i}')
        gen_vol.append(int(dfp[(dfp_hit[f'rule_{i}'] == True) & (dfp_hit['TX_FRAUD'] == 0)]['TRANSACTION_ID'].count()))
        frd_vol.append(int(dfp[(dfp_hit[f'rule_{i}'] == True) & (dfp_hit['TX_FRAUD'] == 1)]['TRANSACTION_ID'].count()))
        frd_val.append(int(dfp[(dfp_hit[f'rule_{i}'] == True) & (dfp_hit['TX_FRAUD'] == 1)]['TX_AMOUNT'].sum()))
        FPR.append(
            int(dfp[(dfp_hit[f'rule_{i}'] == True) & (dfp_hit['TX_FRAUD'] == 0)]['TRANSACTION_ID'].count()) /
            max(int(dfp[(dfp_hit[f'rule_{i}'] == True) & (dfp_hit['TX_FRAUD'] == 1)]['TRANSACTION_ID'].count()),1)
        )

    df_rules = pd.DataFrame({
        'rule_num': rule_num,
        'rule_def': rule_def,
        'gen_vol': gen_vol,
        'frd_vol': frd_vol,
        'frd_val': frd_val,
        'FPR': FPR
    })

    gen_thresh = 0.25 * int(dfp[dfp['TX_FRAUD'] == 0]['TRANSACTION_ID'].count())
    fpr_thresh = 20

    df_rules_filtered = df_rules[(df_rules['gen_vol'] <= gen_thresh) & (df_rules['FPR'] <= fpr_thresh)]
    df_rules_sorted = df_rules_filtered.sort_values(by='frd_val', ascending=False)

    df_rules_sorted.index = range(len(df_rules_sorted))

    df_rules_sorted['cumulative_gen_vol'] = df_rules_sorted['gen_vol'].cumsum()
    df_rules_sorted['cumulative_frd_vol'] = df_rules_sorted['frd_vol'].cumsum()
    df_rules_sorted['cumulative_frd_val'] = df_rules_sorted['frd_val'].cumsum()
    df_rules_sorted['cumulative_FPR'] = df_rules_sorted['cumulative_gen_vol'] / df_rules_sorted['cumulative_frd_vol']

    return df_rules_sorted, dfp_hit

  
def evaluate(individual, rule_statistics, config):  
    """  
    Evaluate the fitness of an individual (ruleset).  
      
    Args:  
        individual (list): A list of binary values representing the inclusion of rules.  
        rule_statistics: A pandas DataFrame containing rule statistics.  
          
    Returns:  
        tuple: A single-element tuple containing the fitness score.  
    """  
    # Calculate fraud volume and fraud value based on selected rules  
    fraud_volume = sum(rule_statistics.loc[i, "frd_vol"] for i, included in enumerate(individual) if included)  
    fraud_value = sum(rule_statistics.loc[i, "frd_val"] for i, included in enumerate(individual) if included)  
  
    # Calculate the fitness score  
    fitness = (config["weights"]["fraud_volume"] * fraud_volume +  
               config["weights"]["fraud_value"] * fraud_value)  
  
    # Apply a penalty based on the number of rules selected  
    num_selected_rules = sum(individual)  
    penalty = config["rule_penalty"] * num_selected_rules  
    fitness -= penalty  
  
    return (fitness,)  



# COMMAND ----------

# Pipeline

def anomaly_detection_pipeline():

    # Read and split data into time periods
    feature_pdf = read_data(TABLE_LOCATION)

    feature_pdf_apr = feature_pdf[
        (feature_pdf['TX_DATETIME'] >= '2025-04-01') & 
        (feature_pdf['TX_DATETIME'] < '2025-05-01')
    ]

    feature_pdf_may_first3wk = feature_pdf[
        (feature_pdf['TX_DATETIME'] >= '2025-05-01') & 
        (feature_pdf['TX_DATETIME'] < '2025-05-24')
    ]

    feature_pdf_may_lastwk = feature_pdf[
        (feature_pdf['TX_DATETIME'] >= '2025-05-24') & 
        (feature_pdf['TX_DATETIME'] < '2025-06-01')
    ]

    # test train split for logistic (feeder model)
    train_df_apr, val_df_apr = custom_train_test_split(feature_pdf_apr, CUSTOMER_ID_COL, test_size=0.4, random_state=42)

    logistic_model = train_logistic_model(train_df_apr, MODEL_FEATURES, TARGET)

    # Get threshold for 19% recall
    logistic_threshold, logistic_precision, logistic_recall,  logistic_f1, logistic_accuracy = find_threshold_for_recall(
        logistic_model, val_df_apr, MODEL_FEATURES, TARGET, desired_recall=0.19, step=0.005
    )
    print(f"Precision: {logistic_precision:.2f}")  
    print(f"Recall: {logistic_recall:.2f}")  
    print(f"F1 Score: {logistic_f1}")  
    print(f"Accuracy: {logistic_accuracy}")

    # Train and test split for auto encoder training on first 3 weeks of May on ATL + BTL
    feature_pdf_may_first3wk_train, feature_pdf_may_first3wk_valid = custom_train_test_split(feature_pdf_may_first3wk, CUSTOMER_ID_COL, test_size=0.4, random_state=42)

    # Predict probability on may data for anomaly detection
    feature_pdf_may_lastwk['logistic_prob'] = logistic_model.predict_proba(feature_pdf_may_lastwk[MODEL_FEATURES])[:, 1] # Check if this is correct

    # Fit from Autoencoder
    autoencoder = AutoEncoderObj(feature_list=MODEL_FEATURES, output_feature=TARGET)
    # autoencoder.fit(mode="train", train_df=feature_pdf_may_first3wk_train, valid_df=feature_pdf_may_first3wk_valid)   # Only needs to train once for demo use saved encoder for other runs

    # Get last week of data for anomaly detection
    feature_may_last3wk_btl = feature_pdf_may_lastwk.loc[feature_pdf_may_lastwk['logistic_prob']<logistic_threshold]

    # Run auto encoder inference
    embeddings = autoencoder.generate_embeddings(x=feature_may_last3wk_btl, mode="inference")

    # Convert np.array into pd.DataFrame
    feature_may_last3wk_btl_embs = pd.DataFrame(embeddings, columns=['first_dimension', 'second_dimension', 'third_dimension'])

    assert len(feature_may_last3wk_btl_embs) == len(feature_may_last3wk_btl)

    # Reset the index of the embeddings DataFrame to match the original DataFrame
    feature_may_last3wk_btl_embs.index = feature_may_last3wk_btl.index

    # Add the embeddings as new columns to the original DataFrame
    feature_may_last3wk_btl[['first_dimension', 'second_dimension', 'third_dimension']] = feature_may_last3wk_btl_embs

    # --------- precision_score, recall_score, f1, accuracy_score = evaluate_logistic_model_performance(logistic_model, feature_pdf_apr, MODEL_FEATURES, TARGET, threshold=logistic_threshold)

    # --------- feature_may_first3wk_btl = feature_pdf_may_first3wk.loc[feature_pdf_may_first3wk['logistic_prob']<logistic_threshold]

    # --------- feature_pdf_may_first3wk_train, feature_pdf_may_first3wk_valid = custom_train_test_split(feature_pdf_may_first3wk, CUSTOMER_ID_COL, test_size=0.4, random_state=42)

    # Get fraud rate from first 3 week of May on which auto encoder was trained - we also dont know the fraud rate in lastest data
    expected_fraud_rate = feature_pdf_may_first3wk.TX_FRAUD.sum()/len(feature_pdf_may_first3wk)

    # Run Anomaly Detection and get tags for anomalies
    if_model = IsolationForest(random_state=42, contamination=expected_fraud_rate)  
    if_model.fit(feature_may_last3wk_btl[['first_dimension', 'second_dimension', 'third_dimension']])

    feature_may_last3wk_btl = get_anomaly_tags(feature_may_last3wk_btl, ['first_dimension', 'second_dimension', 'third_dimension'], if_model)


    save_artefacts(logistic_model, "logistic_model.pkl")
    save_artefacts(if_model, "if_model.pkl")
    save_artefacts(train_df_apr, "train_df_apr")
    save_artefacts(val_df_apr, "val_df_apr")
    save_artefacts(feature_pdf_may_first3wk_train, "feature_pdf_may_first3wk_train")
    save_artefacts(feature_pdf_may_first3wk_valid, "feature_pdf_may_first3wk_valid")
    save_artefacts(feature_may_last3wk_btl_embs, "feature_may_last3wk_btl_embs")
    save_artefacts(feature_may_last3wk_btl, "feature_may_last3wk_btl")

    return feature_may_last3wk_btl


def rule_pipeline(tagged_anomalies):
    X = tagged_anomalies.drop(columns=['TX_FRAUD','TRANSACTION_ID','TX_DATETIME','dw_bus_dt','CUSTOMER_ID','TX_FRAUD_SCENARIO', 'anomaly_scores', 'anomaly_tag', 'first_dimension', 'second_dimension', 'third_dimension', 'random_score'])
    y = tagged_anomalies['TX_FRAUD']

    COLUMNS_TO_REMOVE = ['TX_FRAUD','TRANSACTION_ID','TX_DATETIME','dw_bus_dt','CUSTOMER_ID','TX_FRAUD_SCENARIO', 'anomaly_scores', 'anomaly_tag', 'first_dimension', 'second_dimension', 'third_dimension', 'random_score']

    CATEGORICAL_COLS = [
    "merchantName", "merchantCategory", "transactionChannel"
    ]

    NON_MODEL_COLS = [
        'TX_FRAUD','TRANSACTION_ID','TX_DATETIME','dw_bus_dt','CUSTOMER_ID','TX_FRAUD_SCENARIO', 'anomaly_scores', 'anomaly_tag', 'first_dimension', 'second_dimension', 'third_dimension', 'random_score'
        ]

    NUMERIC_COLS = [i for i in tagged_anomalies.columns if i not in NON_MODEL_COLS and i not in CATEGORICAL_COLS and i not in COLUMNS_TO_REMOVE]

    TARGET_FOR_DT = 'anomaly_tag'

    pipeline, features = train_tree_pipeline(
        tagged_anomalies,TARGET_FOR_DT, NUMERIC_COLS, CATEGORICAL_COLS, NON_MODEL_COLS
    )

    pred = pipeline.predict(tagged_anomalies.drop(NON_MODEL_COLS, axis=1))

    visualise_trained_tree(pipeline, features)

    rules = get_rules(pipeline['decision_tree'], features)
    rules_list = list(rules.values())
    print(rules_list)

    return rules_list, pipeline, features

def rule_stats_pipeline(anomalies_df, all_suggested_ruleset, all_existing_ruleset):
    suggested_rule_stats, suggested_hit_data = generate_rule_stats(anomalies_df, all_suggested_ruleset)
    existing_rule_stats, existing_hit_data = generate_rule_stats(anomalies_df, all_existing_ruleset)

    suggested_rule_stats['rule_num'] = "suggested_"+suggested_rule_stats['rule_num']
    existing_rule_stats['rule_num'] = "existing_"+existing_rule_stats['rule_num']

    all_ruleset = pd.concat([suggested_rule_stats, existing_rule_stats])
    return all_ruleset


def genetic_optimization_pipeline(ruleset_df):  
    """  
    Main function to set up and execute the genetic algorithm.  
    """  

    config = {  
        "num_rules": len(ruleset_df),  # Total number of rules to consider  
        "rule_penalty": 10000,   # Penalty factor for including more rules  
        "weights": {"fraud_volume": 0.5, "fraud_value": 0.5},  # Weights for fitness calculation  
        "cxpb": 0.7,  # Probability for crossover operation  
        "mutpb": 0.2,  # Probability for mutation operation  
        "ngen": 50,    # Number of generations to run the algorithm  
        "pop_size": 20 # Size of the population  
    }

    ruleset_df = ruleset_df.reset_index(drop=True)  
     # Check if classes are already created  
    if "FitnessMax" not in creator.__dict__:  
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    if "Individual" not in creator.__dict__:  
        creator.create("Individual", list, fitness=creator.FitnessMax)  
  
    toolbox = base.Toolbox()  
    toolbox.register("attr_bool", random.randint, 0, 1)  
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=config["num_rules"])  
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
  
    # Register the evaluation, crossover, mutation, and selection functions  
    toolbox.register("evaluate", functools.partial(evaluate, rule_statistics=ruleset_df, config=config))  
    toolbox.register("mate", tools.cxOnePoint)  
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  
    toolbox.register("select", tools.selRoulette)  
  
    random.seed(42)  
  
    # Create initial population  
    pop = toolbox.population(n=config["pop_size"])  
  
    # Define the hall of fame (elitism) to store the best individual  
    hof = tools.HallOfFame(1)  
  
    # Define statistics to track during the GA run  
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Extract the first element of the tuple  
    stats.register("avg", lambda x: sum(x) / len(x))  # Track average fitness  
    stats.register("min", min)  # Track minimum fitness  
    stats.register("max", max)  # Track maximum fitness  
  
    # Run the genetic algorithm  
    algorithms.eaSimple(pop, toolbox, cxpb=config["cxpb"], mutpb=config["mutpb"], ngen=config["ngen"],  
                        stats=stats, halloffame=hof, verbose=True)  
  
    # Print the best solution found  
    best_individual = hof[0]  
    print(f"Best individual is {best_individual} with fitness {best_individual.fitness.values[0]}")  
  
    # Filter the DataFrame to keep only the rules in the best solution  
    selected_rules_indices = [i for i, included in enumerate(best_individual) if included]  
    selected_rules_df = ruleset_df.iloc[selected_rules_indices]  
  
    print("Filtered DataFrame with selected rules:")  
    print(selected_rules_df)  
  
    return selected_rules_df  


# COMMAND ----------

if __name__=='__main__':
    anomalies_tagged = anomaly_detection_pipeline()
    all_suggested_rules_set, decision_tree_model, feature_set = rule_pipeline(anomalies_tagged)
    combined_ruleset = rule_stats_pipeline(anomalies_tagged, all_suggested_rules_set, EXISTING_RULE_SET)
    selected_rules_stats = genetic_optimization_pipeline(combined_ruleset)

    display(selected_rules_stats)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Genetic Algo for Rule Optimization

# COMMAND ----------

!pip install deap

# COMMAND ----------


# df = combined_ruleset  
  
  
def evaluate(individual, rule_statistics, config):  
    """  
    Evaluate the fitness of an individual (ruleset).  
      
    Args:  
        individual (list): A list of binary values representing the inclusion of rules.  
        rule_statistics: A pandas DataFrame containing rule statistics.  
          
    Returns:  
        tuple: A single-element tuple containing the fitness score.  
    """  
    # Calculate fraud volume and fraud value based on selected rules  
    fraud_volume = sum(rule_statistics.loc[i, "frd_vol"] for i, included in enumerate(individual) if included)  
    fraud_value = sum(rule_statistics.loc[i, "frd_val"] for i, included in enumerate(individual) if included)  
  
    # Calculate the fitness score  
    fitness = (config["weights"]["fraud_volume"] * fraud_volume +  
               config["weights"]["fraud_value"] * fraud_value)  
  
    # Apply a penalty based on the number of rules selected  
    num_selected_rules = sum(individual)  
    penalty = config["rule_penalty"] * num_selected_rules  
    fitness -= penalty  
  
    return (fitness,)  
  
def genetic_optimization_pipeline(ruleset_df):  
    """  
    Main function to set up and execute the genetic algorithm.  
    """  

    config = {  
        "num_rules": len(ruleset_df),  # Total number of rules to consider  
        "rule_penalty": 10000,   # Penalty factor for including more rules  
        "weights": {"fraud_volume": 0.5, "fraud_value": 0.5},  # Weights for fitness calculation  
        "cxpb": 0.7,  # Probability for crossover operation  
        "mutpb": 0.2,  # Probability for mutation operation  
        "ngen": 50,    # Number of generations to run the algorithm  
        "pop_size": 20 # Size of the population  
    }

    ruleset_df = ruleset_df.reset_index(drop=True)  
     # Check if classes are already created  
    if "FitnessMax" not in creator.__dict__:  
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
    if "Individual" not in creator.__dict__:  
        creator.create("Individual", list, fitness=creator.FitnessMax)  
  
    toolbox = base.Toolbox()  
    toolbox.register("attr_bool", random.randint, 0, 1)  
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=config["num_rules"])  
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  
  
    # Register the evaluation, crossover, mutation, and selection functions  
    toolbox.register("evaluate", functools.partial(evaluate, rule_statistics=ruleset_df, config=config))  
    toolbox.register("mate", tools.cxOnePoint)  
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  
    toolbox.register("select", tools.selRoulette)  
  
    random.seed(42)  
  
    # Create initial population  
    pop = toolbox.population(n=config["pop_size"])  
  
    # Define the hall of fame (elitism) to store the best individual  
    hof = tools.HallOfFame(1)  
  
    # Define statistics to track during the GA run  
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])  # Extract the first element of the tuple  
    stats.register("avg", lambda x: sum(x) / len(x))  # Track average fitness  
    stats.register("min", min)  # Track minimum fitness  
    stats.register("max", max)  # Track maximum fitness  
  
    # Run the genetic algorithm  
    algorithms.eaSimple(pop, toolbox, cxpb=config["cxpb"], mutpb=config["mutpb"], ngen=config["ngen"],  
                        stats=stats, halloffame=hof, verbose=True)  
  
    # Print the best solution found  
    best_individual = hof[0]  
    print(f"Best individual is {best_individual} with fitness {best_individual.fitness.values[0]}")  
  
    # Filter the DataFrame to keep only the rules in the best solution  
    selected_rules_indices = [i for i, included in enumerate(best_individual) if included]  
    selected_rules_df = ruleset_df.iloc[selected_rules_indices]  
  
    print("Filtered DataFrame with selected rules:")  
    print(selected_rules_df)  
  
    return selected_rules_df  


# COMMAND ----------

selected_rules_stats = genetic_optimization_pipeline(combined_ruleset)  

# COMMAND ----------

display(selected_rules_stats)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Rough Work to test the pipeline - DELETE LATER

# COMMAND ----------

X = dfp.drop(columns=['TX_FRAUD','TRANSACTION_ID','TX_DATETIME','dw_bus_dt','CUSTOMER_ID','TX_FRAUD_SCENARIO'])

y = dfp['TX_FRAUD']

# COMMAND ----------

display(feature_may_last3wk_btl.head())

# COMMAND ----------

