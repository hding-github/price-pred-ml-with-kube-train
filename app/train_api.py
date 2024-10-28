from xgboost import XGBRegressor
import model_evaluation
import sub_data
from sklearn.ensemble import RandomForestRegressor
import json

df_data = sub_data.get_data_locally()
tD_Datasets = sub_data.get_df_training_datasets(df_data, "MEDV")

list_variables_required_xgboost = ["n_estimators","max_depth","learning_rate","subsample"]
list_variables_required_randomforest = ["n_estimators","max_depth","min_samples_split","min_samples_leaf"]

def validate_input(tD_Input):
    if "model" not in tD_Input:
        return False
    if tD_Input["model"] == "xgboost":
        for tVariable in list_variables_required_xgboost:
            if tVariable not in tD_Input:
                return False
        return True
    if tD_Input["model"] == "randomforest":
        for tVariable in list_variables_required_randomforest:
            if tVariable not in tD_Input:
                return False
        return True
    return False

def input(strData, strModel):
    strError = "Error"
    strSME = strError
    tD_Input = {}
    try:
        tD_Input = json.loads(strData)
    except:
        return strError
    tD_Input["model"] = strModel

    if validate_input(tD_Input) == False:
        return strError

    if "model" == "xgboost":
        tModel = XGBRegressor(n_estimators=tD_Input["n_estimators"], max_depth=tD_Input["max_depth"], 
                           learning_rate=tD_Input["learning_rate"], subsample=tD_Input["subsample"])
    if "model" == "randomforest":
        tModel = RandomForestRegressor(n_estimators=tD_Input["n_estimators"], max_depth=tD_Input["n_estimators"], 
                                       min_samples_leaf=tD_Input["min_samples_leaf"], min_samples_split = tD_Input["min_samples_split"])

    tModel.fit(tD_Datasets["X_train"], tD_Datasets["y_train"])
    # make predictions
    try:
        preds = tModel.predict(tD_Datasets["X_test"])
        tMSE = model_evaluation.calculate_mse(tD_Datasets["y_test"], preds)
        tR2 = model_evaluation.calculate_r2(tD_Datasets["y_test"], preds)
        strResults = json.dumps({"mse": tMSE, "R2": tR2})
        return strResults
    except:
        return strError
