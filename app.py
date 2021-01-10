import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shap

pd.set_option('display.max_columns', None)
import gc
gc.collect()

df = pd.read_csv(r"C:\Users\abhishek_sharma39\Downloads\BostonHousing.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()

cat_features = df.select_dtypes(include=['object','category']).columns.tolist()

class what_if_analyzer():
    def __init__(self, df, target, cat_features, force_plot = True, waterfall = False, split_ratio=0.75):
        self.df = df
        self.split_ratio = split_ratio
        self.target = target
        self.cat_features = cat_features
        self.force_plot = force_plot
        self.waterfall = waterfall
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        self.X_test_updated = None
        self.y_test_updated = None
    
    def regression(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target]
        
        self.class_names = self.df[self.target].unique().tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y,
                                                            test_size=self.split_ratio, 
                                                            random_state=42)
        
        self.model = CatBoostRegressor(iterations=1000, learning_rate=0.005,
                                   loss_function='RMSE', thread_count=-1,
                                   early_stopping_rounds=50, random_seed=42)
        
        if len(self.cat_features):
            self.model.fit(self.X_train, self.y_train, 
                      cat_features = self.cat_features,
                      verbose=200,
                      eval_set=(self.X_test, self.y_test))
        else:
            self.model.fit(self.X_train, self.y_train,
                      verbose=200,
                      eval_set=(self.X_test, self.y_test))
        print("Model R2 :",metrics.r2_score(self.model.predict(self.X_test), self.y_test))
        
    def what_if(self, index_to_modify, col_to_change, value):

        self.X_test_updated = self.X_test.copy()
        self.y_test_updated = self.y_test.copy()

        self.X_test.reset_index(inplace=True, drop=True)
        self.y_test.reset_index(inplace=True, drop=True)
        self.X_test_updated.reset_index(inplace=True, drop=True)
        self.y_test_updated.reset_index(inplace=True, drop=True)
        
        from pandas.testing import assert_frame_equal
        print(assert_frame_equal(self.X_test, self.X_test_updated, check_dtype=False))
        
        self.X_test_updated.loc[index_to_modify, col_to_change] = value
        
        temp = self.model.predict(self.X_test_updated.iloc[index_to_modify:index_to_modify+1, :])[0]
        self.y_test_updated.iloc[index_to_modify] = temp
        
        shap_values = self.model.get_feature_importance(Pool(self.X_test, 
                                                             self.y_test, 
                                                             cat_features = self.cat_features), 
                                                        type='ShapValues')
        
        expected_value = shap_values[:, -1][0]
        shap_values = shap_values[:, :-1]
        
        print(f"Actual : X-Value, Y-Value :", (self.X_test.loc[index_to_modify, col_to_change], self.y_test.iloc[index_to_modify]))
        
        print(f"Updated : X-Value , Y-Value :", (self.X_test_updated.loc[index_to_modify, col_to_change],
                                                    np.round(self.y_test_updated.iloc[index_to_modify], 2)))
        
        # Check
        print()
        #print("Equal:",np.array_equal(np.array(self.X_test), np.array(self.X_test_updated)))
        print()
        
        shap_values_modified = self.model.get_feature_importance(Pool(self.X_test_updated, 
                                                                      self.y_test_updated, 
                                                                      cat_features = self.cat_features), 
                                                                 type='ShapValues')
        
        expected_value_modified = shap_values_modified[:, -1][0]
        #print("Expected Value:", expected_value_modified)
        shap_values_modified = shap_values_modified[:, :-1]
        
        # Test
        """
        print("Equal SHAP at Index:",np.array_equal(np.array(shap_values[index_to_modify,:]), 
                                                    np.array(shap_values_modified[index_to_modify,:])))
        
        print("Equal X at Index:",np.array_equal(np.array(self.X_test.iloc[index_to_modify,:]), 
                                                 np.array(self.X_test_updated.iloc[index_to_modify,:])))
        """
        
        if self.force_plot:
            shap.force_plot(expected_value, 
                            shap_values[index_to_modify,:], 
                            self.X_test.iloc[index_to_modify,:],
                            feature_names=self.X_train.columns.tolist(),
                            matplotlib=True)

            print()
            shap.force_plot(expected_value_modified, 
                            shap_values_modified[index_to_modify,:], 
                            self.X_test_updated.iloc[index_to_modify,:],
                            feature_names=self.X_train.columns.tolist(),
                            matplotlib=True)
            
        if self.waterfall:
        
            shap.waterfall_plot(expected_value, 
                                shap_values[index_to_modify,:], 
                                self.X_test.iloc[index_to_modify,:],
                                feature_names=self.X_train.columns.tolist())



            shap.waterfall_plot(expected_value_modified, 
                                shap_values_modified[index_to_modify,:], 
                                self.X_test_updated.iloc[index_to_modify,:],
                                feature_names=self.X_train.columns.tolist(),
                                max_display=15)
        
        del shap_values_modified, self.X_test_updated, self.y_test_updated
        
        
if __name__ == '__main__':
    model = what_if_analyzer(df, target='medv', cat_features=cat_features, force_plot=True, waterfall = False)
    model.regression()
    model.what_if(index_to_modify = 10, col_to_change = 'lstat', value=15)