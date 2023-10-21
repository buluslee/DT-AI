from sklearn.externals import joblib
import config

model_path = config.model_path
model_name = "fasttext_model.pkl"

model = joblib.load(model_path + model_name)

print(model)