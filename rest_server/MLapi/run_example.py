import learningFunction
from ikzziML.common import calcFunctions

inference_testX_path = 'data/json/bodychart.json'
MODEL_PATH = 'patients_2layerNN/2layerNN.ckpt'
DATA_PREPRO_PATH = 'patients_2layerNN/dataPreprocessing.csv'

predict_data = learningFunction.supervised_learning_inference(inference_testX_path, MODEL_PATH, DATA_PREPRO_PATH)
print(predict_data)

predict_score = calcFunctions.setScore(predict_data)
print(predict_score)
