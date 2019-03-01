import tensorflow
from tensorflow import keras
import numpy as np
import requests
import gzip

url = 'https://courses.engr.illinois.edu/ece498icc/sp2019/lab2_request_dataset.php'

def get_testset():
    values = {'request': 'testdata', 'netid':'jyang223'}
    r = requests.post(url, data=values, allow_redirects=True)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f:
        f.write(r.content)
    return load_dataset(filename), testset_id

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 28,28)
    return data

test_images, id = get_testset()
print "TESTID: " + id
test_images = np.expand_dims(test_images, axis=3)
new_model6 = keras.models.load_model('keras_model.h5')
out = []
predictions = new_model6.predict(test_images)
for i in range(len(predictions)):
    out.append(np.argmax(predictions[i]))
predict = ""
for i in range(len(out)):
    predict += str(out[i])
print predict

values = {'request': 'verify', 'netid':'jyang223', 'testset_id':id,'prediction':predict}
r_2 = requests.post(url,data=values,allow_redirects=True)
print r_2.text
print "Percentage:" + str(float(int(r_2.text)/10.0))
