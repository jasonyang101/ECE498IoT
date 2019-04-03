import tensorflow
from tensorflow import keras
import numpy as np
import requests
import gzip
import time

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
model = keras.models.load_model('keras_modified.h5')
model_1 = keras.models.load_model('keras_modified_1.3.h5')
out = []
# get time NEW MODEL
start1 = time.time()
predictions = model.predict(test_images)
rst = ''.join(map(str, np.argmax(predictions,axis=1).tolist()));
end1 = time.time()
# end getting time
time1 = end1-start1
#get time OLD MODEL
start2 = time.time()
predictions2 = model_1.predict(test_images)
rst2 = ''.join(map(str, np.argmax(predictions,axis=1).tolist()));
end2 = time.time()
# end getting time
time2 = end2-start2

for i in range(len(predictions)):
    out.append(np.argmax(predictions[i]))
predict = ""
for i in range(len(out)):
    predict += str(out[i])
print predict

values = {'request': 'verify', 'netid':'jyang223', 'testset_id':id,'prediction':predict, 'team': 'nwa', 'latency': time1}
r_2 = requests.post(url,data=values,allow_redirects=True)
print r_2.text
print "Percentage:" + str(float(int(r_2.text)/10.0))

score = float((int(r_2.text)/1000.0)/(5*time1))
print score
print "SCORE: " + str(score)
print "Inference Exe. time New Model: ", time1, "Sec."
print "Inference Exe. time Old Model: ", time2, "Sec."
