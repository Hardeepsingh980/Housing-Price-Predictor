from joblib import load
import numpy as np

model = load('Dragon.joblib')



## Example Data Has Been Added To The 'features' Variable. 
## You May Add Your Own Value For Testing.

features = np.array([[-0.64649223,  0.18716752, -1.12581552, -0.27288841, -1.42038605,
       -10.04601796, -1.7412613 ,  2.79774223, -2.42387223, -0.57387797,
       -0.99428207,  0.43852974, -0.49833679]])

print(model.predict(features))
