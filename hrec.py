import os 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf

mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
model.save('handwritten-model.h5')

model=tf.keras.models.load_model('handwritten-model.h5')
loss,accuracy=model.evaluate(x_test,y_test)
print(loss)
print(accuracy)

image_number=0
while os.path.isfile(f"#file path.png"):
   try:
        img=cv2.imread(f"digits\digit1.png")
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"this digit is mostly a {np.argmax(prediction)}")
        plt.imsave(img[0],cmap=plt.cm.binary)
        plt.show()
   except:
         print("Error!!")
   finally:
         image_number+=1
         