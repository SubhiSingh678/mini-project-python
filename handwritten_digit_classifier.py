import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
#mnist data set of 70000 handwritten digits(0 to 9) each 28x28 grayscale image
(x_train,y_train), (x_test,y_test)= tf.keras.datasets.mnist.load_data() 
'''x_train: 60000 images to train, y_train: labels(0 to 9) x_test:10000 images to test y_test:labels(0 to 9)'''
print("shape of training images:",x_train.shape)  
print("shape of testing images :",x_test.shape)
print("shape of training labels:",y_train.shape)
print("Shape of testing labels:",y_test.shape)

plt.imshow(x_train[0],cmap='gray')
plt.title(f"label: {y_train[0]}")
plt.axis('off')
plt.show()
#default figsize 6.4,4.8 inches 
plt.figure(figsize=(10,4))  #creates a blank canvas to hold figure 10-width in inches 4-height
for i in range(12):
    idx= np.random.randint(0,len(x_train)-1)
    plt.subplot(3,4,i+1)  # makes 3 rows 4 columns and i+1 selects which slot to place the image 
    plt.imshow(x_train[idx], cmap='gray')
    plt.title(f"label: {y_train[idx]}")
    plt.axis('off')

plt.tight_layout() #automatic even spacing 
plt.show()

#preprocessing the data
#normalizing the value-- 0-255--> 0-1 (small values learn faster)
x_train= x_train/255.0
x_test = x_test/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(10, activation="softmax")])
#flattens - 2D to 1D dense(128,relu)-learns pattern of images dense(10,softmax) - predict 0-9

#compile the model
model.compile(
    optimizer = 'adam',  # adjusts learning rate automatically 
    loss= 'sparse_categorical_crossentropy',
    metrics = ['accuracy'] #measures accuracy 
)
#training the model
model.fit(x_train,y_train,epochs =5) #5 times learning

#testing the model
test_loss, test_acc = model.evaluate(x_test,y_test)
print("accuracy:",test_acc)

#predict a digit
index= np.random.randint(0,len(x_test))
image = x_test[index]
label= y_test[index]

prediction = model.predict(np.expand_dims(image,axis=0))
plt.imshow(image,cmap='gray')
plt.title(f"predicted:{np.argmax(prediction)}, Actual: {label}")
plt.show()