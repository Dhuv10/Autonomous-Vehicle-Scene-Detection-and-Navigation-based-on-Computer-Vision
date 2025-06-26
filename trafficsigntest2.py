import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

path = "D:/Dataset"
labelFile = "D:/labels.csv"
batch_size_val = 32
epochs_val = 10
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)

data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def data_augmentation(images, labels, width_shift_range=0.1, height_shift_range=0.1,
                      zoom_range=0.2, shear_range=0.1, rotation_range=10, target_size=(32, 32)):
    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
         # Resize image to target size
        image = cv2.resize(image, target_size)
        augmented_images.append(image)
        augmented_labels.append(label)

        # Apply width shift
        if width_shift_range > 0:
            width_shift = np.random.uniform(-width_shift_range, width_shift_range) * image.shape[1]
            matrix = np.float32([[1, 0, width_shift], [0, 1, 0]])
            augmented_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))

        # Apply height shift
        if height_shift_range > 0:
            height_shift = np.random.uniform(-height_shift_range, height_shift_range) * image.shape[0]
            matrix = np.float32([[1, 0, 0], [0, 1, height_shift]])
            augmented_image = cv2.warpAffine(augmented_image, matrix, (image.shape[1], image.shape[0]))

        # Apply zoom
        if zoom_range > 0:
            zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            zoomed_width = int(image.shape[1] * zoom_factor)
            zoomed_height = int(image.shape[0] * zoom_factor)
            zoomed_image = cv2.resize(augmented_image, (zoomed_width, zoomed_height))
            augmented_image = cv2.resize(zoomed_image, (image.shape[1], image.shape[0]))

        # Apply shear
        if shear_range > 0:
            shear_factor = np.random.uniform(-shear_range, shear_range)
            matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            augmented_image = cv2.warpAffine(augmented_image, matrix, (image.shape[1], image.shape[0]))

        # Apply rotation
        if rotation_range > 0:
            rotation_angle = np.random.uniform(-rotation_range, rotation_range)
            matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), rotation_angle, 1)
            augmented_image = cv2.warpAffine(augmented_image, matrix, (image.shape[1], image.shape[0]))

        augmented_images.append(augmented_image)
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

X_train_augmented, y_train_augmented = data_augmentation(X_train, y_train)

X_train = np.array(list(map(preprocessing, X_train_augmented)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_train = to_categorical(y_train_augmented, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())
history = model.fit(X_train, y_train, batch_size=batch_size_val, epochs=50,
                    validation_data=(X_validation, y_validation), shuffle=True)

model.save("TrafficSignDetectormodel.h5")

# Evaluating the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculating and printing metrics
accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Printing full confusion matrix
pd.set_option('display.max_rows', None) # To display all rows
pd.set_option('display.max_columns', None) # To display all columns
pd.set_option('display.width', None) # To remove column width limitation
print(pd.DataFrame(conf_matrix))


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

