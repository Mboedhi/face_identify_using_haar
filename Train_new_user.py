import cv2, pickle, os

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from Register import face_detect

def training(dir='faces'):

    if not os.path.exists(dir):
        print(f"{dir} not found!")
        return
    
    f = []
    n = []

    for name in os.listdir(dir) :
        user_directory = os.path.join(dir, name)
        if os.path.isdir(user_directory) :
            for image_name in os.listdir(user_directory) :

                if image_name.endswith('.jpg') :
                    image_path = os.path.join(user_directory, image_name)
                    image_file = cv2.imread(image_path)
                    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
                    detected_face = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                    for (a, b, c, d) in detected_face :
                        face = gray[b:b+d, a:a+c]
                        face = cv2.resize(face, (100, 100)).flatten()
                        f.append(face)
                        n.append(name)

    X_train, X_test, y_train, y_test = train_test_split(f, n, test_size = 0.2, random_state=42)

    data_training = neighbors.KNeighborsClassifier(n_neighbors = 3)
    data_training.fit(X_train, y_train)

    with open('trained_face_identify.pkl', 'wb') as f:
        pickle.dump(data_training, f)

    y_pred = data_training.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy}")

    matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix : \n{matrix}")

    print(classification_report(y_test, y_pred))

# training()
