import os, pickle, cv2

from Register import face_detect, register
from Train_new_user import training

def delete_data(name) :
    user_directory = f"faces/{name}"

    if os.path.exists(user_directory) :
        for file in os.listdir(user_directory):
            os.remove(os.path.join(user_directory, file))
        os.rmdir(user_directory)
        print(f"Data named {user_directory} has been deleted.")
        training()
    else :
        print(f"no data named {user_directory}.")


def identify(vid_src = 0) :
    with open('trained_face_identify.pkl', 'rb') as f:
        model = pickle.load(f)

    cam = cv2.VideoCapture(vid_src)
    while True :
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_face = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (a, b, c, d) in detected_face :
            face = gray[b:b+d, a:a+c]
            face = cv2.resize(face, (100, 100)).flatten()
            user_identify = model.predict([face])[0]

            cv2.rectangle(frame, (a, b), (a+c, b+d), (255, 0, 0), 2)
            cv2.putText(frame, user_identify, (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Identifying face", frame)

        button = cv2.waitKey(1) & 0xFF

        if button == ord('q') :
            break

    cam.release()
    cv2.destroyAllWindows

def main() :
    # training()
    while True :
        print("\nMenu : ")
        print("1. Register New Face")
        print("2. Update Face Dataset")
        print("3. Face Identify")
        print("4. Delete Face")
        print("0. Exit")

        choice = input("input choice : ")

        if choice == '1' :
            name = input("Enter your name : ")
            camera = int(input("Input camera source (ex : 0): "))
            register(name, camera)

            print("Training new face, please wait . . .")
            training()

            print("New data saved")

        elif choice == '2' :
            print(f"updating, please wait")
            training()
            print(f'Update done !!!')
            
        elif choice == '3' :
            camera = int(input("Input camera source (ex : 0): "))
            try :
                camera = int(camera)
            except ValueError :
                pass
            
            print(f"press(q) to quit")
            identify(camera)
        
        elif choice == '4' :
            name = input("Enter your name : ")
            delete_data(name)
        
        elif choice == '0' :
            break

        else :
            print("Input error !! Please try again")

if __name__ == '__main__' :
    main()
