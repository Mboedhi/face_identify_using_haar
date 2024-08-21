import cv2, os, time

cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detect = cv2.CascadeClassifier(cascade)

def register(name, vid_src = 0, num = 30):
    cam = cv2.VideoCapture(vid_src)
    count = 0
    user_directory = f"faces/{name}"

    if not os.path.exists(user_directory) :
        os.makedirs(user_directory)

    img_exist = len([img for img in os.listdir(user_directory) if img.endswith('.jpg')])
    count = img_exist

    while count < img_exist + num :
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_face = face_detect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (a, b, c, d) in detected_face :
            face = frame[b:b+d, a:a+c]
            save_path = os.path.join(user_directory, f"{name}_{count}.jpg")
            cv2.imwrite(save_path, face)
            count+=1
            print(f"taking image . . . {count - img_exist}/{num}")
            time.sleep(0.25)

        cv2.imshow('Register New Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows