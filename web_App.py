from flask import Flask, render_template, Response, request, flash, stream_with_context
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import time
import threading as th


########         Functions         ##############
def timer():
    global t
    t = 5
    while t > 0:
        t = t - 1
        # print("ΤΙ ΦΑΣΗ ",t)
        print(t)
        time.sleep(1)
        if t == 0:
            t = 5

def choices(emotion):
    if emotion=="Angry":
        message_emotion = "Δείχνεις λίγο νευριασμένος. Πάρε λίγες βαθίες ανάσες και μέτρα μέχρι το δέκα!"
        print("Δείχνεις λίγο νευριασμένος. Πάρε λίγες βαθίες ανάσες και μέτρα μέχρι το δέκα!")
    elif emotion=="Disgust":
        message_emotion = "Είδες κάτι αδηδιαστικό;"
        print("Είδες κάτι αδηδιαστικό;")
    elif emotion=="Fear":
        message_emotion = "Μη φοβάσαι"
        print("Μη φοβάσαι")
    elif emotion=="Happy":
        message_emotion = "Είσαι χαρούμενος"
        print("Είσαι χαρούμενος")
    elif emotion=="Neutral":
        message_emotion = "Τι κάνεις;"
        print("Τι κάνεις;")
    elif emotion=="Sad":
        message_emotion = "Φαίνεσαι λυπημένος γιατί δεν βάζεις λίγο μουσική;"
        print("Φαίνεσαι λυπημένος γιατί δεν βάζεις λίγο μουσική;")
    elif emotion=="Surprise":
        message_emotion = "Σε βλέπω ενθουσιασμένο έγινε κάτι καλό;"
        print("Σε βλέπω ενθουσιασμένο έγινε κάτι καλό;")
    else:
        message_emotion = "Δεν υπάρχει πρόσωπο. Φτιάξε την κάμερα"
        print("Δεν υπάρχει πρόσωπο. Φτιάξε την κάμερα")

    return message_emotion

def most_frequent(List):
    # counter = 0
    emotion = ""

    for em in List:
        curr_frequency = List.count(em)
        if (curr_frequency / 5) > 0.5:
            emotion = em
            break
    #         if(curr_frequency> counter):
    #             counter = curr_frequency
    #             emotion = em

    return emotion

# Load Haarcascade for face detection
try:
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("Cascade for face detection loaded")
except Exception as e:
    print(e)


classifier = load_model('cnn_model200_32.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
Current_emotion = []
last5frames = []
count = []
global message
message = ""
thread = th.Thread(target=timer, daemon=True)
thread.start()


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture(1)
def gen_frames():
    global message

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]

                if len(last5frames) < 5:
                    last5frames.append(label)  # New
                elif len(last5frames) == 5:
                    count = last5frames
                    last5frames.pop(0)
                # else:
                #     last5frames.pop(0)  # New


                # print(last5frames)
                Current_emotion = most_frequent(last5frames)
                # print(f"Το κύριο συναίσθημα ειναι {Current_emotion}")

                global t

                if t == 0:
                    message = choices(Current_emotion)
                    t = 5

                percent = int(prediction[prediction.argmax()] * 100)  # New
                percent = str(percent) + '%'
                label_position = (x, y - 10)
                percent_position = (x + w, y - 10)  # New
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(percent), percent_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # New
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



########         Flask         ##############
@app.route('/')
def index():
    return render_template('index.html')

def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.enable_buffering(5)
    return

# def generate():
#     for item in Current_emotion:
#         yield str(item)

# @app.route('/stream')
# def streamed_response():
#     # def generate():
#         # yield 'Hello '
#         # yield request.args['name']
#         # yield '!'
#     return app.response_class(stream_with_context(generate()))
########         Video         ##############

@app.route('/video_feed')
def video_feed():
    # @stream_with_context
    # def gen_frames():
    #     global message
    #
    #     while True:
    #         _, frame = cap.read()
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    #
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    #             roi_gray = gray[y:y + h, x:x + w]
    #             roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    #
    #             if np.sum([roi_gray]) != 0:
    #                 roi = roi_gray.astype('float') / 255.0
    #                 roi = img_to_array(roi)
    #                 roi = np.expand_dims(roi, axis=0)
    #
    #                 prediction = classifier.predict(roi)[0]
    #                 label = emotion_labels[prediction.argmax()]
    #
    #                 if len(last5frames) < 5:
    #                     last5frames.append(label)  # New
    #                 elif len(last5frames) == 5:
    #                     count = last5frames
    #                     last5frames.pop(0)
    #                 # else:
    #                 #     last5frames.pop(0)  # New
    #
    #                 # print(last5frames)
    #                 Current_emotion = most_frequent(last5frames)
    #                 # print(f"Το κύριο συναίσθημα ειναι {Current_emotion}")
    #
    #                 global t
    #
    #                 if t == 0:
    #                     message = choices(Current_emotion)
    #                     t = 5
    #
    #                 percent = int(prediction[prediction.argmax()] * 100)  # New
    #                 percent = str(percent) + '%'
    #                 label_position = (x, y - 10)
    #                 percent_position = (x + w, y - 10)  # New
    #                 cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #                 cv2.putText(frame, str(percent), percent_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #                             2)  # New
    #             else:
    #                 cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         # cv2.imshow('Emotion Detector', frame)
    #
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             cap.release()
    #             cv2.destroyAllWindows()
    #             break
    #
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    #
    # global message
    # global t
    # # yield message
    # if t == 0:
    #     message = choices(Current_emotion)
    #     t = 5
    # return Response(stream_with_context(stream_template('video_stream.html',message=message)))
    # return app.response_class(stream_with_context(stream_template('video_stream.html',message=message)))
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen_messages():
#     global message
#     global t
#
#     if t == 0:
#         message = choices(Current_emotion)
#         t = 5
#
# @app.route('/message_feed')
# def message_feed():
#     return Response(gen_messages())


@app.route('/video_stream', methods=["POST", "GET"])
def video_stream():
    request_method = request.method
    # global message
    # global t
    #
    # if t == 0:
    #     message = choices(Current_emotion)
    #     t = 5
    #
    # if request_method == 'GET':
    #     request.args.get()
    #     mes = request.form.get()
    #     print(mes)
    # params = {
    #     'thing1': request.values.get('thing1'),
    #     'thing2': request.get_json().get('thing2')
    # }
    # return render_template('video_stream.html',request_method = request_method, data = message)
    return render_template('video_stream.html',message = message)

########         Image         ##############
# image.html --> predict.html
# image.html inherited for base.html
@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # try:
    image = request.files['select_file']
    # except KeyError as e:
    #     print(e)
    image.save('static/file.jpg')
    image = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 3)

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = image[y:y + h, x:x + w]

    cv2.imwrite('static/after.jpg', image)
    try:
        cv2.imwrite('static/cropped.jpg', cropped)
    except:
        pass

    try:
        img = cv2.imread('static/cropped.jpg', 0)
    except:
        img = cv2.imread('static/file.jpg', 0)

    img = cv2.resize(img, (48, 48))
    img = img / 255

    img = img.reshape(1, 48, 48, 1)

    model = load_model('D:cnn_model200_32.h5')

    pred = model.predict(img)

    label_map = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    pred = np.argmax(pred)
    final_pred = label_map[pred]

    return render_template('predict.html', data=final_pred)

    # return render_template('predict_image.html')




if __name__ == '__main__':
    app.run(debug=True)