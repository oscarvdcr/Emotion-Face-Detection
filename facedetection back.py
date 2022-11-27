import cv2
from deepface import DeepFace
import threading
from functools import partial

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder

class MainScreen(Screen):
    pass
class Manager(ScreenManager):
    pass
class facedetection(App):
    def build(self):
        self.window = GridLayout()
        self.window.rows = 2
        self.image = Image(
            source="logo.png"
        )
        self.button = Button(
            text = "Start",
            size_hint = (1,0.3),
            bold = True,
            background_color = '#AAAAAA'
        )
        self.label = Label(
            text = "Neutral"
        )
        self.button.bind(on_press= self.pressed)
        self.window.add_widget(self.image)
        self.window.add_widget(self.button)
        face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  #aconseguir .xml
        self.face_cascade = cv2.CascadeClassifier()  
        if not self.face_cascade.load(cv2.samples.findFile(face_cascade_name)):  #cas error
            print("Error loading xml file")
        self.video=cv2.VideoCapture(0)
        
        
    
        return self.window
    def pressed(self, instance):
        threading.Thread(target=self.doit, daemon=True).start()

    def doit(self):
        self.do_vid = True  # flag to stop loop

        cv2.namedWindow('Hidden', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.resizeWindow('Hidden', 0, 0)
        cam = cv2.VideoCapture(0)

        while (self.do_vid):
            ret, frame = cam.read()
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #escala grisos
            face=self.face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5)

            for x,y,w,h in face:
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)  #cuadrat a cara
            #trobar errors
            try: 
                self.button.text = DeepFace.analyze(frame,actions= ['emotion'])['dominant_emotion']  #guardem output de deepface
                #posem en pantalla la emocio dominant
            except:
                self.button.text = "no face"
            Clock.schedule_once(partial(self.display_frame, frame))

            cv2.imshow('Hidden', frame)
            cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()

    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False

    def display_frame(self, frame, dt):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        self.image.texture = texture
    
facedetection().run()