import cv2
from matplotlib import image
import mediapipe as mp



class faceDetector():

    def __init__(self,detectionCon=0.5, modleSelc=0):
        self.detectionCon = detectionCon
        self.modleselc = modleSelc


        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_detection = self.mp_face_detection.FaceDetection (self.detectionCon,self.modleselc)

    def faceFinder(self,image):
        image.flags.writeable = False
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        self.results = self.face_detection.process(image)



        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  

        #if self.results.detections:
            #for detection in self.results.detections:
                #self.mp_drawing.draw_detection(image,detection)
        
        return image


       


def main():
    cap = cv2.VideoCapture(0)
    finder = faceDetector()
    while cap.isOpened():
        success,image = cap.read()
        image = finder.faceFinder(image)

        cv2.imshow('MediaPipe',cv2.flip(image,1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()            

    






if __name__ == "__main__":
    main()

            