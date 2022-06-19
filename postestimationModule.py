import cv2
import mediapipe as mp
import time



class PoseDetector():

    def __init__(self,mode = False,model_complexity = 1, smooth_lm = True, segmentEnable = True, detectionCon = 0.5,TrackeCon = 0.5):

        self.mode = mode
        self.segmentEnable = segmentEnable
        self.model_complexity = model_complexity
        self.smooth_lm = smooth_lm
        self.detectionCon = detectionCon
        self.TrackeCon = TrackeCon

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode,self.model_complexity,self.smooth_lm,self.segmentEnable,self.detectionCon,self.TrackeCon)


        
    def findPose(self,image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image)
        
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())
        return image
    
    def find_marks(self,image):
        marks = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = image.shape
            
                cx, cy = int(lm.x * w), int(lm.y * h)
                marks.append([id,cx,cy])
        return marks

def main():
    ptime = 0 
    cap = cv2.VideoCapture(0)
    PoseTracker = PoseDetector()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
      
            continue
        
        image = PoseTracker.findPose(image)
        marks = PoseTracker.find_marks(image)
            

        if len(marks) != 0:

            print(marks)    


        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        # Flip the image horizontally for a selfie-view display.
        cv2.putText(image,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),4)
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break  

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

