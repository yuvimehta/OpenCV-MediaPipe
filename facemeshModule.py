 

import imghdr
import cv2 
import mediapipe as mp
import time 


class facemesh():


    def __init__(self,static_image_mode=False, max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
    
    
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    
        self.face_mesh  = self.mp_face_mesh.FaceMesh(self.static_image_mode,self.max_num_faces,self.refine_landmarks,self.min_detection_confidence,
        self.min_tracking_confidence)
        

    def findmesh(self,img):
        img.flags.writeable = False
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.result = self.face_mesh.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
        return img

    def drawmesh(self,img,thickness = 1,circle_radius = 1):
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.drawingSpec = self.mp_drawing.DrawingSpec(self.thickness ,self.circle_radius)
        if self.result.multi_face_landmarks:
            for face_landmarks in self.result.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return img


    def meshlm(self,img):
        meshlist = []
        if self.result.multi_face_landmarks:
        
            for face_landmarks in self.result.multi_face_landmarks:
            
                for id, lm in enumerate (face_landmarks.landmark):
                    ih,iw,ic = img.shape
                    x ,y = int(lm.x*iw), int(lm.y*ih)
                    #print(id, x,y)
                    meshlist.append([id, x, y])
                    #print(meshlist)
        return meshlist
                




    
def main():
    cap = cv2.VideoCapture(0)

    mesh = facemesh()
    while cap.isOpened():
        ret,frame = cap.read()
        
        frame = mesh.findmesh(frame)
        frame = mesh.drawmesh(frame)




        cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))


        #cv2.imshow("frame",frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()