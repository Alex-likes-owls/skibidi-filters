import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class FaceFilterApp:
  
    def __init__(self, accessories, max_num_faces=1, window_name="Face Filter"):
        self.accessories = accessories
        self.window_name = window_name
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cap = cv2.VideoCapture(0)

    def run(self):
        with self.face_mesh as face_mesh:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = face_mesh.process(rgb)
                rgb.flags.writeable = True
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        lm = face_landmarks.landmark
                        for accessory in self.accessories:
                            frame = accessory.apply(frame, lm, w, h)

                cv2.imshow(self.window_name, cv2.flip(frame, 1))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        self.cap.release()
        cv2.destroyAllWindows()
