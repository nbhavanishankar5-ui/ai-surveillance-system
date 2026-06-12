import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('surveillance_model.h5')

def detect_suspicious_activity(video_path):
    """
    Detects suspicious activity in video feeds using deep learning
    """
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    alerts = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess frame
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        
        # Predict using CNN model
        prediction = model.predict(np.expand_dims(normalized, axis=0))
        confidence = prediction[0][0]
        
        # Flag suspicious activity
        if confidence > 0.5:
            alerts += 1
            cv2.putText(frame, f"ALERT: Suspicious Activity ({confidence:.2f})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('AI Surveillance System', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Video Analysis Complete")
    print(f"Total Frames: {frame_count}")
    print(f"Alerts Detected: {alerts}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 'input_video.mp4'
    
    detect_suspicious_activity(video_path)
