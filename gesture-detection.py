import cv2
import numpy as np
import math
import time

class GestureController:
    def __init__(self):
        # Initialize gesture parameters
        self.prev_gesture = None
        self.gesture_start_time = time.time()
        self.GESTURE_HOLD_TIME = 0.5
        
        # ROI parameters
        self.roi_top = 100
        self.roi_bottom = 300
        self.roi_right = 300
        self.roi_left = 100
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
    def get_hull_defects(self, contour):
        # Get convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            return defects
        return None
        
    def get_gesture_direction(self, contour, defects):
        if defects is None:
            return "UNKNOWN"
            
        # Get contour moments for center point
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            return "UNKNOWN"
            
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        
        # Get extreme points
        top = tuple(contour[contour[:, :, 1].argmin()][0])
        bottom = tuple(contour[contour[:, :, 1].argmax()][0])
        left = tuple(contour[contour[:, :, 0].argmin()][0])
        right = tuple(contour[contour[:, :, 0].argmax()][0])
        
        # Count valid defects
        count = 0
        gesture = "UNKNOWN"
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate angle between fingers
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            angle = math.degrees(math.acos((b**2 + c**2 - a**2)/(2*b*c)))
            
            # Filter defects based on angle
            if angle <= 90:
                count += 1
        
        # Determine gesture based on defect count and extreme points
        if count == 0:  # Pointing gesture
            if top[1] < cy - 40:  # Point up
                gesture = "FORWARD"
            elif bottom[1] > cy + 40:  # Point down
                gesture = "BACKWARD"
            elif right[0] > cx + 40:  # Point right
                gesture = "RIGHT"
            elif left[0] < cx - 40:  # Point left
                gesture = "LEFT"
        elif count >= 4:  # Open palm
            gesture = "STOP"
            
        return gesture
        
    def process_frame(self, frame):
        # Define region of interest
        roi = frame[self.roi_top:self.roi_bottom, self.roi_left:self.roi_right]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(roi)
        mask = cv2.bitwise_and(mask, fg_mask)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        gesture = "UNKNOWN"
        if contours:
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(contour) > 1000:  # Minimum area threshold
                defects = self.get_hull_defects(contour)
                gesture = self.get_gesture_direction(contour, defects)
                
                # Draw contour and convex hull
                cv2.drawContours(roi, [contour], -1, (0,255,0), 2)
                hull = cv2.convexHull(contour)
                cv2.drawContours(roi, [hull], -1, (0,0,255), 2)
        
        # Implement gesture debouncing
        if gesture != self.prev_gesture:
            self.gesture_start_time = time.time()
        elif time.time() - self.gesture_start_time < self.GESTURE_HOLD_TIME:
            gesture = self.prev_gesture
            
        self.prev_gesture = gesture
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (self.roi_left, self.roi_top), 
                     (self.roi_right, self.roi_bottom), (0,255,0), 2)
        
        return frame, mask, gesture

def main():
    cap = cv2.VideoCapture(0)
    controller = GestureController()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        frame, mask, gesture = controller.process_frame(frame)
        
        # Display gesture
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Display frames
        cv2.imshow('Robot Control Gestures', frame)
        cv2.imshow('Mask', mask)
        
        # Here you would implement robot control based on gesture
        # Example: send_command_to_robot(gesture)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()