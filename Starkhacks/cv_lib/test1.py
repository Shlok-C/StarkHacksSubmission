import cv2
import numpy as np

# --- CONFIGURATION ---
# Set these to the physical dimensions of the rectangle you taped to the table (in mm)
# We will map 1mm to 1 pixel for the output window to keep it simple.
RECT_WIDTH_MM = 300 
RECT_HEIGHT_MM = 200

# Global list to store mouse clicks
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    # Listen for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Recorded point {len(clicked_points)}: Pixel ({x}, {y})")

def main():
    global clicked_points
    
    # Open the Logitech Brio feed (usually index 0 or 1)
    cap = cv2.VideoCapture(1)
    
    # Set resolution to 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow("Brio Live Feed")
    cv2.setMouseCallback("Brio Live Feed", mouse_callback)

    print("--- HOMOGRAPHY TEST ---")
    print("Click the 4 corners of your physical grid/rectangle in this EXACT order:")
    print("1. Top-Left  2. Top-Right  3. Bottom-Left  4. Bottom-Right")
    print("Press 'R' to reset clicks. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check camera connection.")
            break

        # Draw a red circle at every clicked point
        for pt in clicked_points:
            cv2.circle(frame, (pt[0], pt[1]), 5, (0, 0, 255), -1)

        # Once 4 points are clicked, perform the perspective warp
        if len(clicked_points) == 4:
            # Source points (the pixels you clicked on the angled camera feed)
            pts_src = np.float32(clicked_points)
            
            # Destination points (a perfect flat rectangle representing real physical dimensions)
            pts_dst = np.float32([
                [0, 0], 
                [RECT_WIDTH_MM, 0], 
                [0, RECT_HEIGHT_MM], 
                [RECT_WIDTH_MM, RECT_HEIGHT_MM]
            ])

            # Calculate the 3x3 Homography Matrix
            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            
            # Warp the image to the flat perspective
            warped = cv2.warpPerspective(frame, matrix, (RECT_WIDTH_MM, RECT_HEIGHT_MM))

            cv2.imshow("Un-Warped Top-Down View", warped)

        # Show the live feed
        cv2.imshow("Brio Live Feed", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting points...")
            clicked_points = []
            cv2.destroyWindow("Un-Warped Top-Down View")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()