import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import glob  # For loading multiple files

url = "http://192.168.1.10:8080/video"

def preprocess_image(image):
    """Convert image to grayscale and resize."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (800, 800))  # Adjust size as needed
    return resized

def load_images(folder_path):
    """Load and preprocess all images from a folder."""
    images = []
    image_names = []
    for file_path in glob.glob(f"{folder_path}/*.jpg"):  # Adjust extension as needed
        image = cv2.imread(file_path)
        if image is not None:
            preprocessed_image = preprocess_image(image)
            images.append(preprocessed_image)
            image_names.append(file_path.split('/')[-1])  # Extract file name
    return images, image_names

def compare_with_images(frame, images, image_names):
    """Compare the frame with a list of images and return the highest similarity."""
    best_score = 0
    best_match_name = "Unknown"
    for image, name in zip(images, image_names):
        score, _ = compare_ssim(image, frame, full=True)
        if score > best_score:
            best_score = score
            best_match_name = name
    return best_score, best_match_name

def main():
    # Introductory messages
    print("SIDDAGANGA INSTITUTE OF TECHNOLOGY")
    print("By Mechanical Engineers")
    print("Welcome to FITWEL TOOLS & FORGINGS")
    print("Quality Inspection")

    # Input folder paths
    cam_1_perfect_folder = input("Cam_1 Enter the Folder of PERFECT PARTS: ")
    cam_1_defect_folder = input("Cam_1 Enter the Folder of DEFECTED PARTS: ")
    cam_2_perfect_folder = input("Cam_2 Enter the Folder of PERFECT PARTS: ")
    cam_2_defect_folder = input("Cam_2 Enter the Folder of DEFECTED PARTS: ")

    # Load images for both cameras
    perfect_images1, cam_1_perfect_names = load_images(cam_1_perfect_folder)
    defect_images1, cam_1_defect_names = load_images(cam_1_defect_folder)
    perfect_images2, cam_2_perfect_names = load_images(cam_2_perfect_folder)
    defect_images2, cam_2_defect_names = load_images(cam_2_defect_folder)

    # Check if images were loaded
    if not perfect_images1 and not defect_images1:
        print("No images found in the specified folders for Cam_1. Exiting.")
        return

    if not perfect_images2 and not defect_images2:
        print("No images found in the specified folders for Cam_2. Exiting.")
        return

    # Open video capture
    cap1 = cv2.VideoCapture(0)  # Default camera
    cap2 = cv2.VideoCapture(url)  # IP camera URL

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1:
            print("Failed to capture video from Cam_1.")
            break
        if not ret2:
            print("Failed to capture video from Cam_2.")
            break

        # Preprocess live frames
        processed_frame1 = preprocess_image(frame1)
        processed_frame2 = preprocess_image(frame2)

        # Compare with perfect and defect images
        perfect_score1, perfect_match_name1 = compare_with_images(processed_frame1, perfect_images1, cam_1_perfect_names)
        defect_score1, defect_match_name1 = compare_with_images(processed_frame1, defect_images1, cam_1_defect_names)

        perfect_score2, perfect_match_name2 = compare_with_images(processed_frame2, perfect_images2, cam_2_perfect_names)
        defect_score2, defect_match_name2 = compare_with_images(processed_frame2, defect_images2, cam_2_defect_names)

        # Determine classification for Cam_1
        if perfect_score1 > defect_score1 and perfect_score1 > 0.7:
            label1 = f"Accept Top: {perfect_match_name1}"
            color1 = (0, 255, 0)  # Green for accept
        elif defect_score1 > perfect_score1 and defect_score1 > 0.7:
            label1 = f"Defect Top: {defect_match_name1}"
            color1 = (0, 0, 255)  # Red for defect
        else:
            label1 = "Top: Place Properly"
            color1 = (255, 255, 0)  # Yellow for unclassified

        # Determine classification for Cam_2
        if perfect_score2 > defect_score2 and perfect_score2 > 0.7:
            label2 = f"Accept Side: {perfect_match_name2}"
            color2 = (0, 255, 0)  # Green for accept
        elif defect_score2 > perfect_score2 and defect_score2 > 0.7:
            label2 = f"Defect Side: {defect_match_name2}"
            color2 = (0, 0, 255)  # Red for defect
        else:
            label2 = "Side: Place Properly"
            color2 = (255, 255, 0)  # Yellow for unclassified

        # Overlay text on frames
        cv2.putText(frame1, label1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)
        cv2.putText(frame2, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color2, 2)

        # Concatenate frames side by side
        frame1 = cv2.resize(frame1, (640, 480))  # Resize for consistent display
        frame2 = cv2.resize(frame2, (640, 480))  # Resize for consistent display
        combined_frame = cv2.hconcat([frame1, frame2])

        # Display combined feed
        cv2.imshow("Combined Live Feed", combined_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
