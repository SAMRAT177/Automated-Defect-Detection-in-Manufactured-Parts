import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import glob  # For loading multiple files
url = "http://192.168.1.10:8080/video"

def preprocess_image(image):
    """Convert image to grayscale and resize."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (1000, 1000))  # Adjust size as needed
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
    # Load perfect and defective parts images
    print("SIDDAGANGA INSTITUTE OF TECHNOLOGY")
    print("By Mechanical Engineers")
    print("Welcome to FITWEl TOOLS & FORGINGS")
    print("Quality Inspection")
    perfect_folder = input("Enter the Folder of PERFECT PARTS: ")
    defect_folder = input("Enter the Folder of DEFECTED PARTS: ")

    perfect_images, perfect_names = load_images(perfect_folder)
    defect_images, defect_names = load_images(defect_folder)

    if not perfect_images and not defect_images:
        print("No images found in the specified folders. Exiting.")
        return

    # Open video capture
    cap = cv2.VideoCapture(url) # Use 0 for the default camera
    while True:
        ret, frame = cap.read(0)
        if not ret:
            print("Failed to capture video.")
            break

        # Preprocess live frame
        processed_frame = preprocess_image(frame)

        # Compare with perfect images
        perfect_score, perfect_match_name = compare_with_images(processed_frame, perfect_images, perfect_names)

        # Compare with defect images
        defect_score, defect_match_name = compare_with_images(processed_frame, defect_images, defect_names)

        # Determine the classification
        if perfect_score > defect_score and perfect_score > 0.7:
            label = f"Accept: {perfect_match_name}"
            color = (0, 255, 0)  # Green for accept
        elif defect_score > perfect_score and defect_score > 0.7:
            label = f"Defect: {defect_match_name}"
            color = (0, 0, 255)  # Red for defect
        else:
            label = "Place Properly....Loading....."
            color = (255, 255, 0)  # Yellow for unclassified

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

        # Display result on the live frame
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        # Show live feed
        cv2.imshow("Live Feed", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
