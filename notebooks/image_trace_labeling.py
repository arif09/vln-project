import os
import csv
import glob
import cv2

IMAGE_FOLDER = "test_set/"
OUTPUT_CSV = "annotated_paths.csv"
WINDOW_NAME = "Image Path Annotation"

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

POINT_RADIUS = 10
LINE_THICKNESS = 5

current_points = []
current_image = None
display_image = None


def mouse_callback(event, x, y, flags, param):
    global current_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

        cv2.circle(display_image, (x, y), POINT_RADIUS, (0, 0, 255), -1)

        if len(current_points) > 1:
            cv2.line(
                display_image,
                current_points[-2],
                current_points[-1],
                (0, 255, 0),
                LINE_THICKNESS
            )


def get_image_files(folder):
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(folder, ext)))
    image_files.sort()
    return image_files


def save_points_to_csv(csv_path, image_name, text_prompt, points):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["image_name", "text_prompt", "point_order", "x", "y"])

        for idx, (x, y) in enumerate(points, start=1):
            writer.writerow([image_name, text_prompt, idx, x, y])


def annotate_images(image_folder, output_csv):
    global current_points, current_image, display_image

    image_files = get_image_files(image_folder)

    if not image_files:
        print(f"No images found in folder: {image_folder}")
        return

    print("Instructions:")
    print(" - Left click to add path points")
    print(" - Press Enter to save current image annotations and move to next image")
    print(" - After pressing Enter, type a text prompt in the terminal")
    print(" - Press 'r' to reset points for current image")
    print(" - Press 'q' or Esc to quit early")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    for image_path in image_files:
        current_points = []

        current_image = cv2.imread(image_path)
        if current_image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue

        display_image = current_image.copy()
        image_name = os.path.basename(image_path)

        while True:
            cv2.imshow(WINDOW_NAME, display_image)
            key = cv2.waitKey(20) & 0xFF

            if key == 13:  # Enter
                cv2.imshow(WINDOW_NAME, display_image)
                cv2.waitKey(1)

                text_prompt = input(f"Enter text prompt for {image_name}: ").strip()
                save_points_to_csv(output_csv, image_name, text_prompt, current_points)

                print(f"Saved {len(current_points)} points and prompt for {image_name}")
                break

            elif key == ord('r'):
                current_points = []
                display_image = current_image.copy()
                print(f"Reset points for {image_name}")

            elif key == ord('q') or key == 27:
                print("Exiting early.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"Done. All annotations saved to: {output_csv}")


if __name__ == "__main__":
    annotate_images(IMAGE_FOLDER, OUTPUT_CSV)