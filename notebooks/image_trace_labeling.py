import os
import csv
import glob
import cv2

IMAGE_FOLDER = "test_set/"
OUTPUT_CSV = "annotated_paths.csv"
WINDOW_NAME = "Image Path Annotation"

IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]

POINT_RADIUS = 20
LINE_THICKNESS = 10
START_HEIGHT_RATIO = 0.95

current_points = []
current_image = None
display_image = None


def add_starting_point(image):
    height, width = image.shape[:2]
    return [(width // 2, int(height * START_HEIGHT_RATIO))]


def get_display_point(point, image):
    return (point[0], min(point[1], image.shape[0] - 1))


def draw_points(image, points):
    for idx, point in enumerate(points):
        draw_point = get_display_point(point, image)
        cv2.circle(image, draw_point, POINT_RADIUS, (0, 0, 255), -1)

        if idx > 0:
            previous_draw_point = get_display_point(points[idx - 1], image)
            cv2.line(
                image,
                previous_draw_point,
                draw_point,
                (0, 255, 0),
                LINE_THICKNESS
            )


def mouse_callback(event, x, y, flags, param):
    global current_points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

        draw_point = get_display_point(current_points[-1], display_image)
        cv2.circle(display_image, draw_point, POINT_RADIUS, (0, 0, 255), -1)

        if len(current_points) > 1:
            previous_draw_point = get_display_point(current_points[-2], display_image)
            cv2.line(
                display_image,
                previous_draw_point,
                draw_point,
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
    print(" - A starting point is automatically added at (image_width / 2, 95% image_height)")
    print(" - Left click to add path points")
    print(" - Press Enter to save current image annotations and move to next image")
    print(" - After pressing Enter, type a text prompt in the terminal")
    print(" - Press 'r' to reset points for current image back to the starting point")
    print(" - Press 'q' or Esc to quit early")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1200, 800)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    for image_path in image_files:
        current_image = cv2.imread(image_path)
        if current_image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue

        display_image = current_image.copy()
        current_points = add_starting_point(current_image)
        draw_points(display_image, current_points)
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
                display_image = current_image.copy()
                current_points = add_starting_point(current_image)
                draw_points(display_image, current_points)
                print(f"Reset points for {image_name} back to starting point")

            elif key == ord('q') or key == 27:
                print("Exiting early.")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print(f"Done. All annotations saved to: {output_csv}")


if __name__ == "__main__":
    annotate_images(IMAGE_FOLDER, OUTPUT_CSV)
