import imageio
from PIL import ImageGrab
import time

# Define the region to capture (coordinates are in pixels)
bbox = (0, 0, 640, 480)  # (left, top, right, bottom)

# Define the frame rate and delay between captures
frame_rate = 15
capture_delay = 1 / frame_rate

# Initialize an empty list to store frames
frames = []

while True:
    # Capture the screen within the defined region
    screenshot = ImageGrab.grab(bbox)

    # Convert the screenshot to RGB mode (if it's not already)
    screenshot = screenshot.convert('RGB')

    # Append the captured frame to the list of frames
    frames.append(screenshot)

    # Add a delay between captures to maintain the frame rate
    time.sleep(capture_delay)

    # Break the loop after capturing a certain number of frames (optional)
    if len(frames) == 100:
        break

# Save the list of frames as a video file
imageio.mimsave('output.avi', frames, fps=frame_rate)
