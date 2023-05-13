import os
import cv2
import json
import re

# Path to the dataset/raw folder
json_folder = 'datasets/dmd/Raw'

# Create output directory if it doesn't exist
output_dir = 'datasets/dmd/Labels'
os.makedirs(output_dir, exist_ok=True)


# Function to sanitize folder names
def sanitize_folder_name(folder_name):
    # Replace invalid characters with underscores
    folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
    # Remove leading and trailing whitespaces
    folder_name = folder_name.strip()
    return folder_name

# Process each JSON file in the folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        json_path = os.path.join(json_folder, filename)

        # Load JSON data
        with open(json_path) as json_file:
            data = json.load(json_file)

            # Set video_path based on the JSON filename
            video_filename = filename.replace('ann_drowsiness.json', 'face.mp4')
            video_path = os.path.join(json_folder, video_filename)

            # Process each action in the JSON data
            for action_key, action_value in data['openlabel']['actions'].items():
                action_type = action_value['type']
                action_type_folder = sanitize_folder_name(action_type)
                frame_intervals = action_value['frame_intervals']

                # Create subfolder for the action type if it doesn't exist
                action_folder = os.path.join(output_dir, action_type_folder)
                os.makedirs(action_folder, exist_ok=True)

                # Process frame intervals
                for interval in frame_intervals:
                    frame_start = interval['frame_start']
                    frame_end = interval['frame_end']

                    video = cv2.VideoCapture(video_path)

                    # Set video frame position to the start frame
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

                    # Create video writer for the frame interval
                    output_path = os.path.join(action_folder, f'{frame_start}_{frame_end}.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = video.get(cv2.CAP_PROP_FPS)
                    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    # Read and write frames until the end frame is reached
                    for frame_num in range(frame_start, frame_end + 1):
                        ret, frame = video.read()
                        if ret:
                            writer.write(frame)
                        else:
                            break

                    # Release video resources
                    video.release()
                    writer.release()

                    print(f'Saved video: {output_path}')




