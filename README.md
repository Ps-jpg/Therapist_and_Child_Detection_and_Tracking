Here is the updated README with color codes for the text:

README

roject: Video Tracking using Roboflow and OpenCV

Overview

This project uses Roboflow's object detection API and OpenCV to track objects in videos. The project is designed to track people in videos and assign a unique ID to each person. The tracked objects are then displayed on the video with their corresponding IDs
Custom-Made model
Model Stats-
![image](https://github.com/user-attachments/assets/eff706cc-c55b-493b-b59d-1bbbf236b6a8)
Model Type: Roboflow 3.0 Object Detection (Accurate)
Checkpoint: COCOs
![results](https://github.com/user-attachments/assets/6e03f2ce-8ed1-4b0f-9029-995f2d18ef30)




Features

   Object detection using Roboflow's API
   Oject tracking using OpenCV's SORT algorithm
   Assignment of unique IDs to each tracked object
   Display of tracked objects with their IDs on the video
   Support for processing multiple video files

Requirements

   Python 3.x
   OpenCV 4.x
   Roboflow API key
   SORT algorithm implementation (included in the project)

     Usage

    Clone the repository and navigate to the project directory.
    Install the required dependencies using pip: pip install -r requirements.txt
    Place your video files in the videos directory.
    Run the project using the command: python main.py --file_name your_video_file.mp4 
    The processed video will be saved in the same directory with the suffix _out.mp4.

Command Line Arguments

    --file_name: Specify the video file to scan and track. If not provided, the project will process all video files in the videos directory.

Example Output

A sample output video is available at [[Drive link]](https://drive.google.com/drive/folders/1l0mmqzktha6Z5Mi82ijyJubSghCT0HJ5?usp=drive_link)

Code Structure


API Documentation

The project uses Roboflow's object detection API. The API documentation is available at Roboflow API Documentation.

Troubleshooting

If you are not able to install cuda then try cpu or pip install torch==2.1.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 try with if it still shows wheel failed download it from python site and manually build it.

Ensure that the video files are in the correct format (MP4 or AVI) and are placed in the videos directory.

to check gpu usage 
nvidia-smi
dont believe gpu usage shown in task manager its wrong idk why.

If you encounter any issues with the project, feel free to open an issue on the repository.
