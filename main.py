import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip
from deepface import DeepFace
from pydub import AudioSegment
from pydub.playback import play
import torch
import torchvision.transforms as transforms
from transformers import pipeline


# 1. Face Detection and Emotion Analysis (Using DeepFace)

def extract_frames(video_path, frame_rate=1):
    """Extract frames from video at a given frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps / frame_rate)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frames.append(frame)
    cap.release()
    return frames


def detect_emotions_in_frames(frames):
    """Detect emotions in each frame using DeepFace."""
    emotions = []
    for frame in frames:
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions.append(analysis['dominant_emotion'])
        except:
            emotions.append("no face detected")
    return emotions


# 2. Audio-Based Emotion Detection (Using librosa and a pre-trained model)
# Note: Requires a pre-trained audio emotion model or pre-processed dataset.

def extract_audio_from_video(video_path):
    """Extract audio from video file."""
    video = VideoFileClip(video_path)
    audio_path = video_path.replace(".mp4", ".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path


def get_audio_emotion(audio_path, model="speech_emotion_recognition"):
    """Detect emotion from audio."""
    # Load audio and preprocess
    y, sr = librosa.load(audio_path)
    # Here we assume a pre-trained model for audio emotion, which should be loaded and applied here.
    # Placeholder code, replace with your model inference.
    audio_emotion = "neutral"  # Mock result; replace with model output
    return audio_emotion


# 3. Data Fusion and Multimodal Analysis

def combine_emotions(face_emotions, audio_emotion):
    """Combine detected emotions from face and audio analysis."""
    # Placeholder rule-based fusion: You can create a more sophisticated model for combining.
    combined_emotions = []
    for face_emotion in face_emotions:
        if face_emotion == "no face detected":
            combined_emotions.append(audio_emotion)
        else:
            combined_emotions.append(face_emotion)
    return combined_emotions


# 4. Run the System on a Sample Video

def run_emotion_detection(video_path):
    # Step 1: Extract Frames from Video
    frames = extract_frames(video_path, frame_rate=1)
    print(f"Extracted {len(frames)} frames for analysis.")

    # Step 2: Detect Emotions in Frames (Face-based)
    face_emotions = detect_emotions_in_frames(frames)
    print("Detected Face Emotions:", face_emotions)

    # Step 3: Extract Audio from Video and Analyze Emotion
    audio_path = extract_audio_from_video(video_path)
    audio_emotion = get_audio_emotion(audio_path)
    print("Detected Audio Emotion:", audio_emotion)

    # Step 4: Combine Results
    combined_emotions = combine_emotions(face_emotions, audio_emotion)
    print("Combined Emotions:", combined_emotions)
    return combined_emotions


# Example Usage
video_path = "sample_video.mp4"  # Replace with path to your video file
emotions = run_emotion_detection(video_path)
print("Final Detected Emotions per Frame:", emotions)
