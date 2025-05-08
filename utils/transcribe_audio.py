import os

import whisper
import json
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


VIDEO_PATH = "../data/video.mp4"
AUDIO_PATH = "../data/audio.mp3"
TRANSCRIPT_PATH = "../data/transcript.json"


def extract_audio():

    try:
        with VideoFileClip(VIDEO_PATH) as video:
            video.audio.write_audiofile(
                AUDIO_PATH,
                codec='mp3',
                bitrate='192k',
                ffmpeg_params=[
                    '-ar', '16000',
                    '-ac', '1'
                ]
            )
        print(f"Audio extracted to {os.path.abspath(AUDIO_PATH)}")
        return True

    except Exception as e:
        print(f"Audio extraction failed: {str(e)}")
        print("\nTroubleshooting:")
        print(
            "1. Run this command to clean dependencies: pip uninstall moviepy imageio-ffmpeg -y && pip install moviepy==1.0.3 imageio[ffmpeg]")
        return False

def transcribe_audio():
    try:
        print("Loading Whisper model...")
        model = whisper.load_model("base")

        print("Transcribing audio...")
        result = model.transcribe(AUDIO_PATH)

        transcript = [{
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip()
        } for segment in result["segments"]]


        with open(TRANSCRIPT_PATH, 'w') as f:
            json.dump(transcript, f, indent=2)

        print(f"Transcript saved to {TRANSCRIPT_PATH}")
        return transcript

    except Exception as e:
        print(f" Transcription failed: {str(e)}")
        return None


if __name__ == "__main__":
    if extract_audio():
        transcript = transcribe_audio()
        if transcript:
            print("\nFirst 5 segments:")
            for seg in transcript[:5]:
                print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")