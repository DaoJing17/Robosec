#!/usr/bin/env python3

import os
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from speechbrain.inference import SpeakerRecognition
import rospy
from std_msgs.msg import String
import threading
from gtts import gTTS
from gtts.tokenizer.pre_processors import abbreviations, end_of_line
from pygame import mixer

# Initialize the speaker recognition model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# Global variables to manage state
recognized_face = None
login_in_progress = False
lock = threading.Lock()

# Directories for storing voices and transcriptions
voices_dir = '/home/mustar/robot_ws/src/robot_pkg/src/voices/'
transcriptions_dir = '/home/mustar/robot_ws/src/robot_pkg/src/transcriptions/'

# Record audio from the microphone
def record_audio(duration=7, sample_rate=16000):
    
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording stopped.")
    return audio_data, sample_rate

# Save the audio data to a file
def save_audio_file(audio_data, sample_rate, filename):
    
    file_path = os.path.join(voices_dir, filename)
    sf.write(file_path, audio_data, sample_rate)
    print(f"Recording saved to {file_path}.")

# Extract embedding from an audio file using the pre-trained model
def extract_embedding(file_path):
    
    signal = model.load_audio(file_path)
    embeddings = model.encode_batch(signal)
    return embeddings

# Compare test embedding with base embeddings to find the closest speaker
def compare_embeddings(base_embeddings, test_embedding):
    scores = {}
    for person, embedding in base_embeddings.items():
        score = model.similarity(test_embedding, embedding)
        scores[person] = score.item()
    return scores

# Identify the speaker by creating a temporary audio file, extracting embeddings, and comparing.
def identify_speaker(audio_data, sample_rate, base_embeddings):
    
    temp_path = 'temp_identify.wav'
    sf.write(temp_path, audio_data, sample_rate)
    test_embedding = extract_embedding(temp_path)
    os.remove(temp_path)
    scores = compare_embeddings(base_embeddings, test_embedding)
    identified_person = max(scores, key=scores.get)
    return identified_person, scores

# Transcribe audio data using Google's Speech Recognition
def recognize_speech(audio_data, sample_rate):
    
    recognizer = sr.Recognizer()
    temp_path = 'temp_recognize.wav'
    sf.write(temp_path, audio_data, sample_rate)
    with sr.AudioFile(temp_path) as source:
        audio_data = recognizer.record(source)
    os.remove(temp_path)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio_data)
        print("Transcription: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

# Updates the recognized_face variable if no login is currently in progress.
def handle_recognized_face(msg):
    global recognized_face, login_in_progress
    with lock:
        if not login_in_progress:
            # Update the recognized_face with the data from the message
            recognized_face = msg.data

# Generates an audio message based on whether the identified person was recognized or not and plays the sound.
def speak(identified_person):
    if identified_person:
        text = f"Login successful. Welcome {identified_person}"
    else:
        text = f"Login access denied."

    # Use gTTS to convert the text to speech
    tts = gTTS(text, slow=False, pre_processor_funcs=[abbreviations, end_of_line])
    # Save the audio in an mp3 file
    tts.save('login_status.mp3')

    # Initialize the mixer module for playing the mp3 file
    mixer.init()
    mixer.music.load("login_status.mp3")
    mixer.music.play()

    # Remove the mp3 file after playing to avoid clutter
    os.remove("login_status.mp3")

def main():
    global recognized_face, login_in_progress
    
    # Create node with the name 'voice_biometric_node
    rospy.init_node('voice_biometric_node', anonymous=True)
    # Subscribe to the '/recognized_face' topic which receives recognized face data
    rospy.Subscriber('/recognized_face', String, handle_recognized_face)
    
    print("Waiting for recognized face...")
    while not rospy.is_shutdown():
        if recognized_face is not None:
            with lock: # Ensure only one face is processed at a time
                login_in_progress = True  # Set login in progress to True
                user_name = recognized_face  # Store the recognized face data
                recognized_face = None  # Reset recognized face

                existing_audio_file = voices_dir + f"{user_name}.wav"
                 # Check if the user already exists
                if os.path.exists(existing_audio_file):
                    choice = input(f"Username '{user_name}' exists. Type 'Start' and press Enter to start recording for voice biometrics: ").strip().lower()
                    if choice == 'start':
                        audio_data, sample_rate = record_audio()  # Record new audio
                        base_embeddings = {user_name: extract_embedding(existing_audio_file)}  # Extract embedding from existing audio
                        identified_person, scores = identify_speaker(audio_data, sample_rate, base_embeddings)  # Identify speaker
                        highest_score = max(scores.values())  # Get the highest score

                        if highest_score > 0.60:
                            print(f"Login granted for {identified_person} with a score of {highest_score:.2f}")
                            speak(identified_person)  # Announce login success
                        else:
                            print(f"Login denied. Score: {highest_score:.2f}")
                            speak(None)  # Announce login failure
                        
                        os.remove(f"{user_name}.wav")  # Remove temporary audio file
                    else:
                        print("Recording canceled.")
                
                # If user does not exist
                else:
                    choice = input(f"Username '{user_name}' not found. Type 'Start' and press Enter to start recording for voice biometrics: ").strip().lower()
                    if choice == 'start':
                        audio_data, sample_rate = record_audio()  # Record new audio
                        transcription = recognize_speech(audio_data, sample_rate)  # Transcribe audio
                        if transcription: # If transcription is successful
                            choice = input("Type 'Proceed' to save the recording and transcription or 'Rerecord' to record again: ").strip().lower()
                            
                            if choice == 'proceed': # If user enters 'proceed'
                                save_audio_file(audio_data, sample_rate, f"{user_name}.wav")  # Save audio file
                                file_path = os.path.join(transcriptions_dir, f"{user_name}.txt")
                                with open(file_path, 'w') as file:
                                    file.write(transcription)  # Save transcription
                                print(f"Files saved as {user_name}.wav and {user_name}.txt")
                            elif choice == 'rerecord': # If user enters 'rerecord'
                                continue  # Rerecord if needed
                        else:
                            print("Failed to transcribe. Please try recording again.")
                            continue
                    else:
                        print("Recording canceled.")

            with lock:
                login_in_progress = False # Reset login in progress to False
        rospy.sleep(1)

if __name__ == "__main__":
    main()