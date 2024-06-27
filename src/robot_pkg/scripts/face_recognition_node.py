#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import face_recognition
import numpy as np

# Load known face images and encode them
daojing_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/daojing.jpg")
daojing_face_encoding = face_recognition.face_encodings(daojing_image)[0]

jielun_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/jielun.jpeg")
jielun_face_encoding = face_recognition.face_encodings(jielun_image)[0]

jinseng_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/jinseng.jpeg")
jinseng_face_encoding = face_recognition.face_encodings(jinseng_image)[0]

yuxuan_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/yuxuan.jpg")
yuxuan_face_encoding = face_recognition.face_encodings(yuxuan_image)[0]

uwais_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/uwais.jpeg")
uwais_face_encoding = face_recognition.face_encodings(uwais_image)[0]

xuyang_image = face_recognition.load_image_file("/home/mustar/robot_ws/src/robot_pkg/src/images/xuyang.jpeg")
xuyang_face_encoding = face_recognition.face_encodings(xuyang_image)[0]

known_face_encodings = [daojing_face_encoding, jielun_face_encoding, jinseng_face_encoding,  yuxuan_face_encoding, uwais_face_encoding, xuyang_face_encoding]
known_face_names = ["daojing", "jielun", "jinseng", "yuxuan", "uwais", "xuyang"]

# Function to convert face distance to a confidence score
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        return 0.0
    else:
        return (1.0 - face_distance / face_match_threshold) * 100

def image_callback(msg):
    try:
        # Convert the ROS Image message to a NumPy array
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        rospy.loginfo(f"Array shape before decoding: {np_arr.shape}")
        
        # Reshape the array and make a writable copy of the frame
        frame = np_arr.reshape((msg.height, msg.width, 3)).copy()

    except Exception as e:
        rospy.logerr(f"Error converting ROS Image message to OpenCV2: {e}")
        return

    rospy.loginfo(f"Decoded frame shape: {frame.shape}")

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_names = []

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding) # Compare the face encoding with the known face encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding) # Calculate the face distance to know how similar the faces are

        # Find the best match
        best_match_index = np.argmin(face_distances)
        name = "Unknown"
        confidence = 0.0

        # If the best match is found, get the name and confidence score
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            confidence = face_distance_to_conf(face_distances[best_match_index])
            recognized_names.append(name)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw the name of the detected person
        cv2.putText(frame, f"{name} ({confidence:.2f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # If multiple faces are detected, skip publishing the recognized names
    if len(face_locations) > 1:
        rospy.logwarn("Multiple faces detected. Skipping publishing.")

    # Publish the recognized names if there is exactly one face
    if recognized_names and len(recognized_names) == 1:
        pub.publish(String(recognized_names[0]))

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(1)

def main():
    # Declare a global publisher variable
    global pub
    # Create a node with the name 'face_recognition_node'
    rospy.init_node('face_recognition_node', anonymous=True)
    # Subscribe to the '/usb_cam/image_raw' topic which receives raw images from the USB camera
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    # Create a publisher that will publish messages to the '/recognized_face' topic
    pub = rospy.Publisher('/recognized_face', String, queue_size=10)
    # Keep the node running and processing callbacks
    rospy.spin()

if __name__ == '__main__':
    main()