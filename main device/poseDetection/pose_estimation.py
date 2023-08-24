
"""Main script to run pose classification and pose estimation."""
import sys
import time

import cv2
import utils
import socket
import json

from helperTools import Movenet, Classifier

server_address = ('localhost', 12345)

def run(estimation_model: str, classification_model: str,
        label_file: str, camera_id: int, width: int, height: int, client_socket: socket) -> None:


  # Initialize the pose estimator selected.
  #if estimation_model in ['movenet_lightning', 'movenet_thunder']:
  pose_detector = Movenet(estimation_model)
    #sys.exit('ERROR: Model is not supported.')

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 30  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 3
  font_thickness = 1
  classification_results_to_show = 3
  fps_avg_frame_count = 10
  keypoint_detection_threshold_for_classifier = 0.1
  classifier = None

  # Initialize the classification model
  if classification_model:
    classifier = Classifier(classification_model, label_file)
    classification_results_to_show = min(classification_results_to_show,
                                         len(classifier.pose_class_names))

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)


    # Run pose estimation using a SinglePose model, and wrap the result in an
    # array.
    list_persons = [pose_detector.detect(image)]

    # Draw keypoints and edges on input image
    image = utils.visualize(image, list_persons)

    if classifier:
      # Check if all keypoints are detected before running the classifier.
      # If there's a keypoint below the threshold, show an error.
      person = list_persons[0]
      min_score = min([keypoint.score for keypoint in person.keypoints])
      if min_score < keypoint_detection_threshold_for_classifier:
        error_text = 'Some keypoints are not detected.'
        text_location = (left_margin, 2 * row_size)
        cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
        error_text = 'Make sure the person is fully visible in the camera.'
        text_location = (left_margin, 3 * row_size)
        cv2.putText(image, error_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
      else:
        # Run pose classification
        prob_list = classifier.classify_pose(person)


        max_index = None
        max_value = float('-inf')

        # Show classification results on the image
        # check whether there is a fall in the list
        for i in range(classification_results_to_show):
          class_name = prob_list[i].label
          probability = round(prob_list[i].score, 2)
          if probability > max_value:
            max_value = probability
            max_index = i

          #plot on frame
          result_text = class_name + ' (' + str(probability) + ')'
          text_location = (left_margin, (i + 2) * row_size)
          cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)

        #check whether the max value found is a fall
        if prob_list[max_index].label == "Fall":
          #create json
          message = json.dumps({"Id": 2, "message":1, "intent":1})
          #send message to mailroom to alert for fall
          client_socket.sendto(message.encode(), server_address)




    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps))
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow(estimation_model, image)

  cap.release()
  cv2.destroyAllWindows()


def main():
    model = "movenet_lightning"
    classifier = "pose_estimation.tflite" #Pose detection model
    label_file = "labels.txt"
    cameraID = 0
    frameWidth = 640
    frameHeight = 380

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    # this is left as function within main to allow for threading to be added easily
    run(model, classifier, label_file, int(cameraID), frameWidth, frameHeight, client_socket)

    client_socket.close()


if __name__ == '__main__':
  main()
