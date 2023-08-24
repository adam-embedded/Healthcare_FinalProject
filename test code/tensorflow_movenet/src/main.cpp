#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "movenet_tools.h"
//#include "classifier.h"

#include <iostream>
#include <map>


#include <cstdio>

const int num_keypoints = 17;
const float confidence_threshold = 0.2;

// The Output is a float32 tensor of shape [1, 1, 17, 3].

// The first two channels of the last dimension represents the yx coordinates (normalized
// to image frame, i.e. range in [0.0, 1.0]) of the 17 keypoints (in the order of: [nose,
// left eye, right eye, left ear, right ear, left shoulder, right shoulder, left elbow,
// right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee, left
// ankle, right ankle]).

// The third channel of the last dimension represents the prediction confidence scores of
// each keypoint, also in the range [0.0, 1.0].

const std::vector<std::pair<int, int>> connections = {
        {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7},
        {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12},
        {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

void draw_keypoints(cv::Mat &resized_image, float *output)
{
    int square_dim = resized_image.rows;

    for (int i = 0; i < num_keypoints; ++i) {
        float y = output[i * 3];
        float x = output[i * 3 + 1];
        float conf = output[i * 3 + 2];

        if (conf > confidence_threshold) {
            int img_x = static_cast<int>(x * square_dim);
            int img_y = static_cast<int>(y * square_dim);
            cv::circle(resized_image, cv::Point(img_x, img_y), 2, cv::Scalar(255, 200, 200), 1);
        }
    }

    // draw skeleton
    for (const auto &connection : connections) {
        int index1 = connection.first;
        int index2 = connection.second;
        float y1 = output[index1 * 3];
        float x1 = output[index1 * 3 + 1];
        float conf1 = output[index1 * 3 + 2];
        float y2 = output[index2 * 3];
        float x2 = output[index2 * 3 + 1];
        float conf2 = output[index2 * 3 + 2];

        if (conf1 > confidence_threshold && conf2 > confidence_threshold) {
            int img_x1 = static_cast<int>(x1 * square_dim);
            int img_y1 = static_cast<int>(y1 * square_dim);
            int img_x2 = static_cast<int>(x2 * square_dim);
            int img_y2 = static_cast<int>(y2 * square_dim);
            cv::line(resized_image, cv::Point(img_x1, img_y1), cv::Point(img_x2, img_y2), cv::Scalar(200, 200, 200), 1);
        }
    }
}

int main(int argc, char *argv[]) {

    // model from https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4
    // A convolutional neural network model that runs on RGB images and predicts human
    // joint locations of a single person. The model is designed to be run in the browser
    // using Tensorflow.js or on devices using TF Lite in real-time, targeting
    // movement/fitness activities. This variant: MoveNet.SinglePose.Lightning is a lower
    // capacity model (compared to MoveNet.SinglePose.Thunder) that can run >50FPS on most
    // modern laptops while achieving good performance.
    std::string model_file = "/home/adam/development/university/tensorflow_movenet/movenet_lightning.tflite";//"/home/adam/CLionProjects/fall_detection_2/assets/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite";
    std::string classifierModel_file = "/home/adam/PycharmProjects/fall_detection_movenet/pose_classifier.tflite";
    std::string label_file = "/home/adam/PycharmProjects/fall_detection_movenet/pose_labels.txt";

    printf("%s",cv::getBuildInformation().c_str());


    std::map<std::string, std::string> arguments;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg.find("--") == 0) {
            size_t equal_sign_pos = arg.find("=");
            std::string key = arg.substr(0, equal_sign_pos);
            std::string value = equal_sign_pos != std::string::npos ? arg.substr(equal_sign_pos + 1) : "";

            arguments[key] = value;
        }
    }

    if (arguments.count("--model")) {
        model_file = arguments["--model"];
    }

    if (arguments.count("--label")) {
        label_file = arguments["--label"];
    }

    if (arguments.count("--classifier")) {
        classifierModel_file = arguments["--classifier"];
    }

    // initialise pose detection
    MovenetTools movenet(model_file, classifierModel_file, label_file);

    // initialise classification
    //Classifier classifier(classifierModel_file, label_file);

    cv::VideoCapture video;

    int desiredFPS = 10;
    int delay = int((1.0f / desiredFPS)*1000);

    printf("Delay: %d\n", delay);

    video.open(0);

    // Try to use HD resolution (or closest resolution
    auto resW = video.get(cv::CAP_PROP_FRAME_WIDTH);
    auto resH = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
    video.set(cv::CAP_PROP_FRAME_WIDTH, 640); //640
    video.set(cv::CAP_PROP_FRAME_HEIGHT, 360); //360
    resW = video.get(cv::CAP_PROP_FRAME_WIDTH);
    resH = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

    cv::Mat frame;

    if (!video.isOpened()) {
        std::cout << "Can't open the webcam stream: " << std::endl;
        return -1;
    }

    //set up for FPS counter
    double fps;
    cv::TickMeter timer;



    while (true) {

        timer.start();

        video >> frame;

        cv::flip(frame, frame,1);

        movenet.Run(frame );//, &classifier


        timer.stop();
        fps = 1.0 / timer.getTimeSec();
        timer.reset();

        cv::resize(frame, frame, cv::Size(1920,1080));

        // Put FPS counter on screen
        cv::putText(frame, "FPS: " +std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);

        //cv::waitKey(delay);

        if (cv::waitKey(10) == 27){
            break;
        }

        imshow("Output", frame);

    }

        video.release();

        cv::destroyAllWindows();

    return 0;
}