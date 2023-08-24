#include "yolov8.h"

// Runs object detection on video stream then displays annotated results.

static const bool pose = false;

int main(int argc, char *argv[]) {
    // Parse the command line arguments
    // Must pass the model path as a command line argument to the executable
    if (argc < 2) {
        std::cout << "Error: Must specify the model path" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx" << std::endl;
        return -1;
    }

    if (argc > 3) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx" << std::endl;
    }

    // Ensure the onnx model exists
    const std::string onnxModelPath = argv[1];
    if (!doesFileExist(onnxModelPath)) {
        std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
        return -1;
    }

    // Create our YoloV8 engine
    // Use default probability threshold, nms threshold, and top k
    YoloV8 yoloV8(onnxModelPath, 0.65f);

    // Initialize the video stream
    cv::VideoCapture cap;
    // TODO: Replace this with your video source.
    // 0 is default webcam, but can replace with string such as "/dev/video0", or an RTSP stream URL
    cap.open(0);

    // Try to use HD resolution (or closest resolution
    auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 960); //960 1280
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 540); //540 720
    resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

    if (!cap.isOpened()) {
        throw std::runtime_error("Unable to open video capture!");
    }

    cv::Mat img;

    //set up for FPS counter
    double fps;
    cv::TickMeter timer;



    while (true) {
        timer.start();
        // Grab frame
        cap >> img;

        //cv::resize(img, img, cv::Size(512,288));

        if (img.empty()) {
            throw std::runtime_error("Unable to decode image from video stream.");
        }

        // Run inference
        const auto objects = yoloV8.detectObjects(img, pose);

        // Draw the bounding boxes on the image for object detection
        yoloV8.drawObjectLabels(img, objects);

        timer.stop();
        fps = 1.0 / timer.getTimeSec();
        timer.reset();

        cv::putText(img, "FPS: " +std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);


        cv::resize(img, img, cv::Size(1280,720));

        // Display the results
        cv::imshow("Object Detection", img);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    return 0;
}
