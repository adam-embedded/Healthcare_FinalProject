#include "yolov8.h"

//#define ORIGINAL_ASPECT_RATIO

// Runs object detection on video stream then displays annotated results.

static const bool pose = true;

const std::vector<std::vector<unsigned int>> KPS_COLORS =
        {{0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {0,   255, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {255, 128, 0},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255},
         {51,  153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},
                                                         {14, 12},
                                                         {17, 15},
                                                         {15, 13},
                                                         {12, 13},
                                                         {6,  12},
                                                         {7,  13},
                                                         {6,  7},
                                                         {6,  8},
                                                         {7,  9},
                                                         {8,  10},
                                                         {9,  11},
                                                         {2,  3},
                                                         {1,  2},
                                                         {1,  3},
                                                         {2,  4},
                                                         {3,  5},
                                                         {4,  6},
                                                         {5,  7}};

const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {51,  153, 255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 51,  255},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {255, 128, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0},
                                                            {0,   255, 0}};

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
    YoloV8 yoloV8(onnxModelPath, 0.7f, 0.4f, 80);

    // Initialize the video stream
    cv::VideoCapture cap;

    // TODO: Replace this with your video source.
    // 0 is default webcam, but can replace with string such as "/dev/video0", or an RTSP stream URL
    cap.open(0);

    // Try to use HD resolution (or closest resolution
    auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

    if (!cap.isOpened()) {
        throw std::runtime_error("Unable to open video capture!");
    }

    // Create open cv image object
    cv::Mat img;

    //Define pose size
    cv::Size size = cv::Size{640,640};

    //set up for FPS counter
    double fps;
    cv::TickMeter timer;

    while (true) {
        timer.start();
        // Grab frame
        cap >> img;

        if (img.empty()) {
            throw std::runtime_error("Unable to decode image from video stream.");
        }
#ifdef ORIGINAL_ASPECT_RATIO
        cv::Mat *working_image = &img;
#else
        // Resize Image for inference
        int image_width = img.size().width;
        int image_height = img.size().height;
        int square_dim = std::min(image_width, image_height);
        int delta_height = (image_height - square_dim) / 2;
        int delta_width = (image_width - square_dim) / 2;
        cv::Mat resized_image;
        // crop + resize the input image
        cv::resize(img(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, size);

        cv::Mat *working_image = &resized_image;

#endif
        // Run inference
        const auto objects = yoloV8.detectObjects(*working_image, pose);

        // Draw the bounding boxes and skeleton on the image
        //yoloV8.drawObjectLabels(img, objects);
        yoloV8.drawPose(*working_image, objects,SKELETON, KPS_COLORS, LIMB_COLORS);


        timer.stop();
        fps = 1.0 / timer.getTimeSec();
        timer.reset();
        // Put FPS counter on screen
        cv::putText(*working_image, "FPS: " +std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);

        // Display the results
        cv::imshow("Pose Detection", *working_image);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    return 0;
}
