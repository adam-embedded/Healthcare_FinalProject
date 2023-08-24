#include "yolov8.h"
#include "tlite.h"


// Runs object detection on video stream then displays annotated results.

static const bool pose = false;

// Function Prototypes
void drawPeople(cv::Mat &, const std::vector<Object> &, YoloV8 *, Tlite *);
//void poseDetection(cv::Mat & image, const Object &object, Tlite *);

int main(int argc, char *argv[]) {

    std::string model_file = "/home/adam/CLionProjects/fall_detection_2/assets/lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite";

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

    // Initialise Tensorflow Lite
    Tlite tlite(model_file);

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
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
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

        if (img.empty()) {
            throw std::runtime_error("Unable to decode image from video stream.");
        }

        // Run inference
        const auto objects = yoloV8.detectObjects(img, pose);

        // Draw the bounding boxes on the image for object detection
        //yoloV8.drawObjectLabels(img, objects);
        drawPeople(img, objects, &yoloV8, &tlite);


        timer.stop();
        fps = 1.0 / timer.getTimeSec();
        timer.reset();

        cv::putText(img, "FPS: " +std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);


        cv::resize(img, img, cv::Size(1920,1080));

        // Display the results
        cv::imshow("Object Detection", img);
        if (cv::waitKey(1) >= 0) {
            break;
        }

    }
    return 0;
}

void drawPeople(cv::Mat& image, const std::vector<Object> &objects, YoloV8 *yoloV8, Tlite *tflite){
    unsigned int scale = 2;
    for (auto & object : objects) {
        cv::Scalar color = cv::Scalar(yoloV8->COLOR_LIST[object.label][0], yoloV8->COLOR_LIST[object.label][1],
                                      yoloV8->COLOR_LIST[object.label][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        } else {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto &rect = object.rect;

        if (yoloV8->classNames[object.label] == "person") {

            char text[256];
            sprintf(text, "%s %.1f%%", yoloV8->classNames[object.label].c_str(), object.probability * 100);


            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

            cv::Scalar txt_bk_color = color * 0.7 * 255;


            int x = object.rect.x;
            int y = object.rect.y + 1;

            tflite->poseDetection(image,object.rect);


            cv::rectangle(image, rect, color * 255, scale + 1);

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                          txt_bk_color, -1);

            cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale,
                        txtColor, scale);

            //tflite->poseDetection(image,object.rect);
        }
    }
}

//void poseDetection(cv::Mat & image, const Object &object, Tlite *tlite){
//
//    cv::Mat frame = image(object.rect);
//
//    //cv::imshow("individual Person", frame);
//
//    int image_width = frame.size().width;
//    int image_height = frame.size().height;
//
//    int square_dim = std::min(image_width, image_height);
//    int delta_height = (image_height - square_dim) / 2;
//    int delta_width = (image_width - square_dim) / 2;
//
//    cv::Mat flipped_image;
//
//    cv::flip(frame, flipped_image, 1);
//
//    cv::Mat resized_image;
//
//    // crop + resize the input image
//    cv::resize(flipped_image(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, cv::Size(tlite->input_width, tlite->input_height));
//
//    memcpy(tlite->interpreter->typed_input_tensor<unsigned char>(0), resized_image.data, resized_image.total() * resized_image.elemSize());
//
//    // inference
//    //std::chrono::steady_clock::time_point start, end;
//    //start = std::chrono::steady_clock::now();
//    if (tlite->interpreter->Invoke() != kTfLiteOk) {
//        std::cerr << "Inference failed" << std::endl;
//        //return -1;
//    }
//
//    float *results = tlite->interpreter->typed_output_tensor<float>(0);
//
//    tlite->draw_keypoints(resized_image, results);
//}