//
// Created by adam on 7/7/23.
//

#include "tlite.h"

//#include <tensorflow/lite/model.h>
//#include <tensorflow/lite/interpreter.h>
//#include <tensorflow/lite/kernels/register.h>
//#include <tensorflow/lite/optional_debug_tools.h>
//#include <tensorflow/lite/string_util.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <memory>


Tlite::Tlite(const std::string &model_file, const float threshold)
    : confidence_threshold(threshold) {

    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    auto input_batch_size = interpreter->tensor(input)->dims->data[0];
    input_height = interpreter->tensor(input)->dims->data[1];
    input_width = interpreter->tensor(input)->dims->data[2];
    auto input_channels = interpreter->tensor(input)->dims->data[3];

    std::cout << "The input tensor has the following dimensions: ["<< input_batch_size << ","
              << input_height << ","
              << input_width << ","
              << input_channels << "]" << std::endl;

    auto output = interpreter->outputs()[0];

    auto dim0 = interpreter->tensor(output)->dims->data[0];
    auto dim1 = interpreter->tensor(output)->dims->data[1];
    auto dim2 = interpreter->tensor(output)->dims->data[2];
    auto dim3 = interpreter->tensor(output)->dims->data[3];
    std::cout << "The output tensor has the following dimensions: ["<< dim0 << ","
              << dim1 << ","
              << dim2 << ","
              << dim3 << "]" << std::endl;
}

void Tlite::draw_keypoints(cv::Mat &resized_image, float *output)
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

void Tlite::poseDetection(cv::Mat & image, const cv::Rect_<float> & rectangle){

    //cv::Mat frame = image(rectangle);

    cv::Mat frame = image;

    if (frame.empty()) puts("Frame Empty");


    //int image_width = frame.size().width;
    //int image_height = frame.size().height;

    //int square_dim = std::min(image_width, image_height);
    //int delta_height = (image_height - square_dim) / 2;
    //int delta_width = (image_width - square_dim) / 2;

    int desired = 192;

    //cv::Mat flipped_image;

    //cv::flip(frame, flipped_image, 1);

    cv::Mat resized_image;

    // crop + resize the input image
    //cv::resize(flipped_image(cv::Rect(delta_width, delta_height, square_dim, square_dim)), resized_image, cv::Size(input_width, input_height));

//    double scale_factor = static_cast<double>(input_width) / std::max(frame.cols, frame.rows);
//    int new_width = static_cast<int>(frame.cols * scale_factor);
//    int new_height = static_cast<int>(frame.rows * scale_factor);

    double scale_factor = static_cast<double>(desired) / std::max(frame.cols, frame.rows);
    int new_width = static_cast<int>(frame.cols * scale_factor);
    int new_height = static_cast<int>(frame.rows * scale_factor);

    cv::resize(frame, resized_image, cv::Size(new_width, new_height));

    // Create a blank canvas with the square dimensions
    cv::Mat canvas(desired, desired, image.type(), cv::Scalar(0, 0, 0));

    // Calculate the padding required
    int pad_x = (desired - new_width) / 2;
    int pad_y = (desired - new_height) / 2;

    // Place the resized image onto the padded canvas
    resized_image.copyTo(canvas(cv::Rect(pad_x, pad_y, resized_image.cols, resized_image.rows)));


    // Copy data to interpreter
    memcpy(interpreter->typed_input_tensor<unsigned char>(0), canvas.data, canvas.total() * canvas.elemSize());

//     inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        //return -1;
    }

    float *results = interpreter->typed_output_tensor<float>(0);

    draw_keypoints(canvas, results);

    cv::resize(canvas, canvas, cv::Size(1080,1080));

    cv::imshow("individual Person", canvas);

}