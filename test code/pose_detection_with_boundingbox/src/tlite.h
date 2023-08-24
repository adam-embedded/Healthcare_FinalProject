//
// Created by adam on 7/7/23.
//
#ifndef POSE_DETECTION_WITH_BOUNDINGBOX_TLITE_H
#define POSE_DETECTION_WITH_BOUNDINGBOX_TLITE_H

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <memory>

class Tlite{
public:

    Tlite(const std::string& model_file, const float threshold = 0.2f);


    void poseDetection(cv::Mat & image, const cv::Rect_<float> &);

private:
    const float confidence_threshold;
    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    const std::vector<std::pair<int, int>> connections = {
            {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7},
            {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12},
            {11, 13}, {13, 15}, {12, 14}, {14, 16}
    };

    const int num_keypoints = 17;

    int input_width;
    int input_height;
    void draw_keypoints(cv::Mat &resized_image, float *output);
};


#endif //POSE_DETECTION_WITH_BOUNDINGBOX_TLITE_H
