//
// Created by adam on 7/19/23.
//

//std libraries
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>

//tensorflow
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

//opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//local includes
#include "classifier.h"
#include "personLayout.h"
//#include "numpy_tools.h"

Classifier::Classifier(std::string model_file, std::string label_file) {
    model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

    if (!model) {
        throw std::runtime_error("Failed to load classification model");
    }

    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);
    interpreter->SetNumThreads(1);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    auto input_batch_size = interpreter->tensor(input)->dims->data[0];
    auto input_height = interpreter->tensor(input)->dims->data[1];
    auto input_width = interpreter->tensor(input)->dims->data[2];
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


    loadLabels(label_file);

    puts("----------------Available Classification Labels Start-----------------");
    for (const auto & i : label_list)
        std::cout << i << std::endl;
    puts("----------------Available Classification Labels End-----------------");

}

void Classifier::loadLabels(const std::string& file_name) {
    std::fstream file(file_name);
    int i = 0;
    while(getline(file, label_list[i])) i++;
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& array) {
    std::vector<T> flattened;
    for (const auto& row : array) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

template <typename T>
std::vector<T> expand_dims(const std::vector<T>& array, int axis) {
    std::vector<T> expanded_array;
    expanded_array.reserve(array.size() + 1);

    // Insert a new axis at the specified position.
    for (int i = 0; i < array.size(); i++) {
        if (i == axis) {
            expanded_array.push_back(1);
        }
        expanded_array.push_back(array[i]);
    }

    return expanded_array;
}

template <typename T>
std::vector<T> squeeze(const std::vector<T>& arr, const std::vector<int>& axes) {
    std::vector<T> result;
    for (int i = 0; i < arr.size(); i++) {
        bool keep = true;
        for (int axis : axes) {
            if (i % axis == 0) {
                keep = false;
                break;
            }
        }
        if (keep) {
            result.push_back(arr[i]);
        }
    }
    return result;
}

void Classifier::classify_pose(Person &person) {

    std::vector<std::vector<float>> tmp;

    for (auto & keypoint : person.keypoints) {
        keypoint.coordinate.x;
        keypoint.coordinate.y;
        keypoint.score;

        tmp.push_back({keypoint.coordinate.x,keypoint.coordinate.y,keypoint.score});
    }

    std::vector<float> input_tensor = flatten(tmp);
    input_tensor = expand_dims(input_tensor, 0);

//    puts("here");

    //Copy tensor
    memcpy(interpreter->typed_input_tensor<float>(0), &input_tensor[0], sizeof(float)*input_tensor.size());

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        exit(-1);
    }

    float *results = interpreter->typed_output_tensor<float>(0);

    //printf("%f, %f\n",results[2], results[3]);

}