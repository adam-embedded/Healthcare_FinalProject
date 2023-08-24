//
// Created by adam on 7/19/23.
//

#ifndef TENSORFLOW_MOVENET_CLASSIFIER_H
#define TENSORFLOW_MOVENET_CLASSIFIER_H

#include <iostream>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include "personLayout.h"


class Classifier {

public:

    explicit Classifier(std::string model_file, std::string label_file); //constructor
    void classify_pose(Person &person);

private:
    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::string label_list[2];

    void loadLabels(const std::string& file_name);

};


#endif //TENSORFLOW_MOVENET_CLASSIFIER_H
