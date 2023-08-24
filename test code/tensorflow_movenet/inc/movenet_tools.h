//
// Created by adam on 7/13/23.
//

#ifndef TENSORFLOW_MOVENET_MOVENET_TOOLS_H
#define TENSORFLOW_MOVENET_MOVENET_TOOLS_H

#include <iostream>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include "classifier.h"
#include "personLayout.h"

class MovenetTools{
public:

    explicit MovenetTools(const std::string& model_file, const std::string& classifier_model, const std::string& label_file); // constructor

    typedef struct {
        float y_min;
        float x_min;
        float y_max;
        float x_max;
        float box_height;
        float box_width;
    } Crop_region;

    void Run(cv::Mat &frame); //, Classifier *classifier

private:

    Classifier *classifier;

// Configure how confident the model should be on the detected keypoints to
// proceed with using smart cropping logic.
    const float MIN_CROP_KEYPOINT_SCORE = 0.2;
    const float TORSO_EXPANSION_RATIO = 1.9;
    const float BODY_EXPANSION_RATIO = 1.2;
    const float keypoint_score_threshold = 0.1;
    const float keypoint_detection_threshold_for_classifier = 0.1;

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    std::unique_ptr<tflite::FlatBufferModel> model;

    enum BodyPart {
        NOSE = 0,
        LEFT_EYE = 1,
        RIGHT_EYE = 2,
        LEFT_EAR = 3,
        RIGHT_EAR = 4,
        LEFT_SHOULDER = 5,
        RIGHT_SHOULDER = 6,
        LEFT_ELBOW = 7,
        RIGHT_ELBOW = 8,
        LEFT_WRIST = 9,
        RIGHT_WRIST = 10,
        LEFT_HIP = 11,
        RIGHT_HIP = 12,
        LEFT_KNEE = 13,
        RIGHT_KNEE = 14,
        LEFT_ANKLE = 15,
        RIGHT_ANKLE = 16,
    };

    typedef struct {
        float max_torso_yrange;
        float max_torso_xrange;
        float max_body_yrange;
        float max_body_xrange;
    } TorsoBodyRange;

    typedef struct {
        bool reset_crop_region;
        int input_height;
        int input_width;
        int input_batch_size;
        int input_channels;
        Crop_region cropRegion;
    } MoveStruct;
    MoveStruct moveStruct{};

    typedef std::map<int, std::vector<float>> Target_keypoints;



    const int num_keypoints = 17;

    void detector(cv::Mat &frame, Person *person);
    static void init_crop_region(cv::Mat &, Crop_region *);
    void run_detector(cv::Mat &frame, float *results, int cropHeight, int cropWidth);
    void crop_and_resize(cv::Mat &image, Crop_region *cropRegion, int cropHeight, int cropWidth, cv::Mat &outputImage);
    void determine_crop_region(Crop_region *cropRegion_result, cv::Mat &frame, float * keypoints);
    bool torso_visible(float *keypoints);
    void determine_torso_and_body_range(TorsoBodyRange * torsoBodyRange, float center_y, float center_x, Target_keypoints& targetKeypoints, float * keypoints);
    void person_from_keypoints_with_scores(cv::Mat &frame, float *keypoints_with_scores, Person *person);
    void visualise(cv::Mat &frame, Person &person);
    void Classify(cv::Mat &frame, Person &person);
};

#endif //TENSORFLOW_MOVENET_MOVENET_TOOLS_H
