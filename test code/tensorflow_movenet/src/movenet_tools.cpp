//
// Created by adam on 7/13/23.
//
#include <iostream>
#include <iterator>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/string_util.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "movenet_tools.h"

#include "classifier.h"

MovenetTools::MovenetTools(const std::string& model_file, const std::string& classifier_model, const std::string& label_file){
//load model
    model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
    }

    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);
    interpreter->SetNumThreads(2);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to allocate tensors");
    }

    tflite::PrintInterpreterState(interpreter.get());

    auto input = interpreter->inputs()[0];
    moveStruct.input_batch_size = interpreter->tensor(input)->dims->data[0];
    moveStruct.input_height = interpreter->tensor(input)->dims->data[1];
    moveStruct.input_width = interpreter->tensor(input)->dims->data[2];
    moveStruct.input_channels = interpreter->tensor(input)->dims->data[3];

    std::cout << "The input tensor has the following dimensions: ["<< moveStruct.input_batch_size << ","
              << moveStruct.input_height << ","
              << moveStruct.input_width << ","
              << moveStruct.input_channels << "]" << std::endl;


    auto output = interpreter->outputs()[0];
    auto dim0 = interpreter->tensor(output)->dims->data[0];
    auto dim1 = interpreter->tensor(output)->dims->data[1];
    auto dim2 = interpreter->tensor(output)->dims->data[2];
    auto dim3 = interpreter->tensor(output)->dims->data[3];
    std::cout << "The output tensor has the following dimensions: ["<< dim0 << ","
              << dim1 << ","
              << dim2 << ","
              << dim3 << "]" << std::endl;

    // set floats to zero as they default to 1 with some compilers.
    moveStruct.cropRegion.y_max = 0.0f;
    moveStruct.cropRegion.y_min = 0.0f;
    moveStruct.cropRegion.x_max = 0.0f;
    moveStruct.cropRegion.x_min = 0.0f;
    moveStruct.cropRegion.box_width = 0.0f;
    moveStruct.cropRegion.box_height = 0.0f;
    moveStruct.reset_crop_region = false;

    classifier = new Classifier(classifier_model, label_file);
}

//    Defines the default crop region.
//    The function provides the initial crop region (pads the full image from
//    both sides to make it a square image) when the algorithm cannot reliably
//    determine the crop region from the previous frame.
void MovenetTools::init_crop_region(cv::Mat &frame, Crop_region *cropRegion){

    int image_width = frame.size().width;
    int image_height = frame.size().height;

    float tmp_xmin;
    float tmp_ymin;
    float tmp_height;
    float tmp_width;

    if (image_width > image_height){
        tmp_xmin = 0.0f;
        tmp_width = 1.0f;
        // Pad the vertical dimension to become a square image
        tmp_ymin = (float(image_height) / 2 - float(image_width) / 2) / float(image_height);
        tmp_height = float(image_width) / float(image_height);
    } else {
        tmp_ymin = 0.0f;
        tmp_height = 1.0f;
        //Pad the horizontal dimension to become a square image.
        tmp_xmin = (float(image_width) / 2 - float(image_height) / 2) / float(image_width);
        tmp_width = float(image_height) / float(image_width);
    }

    cropRegion->y_min = tmp_ymin;
    cropRegion->x_min = tmp_xmin;
    cropRegion->y_max = tmp_ymin + tmp_height;
    cropRegion->x_max = tmp_xmin + tmp_width;
    cropRegion->box_height = tmp_height;
    cropRegion->box_width = tmp_width;
}

void MovenetTools::crop_and_resize(cv::Mat &image, Crop_region *cropRegion, int cropHeight, int cropWidth, cv::Mat &outputImage){

    //int image_width = image.size().width;
    //int image_height = image.size().height;

    float tmp_y_min = cropRegion->y_min;
    float tmp_x_min = cropRegion->x_min;
    float tmp_y_max = cropRegion->y_max;
    float tmp_x_max = cropRegion->x_max;

    //height is 0...

    int crop_top = (tmp_y_min < 0) ? 0 : int(tmp_y_min * image.size().height);
    int crop_bottom = (tmp_y_max >= 1)? image.size().height : int(tmp_y_max * image.size().height);
    int crop_left = (tmp_x_min < 0) ? 0 : tmp_x_min * image.size().width;
    int crop_right = (tmp_x_max >= 1) ? image.size().width : tmp_x_max * image.size().width;

    int padding_top = (tmp_y_min < 0) ? (0 - tmp_y_min * image.size().height) : 0;
    int padding_bottom = (tmp_y_max >= 1) ? ((tmp_y_max -1) * image.size().height) : 0;
    int padding_left = (tmp_x_min < 0) ? (0 - tmp_x_min * image.size().width) : 0;
    int padding_right = (tmp_x_max >= 1) ? ((tmp_x_max - 1) * image.size().width) : 0;

    cv::imwrite("before.jpg", image);
//cv::Mat outputImage
    outputImage = image(cv::Range(crop_top, crop_bottom), cv::Range(crop_left, crop_right));

    //cv::imwrite("after_crop.jpg", outputImage);

    cv::copyMakeBorder(outputImage,outputImage, padding_top, padding_bottom, padding_left, padding_right, cv::BORDER_CONSTANT);

    //cv::imwrite("after_make_border.jpg", outputImage);

    cv::resize(outputImage, outputImage, cv::Size(cropHeight, cropWidth), cv::INTER_AREA);

    //cv::imwrite("after_resize.jpg", outputImage);

    //return outputImage;

}

bool MovenetTools::torso_visible(float *keypoints){

    //extract score from keypoints results

    float left_hip_score = keypoints[LEFT_HIP * 3 + 2];
    float right_hip_score = keypoints[RIGHT_HIP * 3 + 2];
    float left_shoulder_score = keypoints[LEFT_SHOULDER * 3 + 2];
    float right_shoulder_score = keypoints[RIGHT_SHOULDER * 3 + 2];

    bool left_hip_visible = left_hip_score > MIN_CROP_KEYPOINT_SCORE;
    bool right_hip_visible = right_hip_score > MIN_CROP_KEYPOINT_SCORE;
    bool left_shoulder_visible = left_shoulder_score > MIN_CROP_KEYPOINT_SCORE;
    bool right_shoulder_visble = right_shoulder_score > MIN_CROP_KEYPOINT_SCORE;

    return (left_hip_visible || right_hip_visible) && (left_shoulder_visible || right_shoulder_visble);

}


//    Calculates the maximum distance from each keypoints to the center.
//
//    The function returns the maximum distances from the two sets of keypoints:
//    full 17 keypoints and 4 torso keypoints. The returned information will
//    be used to determine the crop size. See determine_crop_region for more
//            details.
void MovenetTools::determine_torso_and_body_range(TorsoBodyRange * torsoBodyRange, float center_y, float center_x, Target_keypoints& targetKeypoints, float * keypoints){

    std::vector<BodyPart> torso_joints = {LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP};

    // Reset old stored values if any
    torsoBodyRange->max_torso_yrange = 0.0f;
    torsoBodyRange->max_torso_xrange = 0.0f;

    for (auto & torso_joint : torso_joints){
        float dist_y = abs(center_y - targetKeypoints[torso_joint][0]);
        float dist_x = abs(center_x - targetKeypoints[torso_joint][1]);
        if (dist_y > torsoBodyRange->max_torso_yrange) torsoBodyRange->max_torso_yrange = dist_y;
        if (dist_x > torsoBodyRange->max_torso_xrange) torsoBodyRange->max_torso_xrange = dist_x;
    }

    // Reset old stored values if any
    torsoBodyRange->max_body_yrange = 0.0f;
    torsoBodyRange->max_body_xrange = 0.0f;

    for (int i=0; i < num_keypoints; i++){
        if (keypoints[i * 3 + 2] < MIN_CROP_KEYPOINT_SCORE) continue;
        for (auto & torso_joint : torso_joints){
            float dist_y = abs(center_y - targetKeypoints[torso_joint][0]);
            float dist_x = abs(center_x - targetKeypoints[torso_joint][1]);
            if (dist_y > torsoBodyRange->max_body_yrange) torsoBodyRange->max_body_yrange = dist_y;
            if (dist_x > torsoBodyRange->max_body_xrange) torsoBodyRange->max_body_xrange = dist_x;
        }
    }
}

bool cmp(float x, float y) {
    return (x < y);
}

//    Determines the region to crop the image for the model to run inference on.
//
//    The algorithm uses the detected joints from the previous frame to
//    estimate the square region that encloses the full body of the target
//    person and centers at the midpoint of two hip joints. The crop size is
//    determined by the distances between each joint and the center point.
//    When the model is not confident with the four torso joint predictions,
//    the function returns a default crop which is the full image padded to
//    square.
//    Result written to *cropRegion_result pointer
void MovenetTools::determine_crop_region(Crop_region *cropRegion_result, cv::Mat &frame, float * keypoints){

    int image_height = frame.size().height;
    int image_width = frame.size().width;

    //std::map<int, std::vector<float>> target_keypoints;
    Target_keypoints target_keypoints;

    //position 0 is y axis
    for (int i=0; i < num_keypoints; i++){
        target_keypoints[i] = {(keypoints[i * 3] * image_height), (keypoints[i * 3 +1] * image_width)};
    }

    if (torso_visible(keypoints)){
        float center_y = (target_keypoints[LEFT_HIP][0] + target_keypoints[RIGHT_HIP][0]) / 2;
        float center_x = (target_keypoints[LEFT_HIP][1] + target_keypoints[RIGHT_HIP][1]) / 2;

        TorsoBodyRange bodyRange{};

        determine_torso_and_body_range(&bodyRange, center_y, center_x, target_keypoints, keypoints);

        //calculate the crop length half
        float crop_length_half = std::max({
            bodyRange.max_torso_xrange * TORSO_EXPANSION_RATIO,
            bodyRange.max_torso_yrange * TORSO_EXPANSION_RATIO,
            bodyRange.max_body_yrange * BODY_EXPANSION_RATIO,
            bodyRange.max_body_xrange * BODY_EXPANSION_RATIO
        },cmp);

        //Adjust crop length so that it is still within the image border

        float distances_to_border[] = {center_x, image_width - center_x, center_y, image_height - center_y};

        //float tmp_max_dtb = std::max({distances_to_border[0],distances_to_border[1], distances_to_border[2], distances_to_border[3]});

        crop_length_half = std::max({
            crop_length_half,
            std::max({distances_to_border[0],distances_to_border[1], distances_to_border[2], distances_to_border[3]})
        });

        // If the body is large enough, there's no need to apply cropping logic.
        if (crop_length_half > (std::max(image_width, image_height) / 2)) {
            init_crop_region(frame, cropRegion_result);
            return;
        }
        // Calculate the crop region that nicely covers the full body.

        float crop_length = crop_length_half * 2;

        float crop_corner[] = {center_y - crop_length_half, center_x - crop_length_half};


        // Finalise Results
        cropRegion_result->y_min        = crop_corner[0] / image_height;
        cropRegion_result->x_min        = crop_corner[1] / image_width;
        cropRegion_result->y_max        = (crop_corner[0] + crop_length) / image_height;
        cropRegion_result->x_max        = (crop_corner[1] + crop_length) / image_width;
        cropRegion_result->box_height   = ((crop_corner[0] + crop_length) / image_height) - (crop_corner[0] / image_height);
        cropRegion_result->box_width    = ((crop_corner[1] + crop_length) / image_width) - (crop_corner[1] / image_width);


    } else {
        init_crop_region(frame, cropRegion_result);
        return;
    }
}

// Creates a Person instance from single pose estimation model output.
void MovenetTools::person_from_keypoints_with_scores(cv::Mat &frame, float *keypoints_with_scores, Person *person){
    int image_height = frame.size().height;
    int image_width = frame.size().width;

    //Person person; // this may be supplied as a pointer...

    std::vector<float> kpts_x_vec;
    std::vector<float> kpts_y_vec;
    std::vector<float> scores;

    std::vector<float> kpt_x;
    std::vector<float> kpt_y;

    for (int i=0; i < num_keypoints; i++){
        float kpts_x = int(keypoints_with_scores[i * 3 + 1] * image_width);
        float kpts_y = int(keypoints_with_scores[i * 3] * image_height);

        kpt_x.push_back(keypoints_with_scores[i * 3 + 1]);
        kpt_y.push_back(keypoints_with_scores[i * 3]);

        //add values to separate vectors
        kpts_x_vec.push_back(kpts_x);
        kpts_y_vec.push_back(kpts_y);
        scores.push_back(keypoints_with_scores[i * 3 + 2]);

        person->keypoints.push_back(Keypoint{i, Point{kpts_x, kpts_y}, keypoints_with_scores[i * 3 +2]});
    }


    person->bounding_box.start_point.x = int(*std::min_element(kpt_x.begin(), kpt_x.end()) * image_width);
    person->bounding_box.start_point.y = int(*std::min_element(kpt_y.begin(), kpt_y.end()) * image_height);

    person->bounding_box.end_point.x = int(*std::max_element(kpt_x.begin(), kpt_x.end()) * image_width);
    person->bounding_box.end_point.y = int(*std::max_element(kpt_y.begin(), kpt_y.end()) * image_height);

    scores.erase(std::remove_if(scores.begin(), scores.end(), [&](const float& i) {
        return i < keypoint_score_threshold;
    }), scores.end());

    float person_score;

    if (!scores.empty()){
        person_score = std::accumulate(scores.begin(), scores.end(),0.0) / scores.size();
    } else {
        person_score = 0.0f;
    }

    person->score = person_score;
    person->id = 0;
}


//    Runs model inference on the cropped region.
//    The function runs the model inference on the cropped region and updates
//    the model output to the original image coordinate system.
void MovenetTools::run_detector(cv::Mat &frame, float *results, int cropHeight, int cropWidth){

    cv::Mat input_image;

    //printf("height: %d, width: %d", cropHeight, cropWidth);

    crop_and_resize(frame, &moveStruct.cropRegion, cropHeight, cropWidth, input_image);

    //input_image.convertTo(input_image, CV_8U);

    memcpy(interpreter->typed_input_tensor<unsigned char>(0), input_image.data, input_image.total() * input_image.elemSize());

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        exit(-1);
    }
}

void MovenetTools::detector(cv::Mat &frame, Person *person){

    if (moveStruct.cropRegion.box_width == 0 || moveStruct.reset_crop_region){
        init_crop_region(frame, &moveStruct.cropRegion);
    }

    float *results;

    // Detect pose using the crop region inferred from the detection result in
    // the previous frame
    run_detector(frame, results, moveStruct.input_height, moveStruct.input_width);

    results = interpreter->typed_output_tensor<float>(0);

    // Update coordinates for inference as image was resized
    for (int i = 0; i < num_keypoints; i++){
        results[i * 3] = moveStruct.cropRegion.y_min + moveStruct.cropRegion.box_height * results[i * 3]; //y axis
        results[i * 3 + 1] = moveStruct.cropRegion.x_min + moveStruct.cropRegion.box_width * results[i * 3 + 1]; //x axis
    }

    // Calculate the crop region for the next frame
    determine_crop_region(&moveStruct.cropRegion, frame, results);

    person_from_keypoints_with_scores(frame, results, person);

}

void MovenetTools::Run(cv::Mat &frame){ //, Classifier *classifier

    Person person{};

    detector(frame, &person);

    visualise(frame, person);

    //Classify(frame, person);
}