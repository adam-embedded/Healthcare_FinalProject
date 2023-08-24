//
// Created by adam on 7/18/23.
//
#include <iostream>
#include <vector>
#include <map>

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


std::map<std::pair<int, int>, std::vector<int>> KEYPOINT_EDGE_INDS_TO_COLOUR = {
    {{0,1}, {147,20,255}},
    {{0,2}, {255,255,0}},
    {{1,3}, {147,20,255}},
    {{2,4}, {255,255,0}},
    {{0,5}, {147,20,255}},
    {{0,6}, {255,255,0}},
    {{5,7}, {147,20,255}},
    {{7,9}, {147,20,255}},
    {{6,8}, {255,255,0}},
    {{8,10}, {255,255,0}},
    {{5,6}, {0,255,255}},
    {{5,11}, {147,20,255}},
    {{6,12}, {255,255,0}},
    {{11,12}, {0,255,255}},
    {{11,13}, {147,20,255}},
    {{13,15}, {147,20,255}},
    {{12,14}, {255,255,0}},
    {{14,16}, {255,255,0}},
};

std::vector<std::vector<int>> COLOUR_LIST = {
    {47, 79, 79},
    {139, 69, 19},
    {0, 128, 0},
    {0, 0, 139},
    {255, 0, 0},
    {255, 215, 0},
    {0, 255, 0},
    {0, 255, 255},
    {255, 0, 255},
    {30, 144, 255},
    {255, 228, 181},
    {255, 105, 180},
};

int8_t row_size = 30; //# pixels
int8_t left_margin = 24;  // pixels
cv::Scalar text_color = {0, 0, 255}; // red
int font_size = 3;
int font_thickness = 1;
int classification_results_to_show = 3;
int fps_avg_frame_count = 10;


void MovenetTools::visualise(cv::Mat &frame, Person &person) {

    float keypoint_threshold = 0.1;
    float instance_threshold = 0.1;

    Rectangle &bounding_box = person.bounding_box;

    if (person.score < instance_threshold){
        return;
    }

    int person_colour[] = {0,255,0};

    for (auto &i : person.keypoints){
        if (i.score >= keypoint_threshold){
            cv::circle(
                    frame,
                    cv::Point(i.coordinate.x, i.coordinate.y),
                    2,
                    cv::Scalar(person_colour[0],person_colour[1],person_colour[2]),
                    4);
        }
    }

    for (const auto &ele : KEYPOINT_EDGE_INDS_TO_COLOUR) {
        if (person.keypoints[ele.first.first].score > keypoint_threshold && person.keypoints[ele.first.second].score > keypoint_threshold){
            cv::line(frame,
                     cv::Point(person.keypoints[ele.first.first].coordinate.x,person.keypoints[ele.first.first].coordinate.y),
                     cv::Point(person.keypoints[ele.first.second].coordinate.x,person.keypoints[ele.first.second].coordinate.y),
                     cv::Scalar(ele.second[0],ele.second[1],ele.second[2]),
                     2
                     );
        }
    }

    cv::rectangle(frame, cv::Point(bounding_box.start_point.x,bounding_box.start_point.y), cv::Point(bounding_box.end_point.x, bounding_box.end_point.y), cv::Scalar(person_colour[0],person_colour[1],person_colour[2]),2);
}

void MovenetTools::Classify(cv::Mat &frame, Person &person){

    //float *min_score = std::min_element(&person.keypoints.begin()->score, &person.keypoints.end()->score);

    float min_score = person.keypoints[0].score;
    for (auto &i : person.keypoints) {
        if (i.score < min_score){
            min_score = i.score;
        }
    }

    if (min_score < keypoint_detection_threshold_for_classifier){
        std::string errorMessage = "Some Keypoints are not detected";
        cv::putText(frame, errorMessage, cv::Point(left_margin, 2*row_size), cv::FONT_HERSHEY_SIMPLEX, font_size, text_color,font_thickness);

        errorMessage = "Make Sure the person is fully visible in the camera.";
        cv::putText(frame, errorMessage, cv::Point(left_margin, 3*row_size), cv::FONT_HERSHEY_SIMPLEX, font_size, text_color,font_thickness);
    }
    else {
        //Run Pose Classification

        classifier->classify_pose(person);

    }

}