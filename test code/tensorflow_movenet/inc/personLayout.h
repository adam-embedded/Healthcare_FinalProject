//
// Created by adam on 7/19/23.
//

#ifndef TENSORFLOW_MOVENET_PERSONLAYOUT_H
#define TENSORFLOW_MOVENET_PERSONLAYOUT_H

#include <iostream>
#include <vector>

typedef struct {
    float x;
    float y;
}Point;

typedef struct {
    Point start_point;
    Point end_point;
}Rectangle;

typedef struct {
    int bodyPart;
    Point coordinate;
    float score;
} Keypoint;

typedef struct {
    std::vector<Keypoint> keypoints;
    Rectangle bounding_box;
    float score;
    int id;
} Person;

#endif //TENSORFLOW_MOVENET_PERSONLAYOUT_H
