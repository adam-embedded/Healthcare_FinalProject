//
// Created by adam on 7/11/23.
//

#ifndef OPENPOSE_TENSORRT_OPENPOSE_H
#define OPENPOSE_TENSORRT_OPENPOSE_H

const int POSE_PAIRS[4][26][2] = {
        { // COCO body
                {1,2}, {1,5}, {2,3},
                                     {3,4}, {5,6}, {6,7},
                                                            {1,8}, {8,9}, {9,10},
                                                                                   {1,11}, {11,12}, {12,13},
                                                                                                            {1,0}, {0,14},
                                                                                                                              {14,16}, {0,15}, {15,17}
        },
        { // MPI body
                {0,1}, {1,2}, {2,3},
                                     {3,4}, {1,5}, {5,6},
                                                            {6,7}, {1,14}, {14,8}, {8,9},
                                                                                           {9,10}, {14,11}, {11,12}, {12,13}
        },
        { // hand
                {0,1}, {1,2}, {2,3}, {3,4}, // thumb
                                            {0,5}, {5,6}, {6,7}, {7,8}, // pinkie
                {0,9}, {9,10}, {10,11}, {11,12}, // middle
                                                                                                            {0,13}, {13,14}, {14,15}, {15,16}, // ring
                                                                                                                                               {0,17}, {17,18}, {18,19}, {19,20} // small
        },
        { // Body 25
                {1,8}, {1,2}, {1,5}, {2,3},
                                            {3, 4}, {5, 6}, {6, 7}, {8, 9},
                {9, 10}, {10, 11}, {8, 12}, {12, 13},
                                                                                                            {13, 14}, {1, 0}, {0, 15}, {15, 17},
                                                                                                                                               {0, 16}, {16, 18}, {2, 17}, {5, 18},
                {14,19}, {19,20}, {14,21}, {11,22},
                {22,23}, {11, 24},
        }

};

#endif //OPENPOSE_TENSORRT_OPENPOSE_H