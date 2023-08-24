#include <iostream>
using namespace std;

#include <cstdio>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "openpose.h"

#include "engine.h"

using namespace cv;


int main() {

    //Print opencv build information
    printf("%s",cv::getBuildInformation().c_str());

    String modelTxt = samples::findFile("/home/adam/Downloads/pose_deploy.prototxt");
    String modelBin = samples::findFile("/home/adam/Downloads/pose_iter_584000.caffemodel");
    String imageFile = "/home/adam/Pictures/wedding photos/IMG_7101.png";
    String dataset = "BODY_25";
    puts(modelTxt.c_str());

    Options options;

    // This particular model only supports a fixed batch size of 1
    options.doesSupportDynamicBatchSize = false;
    options.optBatchSize = 1;
    options.maxBatchSize = 10;
    options.maxWorkspaceSize = 2000000000;

    // Use FP16 precision to speed up inference
    options.precision = Precision::FP16;

    // Create our TensorRT inference engine
    //m_trtEngine = std::make_unique<Engine>(options);
    Engine m_trtEngine(options);

    auto succ = m_trtEngine.build(modelBin, modelTxt);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

    succ = m_trtEngine.loadNetwork();
    if (!succ) {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    while (true) {

        // The model expects RGB input
        cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    }

    return 0;
}
