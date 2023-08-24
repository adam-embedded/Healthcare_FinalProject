//
// Created by adam slaymark on 05/08/2023.
//
#include <iostream>
#include <cstdio>
#include <cmath>
#include <thread>

#include "soxAudio.h"

static FILE *pipe1;

void initSox(int sampleRate, uint8_t bitDepth){

    std::string command = "play  -t raw -r "+ std::to_string(sampleRate) +" -e signed-integer -b "+ std::to_string(bitDepth) +" -c 1 -";

    pipe1 = popen(command.c_str(),"w");
    if (!pipe1) {
        std::cerr << "Failed to open SOX pipe." << std::endl;
        throw std::runtime_error("");
    }
}

void writeAudioSox(char * buffer, uint32_t bufferSize){
    // Write audio data to the pipe
    size_t bytesWritten = fwrite(buffer, sizeof(char), bufferSize, pipe1);
    if (bytesWritten != bufferSize) {
        std::cerr << "Failed to write audio data to pipe." << std::endl;
        pclose(pipe1);
        throw std::runtime_error("");
    }
}

void closeSox(){
    // Close the pipe
    int status = pclose(pipe1);
    if (status == -1) {
        std::cerr << "Failed to close pipe." << std::endl;
        throw std::runtime_error("");
    }
}