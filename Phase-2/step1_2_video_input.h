// VidOptix - Steps 1 & 2: video input, metadata, grayscale conversion
//
//   Step 1: Open video, read metadata (resolution, FPS, frame count)
//   Step 2: Read frames one at a time, convert to grayscale (float)
//
// This is the base module of the pipeline. The other modules include this
// header to get BLOCK_SIZE, so commit this file FIRST.

#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Block size used by the whole pipeline (DCT, quantization, ...).
constexpr int BLOCK_SIZE = 8;

struct VideoInfo {
    int    width        = 0;
    int    height       = 0;
    double fps          = 0.0;
    int    frameCount   = 0;

    // Frame size in whole 8x8 blocks (edge pixels that don't fill a block
    // are ignored for now).
    int blocksPerRow  = 0;
    int blocksPerCol  = 0;
    int usableWidth   = 0;
    int usableHeight  = 0;
};

// ---------- Step 1: open video + read metadata ----------
// Returns false (and prints an error) if the video can't be opened.
inline bool openVideo(const std::string& path, cv::VideoCapture& cap, VideoInfo& info) {
    cap.open(path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open video file: " << path << std::endl;
        return false;
    }

    info.width      = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    info.height     = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    info.fps        = cap.get(cv::CAP_PROP_FPS);
    info.frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    info.blocksPerRow = info.width  / BLOCK_SIZE;
    info.blocksPerCol = info.height / BLOCK_SIZE;
    info.usableWidth  = info.blocksPerRow * BLOCK_SIZE;
    info.usableHeight = info.blocksPerCol * BLOCK_SIZE;
    return true;
}

inline void printVideoInfo(const VideoInfo& info) {
    std::cout << "=== Video Metadata ===" << std::endl;
    std::cout << "Resolution:   " << info.width << " x " << info.height << std::endl;
    std::cout << "FPS:          " << info.fps << std::endl;
    std::cout << "Frame count:  " << info.frameCount << std::endl;
    std::cout << "=======================" << std::endl;

    if (info.usableWidth != info.width || info.usableHeight != info.height) {
        std::cout << "NOTE: " << info.width << "x" << info.height << " isn't a multiple of "
                  << BLOCK_SIZE << ". Processing " << info.usableWidth << "x" << info.usableHeight
                  << " and ignoring the leftover edge pixels for now." << std::endl;
    }
}

// ---------- Step 2: read next frame, convert to grayscale ----------
// Fills `grayFloat` with the next frame as a single-channel CV_32F image
// (values 0-255), ready for block splitting + DCT.
// Returns false when there are no more frames.
inline bool readNextGrayFrame(cv::VideoCapture& cap, cv::Mat& grayFloat) {
    cv::Mat frame, grayFrame;
    if (!cap.read(frame) || frame.empty()) {
        return false;
    }
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    grayFrame.convertTo(grayFloat, CV_32F);
    return true;
}
