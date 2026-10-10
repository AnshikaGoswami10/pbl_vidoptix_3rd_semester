// VidOptix - Step 3: split a grayscale frame into 8x8 blocks, apply DCT
//
// Input : CV_32F grayscale frame from step1_2_video_input.h
// Output: 8x8 DCT coefficient blocks, consumed by step4_quantization.h
//
// Depends on: step1_2_video_input.h (for BLOCK_SIZE)

#pragma once

#include "step1_2_video_input.h"

// Returns a view (no copy) of the 8x8 block at grid position (bx, by).
inline cv::Mat extractBlock(const cv::Mat& grayFloat, int bx, int by) {
    cv::Rect region(bx * BLOCK_SIZE, by * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    return grayFloat(region);
}

// Forward 2D DCT of one 8x8 block.
inline cv::Mat forwardDCT(const cv::Mat& block) {
    cv::Mat dctBlock;
    cv::dct(block, dctBlock);
    return dctBlock;
}

// Inverse 2D DCT of one 8x8 block (used for reconstruction/verification,
// and later by the decoder).
inline cv::Mat inverseDCT(const cv::Mat& dctBlock) {
    cv::Mat pixels;
    cv::idct(dctBlock, pixels);
    return pixels;
}
