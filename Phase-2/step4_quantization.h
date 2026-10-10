// VidOptix - Step 4: quantization of DCT coefficients
//
// Quantizes DCT blocks using the standard JPEG luminance matrix (scaled by
// QUALITY_SCALE), and can verify the result by dequantizing + reconstructing
// a block.
//
// Depends on: step3_dct.h (inverseDCT), which pulls in step1_2_video_input.h
//
// Still no zigzag/RLE/Huffman here - those come in later steps.

#pragma once

#include "step3_dct.h"
#include <cmath>

// Standard JPEG luminance quantization matrix (ITU-T T.81 / ISO/IEC 10918-1,
// Annex K, Table K.1). Lower values near top-left preserve low-frequency
// detail more precisely; higher values near bottom-right compress
// high-frequency detail more aggressively (human vision is less sensitive
// to it there).
inline const int BASE_QUANT_MATRIX[BLOCK_SIZE][BLOCK_SIZE] = {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

// QUALITY_SCALE controls the size-vs-quality tradeoff.
//   < 1.0  -> smaller divisors -> less compression, higher quality
//   = 1.0  -> standard JPEG quantization
//   > 1.0  -> larger divisors -> more compression, lower quality
// Tune this later while measuring PSNR/SSIM to hit the 75-80% quality target.
inline constexpr float QUALITY_SCALE = 1.0f;

// Build the scaled quantization matrix once at startup.
inline cv::Mat buildQuantMatrix() {
    cv::Mat q(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            q.at<float>(i, j) = BASE_QUANT_MATRIX[i][j] * QUALITY_SCALE;
    return q;
}

// Element-wise divide by the quantization matrix, then round to the nearest
// integer. This is the lossy step - small coefficients often round to 0.
inline cv::Mat quantizeBlock(const cv::Mat& dctBlock, const cv::Mat& quantMatrix) {
    cv::Mat quantized(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            quantized.at<float>(i, j) =
                std::round(dctBlock.at<float>(i, j) / quantMatrix.at<float>(i, j));
    return quantized;
}

// Multiply back by the quantization matrix (inverse of quantizeBlock,
// minus the rounding loss).
inline cv::Mat dequantizeBlock(const cv::Mat& quantized, const cv::Mat& quantMatrix) {
    cv::Mat dequantized(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            dequantized.at<float>(i, j) =
                quantized.at<float>(i, j) * quantMatrix.at<float>(i, j);
    return dequantized;
}

// Count zeros among the 63 AC coefficients (excludes the DC term at [0][0],
// which is almost never zero).
inline int countZeroACs(const cv::Mat& quantized) {
    int zeros = 0;
    for (int i = 0; i < BLOCK_SIZE; i++)
        for (int j = 0; j < BLOCK_SIZE; j++)
            if (!(i == 0 && j == 0) && quantized.at<float>(i, j) == 0)
                zeros++;
    return zeros;
}

// Sanity check: print DCT, quantized values, then dequantize + inverse DCT
// and compare against the original pixels.
inline void printBlockDebug(const cv::Mat& block,
                            const cv::Mat& dctBlock,
                            const cv::Mat& quantized,
                            const cv::Mat& quantMatrix) {
    std::cout << "\n=== First 8x8 Block - DCT Coefficients (before quantization) ===" << std::endl;
    std::cout << dctBlock << std::endl;

    std::cout << "\n=== First 8x8 Block - Quantized Coefficients ===" << std::endl;
    std::cout << quantized << std::endl;

    std::cout << "\nZeros produced in this block (out of 63 AC positions): "
              << countZeroACs(quantized) << std::endl;

    cv::Mat reconstructed = inverseDCT(dequantizeBlock(quantized, quantMatrix));

    std::cout << "\n=== Reconstructed Pixel Values (after dequantize + inverse DCT) ===" << std::endl;
    std::cout << reconstructed << std::endl;

    std::cout << "\n=== Original Pixel Values (for comparison) ===" << std::endl;
    std::cout << block << std::endl;

    cv::Mat diff;
    cv::absdiff(block, reconstructed, diff);
    double maxError;
    cv::minMaxLoc(diff, nullptr, &maxError);
    std::cout << "\nMax pixel error (original vs reconstructed): "
              << maxError << " (out of 0-255 scale)" << std::endl;
    std::cout << "Expectation: reconstructed values should look close to "
              << "original, with some loss - that's the lossy part working.\n" << std::endl;
}
