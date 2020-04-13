#include <iostream>
#include "cudaWarp-fisheye.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/stitching/detail/seam_finders.hpp" // seam_finder
#include <opencv2/core/utility.hpp>


void testIntrinsicRGBA(std::string, float2, float2, float4);

void testRGBA(std::string, uint32_t);
void testGRAY(std::string, uint32_t);
int main() 
{
  std::string filename = "/home/jetson2/fisheye_dewarp_cuda/test_imgs/original_opencv.png";
  uint32_t focal = 1.4;

  float2 focalLength, principalPoint;
  float4 distortion;

  focalLength.x = 284.487793;
  focalLength.y = 284.542206;

  principalPoint.x = 418.67749;
  principalPoint.y = 401.691589;

  distortion.x = -0.00217492692;
  distortion.y = 0.0334486216;
  distortion.z = -0.0304316599;
  distortion.w = 0.00379009591;

  // std::cout<<"Enter filename\n";
  // std::cin>>filename;

  //testIntrinsicRGBA(filename, focalLength, principalPoint, distortion);
  //std::cin>>focal;
  //testRGBA(filename, focal);
  testGRAY(filename, focal);
  // while(true)
  // {
  //   std::cin>>filename;
  //   std::cin>>focal;
  //   testGRAY(filename, focal);
  //   testRGBA(filename, focal);
  // }
  //testRGBA(filename, focal);
  
  return 0;
}

/*
void testIntrinsicRGBA(std::string filename, float2 focalLength, float2 principalPoint, float4 distortion)
{

    // Source Image
    cv::Mat image;
    image = cv::imread(filename, 0);

    // Image attributes
    uint32_t width = image.cols;
    uint32_t height = image.rows;
    //uint32_t focus = 1.4;

    vsfish::Dewarp dewarp(width, height);

    uchar4 *h_rgbaImage; // Host rgba image (from mat to uchar4*)
    uchar4 *d_rgbaImage, *d_rgbaImageOut; // d_rgbaImage: Cuda host to device will contain h_rgbaImage data
    // d_rgbaImageOut: Cuda buffer cotaining defish image

    uchar4* defish;                       // defish: d_rgbaImageOut into cpu buffer



    // Convert BGR to RGBA
    cv::Mat continuousRGBA;
    //continuousRGBA = image;
    cv::cvtColor(image, continuousRGBA, 9);

    // Init host data
    h_rgbaImage = (uchar4*)continuousRGBA.ptr<unsigned char>(0);
    defish = (uchar4*)continuousRGBA.ptr<unsigned char>(0);

    // Allocating gpu memory and copying image data from cpu to gpu
    cudaMalloc((void**)&d_rgbaImage, sizeof(uchar4) * ((width*height)+500));
    cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * (width*height), cudaMemcpyHostToDevice);

    // Allocating Output defish cuda buffer
    cudaMalloc((void**)&d_rgbaImageOut, sizeof(uchar4) * ((width*height) + 500));

    // Calling dewarp method
    int loopCount = 0;
    while (loopCount<1)
    {
        dewarp.cudaWarpIntrinsic(d_rgbaImage, d_rgbaImageOut, width, height, focalLength, principalPoint, distortion, vsfish::Color::RGBA);

        // Copy Output buffer given by vsDeWarpFisheye to defish (CPU)
        cudaMemcpy(defish, d_rgbaImageOut, sizeof(uchar4) * (width*height), cudaMemcpyDeviceToHost);

        // Creating Mat out of the cpu uchar4* buffer
        cv::Mat img(image.cols, image.rows, CV_8UC4, defish);

        cv::imshow("TEST Intrinsic RGBA", img);
        cv::waitKey(0);
        loopCount ++;
    }

    // Hello World
    //vsfish::test();

    // Cuda Memory Free
    cudaFree(d_rgbaImage);
    cudaFree(d_rgbaImageOut);
}
*/


void testRGBA(std::string filename, uint32_t focus)
{

  // Source Image
  cv::Mat image;
  image = cv::imread(filename, 0);
  
  // Image attributes
  uint32_t width = image.cols;
  uint32_t height = image.rows;
  //uint32_t focus = 1.4;

  vsfish::Dewarp dewarp(width, height);
 
  uchar4 *h_rgbaImage; // Host rgba image (from mat to uchar4*)
  uchar4 *d_rgbaImage, *d_rgbaImageOut; // d_rgbaImage: Cuda host to device will contain h_rgbaImage data
                                        // d_rgbaImageOut: Cuda buffer cotaining defish image 

  uchar4* defish;                       // defish: d_rgbaImageOut into cpu buffer
  


  // Convert BGR to RGBA
  cv::Mat continuousRGBA;
  //continuousRGBA = image;
  cv::cvtColor(image, continuousRGBA, 9);
  
  // Init host data
  h_rgbaImage = (uchar4*)continuousRGBA.ptr<unsigned char>(0);
  defish = (uchar4*)continuousRGBA.ptr<unsigned char>(0);
 
  // Allocating gpu memory and copying image data from cpu to gpu
  cudaMalloc((void**)&d_rgbaImage, sizeof(uchar4) * ((width*height)+500));
  cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * (width*height), cudaMemcpyHostToDevice);

  // Allocating Output defish cuda buffer
  cudaMalloc((void**)&d_rgbaImageOut, sizeof(uchar4) * ((width*height) + 500));

  // Calling dewarp method
 // while (true)
  //{
    dewarp.vsDeWarpFisheye(d_rgbaImage, d_rgbaImageOut, focus, vsfish::Color::RGBA);

    // Copy Output buffer given by vsDeWarpFisheye to defish (CPU)
    cudaMemcpy(defish, d_rgbaImageOut, sizeof(uchar4) * (width*height), cudaMemcpyDeviceToHost);
    
    // Creating Mat out of the cpu uchar4* buffer
    cv::Mat img(image.cols, image.rows, CV_8UC4, defish);
    cv::imwrite("/home/devi/card/fisheye_dewarp_cuda/test_imgs/devi_output.png", img);
    cv::imshow("TEST RGBA", img);
    cv::waitKey(0);
 // }
  
  // Hello World
  //vsfish::test();

  // Cuda Memory Free
  cudaFree(d_rgbaImage);
  cudaFree(d_rgbaImageOut);
}
void testGRAY(std::string filename, uint32_t focus)
{
  // Source Image
  cv::Mat image;
  image = cv::imread(filename, 0);

  // Image attributes
  uint32_t width = image.cols;
  uint32_t height = image.rows;
  //uint32_t focus = 2;
  uint32_t prespwidth = 412;
  uint32_t prespheight = 300;
  vsfish::Dewarp dewarp(width, height, prespwidth, prespheight);

  uchar *h_grayImage; // Host rgba image (from mat to uchar4*)
  uchar *d_grayImage, *d_grayImageOut; // d_rgbaImage: Cuda host to device will contain h_rgbaImage data
                                        // d_rgbaImageOut: Cuda buffer cotaining defish image 

  uchar* defish;                       // defish: d_rgbaImageOut into cpu buffer
  
  

  // Init host data
  h_grayImage = (uchar*)image.ptr<unsigned char>(0);
  defish = (uchar*)malloc(prespwidth*prespheight);
 
  // Allocating gpu memory and copying image data from cpu to gpu
  cudaMalloc((void**)&d_grayImage, sizeof(uchar) * ((width*height)+500));  
  cudaMemcpy(d_grayImage, h_grayImage, sizeof(uchar) * (width*height), cudaMemcpyHostToDevice);

  // Allocating Output defish cuda buffer
  cudaMalloc((void**)&d_grayImageOut, sizeof(uchar) * ((prespwidth*prespheight) + 500));

 // while(true)
  //{
    // Calling dewarp method
    dewarp.vsDeWarpFisheye(d_grayImage, d_grayImageOut, focus, vsfish::Color::GRAY);
    // Copy Output buffer given by vsDeWarpFisheye to defish (CPU)
    cudaMemcpy(defish, (uchar*)d_grayImageOut, sizeof(uchar) * (prespwidth*prespheight), cudaMemcpyDeviceToHost);

    // Creating Mat out of the cpu uchar4* buffer
    cv::Mat img(prespheight, prespwidth, CV_8UC1, defish);

    cv::imshow("TEST GRAY", img);
    cv::waitKey(0);
  //}
  // Hello World
  //vsfish::test();

  // Cuda Memory Free
  free(defish);
  cudaFree(d_grayImage);
  cudaFree(d_grayImageOut);
}
