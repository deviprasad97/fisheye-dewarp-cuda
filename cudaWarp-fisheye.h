#ifndef __CUDA_WARP_H__
#define __CUDA_WARP_H__


#include "cudaUtility.h"
#include <exception>
#include <stdexcept>


#define M_PI 3.14159265358979323846
#define DTOR (M_PI/180)
#define RTOD (180/M_PI)
#define TWOPI (2*M_PI)
#define PID2 (0.5*M_PI)
#define PI (M_PI)

/** @brief Utility function to dewarp a fisheye image

    Detailed description follows here.
    @author Devi Prasad Tripathy
    @date October 2019
*/
namespace vsfish 
{ 
	/**
 * Color type ENUM
 * Image type identifiers need to be passed into the function
 */
	enum Color
	{
		RGBA,
		RGB,
		BGR,
		GRAY
	};

    enum Func
    {
        FISH2PRESP,
        WARP
    };
	class Dewarp
	{
		public:


                Dewarp(uint32_t width, uint32_t height,  const float2& focalLength, const float2& principalPoint, const float4& distortion):
                    width(width), height(height), focalLength(focalLength), principalPoint(principalPoint), distortion(distortion), useIntrinsic(
                        true){ };

                /**
                  * Class Constructor
                  * @param _width (input image width)
                  * @param _height (input image height)
                  * @param prespwidth (output image width, default=800)
                  * @param prespheight (output image height, default=600)
                */
                Dewarp(uint32_t _width, uint32_t _height, uint32_t prespwidth=800, uint32_t prespheight=600) :
                    width(_width), height(_height), prespwidth(prespwidth), prespheight(prespheight), blockDim (8, 8),
                        gridDim(iDivUp(prespwidth,blockDim.x), iDivUp(prespheight,blockDim.y)) { }
			
                /**
                 * Driver function, to entry
                 * @param input (raw image data cuda pointer)
                 * @param output (cuda pointer to output image)
                 * @param focus (Focus in mm, default:1.4)
                 * @param type (input image type RGBA or GRAY, default:GRAY)
                 * @param fishfov (Field of View for fisheye image, default: 180)
                 * @param prespfov (Field of View for Prespective image, default:100)
                 * @param prespwidth (output image width, default:800)
                 * @param prespheight (output image height, default:600)
                 * @param func (Dewarping of choice default:FISH2PRESP choice:FISH2PRESP, WARP)
                 * @return Cuda Error
                 */
				cudaError_t vsDeWarpFisheye( void* input, void* output, float focus=1.4, Color type=Color::GRAY, float fishfov=180,
				        float prespfov=100, Func func=Func::FISH2PRESP);





                cudaError_t vsDeWarpIntrinsic( void* input, void* output, const float2& _focalLength, const float2& _principalPoint, const float4& _distortion , Color _type=Color::GRAY);

                uint32_t get_height(){ return height;}
                uint32_t get_width(){ return width;}
		private:
				// launch kernel
				uint32_t width, height;
                uint32_t prespwidth;
                uint32_t prespheight;
                float fishfov = 180;
				const dim3 blockDim; //(8, 8);
				const dim3 gridDim; //(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));
                const float2 focalLength = {};
                const float2 principalPoint = {};
                const float4 distortion = {};
                bool useIntrinsic = false;

                /**
                  * Apply fisheye lens dewarping to an 8-bit fixed-point RGBA image.
                  * @param input (raw image data cuda pointer)
                  * @param output (cuda pointer to output image)
                  * @param width width of the input image
                  * @param height height of the input image
                  * @param focus focus of the lens (in mm).
                  * @ingroup warping
                */
                cudaError_t fisheyeDeWarp( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float focus);

                /**
                 * Paul's Method for fisheye to rectilinear transformation
                 * @param input (raw image data cuda pointer)
                 * @param output (cuda pointer to output image)
                 * @param width (width of input image)
                 * @param height (height of input image)
                 * @param fishfov (Field of View for fisheye image, default: 180)
                 * @param prespfov (Field of View for Prespective image, default:100)
                 * @param prespwidth (output image width, default:800)
                 * @param prespheight (output image height, default:600)
                 * @return
                 */
                cudaError_t vsFisheye2Presp( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                                             float fishfov=180, float prespfov=100, uint32_t prespwidth=800, uint32_t prespheight=600);

                cudaError_t vsDeWarpIntrinsicRGBA( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                                           const float2& focalLength, const float2& principalPoint, const float4& distortion);
	};
}
							
#endif

