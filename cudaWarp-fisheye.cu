/** @brief This is implemetation dewarp a fisheye image

    Implementation of various Fisheye to conversion in CUDA kernels
    @author Devi Prasad Tripathy
    @date October 2019
*/

#include "cudaWarp-fisheye.h"
using namespace vsfish;

/**
 * Struct to save cx, cy, cz for each pixel
 */
typedef struct {
    double x,y,z;
} XYZ;


/**
 * Helper Function to calculate cx, cy, cz for each pixel
 * @param d1
 * @param p1
 * @param d2
 * @param p2
 * @param d3
 * @param p3
 * @param d4
 * @param p4
 * @return struct XYZ with cx, cy, cz for each pixel
 */
__device__ XYZ VectorSum(double d1,XYZ p1,double d2,XYZ p2,double d3,XYZ p3,double d4,XYZ p4)
{
    XYZ sum;

    sum.x = d1 * p1.x + d2 * p2.x + d3 * p3.x + d4 * p4.x;
    sum.y = d1 * p1.y + d2 * p2.y + d3 * p3.y + d4 * p4.y;
    sum.z = d1 * p1.z + d2 * p2.z + d3 * p3.z + d4 * p4.z;

    return (sum);
}


__device__ float lerp(float y0, float y1, float x0, float x1, float x)
{
    const float m = (y1 - y0) / (x1 - x0);
    const float b = y0;

    return (m *(x-x0) + b);
}


/**
 * To caluclate cx, cy, cz for each pixel
 * @param x
 * @param y
 * @param p
 * @param perspwidth
 * @param perspheight
 * @param prespfov
 */
__device__ void CameraRay(double x, double y, XYZ *p, uint32_t perspwidth, uint32_t perspheight, float prespfov) {
    double h,v;
    double dh,dv;
    XYZ vp = {0,0,0},vd = {0,1,0}, vu = {0,0,1}; // Camera view position, direction, and up
    XYZ right = {1,0,0};

    static XYZ p1,p2,p4; // Corners of the view frustum
    //static XYZ p3;
    static int first = true;
    static XYZ deltah,deltav;
    static double inversew,inverseh;

    // Precompute what we can just once
    if (first) {
        dh = __tanf((prespfov*DTOR) / 2);
        dv = perspheight * dh / perspwidth;
        p1 = VectorSum(1.0,vp,1.0,vd,-dh,right, dv,vu);
        p2 = VectorSum(1.0,vp,1.0,vd,-dh,right,-dv,vu);
        //p3 = VectorSum(1.0,vp,1.0,vd, dh,right,-dv,vu);
        p4 = VectorSum(1.0,vp,1.0,vd, dh,right, dv,vu);
        deltah.x = p4.x - p1.x;
        deltah.y = p4.y - p1.y;
        deltah.z = p4.z - p1.z;
        deltav.x = p2.x - p1.x;
        deltav.y = p2.y - p1.y;
        deltav.z = p2.z - p1.z;

        inversew = 1.0 / perspwidth;
        inverseh = 1.0 / perspheight;
        first = false;
    }

    h = x * inversew;
    v = (perspheight - 1 - y) * inverseh;
    p->x = p1.x + h * deltah.x + v * deltav.x;
    p->y = p1.y + h * deltah.y + v * deltav.y;
    p->z = p1.z + h * deltah.z + v * deltav.z;
}


/**
 * Different Fisheye conversion methods
 */

 
/**
 * Funcion to accomodate for aparature diiferent FOV
 * Method derived form Paul for dualfish2sphere blog post
 */

/**
 * Funcion to accomodate for aparature diiferent FOV
 * @tparam T
 * @param input
 * @param output
 * @param width
 * @param height
 * @param aperture
 */

 template<typename T>
 __global__ void cudaDualFisheye( T* input, T* output, int width, int height, float aperture, uint32_t prespwidth, uint32_t prespheight)
 {
    const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
        blockDim.y * blockIdx.y + threadIdx.y);

    if( uv_out.x >= width || uv_out.y >= height )
        return;
    const float y_dst_norm = lerp(-1, 1, 0, prespheight, uv_out.y);
    const float x_dst_norm = lerp(-1, 1, 0, prespwidth, uv_out.x);
    

    const float longitude = x_dst_norm * M_PI; 
    const float latitude = y_dst_norm *  M_PI/2;

    const float p_x = __cosf(latitude) * __cosf(longitude);
    const float p_y = __cosf(latitude) * __sinf(longitude);
    const float p_z = __sinf(latitude);

    const float p_xz = sqrtf(p_x*p_x + p_z*p_z);
    const float r = 2* atan2f(p_xz, p_y) / aperture;
    const float theta = atan2f(p_z, p_x);

    const float x_src_norm = r * __cosf(theta);
    const float y_src_norm = r * __sinf(theta);

    const float x_src = lerp(0, width, -1, 1, x_src_norm);
    const float y_src = lerp(0, height, -1, 1, y_src_norm);

    output[uv_out.y * width + uv_out.x] = input[(int)y_src * width + (int)x_src];

 }




// cudaFisheye
// It converts spherical cord to equirectilinear
/**
 * To be used to convert imperfect Fisheye to an equirectilinear projection
 * @tparam T
 * @param input
 * @param output
 * @param width
 * @param height
 * @param focus
 */
template<typename T>
__global__ void cudaFisheye( T* input, T* output, int width, int height, float focus)
{
	const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
				               blockDim.y * blockIdx.y + threadIdx.y);
						   
	if( uv_out.x >= width || uv_out.y >= height )
		return;

	// in pixels
	const float fWidth  = width;
	const float fHeight = width;
	float theta = M_PI * (uv_out.x / ( (float) fWidth ) - 0.5);
	float phi = M_PI * (uv_out.y / ( (float) fHeight ) - 0.5);
	
	const float cx = __cosf(phi) * __sinf(theta);	
	const float cy = __cosf(phi) * __cosf(theta);
	const float cz = __sinf(phi);

	theta = atan2f(cz, cx);
	phi     = atan2f(sqrtf(cx*cx+cz*cz), cy);
	float r = ((float)fWidth) * phi / PI;

    auto u =  (float)( 0.5 * ( (float) fWidth ) + r * __cosf(theta) );
    auto v =  (float)( 0.5 * ( (float) fHeight ) + r * __sinf(theta));


	output[uv_out.y * height + uv_out.x] = input[(int)v * width + (int)u];
}

/**
 * To be used to convert Fisheye to an equirectilinear projection
 * @tparam T
 * @param input
 * @param output
 * @param width
 * @param height
 * @param focus
 */
template<typename T>
__global__ void cudaFisheyeSame( T* input, T* output, int width, int height, float focus)
{
    const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                  blockDim.y * blockIdx.y + threadIdx.y);

    if( uv_out.x >= width || uv_out.y >= height )
        return;

    const float fWidth  = width;
    const float fHeight = width;


    // convert to cartesian coordinates
    const float cx = ((uv_out.x / fWidth) - 0.5f)  * 2.0f;
    const float cy = (0.5f - (uv_out.y / fHeight)) * 2.0f;

    const float theta = atan2f(cy, cx);
    const float r     = atanf(sqrtf(cx*cx+cy*cy) * focus);

    const float tx = r * __cosf(theta);
    const float ty = r * __sinf(theta);

    // // convert back out of cartesian coordinates
    float u = (tx * 0.5f + 0.5f) * fWidth;
    float v = (0.5f - (ty * 0.5f)) * fHeight;

    if( u < 0.0f ) u = 0.0f;
    if( v < 0.0f ) v = 0.0f;

    if( u > fWidth  - 1.0f ) u = fWidth - 1.0f;
    if( v > fHeight - 1.0f ) v = fHeight - 1.0f;

    output[uv_out.y * width + uv_out.x] = input[(int)v * width + (int)u];
}


/**
 * CUDA Kernel to convert Fisheye to an equirectilinear projection
 * @tparam T
 * @param input
 * @param output
 * @param width
 * @param height
 * @param fishfov
 * @param prespfov
 * @param prespwidth
 * @param prespheight
 */
template<typename T>
__global__ void cudaFisheyePaul( T* input, T* output, int width, int height, float fishfov, float prespfov, uint32_t prespwidth, uint32_t prespheight)
{
    const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                  blockDim.y * blockIdx.y + threadIdx.y);

    if( uv_out.x >= prespwidth || uv_out.y >= prespheight )
        return;

    XYZ p;

    fishfov /= 2;
    fishfov *= DTOR;

    CameraRay(uv_out.x,uv_out.y,&p,prespwidth,prespheight,prespfov);


    const float theta = atan2f(p.z, p.x);
    const float phi     = atan2f(sqrtf(p.x*p.x+p.z*p.z), p.y);

    const float r = phi/fishfov;

    // // convert back out of cartesian coordinates
    float u = (width/2) + r * (width/2) * __cosf(theta);
    float v = (height/2) + r * (width/2) * __sinf(theta);



    output[uv_out.y * prespwidth + uv_out.x] = input[(int)v * width + (int)u];
}


// gpuIntrinsicWarp
template<typename T>
__global__ void gpuIntrinsicWarp( T* input, T* output, int width, int height,
                                  float2 focalLength, float2 principalPoint, float k1, float k2, float p1, float p2)
{
    const int2 uv_out = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                  blockDim.y * blockIdx.y + threadIdx.y);

    if( uv_out.x >= width || uv_out.y >= height )
        return;

    const float u = uv_out.x;
    const float v = uv_out.y;

    const float _fx = 1.0f / (focalLength.x);
    const float _fy = 1.0f / (focalLength.y);

    const float y      = (v - principalPoint.y)*_fy;
    const float y2     = y*y;
    const float _2p1y  = 2.0*p1*y;
    const float _3p1y2 = 3.0*p1*y2;
    const float p2y2   = p2*y2;

    const float x  = (u - principalPoint.x)*_fx;
    const float x2 = x*x;
    const float r2 = x2 + y2;
    const float d  = 1.0 + (k1 + k2*r2)*r2;
    const float _u = focalLength.x*(x*(d + _2p1y) + p2y2 + (3.0*p2)*x2) + principalPoint.x;
    const float _v = focalLength.y*(y*(d + (2.0*p2)*x) + _3p1y2 + p1*x2) + principalPoint.y;

    const int2 uv_in = make_int2( _u, _v );

    if( uv_in.x >= width || uv_in.y >= height || uv_in.x < 0 || uv_in.y < 0 )
        return;

    output[uv_out.y * height + uv_out.x] = input[uv_in.y * width + uv_in.x];
}

/**
 * Helper CUDA Kernel converts a GRAYSCALE image to RGBA (uchar1 -> uchar4)
 * @param inputImage
 * @param grayImage
 * @param width
 * @param height
 */
__global__ void gray2rgbaCudaKernel(uchar1 *inputImage, uchar4 *grayImage, const int width, const int height)
{


    int tx = (blockIdx.y * blockDim.y) + threadIdx.y;
    int ty = (blockIdx.x * blockDim.x) + threadIdx.x;

    if( (ty < height && tx<width) )
    {
        unsigned int pixel = ty*width+tx;
        unsigned char r = static_cast< unsigned char >(inputImage[pixel].x);
        unsigned char g = static_cast< unsigned char >(inputImage[pixel].x);
        unsigned char b = static_cast< unsigned char >(inputImage[pixel].x);
        unsigned char a = static_cast< unsigned char >(inputImage[pixel].x);


        grayImage[pixel].x = r;
        grayImage[pixel].y = g;
        grayImage[pixel].z = b;
        grayImage[pixel].w = a;

    }
}

/**
 * Helper CUDA Kernel converts a RGBA image to GRAYSCALE (uchar4 -> uchar1)
 * @param inputImage
 * @param grayImage
 * @param width
 * @param height
 */
__global__ void rgb2grayCudaKernel(uchar4 *inputImage, uchar1 *grayImage, const int width, const int height)
{


    int tx = (blockIdx.y * blockDim.y) + threadIdx.y;
    int ty = (blockIdx.x * blockDim.x) + threadIdx.x;

    if( (ty < height && tx<width) )
    {
        unsigned int pixel = ty*width+tx;
        float grayPix = 0.0f;

        unsigned char r = static_cast< unsigned char >(inputImage[pixel].x);
        unsigned char g = static_cast< unsigned char >(inputImage[pixel].y);
        unsigned char b = static_cast< unsigned char >(inputImage[pixel].z);

        grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

        grayImage[pixel].x = static_cast< unsigned char >(grayPix);

    }
}



cudaError_t vsfish::Dewarp::vsDeWarpIntrinsic( void* input, void* output, const float2& focalLength, const float2& principalPoint, const float4& distortion, Color type)
{
    if(!useIntrinsic)
    {
        throw std::invalid_argument("Object was created to not use intrinsic warping methods.\n Please create a new object"
                                    "with appropriate constructor to use this function.");
    }
    if(type == Color::RGBA)
    {
        vsDeWarpIntrinsicRGBA((uchar4*)input, (uchar4*)output, width, height, focalLength, principalPoint, distortion);
    }
    else
    {
        uchar4 *outputRGBA;
        uchar4 *outputRectRGBA;

        const dim3 blockDim1 (8, 8);
        const dim3 gridDim1(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

        cudaMalloc((void **) &outputRGBA, sizeof(uchar4) * ((width * height) + 500));
        cudaMalloc((void **) &outputRectRGBA, sizeof(uchar4) * ((width * height) + 500));

        gray2rgbaCudaKernel<<<gridDim1, blockDim1>>>((uchar1*)input, outputRGBA, width, height);

        vsDeWarpIntrinsicRGBA((uchar4*)input, (uchar4*)output, width, height, focalLength, principalPoint, distortion);

        rgb2grayCudaKernel<<<gridDim1, blockDim1>>>(outputRectRGBA, (uchar1*)output, height, width);

        cudaFree(outputRGBA);
        cudaFree(outputRectRGBA);

    }
    return CUDA(cudaGetLastError());
}




cudaError_t vsfish::Dewarp::vsDeWarpIntrinsicRGBA( uchar4* input, uchar4* output, uint32_t width, uint32_t height,
                                   const float2& focalLength, const float2& principalPoint, const float4& distortion)
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

    gpuIntrinsicWarp<<<gridDim, blockDim>>>(input, output, width, height,
            focalLength, principalPoint,
            distortion.x, distortion.y, distortion.z, distortion.w);

    return CUDA(cudaGetLastError());
}


cudaError_t vsfish::Dewarp::vsDeWarpFisheye(void* input, void* output, float focus, Color type, float fishfov, float prespfov, Func func) {
    if(useIntrinsic)
    {
        throw std::invalid_argument("Object was created to use Intrinsic warping methods.\n Please create a new object"
                                    "with appropriate constructor to use this function.");
    }
    if (func == Func::FISH2PRESP) {
        if (type == Color::RGBA) {
            vsFisheye2Presp((uchar4 *) input, (uchar4 *) output, width, height, fishfov, prespfov, prespwidth,
                            prespheight);
        }
        else {
              uchar4 *outputRGBA;
              uchar4 *outputRectRGBA;

              const dim3 blockDim1 (8, 8);
              const dim3 gridDim1(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

              cudaMalloc((void **) &outputRGBA, sizeof(uchar4) * ((width * height) + 500));
              cudaMalloc((void **) &outputRectRGBA, sizeof(uchar4) * ((prespwidth * prespheight) + 500));

              gray2rgbaCudaKernel<<<gridDim1, blockDim1>>>((uchar1*)input, outputRGBA, width, height);
              vsFisheye2Presp(outputRGBA, outputRectRGBA, width, height, fishfov, prespfov, prespwidth,
                      prespheight);

              rgb2grayCudaKernel<<<gridDim, blockDim>>>(outputRectRGBA, (uchar1*)output, prespheight, prespwidth);

              cudaFree(outputRGBA);
              cudaFree(outputRectRGBA);

        }
    }
    else {
        if (type == Color::RGBA) {
            fisheyeDeWarp((uchar4 *) input, (uchar4 *) output, width, height, focus);
        }
        else {

            uchar4 *outputRGBA;
            uchar4 *outputRectRGBA;

            const dim3 blockDim1 (8, 8);
            const dim3 gridDim1(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

            cudaMalloc((void **) &outputRGBA, sizeof(uchar4) * ((width * height) + 500));
            cudaMalloc((void **) &outputRectRGBA, sizeof(uchar4) * ((width * height) + 500));

            gray2rgbaCudaKernel<<<gridDim1, blockDim1>>>((uchar1*)input, outputRGBA, width, height);
            fisheyeDeWarp(outputRGBA, outputRectRGBA, width, height, focus);
            
            rgb2grayCudaKernel<<<gridDim, blockDim>>>(outputRectRGBA, (uchar1*)output, height, width);

            cudaFree(outputRGBA);
            cudaFree(outputRectRGBA);

         }
    }
	return CUDA(cudaGetLastError());
}


// cudaWarpFisheye
cudaError_t vsfish::Dewarp::fisheyeDeWarp( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float focus)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;
	if (width == height)
	{
		cudaFisheyeSame<<<gridDim, blockDim>>>(input, output, width, height, focus);
	}
	else
	{
		cudaFisheye<<<gridDim, blockDim>>>(input, output, width, height, focus);
	}
	
	return CUDA(cudaGetLastError());
}

// cudaWarpFisheye
cudaError_t vsfish::Dewarp::vsFisheye2Presp( uchar4* input, uchar4* output, uint32_t width, uint32_t height, float fishfov, float prespfov, uint32_t prespwidth, uint32_t prespheight)
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( width == 0 || height == 0 )
        return cudaErrorInvalidValue;

    cudaFisheyePaul<<<gridDim, blockDim>>>(input, output, width, height, fishfov, prespfov, prespwidth, prespheight);

    return CUDA(cudaGetLastError());
}
