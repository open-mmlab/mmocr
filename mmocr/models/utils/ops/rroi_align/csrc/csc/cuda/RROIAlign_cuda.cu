// Copyright (c) Jianqi Ma. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <stdio.h>
#include <math.h>
#include <float.h>
//#include "rroi_alignment_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void RROIAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    int height,
    int width,
    int channels,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    T* top_data,
    float* con_idx_x,
    float* con_idx_y)
{

    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // +0.5 shift removed
        int imageWidth = width;
        int imageHeight = height;

        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;

        const T* offset_bottom_rois = bottom_rois + n * 7; //= 7 is rois dimension 0

        int roi_batch_ind = offset_bottom_rois[0];
        T cx = offset_bottom_rois[1];
        T cy = offset_bottom_rois[2];
        T h = offset_bottom_rois[3];
        T w = offset_bottom_rois[4];
        //T angle = offset_bottom_rois[5]/180.0*3.1415926535;
        T Alpha = offset_bottom_rois[5];
        T Beta = offset_bottom_rois[6];

        //TransformPrepare
        T dx = -pooled_width/2.0;
        T dy = -pooled_height/2.0;
        T Sx = w*spatial_scale/pooled_width;
        T Sy = h*spatial_scale/pooled_height;
        //T Alpha = cos(angle);
        //T Beta = sin(angle);
        T Dx = cx*spatial_scale;
        T Dy = cy*spatial_scale;

        T M[2][3];
        M[0][0] = Alpha*Sx;
        M[0][1] = Beta*Sy;
        M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
        M[1][0] = -Beta*Sx;
        M[1][1] = Alpha*Sy;
        M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

        T P[8];
        P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
        P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
        P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
        P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
        P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
        P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
        P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
        P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

        T leftMost = (max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
        T rightMost= (min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
        T topMost= (max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
        T bottomMost= (min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));

        //float maxval = 0;
        //int maxidx = -1;
        const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

        //float AB[2];
        //AB[0] = P[2] - P[0];
        //AB[1] = P[3] - P[1];
        //float ABAB = AB[0]*AB[0] +AB[1]*AB[1];
        //float AC[2];
        //AC[0] = P[4] - P[0];
        //AC[1] = P[5] - P[1];
        //float ACAC = AC[0]*AC[0] + AC[1]*AC[1];

        float bin_cx = (leftMost + rightMost) / 2.0; // shift
        float bin_cy = (topMost + bottomMost) / 2.0;

        int bin_l = (int)floor(bin_cx);
        int bin_r = (int)ceil(bin_cx);
        int bin_t = (int)floor(bin_cy);
        int bin_b = (int)ceil(bin_cy);

        T lt_value = 0.0;
        if (bin_t > 0 && bin_l > 0 && bin_t < height && bin_l < width)
            lt_value = offset_bottom_data[bin_t * width + bin_l];
        T rt_value = 0.0;
        if (bin_t > 0 && bin_r > 0 && bin_t < height && bin_r < width)
            rt_value = offset_bottom_data[bin_t * width + bin_r];
        T lb_value = 0.0;
        if (bin_b > 0 && bin_l > 0 && bin_b < height && bin_l < width)
            lb_value = offset_bottom_data[bin_b * width + bin_l];
        T rb_value = 0.0;
        if (bin_b > 0 && bin_r > 0 && bin_b < height && bin_r < width)
            rb_value = offset_bottom_data[bin_b * width + bin_r];

        T rx = bin_cx - floor(bin_cx);
        T ry = bin_cy - floor(bin_cy);

        T wlt = (1.0 - rx) * (1.0 - ry);
        T wrt = rx * (1.0 - ry);
        T wrb = rx * ry;
        T wlb = (1.0 - rx) * ry;

        T inter_val = 0.0;

        inter_val += lt_value * wlt;
        inter_val += rt_value * wrt;
        inter_val += rb_value * wrb;
        inter_val += lb_value * wlb;

        atomicAdd(top_data + index, static_cast<T>(inter_val));
        atomicAdd(con_idx_x + index, static_cast<float>(bin_cx));
        atomicAdd(con_idx_y + index, static_cast<float>(bin_cy));

        //top_data[index] = static_cast<T>(inter_val);
        //con_idx_x[index] = bin_cx;
        //con_idx_y[index] = bin_cy;

    }
}

/**
int RROIAlignForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* con_idx_x, float* con_idx_y, const float* im_info, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = num_rois * pooled_height * pooled_width * channels;
    cudaError_t err;


    RROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_rois, top_data, con_idx_x, con_idx_y, im_info);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "RRoI forward cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
*/
//ROIAlign_forward_cuda

//std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
//                                const at::Tensor& rois,
//                                const float spatial_scale,
//                                const int pooled_height,
//                                const int pooled_width)

std::tuple<at::Tensor, at::Tensor, at::Tensor> RROIAlign_forward_cuda(
                                  const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int pooled_width)
{


    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

    auto num_rois = rois.size(0);
    auto channels = input.size(1);
    auto height = input.size(2);
    auto width = input.size(3);

    auto output = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options());
    auto output_size = num_rois * pooled_height * pooled_width * channels;
    auto con_idx_x = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kFloat));
    auto con_idx_y = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kFloat));

    dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
    dim3 block(512);

    //const int kThreadsPerBlock = 1024;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (output.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return std::make_tuple(output, con_idx_x, con_idx_y);//, con_idx_y; //std::make_tuple(
    }

    AT_DISPATCH_FLOATING_TYPES(input.type(), "RROIAlign_forward", [&] {
        RROIAlignForward<scalar_t><<<grid, block, 0, stream>>>(
        output_size,
        input.contiguous().data<scalar_t>(),
        spatial_scale,
        height,
        width,
        channels,
        pooled_height,
        pooled_width,
        rois.contiguous().data<scalar_t>(),
        output.data<scalar_t>(),
        con_idx_x.data<float>(),
        con_idx_y.data<float>());
        }
      );

   THCudaCheck(cudaGetLastError());
   return std::make_tuple(output, con_idx_x, con_idx_y);
}

template <typename T>
__global__ void RROIAlignBackward(
            const int nthreads,
            const T* top_diff,
            const float* con_idx_x,
            const float* con_idx_y,
            const int num_rois,
            const float spatial_scale,
            const int height,
            const int width,
            const int channels,
            const int pooled_height,
            const int pooled_width,
            T* bottom_diff,
            const T* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {

        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        //int w = n % width;
        n /= pooled_width;
        //int h = n % height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;

        const T* offset_bottom_rois = bottom_rois + n * 7;
        int roi_batch_ind = offset_bottom_rois[0];
        T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

        //int bottom_index = argmax_data[index];

        float bw = con_idx_x[index];
        float bh = con_idx_y[index];
        //if (bh > 0.00001 && bw > 0.00001 && bw < height-1 && bw < width-1){

        int bin_xs = int(floor(bw));
        int bin_ys = int(floor(bh));

        float rx = bw - float(bin_xs);
        float ry = bh - float(bin_ys);

        T wlt = (1.0 - rx) * (1.0 - ry);
        T wrt = rx * (1.0 - ry);
        T wrb = rx * ry;
        T wlb = (1.0 - rx) * ry;

        // if(bottom_index >= 0) // original != 0 maybe wrong
        //    bottom_diff[bottom_index]+=top_diff[index] ;

        //int min_x = bin_xs, 0), width - 1);
        //int min_y = min(max(bin_ys, 0), height - 1);
        //int max_x = max(min(bin_xs + 1, width - 1), 0);
        //int max_y = max(min(bin_ys + 1, height - 1), 0);

        int min_x = (int)floor(bw);
        int max_x = (int)ceil(bw);
        int min_y = (int)floor(bh);
        int max_y = (int)ceil(bh);

        T top_diff_of_bin = top_diff[index];

        T v1 = wlt * top_diff_of_bin;
        T v2 = wrt * top_diff_of_bin;
        T v3 = wrb * top_diff_of_bin;
        T v4 = wlb * top_diff_of_bin;

        // Atomic add

        if (min_y > 0 && min_x  > 0 && min_y < height - 1 && min_x < width - 1)
            atomicAdd(offset_bottom_diff + min_y * width + min_x, static_cast<T>(v1));
        if (min_y > 0 && max_x < width - 1 && min_y < height - 1 && max_x > 0)
            atomicAdd(offset_bottom_diff + min_y * width + max_x, static_cast<T>(v2));
        if (max_y < height - 1 && max_x < width - 1 && max_y > 0 && max_x > 0)
            atomicAdd(offset_bottom_diff + max_y * width + max_x, static_cast<T>(v3));
        if (max_y < height - 1 && min_x > 0 && max_y > 0 && min_x < width - 1)
            atomicAdd(offset_bottom_diff + max_y * width + min_x, static_cast<T>(v4));

        //}

  }
}


// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor RROIAlign_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& rois,
                                 const at::Tensor& con_idx_x,
                                 const at::Tensor& con_idx_y,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  // TODO add more checks

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "RROIAlign_backward", [&] {
    RROIAlignBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         con_idx_x.data<float>(),
         con_idx_y.data<float>(),
         num_rois,
         spatial_scale,
         height,
         width,
         channels,
         pooled_height,
         pooled_width,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
