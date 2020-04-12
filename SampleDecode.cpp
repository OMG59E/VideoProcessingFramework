//
// Created by xingwg on 20-4-12.
//

#include "Tasks.hpp"
#include <memory>
#include <cuda.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace VPF;

Buffer* getElementaryVideo(DemuxFrame *demuxer)
{
    Buffer *elementaryVideo = nullptr;
    /**
     * Demuxer may also extracts elementary audio etc. from stream, so we run it
     * until we get elementary video;
     */
    do
    {
        if (TaskExecStatus::TASK_EXEC_FAIL == demuxer->Execute())
        {
            return nullptr;
        }
        elementaryVideo = (Buffer *)demuxer->GetOutput(0U);
    } while (!elementaryVideo);

    return elementaryVideo;
};

Surface* DecodeSingleSurface(NvdecDecodeFrame *decoder, DemuxFrame *demuxer)
{
    Surface *surface = nullptr;
    do
    {
        /* Get encoded frame from demuxer;
         * May be null, but that's ok - it will flush decoder;
         */
        auto elementaryVideo = getElementaryVideo(demuxer);

        /* Kick off HW decoding;
         * We may not have decoded surface here as decoder is async;
         */
        decoder->SetInput(elementaryVideo, 0U);
        if (TaskExecStatus::TASK_EXEC_FAIL == decoder->Execute())
        {
            break;
        }

        surface = (Surface *)decoder->GetOutput(0U);
        /* Repeat untill we got decoded surface;
         */
    } while (!surface);

    return surface;
};

int main()
{
    const char* videoFile = "/home/xingwg/nas/workspace/avs/data/video/ch00.264";

    unique_ptr<DemuxFrame> upDemuxer;
    unique_ptr<NvdecDecodeFrame> upDecoder;
    int gpuID = 0;
    cout << "Decoding on GPU " << gpuID << endl;

    cuInit(0);
    CUstream cuStream{nullptr};
    CUcontext cuContext{nullptr};
    cuCtxCreate(&cuContext, 0, gpuID);
    cuStreamCreate(&cuStream, 0);

    upDemuxer.reset(DemuxFrame::Make(videoFile));

    MuxingParams params;
    upDemuxer->GetParams(params);

    upDecoder.reset(NvdecDecodeFrame::Make(cuStream, cuContext, params.videoContext.codec,
            4, params.videoContext.width, params.videoContext.height));

    CUdeviceptr pFrame{0};
    size_t pitch{0};
    cuMemAllocPitch(&pFrame, &pitch, (size_t)3*params.videoContext.width, (size_t)params.videoContext.height, 16);

    cv::Mat vis(params.videoContext.height, params.videoContext.width, CV_8UC3);

    while (true)
    {
        Surface* surface = DecodeSingleSurface(upDecoder.get(), upDemuxer.get());
        if (!surface)
            break;

        const Npp8u *const pSrc[] = {(const Npp8u *const)surface->PlanePtr(0U),
                                     (const Npp8u *const)surface->PlanePtr(1U)};

        NppiSize oSizeRoi = {(int)surface->Width(), (int)surface->Height()};
        auto err = nppiNV12ToBGR_709HDTV_8u_P2C3R(pSrc, surface->Pitch(), (Npp8u*)pFrame, (int)pitch, oSizeRoi);

        if (NPP_NO_ERROR != err)
        {
            break;
        }
    }

    return 0;
};