// This implementation is from https://github.com/WenmuZhou/PAN.pytorch/blob/master/post_processing/pse.cpp

#include <queue>
#include <math.h>
#include <map>
#include <algorithm>
#include <vector>
#include <array>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

namespace py = pybind11;


namespace panet{
    py::array_t<uint8_t> assign_pixels(
    py::array_t<uint8_t, py::array::c_style> text,
    py::array_t<float, py::array::c_style> similarity_vectors,
    py::array_t<int32_t, py::array::c_style> label_map,
    int label_num,
    float dis_threshold = 0.8)
    {
        auto pbuf_text = text.request();
        auto pbuf_similarity_vectors = similarity_vectors.request();
        auto pbuf_label_map = label_map.request();
        if (pbuf_label_map.ndim != 2 || pbuf_label_map.shape[0]==0 || pbuf_label_map.shape[1]==0)
            throw std::runtime_error("label map must have a shape of (h>0, w>0)");
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];
        if (pbuf_similarity_vectors.ndim != 3 || pbuf_similarity_vectors.shape[0]!=h || pbuf_similarity_vectors.shape[1]!=w || pbuf_similarity_vectors.shape[2]!=4 ||
            pbuf_text.shape[0]!=h || pbuf_text.shape[1]!=w)
            throw std::runtime_error("similarity_vectors must have a shape of (h,w,4) and text must have a shape of (h,w,4)");
        //初始化结果
        auto res = py::array_t<uint8_t>(pbuf_text.size);
        auto pbuf_res = res.request();
        // 获取 text similarity_vectors 和 label_map的指针
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_text = static_cast<uint8_t *>(pbuf_text.ptr);
        auto ptr_similarity_vectors = static_cast<float *>(pbuf_similarity_vectors.ptr);
        auto ptr_res = static_cast<uint8_t *>(pbuf_res.ptr);

        std::queue<std::tuple<int, int, int32_t>> q;
        // 计算各个kernel的similarity_vectors
        std::vector<std::array<float, 5>> kernel_vector(label_num);

        // 文本像素入队列
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_res = ptr_res + i*w;
            auto p_similarity_vectors = ptr_similarity_vectors + i*w*4;
            for(int j = 0, k = 0; j<w && k < w * 4; j++,k+=4)
            {
                int32_t label = p_label_map[j];
                if (label>0)
                {
                    kernel_vector[label][0] += p_similarity_vectors[k];
                    kernel_vector[label][1] += p_similarity_vectors[k+1];
                    kernel_vector[label][2] += p_similarity_vectors[k+2];
                    kernel_vector[label][3] += p_similarity_vectors[k+3];
                    kernel_vector[label][4] += 1;
                    q.push(std::make_tuple(i, j, label));
                }
                p_res[j] = label;
            }
        }

        for(int i=0;i<label_num;i++)
        {
            for (int j=0;j<4;j++)
            {
                kernel_vector[i][j] /= kernel_vector[i][4];
            }
        }

        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        while(!q.empty()){
            //get each queue menber in q
            auto q_n = q.front();
            q.pop();
            int y = std::get<0>(q_n);
            int x = std::get<1>(q_n);
            int32_t l = std::get<2>(q_n);
            //store the edge pixel after one expansion
            auto kernel_cv = kernel_vector[l];
            for (int idx=0; idx<4; idx++)
            {
                int tmpy = y + dy[idx];
                int tmpx = x + dx[idx];
                auto p_res = ptr_res + tmpy*w;
                if (tmpy<0 || tmpy>=h || tmpx<0 || tmpx>=w)
                    continue;
                if (!ptr_text[tmpy*w+tmpx] || p_res[tmpx]>0)
                    continue;
                // 计算距离
                float dis = 0;
                auto p_similarity_vectors = ptr_similarity_vectors + tmpy * w*4;
                for(size_t i=0;i<4;i++)
                {
                    dis += pow(kernel_cv[i] - p_similarity_vectors[tmpx*4 + i],2);
                }
                dis = sqrt(dis);
                if(dis >= dis_threshold)
                    continue;
                q.push(std::make_tuple(tmpy, tmpx, l));
                p_res[tmpx]=l;
            }
        }
        return res;
    }

    std::map<int,std::vector<float>> estimate_text_confidence(
    py::array_t<int32_t, py::array::c_style> label_map,
    py::array_t<float, py::array::c_style> score_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto pbuf_score_map = score_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_score_map = static_cast<float *>(pbuf_score_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::map<int,std::vector<float>> point_dict;
        std::vector<std::vector<float>> point_vector;
        for(int i=0;i<label_num;i++)
        {
            std::vector<float> point;
            point.push_back(0);
            point.push_back(0);
            point_vector.push_back(point);
        }
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_score_map = ptr_score_map + i*w;
            for(int j = 0; j<w; j++)
            {
                int32_t label = p_label_map[j];
                if(label==0)
                {
                    continue;
                }
                float score = p_score_map[j];
                point_vector[label][0] += score;
                point_vector[label][1] += 1;
                point_vector[label].push_back(j);
                point_vector[label].push_back(i);
            }
        }
        for(int i=0;i<label_num;i++)
        {
            if(point_vector[i].size() > 2)
            {
                point_vector[i][0] /= point_vector[i][1];
                point_dict[i] = point_vector[i];
            }
        }
        return point_dict;
    }
    std::vector<int> get_pixel_num(
    py::array_t<int32_t, py::array::c_style> label_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::vector<int> point_vector;
        for(int i=0;i<label_num;i++){
            point_vector.push_back(0);
        }
        for (int i = 0; i<h; i++){
            auto p_label_map = ptr_label_map + i*w;
            for(int j = 0; j<w; j++){
                int32_t label = p_label_map[j];
                if(label==0){
                    continue;
                }
                point_vector[label] += 1;
            }
        }
        return point_vector;
    }
}

PYBIND11_MODULE(pan, m){
    m.def("assign_pixels", &panet::assign_pixels, " assign pixels to text kernels", py::arg("text"), py::arg("similarity_vectors"), py::arg("label_map"), py::arg("label_num"), py::arg("dis_threshold")=0.8);
    m.def("estimate_text_confidence", &panet::estimate_text_confidence, " estimate average confidence for text instances", py::arg("label_map"), py::arg("score_map"), py::arg("label_num"));
    m.def("get_pixel_num", &panet::get_pixel_num, " get each text instance pixel number", py::arg("label_map"), py::arg("label_num"));
}
