// This implementation is from https://github.com/whai362/PSENet/blob/master/pse/adaptor.cpp

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <iostream>
#include <queue>

using namespace std;

namespace py = pybind11;

namespace pse_adaptor {

    class Point2d {
    public:
        int x;
        int y;

        Point2d() : x(0), y(0)
        {}

        Point2d(int _x, int _y) : x(_x), y(_y)
        {}
    };

    void growing_text_line(const int *data,
                           vector<long int> &data_shape,
                           const int *label_map,
                           vector<long int> &label_shape,
                           int &label_num,
                           float &min_area,
                           vector<vector<int>> &text_line) {
        int area[label_num + 1];
        memset(area, 0, sizeof(area));
        for (int x = 0; x < label_shape[0]; ++x) {
            for (int y = 0; y < label_shape[1]; ++y) {
                int label = label_map[x * label_shape[1] + y];
                if (label == 0) continue;
                area[label] += 1;
            }
        }

        queue<Point2d> queue, next_queue;
        for (int x = 0; x < label_shape[0]; ++x) {
            vector<int> row(label_shape[1]);
            for (int y = 0; y < label_shape[1]; ++y) {
                int label = label_map[x * label_shape[1] + y];
                if (label == 0) continue;
                if (area[label] < min_area) continue;

                Point2d point(x, y);
                queue.push(point);
                row[y] = label;
            }
            text_line.emplace_back(row);
        }

        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int kernel_id = data_shape[0] - 2; kernel_id >= 0; --kernel_id) {
            while (!queue.empty()) {
                Point2d point = queue.front();
                queue.pop();
                int x = point.x;
                int y = point.y;
                int label = text_line[x][y];

                bool is_edge = true;
                for (int d = 0; d < 4; ++d) {
                    int tmp_x = x + dx[d];
                    int tmp_y = y + dy[d];

                    if (tmp_x < 0 || tmp_x >= (int)text_line.size()) continue;
                    if (tmp_y < 0 || tmp_y >= (int)text_line[1].size()) continue;
                    int kernel_value = data[kernel_id * data_shape[1] * data_shape[2] + tmp_x * data_shape[2] + tmp_y];
                    if (kernel_value == 0) continue;
                    if (text_line[tmp_x][tmp_y] > 0) continue;

                    Point2d point(tmp_x, tmp_y);
                    queue.push(point);
                    text_line[tmp_x][tmp_y] = label;
                    is_edge = false;
                }

                if (is_edge) {
                    next_queue.push(point);
                }
            }
            swap(queue, next_queue);
        }
    }

    vector<vector<int>> pse(py::array_t<int, py::array::c_style | py::array::forcecast> quad_n9,
                            float min_area,
                            py::array_t<int32_t, py::array::c_style> label_map,
                            int label_num) {
        auto buf = quad_n9.request();
        auto data = static_cast<int *>(buf.ptr);
        vector<long int> data_shape = buf.shape;

        auto buf_label_map = label_map.request();
        auto data_label_map = static_cast<int32_t *>(buf_label_map.ptr);
        vector<long int> label_map_shape = buf_label_map.shape;

        vector<vector<int>> text_line;

        growing_text_line(data,
                          data_shape,
                          data_label_map,
                          label_map_shape,
                          label_num,
                          min_area,
                          text_line);

        return text_line;
    }
}

PYBIND11_PLUGIN(pse) {
    py::module m("pse", "pse");

    m.def("pse", &pse_adaptor::pse, "pse");

    return m.ptr();
}
