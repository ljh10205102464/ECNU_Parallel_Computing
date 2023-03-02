#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/layers/poolinglayer.h>
#include <yannpp/common/log.h>
#include <yannpp/common/utils.h>

#include <omp.h>
#include <time.h>

using namespace yannpp;

int fill_array(array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        slice.at(*it) = (float)i;
    }

    return (i - 1);
}

array3d_t<float> maxpool2d(array3d_t<float> const &input, int window_size_, int stride) {
    point3d_t<int> stride_(stride, stride, 1);
    auto &input_shape_ = input.shape();
    // downsample input using window with step stride
    shape3d_t output_shape(POOL_DIM(input_shape_.x(), window_size_, stride_.x()),
                           POOL_DIM(input_shape_.y(), window_size_, stride_.y()),
                           input_shape_.z());
    array3d_t<float> result(output_shape, 0.f);
    array3d_t<index3d_t> max_index_(output_shape, index3d_t(0, 0, 0));

    // z axis corresponds to each filter from convolution layer
#   pragma omp parallel num_threads(4)
{
    size_t my_rank = omp_get_thread_num();
    size_t thread_count = omp_get_num_threads();
    size_t local_x = output_shape.z() / thread_count;
    size_t sub = output_shape.z() % thread_count;
    size_t start = my_rank * local_x + (sub > my_rank ? my_rank : sub);
    size_t end = start + local_x + (sub > my_rank ? 1 : 0);
    for (int z = start; z < end; z++) {
        // 2D loop over convoluted image from each filter
        for (int y = 0; y < output_shape.y(); y++) {
            int ys = y * stride_.y();

            for (int x = 0; x < output_shape.x(); x++) {
                int xs = x * stride_.x();
                // pooling layer does max-pooling, selecting a maximum
                // activation within the bounds of it's "window"
                auto input_slice = const_cast<array3d_t<float>&>(input)
                                   .slice(
                                       index3d_t(xs, ys, z),
                                       index3d_t(xs + window_size_ - 1,
                                                 ys + window_size_ - 1,
                                                 z));
                max_index_(x, y, z) = input_slice.argmax();
                result(x, y, z) = input_slice.at(max_index_(x, y, z));
            }
        }
    }
}

    return result;
}

array3d_t<float> create_input(shape3d_t const &shape) {
    array3d_t<float> arr(shape, 0.f);
    fill_array(arr);
    return arr;
}

int main(int argc, char* argv[]) {
    using namespace yannpp;

    array3d_t<float> test = create_input(shape3d_t(2000, 2000, 20));

    double start, end;

    start = omp_get_wtime();

//    [[[[18. 19. 20.]
//       [21. 22. 23.]
//       [24. 25. 26.]
//       [27. 28. 29.]]

//      [[33. 34. 35.]
//       [36. 37. 38.]
//       [39. 40. 41.]
//       [42. 43. 44.]]

//      [[48. 49. 50.]
//       [51. 52. 53.]
//       [54. 55. 56.]
//       [57. 58. 59.]]

//      [[63. 64. 65.]
//       [66. 67. 68.]
//       [69. 70. 71.]
//       [72. 73. 74.]]]]
    for (int i=0;i<10;i++)
        maxpool2d(test,
                  2,
                  2);

    end = omp_get_wtime();
    std::cout<<"time = "<<(end-start)/10<<"s"<<std::endl;

    return 0;
}
