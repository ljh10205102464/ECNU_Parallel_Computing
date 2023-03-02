#include <iostream>
#include <vector>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/shape.h>
#include <yannpp/common/log.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/network/activator.h>

#include <omp.h>
#include <time.h>

using namespace yannpp;

int fill_array(yannpp::array3d_t<float> &arr, int start=0) {
    int i = start;
    auto slice = arr.slice();
    auto it = slice.iterator();
    for (; it.is_valid(); ++it, i++) {
        auto index = *it;
        //printf("(%d, %d, %d) = %d\n", index.x(), index.y(), index.z(), i);
        slice.at(*it) = i;
    }

    return (i - 1);
}

std::vector<array3d_t<float>> func1(array3d_t<float> array){
    return {array};
}

array3d_t<float> func2(array3d_t<float> array){
    return array;
}

int main() {

    int input_shape = 40000;
    int output_shape = 40000;
    activator_t<float> relu_activator(relu_v<float>, relu_v<float>);
    fully_connected_layer_t<float> dense(input_shape, output_shape, relu_activator);

    array3d_t<float> weight(shape3d_t(input_shape, output_shape, 1), 0.f);
    fill_array(weight);
    // log(weight);
    // for (auto &v: weight.data()) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;
    array3d_t<float> bias(shape3d_t(output_shape, 1, 1), 0.f);

    std::vector<array3d_t<float>> weights = { weight };
    std::vector<array3d_t<float>> biases = { bias };

    dense.load(func1(weight), func1(bias));

    array3d_t<float> input(shape3d_t(input_shape, 1, 1), 1.f);

    double start, end;
    int times = 1;

    start = omp_get_wtime();
    
    for (int i=0;i<times;i++)
        auto output = dense.feedforward(func2(input));

    end = omp_get_wtime();
    std::cout<<"time = "<<(end-start)/times<<"s"<<std::endl;

    // log(output);

    // log(weight.flatten());

    // for (auto &v: weight.data()) {
    //     std::cout << v << " ";
    // }
    // std::cout << std::endl;

    return 0;
}
