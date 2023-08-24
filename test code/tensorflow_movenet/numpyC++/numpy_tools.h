//
// Created by adam on 7/19/23.
//

#ifndef TENSORFLOW_MOVENET_NUMPY_TOOLS_H
#define TENSORFLOW_MOVENET_NUMPY_TOOLS_H

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& array);

template <typename T>
std::vector<T> expand_dims(const std::vector<T>& array, int axis);

#endif //TENSORFLOW_MOVENET_NUMPY_TOOLS_H
