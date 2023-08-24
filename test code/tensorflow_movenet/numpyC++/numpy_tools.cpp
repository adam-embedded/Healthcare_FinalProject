//
// Created by adam on 7/19/23.
//


#include <vector>
#include "numpy_tools.h"


template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& array) {
    std::vector<T> flattened;
    for (const auto& row : array) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

template <typename T>
std::vector<T> expand_dims(const std::vector<T>& array, int axis) {
    std::vector<T> expanded_array;
    expanded_array.reserve(array.size() + 1);

    // Insert a new axis at the specified position.
    for (int i = 0; i < array.size(); i++) {
        if (i == axis) {
            expanded_array.push_back(1);
        }
        expanded_array.push_back(array[i]);
    }

    return expanded_array;
}