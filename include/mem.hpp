#pragma once

#include <string>
#include <iostream>
// #include <format> // Requires C++20 or later

struct X {
    std::string s;

    X(std::string_view s): s{s} {
        std::cout << "X::X({}) -> " << s << std::endl;
    }

    ~X() {
        std::cout << "X::~X({}) -> " << s << std::endl;
    }
};

X glob { "glob" };
void  g() {
    X xg{ "g()" };
}