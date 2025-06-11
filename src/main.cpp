#include <iostream>
#include "mem.hpp"

int main() {
    std::cout << "main()\n";
    X xm{ "main()" };
    g();

    X* p = new X{ "new X" };
    delete p;

    std::cout << "end of main()\n";
    return 0;
}
