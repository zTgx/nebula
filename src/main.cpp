#include <iostream>
#include <boost/asio.hpp>
#include "server.hpp"

int main() {
    std::cout << "Server started on port 12345" << std::endl;
   
    boost::asio::io_context io_context;
    Server server(io_context, 12345);
    io_context.run();

    return 0;
}
