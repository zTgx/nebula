#pragma once

#include <string>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <boost/thread.hpp>

void task(int x=0) {
    std::cout << "thread[" << boost::this_thread::get_id() << "] is running" << std::endl;
    std::cout << "x = " << x << std::endl;
}

void handle_joins(const std::string& input) {
    // This function processes the input string to handle joins.
    // The implementation details would depend on the specific requirements of the join operation.
    
    // Example placeholder logic:
    if (input.empty()) {
        throw std::invalid_argument("Input cannot be empty");
    }
    
    // Process the input string
    std::cout << "Processing join for input: " << input << std::endl;
    
    // Further processing logic would go here...

    boost::thread thread1(task, 1);
    boost::thread thread2(task, 2);
    
    thread1.join();
    thread2.join();
}