#pragma once

#include <boost/asio.hpp>
#include <boost/json.hpp>
#include <memory>
#include <queue>
#include <iostream>
#include "cuda_engine.hpp" // 包含 CUDA 计算引擎的头文件
#include <boost/beast.hpp>
#include <boost/beast/version.hpp>
#include <boost/beast/http.hpp>

class Server {
public:
    Server(boost::asio::io_context& io_context, short port);
    
private:
    void do_accept();
    void handle_request(std::shared_ptr<boost::asio::ip::tcp::socket> socket);
    boost::asio::ip::tcp::acceptor acceptor_;
    std::queue<boost::json::value> task_queue_;

private:
    void process_http_request(
    std::shared_ptr<boost::asio::ip::tcp::socket> socket,
    std::shared_ptr<boost::beast::http::request_parser<boost::beast::http::string_body>> parser,
    const boost::system::error_code& ec);

    private:
    void send_http_error(
        std::shared_ptr<boost::asio::ip::tcp::socket> socket,
        unsigned version,
        boost::beast::http::status status,
        const std::string& message)
    {
        boost::beast::http::response<boost::beast::http::string_body> res{status, version};
        res.set(boost::beast::http::field::content_type, "text/plain");
        res.body() = message;
        res.prepare_payload();

        boost::beast::http::async_write(*socket, res,
            [socket](const boost::system::error_code& ec, size_t) {
                if (ec) {
                    std::cerr << "Error sending error response: " << ec.message() << std::endl;
                }
                socket->close();
            });
    }

private:
    CudaEngine cuda_engine_; // CUDA 计算引擎实例
};
