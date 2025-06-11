#include "server.hpp"
#include <boost/beast.hpp>
#include <boost/beast/http.hpp>

using namespace boost::asio;

Server::Server(io_context& io_context, short port)
    : acceptor_(io_context, ip::tcp::endpoint(ip::tcp::v4(), port)) {
    do_accept();
}

void Server::do_accept() {
    auto socket = std::make_shared<ip::tcp::socket>(acceptor_.get_executor());
    acceptor_.async_accept(*socket, [this, socket](const boost::system::error_code& ec) {
        if (!ec) {
            handle_request(socket);
        }
        do_accept();
    });
}


void Server::handle_request(std::shared_ptr<ip::tcp::socket> socket) {
    auto parser = std::make_shared<boost::beast::http::request_parser<boost::beast::http::string_body>>();
    auto buffer = std::make_shared<boost::beast::flat_buffer>();

    boost::beast::http::async_read_header(*socket, *buffer, *parser,
        [this, socket, parser, buffer](const boost::system::error_code& ec, size_t) {
            if (ec) {
                std::cerr << "Header error: " << ec.message() << std::endl;
                return;
            }

            // Read remaining body if content-length exists
            if (parser->content_length()) {
                parser->body_limit(parser->content_length());
                boost::beast::http::async_read(*socket, *buffer, *parser,
                    [this, socket, parser](const boost::system::error_code& ec, size_t) {
                        process_http_request(socket, parser, ec);
                    });
            } else {
                process_http_request(socket, parser, ec);
            }
        });
}

void Server::process_http_request(
    std::shared_ptr<ip::tcp::socket> socket,
    std::shared_ptr<boost::beast::http::request_parser<boost::beast::http::string_body>> parser,
    const boost::system::error_code& ec)
{
    if (ec) {
        std::cerr << "Read error: " << ec.message() << std::endl;
        return;
    }

    try {
        auto request = boost::json::parse(parser->get().body());
        auto result = cuda_engine_.compute(request);

        auto res = std::make_shared<boost::beast::http::response<boost::beast::http::string_body>>(
            boost::beast::http::status::ok, 
            parser->get().version()
        );

        res->set(boost::beast::http::field::content_type, "application/json");
        res->body() = boost::json::serialize(result);
        res->prepare_payload();

        std::cout << "Sending response: " << res->body() << std::endl;

        // Capture res in the lambda to keep it alive
        boost::beast::http::async_write(*socket, *res,
            [socket, res](const boost::system::error_code& ec, size_t) {
                if (ec) {
                    std::cerr << "Write error: " << ec.message() << std::endl;
                }
                socket->close();
            });
    } catch (const std::exception& e) {
        std::cerr << "Processing error: " << e.what() << std::endl;
        send_http_error(socket, parser->get().version(), 
                       boost::beast::http::status::bad_request, 
                       e.what());
    }
}