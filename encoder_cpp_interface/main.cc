#define PROLOG_MODLUE "encoder"
#include <iostream>
#include <boost/asio.hpp>
#include <SWI-Prolog.h>
#include <SWI-cpp.h>
#include <memory>

#define PORT 2042
#define HOST "127.0.0.1"

using namespace boost::asio;

std::string read_(ip::tcp::socket &socket)
{
    streambuf buf;
    read_until(socket, buf, "\n");
    std::string data = buffer_cast<const char *>(buf.data());
    return data;
}

void send_(ip::tcp::socket &socket, const std::string &message)
{
    boost::system::error_code error;
    write(socket, buffer(message), error);
    if (error)
    {
        std::cout << "Error while querying server: " << error.message() << std::endl;
    }
}

std::unique_ptr<std::vector<double>> process_embedding(std::string &response)
{
    auto result = std::make_unique<std::vector<double>>();
    char delimiter = ',';
    std::string token;

    // Splitting the string
    size_t pos = 0;
    while ((pos = response.find(delimiter)) != std::string::npos)
    {
        token = response.substr(0, pos);
        result->push_back(std::stod(token));
        response.erase(0, pos + 1);
    }

    return result;
}

std::unique_ptr<std::vector<double>> get_embedding(const std::string &clause)
{

    // Connect to server
    io_service service;
    ip::tcp::socket socket(service);
    socket.connect(
        ip::tcp::endpoint(
            ip::address::from_string(HOST),
            PORT));

    // Query server
    send_(socket, clause);

    // Server response
    auto response = read_(socket);
    return process_embedding(response);
}

PREDICATE(test2, 2)
{
    PlTail tail(A1);
    PlTerm e;

    while (tail.next(e))
    {
        std::cout << "Received term: ";
        std::cout << (char *)e << std::endl;
    }

    return TRUE;
}

PREDICATE(encode_clause, 2)
{
    PlTerm e(A1);
    std::cout << "The receved term: " << (char *)e << std::endl;
    std::string clause_str((char *)e);
    auto embedding = get_embedding(clause_str);

    PlTail l2(A2);
    for (auto d : *embedding)
    {
        if (!l2.append(d))
        {
            return FALSE;
        }
    }
    l2.close();
    return TRUE;
}