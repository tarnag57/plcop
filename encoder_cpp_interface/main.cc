#define PROLOG_MODLUE "encoder"
#include <iostream>
#include <boost/asio.hpp>
#include <chrono>
#include <SWI-Prolog.h>
#include <SWI-cpp.h>
#include <memory>
#include <regex>

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

// Replaces each skolem function with SKLM and variable with VAR
std::string process_skolemisation(std::string &clause)
{
    // Remove '#' (used internally in leancop)
    std::regex match("#,|#");
    clause = std::regex_replace(clause, match, "");

    // Replace skolemized variables with VAR symbol
    match = std::regex("[0-9]+\\^\\[\\]");
    clause = std::regex_replace(clause, match, "VAR");

    // Replace numbered variables with VAR symbol
    // There are two types of numbered variables:
    //     - _1234: Used by Prolog as an uninitialised var
    //     - Q42: Used by leancop to mark substitutions in the extension steps
    match = std::regex("([^a-zA-Z0-9])_[0-9]+");
    clause = std::regex_replace(clause, match, "$1VAR");
    match = std::regex("[A-Z][0-9]+");
    clause = std::regex_replace(clause, match, "VAR");

    // Replace (potentially nested) skolem functions with SKLM symbol
    match = std::regex("[0-9]+\\^\\[[^\\[\\]]*\\]");
    std::smatch sm;
    while (std::regex_search(clause, sm, match))
    {
        clause = std::regex_replace(clause, match, "SKLM");
    }

    return clause;
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

std::unique_ptr<std::vector<double>> get_embedding(std::string &clause)
{

    auto begin_time = std::chrono::high_resolution_clock::now();

    // Connect to server
    io_service service;
    ip::tcp::socket socket(service);
    socket.connect(
        ip::tcp::endpoint(
            ip::address::from_string(HOST),
            PORT));

    // Reformat input clause (get rid off skolem functions and variables)
    clause = process_skolemisation(clause);
    std::cout << "Sending to server: " << clause << std::endl;

    // Query server
    send_(socket, clause);

    // Server response
    auto response = read_(socket);
    auto result = process_embedding(response);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_duration = end_time - begin_time;
    std::cout << "Elapsed (ms): " << ms_duration.count() << std::endl;

    return result;
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
    std::cout << "The received term: " << (char *)e << std::endl;
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