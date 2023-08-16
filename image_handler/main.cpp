#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "json/json.h"

int main(void)
{
    std::string path;
    path = "hello world!";

    std::string str;
    Json::Value root;
    Json::StyledWriter writer;
    str = writer.write(root);
    

    std::cout << path << std::endl;
    std::cout << str << std::endl;

    return 0;
}