#include <boost/function.hpp>
#pragma once

#include <string>

class Inferencer {
public:
    virtual ~Inferencer() = default;

    virtual bool loadModel(const std::string& path) = 0;

    virtual unsigned getInputBuffer(const std::string& name, void** buffer) = 0;

    virtual unsigned getOutputBuffer(const std::string& name, void** buffer) = 0;

    virtual bool infer() = 0;

    virtual const char* getErrorString() const = 0;
};