/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Visualizing and Communicating Errors in Rendered Images
// Ray Tracing Gems II, 2021,
// by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
// Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

// Visualizing Errors in Rendered High Dynamic Range Images
// Eurographics 2021,
// by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
// Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

// FLIP: A Difference Evaluator for Alternating Images
// High Performance Graphics 2020,
// by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
// Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
// Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.

// Code by Pontus Ebelin (formerly Andersson), Jim Nilsson, and Tomas Akenine-Moller.

#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <regex>
#include <cctype>

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

namespace FLIP
{

    class filename
    {
    private:
        std::string mDirectory;
        std::string mName;
        std::string mExtension;

        bool mIsNumbered;
        size_t mNumber;
        size_t mNumberWidth;

    private:
        const std::string numberString(void) const
        {
            std::stringstream ss;
            ss << std::setw(mNumberWidth) << std::setfill('0') << mNumber;
            return ss.str();
        }


    public:
        filename() { init(); };
        filename(const std::string& path) { parse(path); }
        filename(const filename& fileName) { *this = fileName; }
        filename(const std::string& path0, const std::string& path1, const std::string& newDirectory = "") { merge(path0, path1, newDirectory); }
        filename(const filename& fileName0, const filename& fileName1, const std::string& newDirectory = "") { merge(fileName0, fileName1, newDirectory); }
        ~filename() {};

        filename operator=(const std::string& path) { parse(path); return *this; }

        bool operator==(const filename& fileName) const
        {
            return this->mDirectory == fileName.mDirectory &&
                this->mName == fileName.mName &&
                this->mExtension == fileName.mExtension &&
                this->mIsNumbered == fileName.mIsNumbered && (this->mIsNumbered == false || (this->mNumber == fileName.mNumber && this->mNumberWidth == fileName.mNumberWidth));
        }
        bool operator!=(const filename& fileName) const { return !(*this == fileName); }

        inline void setDirectory(const std::string directory)
        {
            // Remove trailing slashes.
            this->mDirectory = std::regex_replace(directory, std::regex("[\\\\/]+$"), "");
        }
        inline const std::string& getDirectory(void) const { return this->mDirectory; }

        inline void setName(std::string name)
        {
            // Remove forbidden characters from the name.
            this->mName = std::regex_replace(name, std::regex("\\\\|/|:|\\*|\\?|\\||\"|<|>"), "_");
        }
        inline const std::string& getName(void) const { return this->mName; }

        inline void setExtension(std::string extension) { this->mExtension = extension; }
        inline const std::string& getExtension(void) const { return this->mExtension; }

        inline void setIsNumbered(bool isNumbered, size_t number = std::string::npos) { this->mIsNumbered = isNumbered; if (number != std::string::npos) this->mNumber = number; }
        inline bool getIsNumbered(void) const { return this->mIsNumbered; }

        inline void setNumber(size_t number) { this->mNumber = number; }
        inline size_t getNumber(void) const { return this->mNumber; }
        inline size_t incNumber(void) { return this->mNumber++; }
        inline size_t addNumber(size_t number) { return (this->mNumber += number); }

        inline void setNumberWidth(size_t numberWidth) { this->mNumberWidth = numberWidth; }
        inline const size_t& getNumberWidth(void) const { return this->mNumberWidth; }

        bool hasWildcard(void) const { return this->toString().find_first_of("*?[]{}") != std::string::npos; }

        bool isEmpty(void) const
        {
            return *this == empty();
        }

        std::string getNameExtension(bool toLower = false) const
        {
            if (*this == this->empty())
                return "";

            std::string fn = this->mName +
                (this->mIsNumbered ? "." + this->numberString() : "") +
                (this->mExtension.empty() ? "" : "." + this->mExtension);
            if (toLower)
            {
                std::transform(fn.begin(), fn.end(), fn.begin(), ::tolower);
            }

            return fn;
        }


        std::string getCompactPath(size_t maxLen = 64, bool toLower = false) const
        {
            maxLen = std::min(maxLen, size_t(MAX_PATH));

            std::string directoryString = this->directoryToString(toLower);
            std::string fileNameString = "/" + this->fileNameToString(toLower);

            int directoryStringLength = int(directoryString.length());
            int fileNameStringLength = int(fileNameString.length());
            int totalLength = directoryStringLength + fileNameStringLength;

            if (totalLength > maxLen)
            {
                //  First shorten the directory.
                int overflow = totalLength - int(maxLen);

                if (directoryStringLength > overflow)
                {
                    directoryString = directoryString.substr(0, std::min(std::max(directoryStringLength - 3 - overflow, 0), directoryStringLength)) + "...";
                }
                else
                {
                    //  We can't keep any of the directory.
                    if (maxLen == 0)
                    {
                        directoryString = "";
                    }
                    else if (maxLen == 1)
                    {
                        directoryString = ".";
                    }
                    else if (maxLen == 2)
                    {
                        directoryString = "..";
                    }
                    else
                    {
                        directoryString = "...";
                    }
                }

                directoryStringLength = int(directoryString.length());
                totalLength = directoryStringLength + fileNameStringLength;

                overflow = totalLength - int(maxLen);

                if (overflow > 0)
                {
                    //  We can't keep the whole filename.
                    if (maxLen < 4)
                    {
                        fileNameString = "";
                    }
                    else if (maxLen == 4)
                    {
                        fileNameString = "/";
                    }
                    else if (maxLen == 5)
                    {
                        fileNameString = "/.";
                    }
                    else if (maxLen == 6)
                    {
                        fileNameString = "/..";
                    }
                    else if (maxLen == 7)
                    {
                        fileNameString = "/...";
                    }
                    else
                    {
                        fileNameString = fileNameString.substr(0, std::min(std::max(fileNameStringLength - 3 - overflow, 0), fileNameStringLength)) + "...";
                    }
                }
            }

            std::string pathString = directoryString + fileNameString;

            if (toLower)
            {
                std::transform(pathString.begin(), pathString.end(), pathString.begin(), ::tolower);
            }

            return pathString;
        }

        // Returns a relative path to the file by attempting to remove basePath part.
        // Unlike PathRelativePathTo function provided by Windows, this function will
        // give up and return an empty string if current file is not in basePath.
        std::string getRelativePath(std::string const& basePath, bool caseInsensitive = false) const
        {
            uint32_t relativePathStart = 0;
            auto basePathLength = basePath.size();
            // Remove trailing slashes.
            while (basePath[basePathLength - 1] == '\\' || basePath[basePathLength - 1] == '/')
                --basePathLength;

            for (uint32_t i = 0; i < basePath.size(); ++i)
            {
                if (basePath[i] == mDirectory[i])
                    ++relativePathStart;
                else if (caseInsensitive && std::tolower(basePath[i]) == std::tolower(mDirectory[i]))
                    ++relativePathStart;
                else
                    break;
            }
            std::string relativePath;
            if (relativePathStart == basePath.size())
                relativePath = toString().substr(relativePathStart);
            return relativePath;
        }


        std::string directoryToString(bool toLower = false) const
        {
            if (*this == this->empty())
                return "";

            std::stringstream ss;
            if (this->mDirectory != "")
            {
                ss << this->mDirectory;
            }

            std::string string = ss.str();

            if (toLower)
            {
                std::transform(string.begin(), string.end(), string.begin(), ::tolower);
            }

            return string;
        }


        std::string fileNameToString(bool toLower = false) const
        {
            if (*this == this->empty())
                return "";

            std::stringstream ss;
            ss << this->mName;
            if (this->mIsNumbered)
            {
                ss << "." << this->numberString();
            }
            if (this->mExtension != "")
            {
                ss << "." << this->mExtension;
            }

            std::string string = ss.str();

            if (toLower)
            {
                std::transform(string.begin(), string.end(), string.begin(), ::tolower);
            }

            return string;
        }


        std::string toString(bool toLower = false) const
        {
            if (*this == this->empty())
                return "";

            std::stringstream ss;
            if (this->mDirectory != "")
            {
#if _WIN32
                ss << this->mDirectory << "\\";
#else
                ss << this->mDirectory << "/";
#endif
            }
            ss << this->mName;
            if (this->mIsNumbered)
            {
                ss << "." << this->numberString();
            }
            if (this->mExtension != "")
            {
                ss << "." << this->mExtension;
            }

            std::string string = ss.str();

            if (toLower)
            {
                std::transform(string.begin(), string.end(), string.begin(), ::tolower);
            }

            return string;
        }


        std::string toStringNumberWildcard(bool toLower = false) const
        {
            if (*this == this->empty())
                return "";

            std::stringstream ss;
            if (this->mDirectory != "")
            {
#if _WIN32
                ss << this->mDirectory << "\\";
#else
                ss << this->mDirectory << "/";
#endif
            }
            ss << this->mName;
            if (this->mIsNumbered)
            {
                ss << ".*";
            }
            if (this->mExtension != "")
            {
                ss << "." << this->mExtension;
            }

            std::string string = ss.str();

            if (toLower)
            {
                std::transform(string.begin(), string.end(), string.begin(), ::tolower);
            }

            return string;
        }


        void init(void)
        {
            this->mDirectory = "";
            this->mName = "";
            this->mExtension = "";
            this->mIsNumbered = false;
            this->mNumber = 0;
            this->mNumberWidth = 4;
        }


        static const filename& empty(void)
        {
            static const filename emptyFileName = filename();
            return emptyFileName;
        }


        bool parse(std::string path)
        {
            init();

            // Must be at least one slash to contain a directory.
            size_t iLastSlash = path.find_last_of("\\/");
            if (iLastSlash != std::string::npos)
            {
                this->mDirectory = path.substr(0, iLastSlash);
                this->mDirectory = this->mDirectory.substr(0, this->mDirectory.find_last_not_of("\\/") + 1);

                // No wildcards allowed in the directory.
                if (this->mDirectory.find_first_of("*?[]{}") != std::string::npos)
                    return false;
            }

            // Period, plus no later slash, constitutes an extension.
            size_t iLastPeriod = path.find_last_of(".");
            if (iLastPeriod != std::string::npos && (iLastSlash == std::string::npos || iLastPeriod > iLastSlash))
            {
                this->mExtension = path.substr(path.find_last_of(".") + 1);
            }

            path = path.substr(0, iLastPeriod);  // Remove extension.
            path = path.substr(path.find_last_of("\\/") + 1);  // Remove directory.

            this->mIsNumbered = false;

            iLastPeriod = path.find_last_of(".");
            if (iLastPeriod != std::string::npos)
            {
                std::string numberString = path.substr(iLastPeriod + 1);
                if (numberString.find_last_not_of("0123456789") == std::string::npos)
                {
                    this->mIsNumbered = true;
                    this->mNumber = atoi(numberString.c_str());
                    this->mNumberWidth = numberString.size();
                    path = path.substr(0, path.find_last_of("."));  // Remove number.
                }
            }
            this->mName = path;
            return true;
        }


        void merge(const filename& fn0, const filename& fn1, const std::string& newDirectory = "")
        {
            assert(fn0.getIsNumbered() == fn1.getIsNumbered());

            if (newDirectory == "")
            {
                this->mDirectory = fn0.getDirectory();
            }
            else
            {
                filename tmpFilename(newDirectory + "/");
                this->mDirectory = tmpFilename.getDirectory();
            }

            // Remove common preamble, delimited by periods.
            size_t last = 0;
            size_t lastPeriod = 0;
            std::string name0 = fn0.getName();
            std::string name1 = fn1.getName();
            while (last < name0.size() && last < name1.size() && name0[last] == name1[last])
            {
                if (name0[last] == '.')
                    lastPeriod = last;
                last++;
            }
            if (lastPeriod > 0)
                lastPeriod++;

            this->mName = fn0.getName() + "-" + fn1.getName().substr(lastPeriod);

            this->mIsNumbered = fn0.getIsNumbered();
            this->mNumberWidth = fn0.getNumberWidth();
            this->mNumber = fn0.getNumber();

            this->mExtension = fn0.getExtension();
        }


        void merge(const std::string& fullName0, const std::string& fullName1, const std::string& newDirectory = "")
        {
            filename fn0(fullName0);
            filename fn1(fullName1);

            merge(fn0, fn1, newDirectory);
        }


        void shiftExtension(void)
        {
            size_t index = this->mName.find_last_of(".");
            if (index != std::string::npos)
            {
                this->mExtension = this->mName.substr(index + 1);
                this->mName = this->mName.substr(0, index);
            }
        }


        static void sortFileNamesNatural(std::vector<std::string>& filenames)
        {
            std::sort(filenames.begin(), filenames.end(),
                [](std::string s0, std::string s1)
                {
                    // Favor short strings, e.g., img_2.png will come before img_10.png.
                    // -- not the ideal solution, but some people really wanted this. (TAM)
                    return s0.length() == s1.length() ? s0 < s1 : s0.length() < s1.length();
                    //return s0 < s1;
                });
        }

    };

}
