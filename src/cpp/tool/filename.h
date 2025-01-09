/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES
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
#include <cstdint>
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

    public:
        filename() { init(); };
        filename(const std::string& path) { parse(path); }
        filename(const filename& fileName) { *this = fileName; }
        ~filename() {};

        filename operator=(const std::string& path) { parse(path); return *this; }

        bool operator==(const filename& fileName) const
        {
            return this->mDirectory == fileName.mDirectory &&
                this->mName == fileName.mName &&
                this->mExtension == fileName.mExtension;
        }
        bool operator!=(const filename& fileName) const { return !(*this == fileName); }

        inline void setName(std::string name)
        {
            // Remove forbidden characters from the name.
            this->mName = std::regex_replace(name, std::regex("\\\\|/|:|\\*|\\?|\\||\"|<|>"), "_");
        }
        inline const std::string& getName(void) const { return this->mName; }

        inline void setExtension(std::string extension) { this->mExtension = extension; }
        inline const std::string& getExtension(void) const { return this->mExtension; }


        static const filename& empty(void)
        {
            static const filename emptyFileName = filename();
            return emptyFileName;
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
        }

        bool parse(std::string path)
        {
            init();
            size_t periodCount = std::count_if(path.begin(), path.end(), [](char c) {return c == '.'; });
            if (path[0] == '.' && periodCount == 1)
            {
                mExtension = path.substr(1);
                return true;
            }

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

            this->mName = path;
            return true;
        }

    };

}
