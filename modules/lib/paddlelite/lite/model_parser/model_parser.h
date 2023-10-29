// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file contains model format related operations, such as load a model,
// parse an operator definitions and so on.

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "lite/core/scope.h"
#include "lite/core/variable.h"
#include "lite/model_parser/compatible_pb.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {


void LoadParamNaive(const std::string& path,
                    lite::Scope* scope,
                    const std::string& name);

// warning:this old inference will be abandened in release/v3.0.0
// and LoadModelNaiveFromFile is suggested.
void LoadModelNaive(const std::string& model_dir,
                    lite::Scope* scope,
                    cpp::ProgramDesc* prog,
                    bool combined = true);
void LoadModelNaiveFromFile(const std::string& filename,
                            lite::Scope* scope,
                            cpp::ProgramDesc* prog);
void LoadModelNaiveFromMemory(const std::string& model_buffer,
                              const std::string& param_buffer,
                              lite::Scope* scope,
                              cpp::ProgramDesc* cpp_prog);
void LoadModelNaiveFromMemory(const std::string& model_buffer,
                              lite::Scope* scope,
                              cpp::ProgramDesc* cpp_prog);

}  // namespace lite
}  // namespace paddle
