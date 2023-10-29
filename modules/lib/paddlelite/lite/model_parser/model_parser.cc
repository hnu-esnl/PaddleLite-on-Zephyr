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

#include "lite/model_parser/model_parser.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <set>
#include <unordered_set>
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/variable.h"
#include "lite/core/version.h"
#include "lite/model_parser/desc_apis.h"
#include "lite/model_parser/naive_buffer/combined_params_desc.h"
#include "lite/model_parser/naive_buffer/param_desc.h"
#include "lite/model_parser/naive_buffer/program_desc.h"
#include "lite/model_parser/naive_buffer/var_desc.h"
#include "lite/utils/io.h"




#include<string.h>
#include<stdio.h>
#include<unistd.h>
#include<iostream>
using namespace std;
#include <memory>
#include <string>
#include <vector>
size_t mem_read(void *buffer,size_t size,void *file)
{
    int i;
    if(buffer == NULL || size == 0 || file == NULL)
    {
        printf("error\n");
        return 0;
    }
    for(i=0;i<size;i++)
    {
    memcpy(buffer,file,1);
    buffer = buffer+1;
    file = file+1;
    }

    //memcpy(buffer,file,size);
    return 0;

}

namespace paddle {
namespace lite {



template <typename T>
void SetTensorDataNaive(T *out, size_t size, const std::vector<T> &src) {
  CHECK(out);
  CHECK(size == src.size());
  for (size_t i = 0; i < size; ++i) {
    out[i] = src[i];
  }
}

void GetParamInfoNaive(const naive_buffer::ParamDesc &desc,
                       lite::Scope *scope,
                       const std::string &name) {
  CHECK(scope);
  CHECK_EQ(desc.Name(), name)
      << "Var name not equal: ParamDesc.name=" << desc.Name()
      << "vs filename=" << name;
 
  auto *tensor = scope->Var(name)->GetMutable<lite::Tensor>();
  
  VLOG(3) << "model version " << desc.ModelVersion();
  CHECK_EQ(desc.TensorVersion(), 0U) << "Only version 0 is supported";
  
  // Load LoD info
  auto *tgt_lod = tensor->mutable_lod();
  auto desc_lod = desc.LoD();
  tgt_lod->assign(desc_lod.begin(), desc_lod.end());
   
  // Load Dim info
  tensor->Resize(lite::DDim(desc.Dim()));
 
  // Load data
  switch (desc.GetDataType()) {
#define SET_TENSOR(data_type__, T, precision)                            \
  case VarDescAPI::VarDataType::data_type__:                             \
    SetTensorDataNaive<T>(                                               \
        tensor->mutable_data<T>(), tensor->data_size(), desc.Data<T>()); \
    tensor->set_precision(precision);                                    \
    break
    // SET_TENSOR(BOOL, bool, PRECISION(kBool));
    SET_TENSOR(FP32, float, PRECISION(kFloat));
    SET_TENSOR(INT8, int8_t, PRECISION(kInt8));
    SET_TENSOR(INT16, int16_t, PRECISION(kInt16));
    SET_TENSOR(INT32, int32_t, PRECISION(kInt32));
    SET_TENSOR(INT64, int64_t, PRECISION(kInt64));
#undef SET_TENSOR
    default:
      
      LOG(FATAL) << "unknown type";
  }
 
  tensor->set_persistable(true);
 
}

void LoadParamNaive(const std::string &path,
                    lite::Scope *scope,
                    const std::string &name) {
  // Load param
  naive_buffer::BinaryTable table;
  table.LoadFromFile(path);
  naive_buffer::proto::ParamDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::ParamDesc desc(&pt_desc);
  GetParamInfoNaive(desc, scope, name);
}

void LoadCombinedParamsNaive(const std::string &path,
                             const uint64_t &offset,
                             lite::Scope *scope,
                             const cpp::ProgramDesc &cpp_prog,
                             bool params_from_memory) {
  naive_buffer::BinaryTable table;
  if (params_from_memory) {
    table.LoadFromMemory(path.c_str() + offset, path.length() - offset);
  } else {
    table.LoadFromFile(path, offset, 0);
  }
  naive_buffer::proto::CombinedParamsDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::CombinedParamsDesc desc(&pt_desc);

  std::set<std::string> param_names;
  for (size_t i = 0; i < desc.ParamsSize(); ++i) {
    naive_buffer::ParamDesc param_desc(desc.GetParam(i));
    GetParamInfoNaive(param_desc, scope, param_desc.Name());
    param_names.insert(param_desc.Name());
  }

  // Check all params loaded
  auto prog = cpp_prog;
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
      continue;
    CHECK(param_names.count(var.Name())) << "Persistable var[" << var.Name()
                                         << "] not found";
  }
}


void LoadCombinedParamsNaive(const std::string &path,
                             const uint64_t &offset,
                             lite::Scope *scope,
                             const cpp::ProgramDesc &cpp_prog,
                             bool params_from_memory,uint8_t*start) {
  naive_buffer::BinaryTable table;
  if (params_from_memory) {
    table.LoadFromMemory(path.c_str() + offset, path.length() - offset);
  } else {
     //uint64_t  topo_size1 = 100000;
     uint64_t  topo_size1 = 103000000;
    //table.LoadFromFile(path, offset, 0);
   // table.LoadFromFile(topo_size1,(uint8_t **)start);
    table.LoadFromFile(topo_size1,start);
    
  }
  naive_buffer::proto::CombinedParamsDesc pt_desc(&table);
  pt_desc.Load();
  naive_buffer::CombinedParamsDesc desc(&pt_desc);
  
  std::set<std::string> param_names;
  for (size_t i = 0; i < desc.ParamsSize(); ++i) {
    
    naive_buffer::ParamDesc param_desc(desc.GetParam(i));
   
    GetParamInfoNaive(param_desc, scope, param_desc.Name());
   
    param_names.insert(param_desc.Name());
   
  }
  
  // Check all params loaded
  auto prog = cpp_prog;
 
  
  auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);


  for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
    auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
    if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
      continue;
    CHECK(param_names.count(var.Name())) << "Persistable var[" << var.Name()
                                         << "] not found";
  }
  
 
}


void LoadModelNaive(const std::string &model_dir,
                    Scope *scope,
                    cpp::ProgramDesc *cpp_prog,
                    bool combined) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  LOG(WARNING)
      << "WARNING: MobileConfig::set_model_dir and "
         "MobileConfig::set_model_buffer are deprecated APIs "
         "and will be removed in latter release. \n"
         "    MobileConfig::set_model_from_file(const std::string& model_file)"
         " and MobileConfig::set_model_from_buffer(const std::string& "
         "model_buffer) are recommended.";
  // Load model
  const std::string prog_path = model_dir + "/__model__.nb";
  naive_buffer::BinaryTable table;
  table.LoadFromFile(prog_path);
  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  if (combined) {
    const std::string combined_params_path = model_dir + "/param.nb";
    LoadCombinedParamsNaive(combined_params_path, 0, scope, *cpp_prog, false);
  } else {
    auto &prog = *cpp_prog;
    auto &main_block_desc = *prog.GetBlock<cpp::BlockDesc>(0);
    for (size_t i = 0; i < main_block_desc.VarsSize(); ++i) {
      auto &var = *main_block_desc.GetVar<cpp::VarDesc>(i);
      if (var.Name() == "feed" || var.Name() == "fetch" || !var.Persistable())
        continue;

      std::string file_path = model_dir + "/" + var.Name() + ".nb";
      VLOG(4) << "reading weight " << var.Name();

      switch (var.GetType()) {
        case VarDescAPI::Type::LOD_TENSOR:
          LoadParamNaive(file_path, scope, var.Name());
          break;
        default:
          CHECK(false) << "unknown weight type";
      }
    }
  }

  VLOG(4) << "Load naive buffer model in '" << model_dir << "' successfully";
}

/*
 * Binary structure of naive_buffer model: model.nb
 * ----------------------------------------------------------
 * |       |    PART         |   Precision |   Length(byte) |
 * |   1   |  meta_version   |   uint16_t  |       2        |
 * |   2   |  opt_version    |   char[16]  |      16        |
 * |   3   |  topo_size      |   uint64_t  |       8        |
 * |   4   |  topo_data      |   char[]    | topo_size byte |
 * |   5   |  param_data     |   char[]    |                |
 * ----------------------------------------------------------
 *  Meaning of each part:
 *      meta_version: meata_version, 0 default.
 *      opt_version:  lite_version of opt tool that transformed this model.
 *      topo_size:    length of `topo_data`.
 *      topo_data:    contains model's topology data.
 *      param_data:   contains model's params data.
*/

// usage: LoadModelNaiveFromFile is used for loading model from file.
template <typename T>
void ReadModelDataFromFile(T *data,
                           const std::string &prog_path,
                           uint64_t *offset,
                           const uint64_t &size) {
  naive_buffer::BinaryTable data_table;
  data_table.LoadFromFile(prog_path, *offset, size);
  memcpy(data, data_table.cursor(), size);
  *offset = *offset + size;
}

void LoadModelNaiveFromFile(const std::string &filename,
                            Scope *scope,
                            cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();
  // ModelFile
   const std::string prog_path = filename;
  
  //int fd = open(filename.c_str(),O_RDWR);
 
 // uint8_t* start = ( uint8_t *)mmap(NULL,40000000,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
  uint8_t* start = (uint8_t*)0x70000000;

  // Offset
  uint64_t offset = 0;

  // (1)get meta version
  uint16_t meta_version;
  mem_read(&meta_version,2,start);
 // mem_read(&meta_version,2,(uint8_t **)start);
  //cout<<"meta_version is "<<meta_version<<endl;
  
  // (2)get opt version
  char opt_version[16];
  start+=2;

  //mem_read(opt_version,16,(uint8_t **)start);
  mem_read(opt_version,16,start);
  //cout<<"opt_version is "<<opt_version<<endl;
  

  // (3)get topo_size
  uint64_t topo_size;
  start+=16;
  //mem_read(&topo_size,8,(uint8_t **)start);
  mem_read(&topo_size,8,start);
  //cout<<"topo_size is "<<topo_size<<endl;
  

  // (4)get topo data
  naive_buffer::BinaryTable topo_table;
  
  start+=8;
 
  //topo_table.LoadFromFile(topo_size,(uint8_t **)start);
  topo_table.LoadFromFile(topo_size,start);

  //uint64_t topo_size2 = 200000;
  //topo_table.LoadFromFile(topo_size2,start);

  offset = offset + topo_size +2 + 16 + 8;


  // transform topo_data into cpp::ProgramDesc

  naive_buffer::proto::ProgramDesc nb_proto_prog(&topo_table);


  nb_proto_prog.Load();

  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);


  // (5)Load Params

  start+=topo_size;
  //LoadCombinedParamsNaive(prog_path, offset, scope, *cpp_prog, false,(uint8_t **)start);
  LoadCombinedParamsNaive(prog_path, offset, scope, *cpp_prog, false,start);

  VLOG(4) << "Load naive buffer model in '" << filename << "' successfully";

}

// warning: this is an old inference and is not suggested.
// todo: this inference will be abandened in release/v3.0.0
void LoadModelNaiveFromMemory(const std::string &model_buffer,
                              const std::string &param_buffer,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  // Load model

  naive_buffer::BinaryTable table;
  table.LoadFromMemory(model_buffer.c_str(), model_buffer.length());

  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  // only combined Params are supported in Loading Model from memory
  LoadCombinedParamsNaive(param_buffer, 0, scope, *cpp_prog, true);

  VLOG(4) << "Load model from naive buffer memory successfully";
}

// usage: LoadModelNaiveFromMemory is used for loading naive model from memory
template <typename T>
void ReadModelDataFromBuffer(T *data,
                             const std::string &model_buffer,
                             uint64_t *offset,
                             const uint64_t &size) {
  naive_buffer::BinaryTable data_table;
  data_table.LoadFromMemory(model_buffer.c_str() + *offset, size);
  memcpy(data, data_table.cursor(), size);
  *offset = *offset + size;
}
void LoadModelNaiveFromMemory(const std::string &model_buffer,
                              Scope *scope,
                              cpp::ProgramDesc *cpp_prog) {
  CHECK(cpp_prog);
  CHECK(scope);
  cpp_prog->ClearBlocks();

  // Offset
  uint64_t offset = 0;

  // (1)get meta version
  uint16_t meta_version;
  ReadModelDataFromBuffer<uint16_t>(
      &meta_version, model_buffer, &offset, sizeof(uint16_t));
  VLOG(4) << "Meta_version:" << meta_version;

  // (2)get opt version
  char opt_version[16];
  const uint64_t paddle_version_length = 16 * sizeof(char);
  ReadModelDataFromBuffer<char>(
      opt_version, model_buffer, &offset, paddle_version_length);
  VLOG(4) << "Opt_version:" << static_cast<const char *>(opt_version);

  // (3)get topo_size and topo_data
  uint64_t topo_size;
  ReadModelDataFromBuffer<uint64_t>(
      &topo_size, model_buffer, &offset, sizeof(uint64_t));
  naive_buffer::BinaryTable table;
  table.LoadFromMemory(model_buffer.c_str() + offset, topo_size);
  offset = offset + topo_size;

  naive_buffer::proto::ProgramDesc nb_proto_prog(&table);
  nb_proto_prog.Load();
  naive_buffer::ProgramDesc nb_prog(&nb_proto_prog);

  // Transform to cpp::ProgramDesc
  TransformProgramDescAnyToCpp(nb_prog, cpp_prog);

  // Load Params
  // NOTE: Only main block be used now.
  // only combined Params are supported in Loading Model from memory
  LoadCombinedParamsNaive(model_buffer, offset, scope, *cpp_prog, true);

  VLOG(4) << "Load model from naive buffer memory successfully";
}

}  // namespace lite
}  // namespace paddle
