// Tencent is pleased to support the open source community by making TNN
// available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>

#include <fstream>
#include <iostream>
int main(int argc, char *argv[]) {

  if (argc < 4) {
    std::cout << "Usage: ./mnnTest.out model.mnn input.txt output.txt"
              << std::endl;
  }
  const auto model_path = argv[1];
  const auto input_path = argv[2];
  const auto output_path = argv[3];
  auto mnnNEt = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path));
  MNN::ScheduleConfig netConfig;
  netConfig.type = MNN_FORWARD_CPU;
  netConfig.numThread = 1;
  auto session = mnnNEt->createSession(netConfig);
  auto input = mnnNEt->getSessionInput(session, nullptr);
  MNN::Tensor givenTensor(input, MNN::Tensor::CAFFE);
  std::ifstream inputFile(input_path);
  const int inputSize = givenTensor.elementSize();
  auto inputData = givenTensor.host<float>();
  float pixel = 0.0f;
  for (int i = 0; i < inputSize; ++i) {
    inputFile >> pixel;
    inputData[i] = pixel;
  }
  inputFile.close();
  input->copyFromHostTensor(&givenTensor);
  // run
  mnnNEt->runSession(session);

  // get output
  auto output_map = mnnNEt->getSessionOutputAll(session);
  std::ofstream outputFile(output_path);
  for (auto iter : output_map) {
    printf("the output_name: %s", iter.first.c_str());
    iter.second->printShape();
    auto nchwTensor = new MNN::Tensor(iter.second, MNN::Tensor::CAFFE);
    iter.second->copyToHostTensor(nchwTensor);
    auto data_ptr = nchwTensor->host<float>();
    for (int i = 0; i < nchwTensor->elementSize(); ++i) {
      outputFile << * (data_ptr + i) << "\n";
    }
    delete nchwTensor;
  }
  outputFile.close();
}
