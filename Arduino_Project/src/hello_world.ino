/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"
#include "Arduino.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
//#include "tensorflow/lite/micro/testing/micro_test.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* enc_model_fw = nullptr;
const tflite::Model* enc_model_bw = nullptr;
const tflite::Model* dec_model = nullptr;

tflite::MicroInterpreter* enc_interpreter_fw = nullptr;
tflite::MicroInterpreter* enc_interpreter_bw = nullptr;
tflite::MicroInterpreter* dec_interpreter = nullptr;

TfLiteTensor* enc_embedding_fw = nullptr;
TfLiteTensor* enc_state_h_fw = nullptr;
TfLiteTensor* enc_state_c_fw = nullptr;
TfLiteTensor* enc_state_h_out_fw = nullptr;
TfLiteTensor* enc_state_c_out_fw = nullptr;
TfLiteTensor* enc_output_out_fw = nullptr;

TfLiteTensor* enc_embedding_bw = nullptr;
TfLiteTensor* enc_state_h_bw = nullptr;
TfLiteTensor* enc_state_c_bw = nullptr;
TfLiteTensor* enc_state_h_out_bw = nullptr;
TfLiteTensor* enc_state_c_out_bw = nullptr;
TfLiteTensor* enc_output_out_bw = nullptr;

TfLiteTensor* dec_enc_outputs = nullptr;
TfLiteTensor* dec_embedding = nullptr;
TfLiteTensor* dec_state_h = nullptr;
TfLiteTensor* dec_state_c = nullptr;
TfLiteTensor* dec_logit = nullptr;
TfLiteTensor* dec_context = nullptr;
TfLiteTensor* dec_state_h_out = nullptr;
TfLiteTensor* dec_state_c_out = nullptr;

// to avoid overwrites
float enc_state_h_fw_buffer[enc_hidden_size];
float enc_state_c_fw_buffer[enc_hidden_size];
float enc_state_h_bw_buffer[enc_hidden_size];
float enc_state_c_bw_buffer[enc_hidden_size];
float enc_outputs_buffer[max_len_src][dec_hidden_size];
float dec_state_h_buffer[dec_hidden_size];
float dec_state_c_buffer[dec_hidden_size];
float dec_context_buffer[dec_hidden_size];

constexpr int kTensorArenaSize = 100*1000;//2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  enc_model_fw = tflite::GetModel(g_enc_model_fw);
  if (enc_model_fw->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         enc_model_fw->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  
  enc_model_bw = tflite::GetModel(g_enc_model_bw);
  if (enc_model_bw->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         enc_model_bw->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  dec_model = tflite::GetModel(g_dec_model);
  if (dec_model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         dec_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter enc_static_interpreter_fw(
      enc_model_fw, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  enc_interpreter_fw = &enc_static_interpreter_fw;
  
  static tflite::MicroInterpreter enc_static_interpreter_bw(
      enc_model_bw, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  enc_interpreter_bw = &enc_static_interpreter_bw;


  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter dec_static_interpreter(
      dec_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  dec_interpreter = &dec_static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus enc_allocate_status_fw = enc_interpreter_fw->AllocateTensors();
  if (enc_allocate_status_fw != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  TfLiteStatus enc_allocate_status_bw = enc_interpreter_bw->AllocateTensors();
  if (enc_allocate_status_bw != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus dec_allocate_status = dec_interpreter->AllocateTensors();
  if (dec_allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  enc_embedding_fw = enc_interpreter_fw->input(0);
  enc_state_h_fw = enc_interpreter_fw->input(1);
  enc_state_c_fw = enc_interpreter_fw->input(2);
  enc_output_out_fw = enc_interpreter_fw->output(0);
  enc_state_h_out_fw = enc_interpreter_fw->output(1);
  enc_state_c_out_fw = enc_interpreter_fw->output(2);

  enc_embedding_bw = enc_interpreter_bw->input(0);
  enc_state_h_bw = enc_interpreter_bw->input(1);
  enc_state_c_bw = enc_interpreter_bw->input(2);
  enc_output_out_bw = enc_interpreter_bw->output(0);
  enc_state_h_out_bw = enc_interpreter_bw->output(1);
  enc_state_c_out_bw = enc_interpreter_bw->output(2);

  dec_enc_outputs = dec_interpreter->input(0);
  dec_embedding = dec_interpreter->input(1);
  dec_state_h = dec_interpreter->input(2);
  dec_state_c = dec_interpreter->input(3);
  dec_logit = dec_interpreter->output(2);
  dec_context = dec_interpreter->output(3);
  dec_state_h_out = dec_interpreter->output(0);
  dec_state_c_out = dec_interpreter->output(1);

  
  Serial.begin(9600);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  String inString = "";
  while (Serial.available() > 0) {
    char inChar = Serial.read();
    inString += inChar;
    delay(10);
  }
  if (inString != "") {
    Serial.print("Input: ");
    Serial.println(inString);

    int len = inString.length();
    // Forward encoder rnn (fw)
    // Initialize Hidden State to zeros
    for (int j = 0; j < enc_hidden_size; j++) {
      enc_state_h_fw->data.f[j] = 0.0;
      enc_state_c_fw->data.f[j] = 0.0;
    }
    for (int i = 0; i < max_len_src; i++) {
      String c = "";
      if (i < len) {
        c += inString[i];
      } else {
        c = "pad";
      }
      // Set embeddings
      set_enc_embed(enc_embedding_fw, c);
      // Call RNN
      TfLiteStatus invoke_status = enc_interpreter_fw->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
      }
      // Get outputs
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_outputs_buffer[i][j] = enc_output_out_fw->data.f[j];
      }
      // Update hidden
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_state_h_fw_buffer[j] = enc_state_h_out_fw->data.f[j];
        enc_state_c_fw_buffer[j] = enc_state_c_out_fw->data.f[j];
      }
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_state_h_fw->data.f[j]  = enc_state_h_fw_buffer[j];
        enc_state_c_fw->data.f[j]  = enc_state_c_fw_buffer[j];
      }
    }
    // Forward encoder rnn (bw)
    // Initialize Hidden State to zeros
    for (int j = 0; j < enc_hidden_size; j++) {
      enc_state_h_bw->data.f[j] = 0.0;
      enc_state_c_bw->data.f[j] = 0.0;
    }
    for (int i = max_len_src-1; i >=0 ; i--) {
      String c = "";
      if (i < len) {
        c += inString[i];
      } else {
        c = "pad";
      }
      set_enc_embed(enc_embedding_bw, c);
      // Call RNN
      TfLiteStatus invoke_status = enc_interpreter_bw->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
      }
      // Get outputs
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_outputs_buffer[i][enc_hidden_size+j] = enc_output_out_bw->data.f[j];
      }
      // Update hidden
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_state_h_bw_buffer[j] = enc_state_h_out_bw->data.f[j];
        enc_state_c_bw_buffer[j] = enc_state_c_out_bw->data.f[j];
      }
      for (int j = 0; j < enc_hidden_size; j++) {
        enc_state_h_bw->data.f[j]  = enc_state_h_bw_buffer[j];
        enc_state_c_bw->data.f[j]  = enc_state_c_bw_buffer[j];
      }
    }
    // Set decoder initial hidden state to be encoder final states
    for (int j = 0; j < enc_hidden_size; j++) {
      dec_state_h->data.f[j] = enc_state_h_fw_buffer[j];
      dec_state_c->data.f[j] = enc_state_c_fw_buffer[j];
    }
    for (int j = 0; j < enc_hidden_size; j++) {
      dec_state_h->data.f[enc_hidden_size+j] = enc_state_h_bw_buffer[j];
      dec_state_c->data.f[enc_hidden_size+j] = enc_state_c_bw_buffer[j];
    }
    int prev_word_id = bos_word_id;
    
    String result = "Result: ";
    for (int t = 0; t < max_len_tgt; t++) {
      // Set encoder outputs
      for (int i = 0; i < max_len_src; i++) {
        for (int j = 0; j < dec_hidden_size; j++) {
          dec_enc_outputs->data.f[i*dec_hidden_size+j] = enc_outputs_buffer[i][j];
        }
      }
      String prev_word = id_to_word(prev_word_id);
      set_dec_embed(dec_embedding, prev_word);
      if (t > 0) {
        for (int j = 0; j < dec_hidden_size; j++) {
          dec_embedding->data.f[j] += dec_context_buffer[j];
        }
      }
      TfLiteStatus invoke_status = dec_interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
        return;
      }
      int max_id = 0;
      float max_val;
      for (int j = 0; j < tgt_vocab_size; j++) {
        if (j == 0) {
          max_val = dec_logit->data.f[j];
          max_id = j;
        } else {
          if (dec_logit->data.f[j] > max_val) {
            max_id = j;
            max_val = dec_logit->data.f[j];
          }
        }
      }
      prev_word_id = max_id;

      if (prev_word_id == eos_word_id || prev_word_id == 0) {
        break;
      }

      result += id_to_word(prev_word_id);
      result += " ";
      
      // Update hidden
      for (int j = 0; j < dec_hidden_size; j++) {
        dec_context_buffer[j] = dec_context->data.f[j];
      }
      for (int j = 0; j < dec_hidden_size; j++) {
        dec_state_h_buffer[j] = dec_state_h_out->data.f[j];
        dec_state_c_buffer[j] = dec_state_c_out->data.f[j];
      }
      for (int j = 0; j < dec_hidden_size; j++) {
        dec_state_h->data.f[j] = dec_state_h_buffer[j];
        dec_state_c->data.f[j] = dec_state_c_buffer[j];
      }
    }
    Serial.println(result);
  }
}
