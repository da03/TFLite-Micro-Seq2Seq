
#ifndef MODEL_H_
#define MODEL_H_

#include "Arduino.h"
#include "tensorflow/lite/c/common.h"

extern const unsigned char g_enc_model_fw[];
extern const unsigned char g_enc_model_bw[];
extern const unsigned char g_dec_model[];
void set_enc_embed(TfLiteTensor* ptr, String token);
void set_dec_embed(TfLiteTensor* ptr, String token);
String id_to_word(int idx);

const int max_len_src = 10;
const int max_len_tgt = 21;
const int bos_word_id = 3;
const int eos_word_id = 4;
const int enc_hidden_size = 32;
const int dec_hidden_size = 64;
const int src_embedding_size = 64;
const int tgt_embedding_size = 64;
const int src_vocab_size = 11;
const int tgt_vocab_size = 35;
#endif
