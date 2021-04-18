# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import math
import sys

import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, ACT2FN
from torch import nn
from torch.nn import CrossEntropyLoss


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertLayerNormNoVar(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNormNoVar, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = x - u
        # IMPORTANT: what I didn't get is that this is an ELEMENT WISE multiplication!
        # self.weights is not a matrix! it's a vector.
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        # https://medium.com/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a
        # Total embeddings = word embeddings + position embedding + token/segment embedding
        # Word embeddings: embedding of the individual words and specials indicators ([CLS], [SEP])
        # Position embeddings: embeddings that depend on the position of the word
        # Token/Segment embeddings: embeddings use to disambiguage between different parts of the input
        #                           for example, a pair of sentences (question/answer, or 2 sentences for which
        #                           we want to detect if they are about the same topic)
        #
        # In the adversarial attacks, I think the word embeddings will be perturbed. The segment embeddings will
        # definitely not be perturbed. The position embeddings but maybe not (just switching the embeddings of
        # the word might be enough and equivalent). For now, I'll assume that the only thing that will be perturbed
        # is the word embeddings. Thus, the sums with constants don't matter much, but I need to handle
        # the normalization layer.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.config = config

        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.hidden_size, eps=1e-12)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # Original size: B x n x (A x H)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # B x n x A x h
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # B x A x n x h

    def forward(self, hidden_states, attention_mask, values_storage=None):
        # What I learned: when doing tensor operations where
        # 1) tensor1 has dimensions (A1 x A2 x ... x An) x B x C
        # 2) tensor2 has dimensions (A1 x A2 x ... x An) x C x D
        # what actually happens is that each pair of matrices (there are A1 x ... x An  such pairs) gets
        # multiplied together (obtaining a new matrix of size B x D) and then the results is stored in a
        # result tensor of size (A1 x A2 x ... x An) x B x D
        # Therefore it's much more straightforward than I though
        # It's just matrix multiplications, done many times. If I can implement matrix multiplications
        # then I can handle arbitrary tensor multiplications very easily

        # Glossary: B = batch size, A = num attention heads, n = num words, H = embedding size
        mixed_query_layer = self.query(hidden_states)  # Size: B x n x (A x H)
        mixed_key_layer = self.key(hidden_states)  # Size: B x n x (A x H)
        mixed_value_layer = self.value(hidden_states)  # Size: B x n x (A x H)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # B x A x n x H
        key_layer = self.transpose_for_scores(mixed_key_layer)  # B x A x n x H
        value_layer = self.transpose_for_scores(mixed_value_layer)  # B x A x n x H

        if values_storage is not None: values_storage.append(query_layer)
        if values_storage is not None: values_storage.append(key_layer)
        if values_storage is not None: values_storage.append(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B x A x n x n
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # B x A x n x n

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask  # B x A x n x n (TODO / Question: attention mask in n²?)
        if values_storage is not None: values_storage.append(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # B x A x n x n (normalized, compute 1 softmax per row)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # B x A x n x n
        if values_storage is not None: values_storage.append(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # (B x A x n x n) @ (B x A x n x H) = B x A x n x H
        if values_storage is not None: values_storage.append(context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # B x n x A x H
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # B x n x (A x H)
        context_layer = context_layer.view(*new_context_layer_shape)
        if values_storage is not None: values_storage.append(context_layer)

        return context_layer, attention_scores, attention_probs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.hidden_size, eps=1e-12)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, values_storage=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if values_storage is not None: values_storage.append(hidden_states)

        hidden_states = hidden_states + input_tensor
        if values_storage is not None: values_storage.append(hidden_states)

        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            hidden_states = self.LayerNorm(hidden_states)
            if values_storage is not None: values_storage.append(hidden_states)

        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, values_storage=None):
        self_output, attention_scores, attention_probs = self.self(input_tensor, attention_mask, values_storage)
        attention_output = self.output(self_output, input_tensor, values_storage)

        return attention_output, self_output, attention_scores, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, values_storage=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if values_storage is not None: values_storage.append(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        if hasattr(config, "layer_norm") and config.layer_norm == "no_var":
            self.LayerNorm = BertLayerNormNoVar(config.hidden_size, eps=1e-12)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, values_storage=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if values_storage is not None: values_storage.append(hidden_states)

        if hasattr(self.config, "layer_norm") and self.config.layer_norm == "no":
            pass
        else:
            hidden_states = self.LayerNorm(hidden_states)
            if values_storage is not None: values_storage.append(hidden_states)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, values_storage=None):
        attention_output, self_output, attention_scores, attention_probs = self.attention(hidden_states, attention_mask, values_storage)
        intermediate_output = self.intermediate(attention_output, values_storage)
        layer_output = self.output(intermediate_output, attention_output, values_storage)

        return layer_output, self_output, attention_scores, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, values_storage=None):
        all_encoder_layers = []
        all_self_output = []  # right after summation weighted by softmax probs
        all_attention_scores = []
        all_attention_probs = []
        all_attention_output = []
        for layer_module in self.layer:
            hidden_states, self_output, attention_scores, attention_probs = layer_module(hidden_states, attention_mask, values_storage)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_self_output.append(self_output)
                all_attention_scores.append(attention_scores)
                all_attention_probs.append(attention_probs)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_self_output.append(self_output)
            all_attention_scores.append(attention_scores)
            all_attention_probs.append(attention_probs)
        return all_encoder_layers, all_self_output, all_attention_scores, all_attention_probs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # print("With Tanh")
        # self.activation = nn.ReLU()
        # print("With RELU")

    def forward(self, hidden_states, values_storage=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if values_storage is not None: values_storage.append(first_token_tensor)

        pooled_output = self.dense(first_token_tensor)
        if values_storage is not None: values_storage.append(pooled_output)

        pooled_output = self.activation(pooled_output)
        if values_storage is not None: values_storage.append(pooled_output)

        return pooled_output


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                embeddings=None, values_storage=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if embeddings is None:
            embeddings = self.embeddings(input_ids, token_type_ids)
        else:
            if values_storage is not None: values_storage.append(embeddings)

            # If we provide perturbed embedding, we don't call self.embeddings
            # but we still have to apply the layer norm. We do that manually here
            embeddings = self.embeddings.LayerNorm(embeddings)
            embeddings = self.embeddings.dropout(embeddings)
            if values_storage is not None: values_storage.append(embeddings)



        encoded_layers, self_output, attention_scores, attention_probs = \
            self.encoder(embeddings, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers, values_storage=values_storage)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output, values_storage=values_storage)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
            attention_scores = attention_scores[-1]
            attention_probs = attention_probs[-1]
        return embeddings, encoded_layers, attention_scores, attention_probs, \
               pooled_output, self_output


class BertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, embeddings=None, values_storage=None):
        embedding_output, encoded_layers, attention_scores, attention_probs, pooled_output, self_output = \
            self.bert(input_ids, token_type_ids, attention_mask,
                      output_all_encoded_layers=True, embeddings=embeddings, values_storage=values_storage)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if values_storage is not None: values_storage.append(logits)

        # assert (labels is None)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits, embedding_output, encoded_layers, attention_scores, attention_probs, self_output, pooled_output
