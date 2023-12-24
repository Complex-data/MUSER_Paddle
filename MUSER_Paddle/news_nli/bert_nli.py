import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import os
import numpy as np
from tqdm import tqdm

from paddlenlp.transformers import *
from utils.utils_paddle import build_batch


class BertNLIModel(nn.Layer):
    """Performs prediction, given the input of BERT embeddings. """

    def __init__(self, model_path=None, gpu=True, bert_type='bert-large', label_num=2, batch_size=8, reinit_num=0,
                 freeze_layers=False):
        super(BertNLIModel, self).__init__()
        self.bert_type = bert_type
        if 'bert-base' in bert_type:
            self.bert, self.bert_config = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif 'bert-large' in bert_type:
            # self.bert, self.bert_config = BertModel.from_pretrained('bert-large-uncased')
            # self.bert = BertModel.from_pretrained('bert-large-uncased')
            self.bert = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=label_num)
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        elif 'albert' in bert_type:
            self.bert, self.bert_config = AlbertModel.from_pretrained(bert_type)
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_type)
        else:
            print('illegal bert type {}!'.format(bert_type))
        # self.num_hidden_layers = self.bert.config.num_hidden_layers
        # self.vdim = self.bert.config.hidden_size
        # self.vdim = next(iter(self.bert.parameters())).shape[-1]
        # self.vdim = 1024
        # self.num_hidden_layers = 24
        # weight_attr = paddle.framework.ParamAttr(name="linear_weight", initializer=paddle.nn.initializer.XavierNormal())
        # self.nli_head = nn.Linear(self.vdim, label_num, weight_attr=weight_attr)
        self.gpu = gpu
        self.batch_size = batch_size
        self.sm = nn.Softmax(axis=1)
        self.reinit(layer_num=reinit_num, freeze=freeze_layers)
        # load trained model
        if model_path is not None:
            if gpu:
                sdict = paddle.load(model_path)
                self.set_state_dict(sdict)
                paddle.device.set_device('gpu:0')
            else:
                sdict = paddle.load(model_path)
                self.set_state_dict(sdict)
        else:
            if self.gpu:
                paddle.device.set_device('gpu:0')

    def reinit(self, layer_num, freeze):
        """Reinitialise parameters of last N layers and freeze all others"""
        if freeze:
            for _, pp in self.bert.named_parameters():
                pp.stop_gradient = True
        if layer_num > 0:
            layer_idx = [self.num_hidden_layers - 1 - i for i in range(layer_num)]
            layer_names = ['encoder.layer.{}'.format(j) for j in layer_idx]
            for pn, pp in self.bert.named_parameters():
                if any([ln in pn for ln in layer_names]) or 'pooler.' in pn:
                    # pp.set_value(paddle.randn(pp.shape)*0.02)
                    paddle.disable_static()
                    pp.set_value((paddle.randn(pp.shape) * 0.02).numpy())
                    pp.stop_gradient = False

    def load_model(self, sdict):
        if self.gpu:
            self.set_state_dict(sdict)
            self.cuda()
        else:
            self.set_state_dict(sdict)

    def forward(self, sent_pair_list, checkpoint=True, bs=None):
        all_probs = None
        if bs is None:
            bs = self.batch_size
            no_prog_bar = True
        else:
            no_prog_bar = False
        for batch_idx in tqdm(range(0, len(sent_pair_list), bs), disable=no_prog_bar, desc='evaluate'):
            probs = self.ff(sent_pair_list[batch_idx:batch_idx + bs], checkpoint)[1].detach().cpu().numpy()
            if all_probs is None:
                all_probs = probs
            else:
                all_probs = np.append(all_probs, probs, axis=0)
        labels = []
        for pp in all_probs:
            ll = np.argmax(pp)
            labels.append(ll)
        return labels, all_probs

    def step_bert_encode(self, module, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(module):
            # if module.output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = checkpoint.checkpoint(layer_module, hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
        all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        outputs = outputs + (all_hidden_states,)
        # outputs = outputs + (all_attentions,)

        # if module.output_hidden_states:
        #     outputs = outputs + (all_hidden_states,)
        # if module.output_attentions:
        #     outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def step_checkpoint_bert(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None,
                             head_mask=None):
        modules = [module for k, module in self.bert.named_children()]

        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = paddle.cast(extended_attention_mask, dtype=self.parameters()[0].dtype)
        
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = modules[0](input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.step_bert_encode(modules[1], embedding_output, extended_attention_mask, head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = modules[2](sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def ff(self, sent_pair_list, checkpoint):
        ids, types, masks = build_batch(self.tokenizer, sent_pair_list, self.bert_type)
        # logger.info(f'sent_pair_list shape: {len(sent_pair_list)}')

        if ids is None:
            return None
        ids_tensor = paddle.to_tensor(ids)
        types_tensor = paddle.to_tensor(types)
        masks_tensor = paddle.to_tensor(masks)

        if self.gpu:
            ids_tensor = ids_tensor.cuda()
            types_tensor = types_tensor.cuda()
            masks_tensor = masks_tensor.cuda()
        if checkpoint:
            cls_vecs = \
            self.step_checkpoint_bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)[1]
        else:
            cls_vecs = self.bert(input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)
        # logits = self.nli_head(cls_vecs)
        logger.info(cls_vecs.shape)
        logits = cls_vecs.reshape((len(sent_pair_list), -1))

        probs = self.sm(logits)

        # to reduce gpu memory usage
        # del ids_tensor
        # del types_tensor
        # del masks_tensor
        # torch.cuda.empty_cache() # releases all unoccupied cached memory

        return logits, probs

    def save(self, output_path, config_dic=None, acc=None):
        if acc is None:
            model_name = 'nli_model.state_dict'
        else:
            model_name = 'nli_model_acc{}.state_dict'.format(acc)
        opath = os.path.join(output_path, model_name)
        if config_dic is None:
            paddle.save(self.state_dict(), opath)
        else:
            paddle.save(config_dic, opath)

    @staticmethod
    def load(input_path, gpu=True, bert_type='bert-large', label_num=2, batch_size=16):
        if gpu:
            sdict = paddle.load(input_path)
        else:
            sdict = paddle.load(input_path, map_location=lambda storage, loc: storage)
        model = BertNLIModel(gpu, bert_type, label_num, batch_size)
        model.load_state_dict(sdict)
        return model
