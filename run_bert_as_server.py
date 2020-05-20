import threading
from flask import Flask, request, jsonify
import modeling
import tokenization
import tensorflow as tf
import json
import collections
import time
import sys


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text

def get_kor_examples(sent):
    examples = []
    ori_sents = []
    sent = sent.strip()
    ori_sents.append(sent)
    tokens = tokenizer.tokenize(sent)

    guid = "%s-%s" % ('test', 0)
    examples.append(InputExample(guid=guid, text=tokens))
    return examples, ori_sents




def graph_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, use_one_hot_embeddings):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    all_output_layer = model.get_all_encoder_layers()
    embedding_output = model.get_embedding_output()
        
    return all_output_layer, embedding_output


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    tokens = example.text
    if len(tokens) >= max_seq_length:
        tokens = tokens[0:(max_seq_length)]
    ntokens = []
    segment_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
    ntokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        ntokens.append("**NULL**")
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
    )
    return feature
	

	
def convert_examples_to_features(examples, tokenizer, max_seq_length):
    features = []
    example_texts = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features.append(feature)
        example_texts.append(example.text)
    return features, example_texts
		
def get_feed_dict(features):
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    for feature in features:
        input_ids_list.append(feature.input_ids)
        input_mask_list.append(feature.input_mask)
        segment_ids_list.append(feature.segment_ids)
    return {input_ids: input_ids_list, input_mask:input_mask_list, segment_ids:segment_ids_list}


bert_config = modeling.BertConfig.from_json_file("../multilingual_model/bert_config.json")
tokenizer = tokenization.FullTokenizer(vocab_file="../multilingual_model/vocab.txt", do_lower_case=False)

input_ids = tf.placeholder(shape=[None,None], dtype=tf.int32)
input_mask = tf.placeholder(shape=[None,None], dtype=tf.int32)
segment_ids = tf.placeholder(shape=[None,None], dtype=tf.int32)
use_one_hot_embeddings=False

predict_ = graph_model(
            bert_config, False, input_ids, input_mask, segment_ids, use_one_hot_embeddings)

tf_config = tf.ConfigProto()

sess = tf.Session(config = tf_config)
sess.run(tf.global_variables_initializer())
model_loader = tf.train.Saver()
model_loader.restore(sess, "../multilingual_model/bert_model.ckpt")
app = Flask(__name__)
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    with threading.Semaphore(1):
        ori_input_sent = ''
    if request.method == 'POST':
        json_data = request.get_json(force=True)
        sentence = json_data['sentence']
        layer_num = int(json_data['layer'])
        ori_input_sent = sentence
    else:
        sentence = request.args.get('sentence')
        layer_num = int(request.args.get('layer'))
        ori_input_sent = sentence
    
    eval_examples, ori_sents = get_kor_examples(sentence)

    eval_features, tokens_list = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    result_all_output_layer, result_embedding_output = sess.run(predict_, feed_dict=get_feed_dict(eval_features))

    #print(result_embedding_output)
    return_result = ''
    if layer_num == -1:
        return_result = json.dumps({"input_sentence": ori_input_sent, "tokens":tokens_list[0], "return_layer": layer_num, "vectors": result_embedding_output[0].tolist()}, ensure_ascii=False)
    else:
        return_result = json.dumps({"input_sentence": ori_input_sent, "tokens":tokens_list[0], "return_layer": layer_num, "vectors": result_all_output_layer[layer_num][0].tolist()}, ensure_ascii=False)

    return return_result



if __name__ == '__main__':
    port_num = 9090
    app.run(host = '0.0.0.0', port=port_num, threaded=True)
