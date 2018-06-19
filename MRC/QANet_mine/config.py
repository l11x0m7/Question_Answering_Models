from prepro import prepro

import tensorflow as tf

import os

flags = tf.flags

home = os.path.join(os.getcwd(), '..')

train_file = os.path.join(home, 'datasets', 'squad2.0', 'train-v2.0.json')
dev_file = os.path.join(home, 'datasets', 'squad2.0', 'dev-v2.0.json')
test_file = os.path.join(home, 'datasets', 'squad2.0', 'dev-v2.0.json')
glove_word_file = os.path.join(home, 'datasets', 'glove', 'glove.840B.300d.txt')


train_dir = 'train'
model_name = 'FRC'
dir_name = os.path.join(train_dir, model_name)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

target_dir = 'data'
log_dir = os.path.join(target_dir, 'event')
save_dir = os.path.join(dir_name, "model")
answer_dir = os.path.join(dir_name, "answer")
train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
word_dictionary = os.path.join(target_dir, "word_dictionary.json")
char_dictionary = os.path.join(target_dir, "char_dictionary.json")
answer_file = os.path.join(answer_dir, "answer.json")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer")
flags.DEFINE_string("word_dictionary", word_dictionary, "Word dictionary")
flags.DEFINE_string("char_dictionary", char_dictionary, "Character dictionary")

flags.DEFINE_integer("glove_char_size", 94, "Corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 64, "Embedding dimension for char")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("ans_limit", 30, "Limit length for answers")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

glove_char_file = os.path.join(home, "datasets", "glove", "glove.840B.300d-char.txt")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", False, "Whether to use pretrained character embedding")

fasttext_file = os.path.join(home, "datasets", "fasttext", "wiki-news-300d-1M.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("use_fasttext", False, "Whether to use fasttext")

def main(_):
    config = flags.FLAGS

    if config.mode == 'prepro':
        prepro(config)
    elif config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    else:
        raise ValueError('The mode must choose from [prepro/train/test]!')


if __name__ == '__main__':
    tf.app.run()




