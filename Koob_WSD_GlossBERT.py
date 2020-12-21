import pandas as pd
import argparse
from tokenization import BertTokenizer
from modeling import BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import sys
from nltk.corpus import wordnet
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

import json

def construct_context_gloss_pairs_through_nltk(input, target_start_id, target_end_id, gloss_dict):
    """
    construct context gloss pairs like sent_cls_ws
    :param input: str, a sentence
    :param target_start_id: int
    :param target_end_id: int
    :param lemma: lemma of the target word
    :return: candidate lists
    """
    sent = input.split(" ")
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])
    if len(sent) > target_end_id:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"'] + sent[target_end_id:]
    else:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"']

    sent = " ".join(sent)

    candidate = []
    for gloss_def in gloss_dict['synsets']:
        candidate.append((sent, f"{target} : {gloss_def}", target, gloss_def))
    #syns = wordnet.synsets(target)
    #for syn in syns:
    #    gloss = syn.definition()
    #    candidate.append((sent, f"{target} : {gloss}", target, gloss))

    assert len(candidate) != 0, f'there is no candidate sense of "{target}" in WordNet, please check'
    #print(f'there are {len(candidate)} candidate senses of "{target}"')


    return candidate

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_to_features(candidate, tokenizer, max_seq_length=512):

    candidate_results = []
    features = []
    for item in candidate:
        text_a = item[0] # sentence
        text_b = item[1] # gloss
        candidate_results.append((item[-2], item[-1])) # (target, gloss)


        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))


    return features, candidate_results

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def infer(input, target_start_id, target_end_id, gloss_dict, args):

    sent = input.split(" ")
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    label_list = ["0", "1"]
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                          num_labels=num_labels)
    model.to(device)

    #print(f"input: {input}\ntarget: {target}")

    examples = construct_context_gloss_pairs_through_nltk(input, target_start_id, target_end_id, gloss_dict)
    eval_features, candidate_results = convert_to_features(examples, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)


    model.eval()
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    logits_ = F.softmax(logits, dim=-1)
    logits_ = logits_.detach().cpu().numpy()
    output = np.argmax(logits_, axis=0)[1]
    print(f"results:\ngloss: {candidate_results[output][1]}")
    return candidate_results[output][1]


### to get context sentense from paragraph
def get_context_sentence(word, para):
    """
    Description: Get context sentense for the given word and paragraph

    Input:
        word: (str), target word
        para: (str), context paragraph

    Output:
        word: (str), target word
        sent: (str), context sentence
        target_start_id: (int), starting index of target word
        target_end_id: (int), end index of target word
    """
    # sentence tokenization
    sentences = sent_tokenize(para)

    # select context sentence for target word
    sent = [sent.lower() for sent in sentences if word.lower() in sent.lower()][0]

    # regular expression to substitute punctuations
    sent = re.sub(r'[!(){};:"’“”\,<>./?@#$%^&*_~]', '', sent)

    # tokenization
    target_tokens = word_tokenize(word)  # create word tokens for input target string
    sent_tokens = word_tokenize(sent)  # create word tokens for context sentence

    # calculate start position and end position of target word in context sentence
    target_start_id = [ind for ind, token in enumerate(sent_tokens) if target_tokens[0] in token][0]
    target_end_id = target_start_id + len(target_tokens)

    return word, sent, target_start_id, target_end_id


def process_input_json(item):
    """
    Description: (dict), process input json object

    Input:
        (dict), input json object

    Output:
        word_id: (int), target word id
        word: (str), target word
        teacher_vote: (str), teacher voted sense
        ai_vote: (str), AI voted sense
    """
    word = item['word'].lower()  # get target word

    # get context sentence for target word and index details of target word in context sentence
    # word, sent, target_start_id, target_end_id, target_pos = get_context_sentence(word, item['paragraph'])
    word, sent, target_start_id, target_end_id = get_context_sentence(word, item['paragraph'])

    gloss_dict = {}
    gloss_data = []
    for gloss in item['dictionaryData']:

        # if (gloss['fl'].lower() == target_pos or target_pos == 'None'):
        gloss_data.append(gloss['meaning'])

        if gloss['isTeacher'] == 1:
            word_id = gloss['word_id']
            teacher_vote = gloss['meaning']

    gloss_dict['synsets'] = gloss_data
    ai_vote = infer(sent, target_start_id, target_end_id, gloss_dict, args)
    return word_id, word, teacher_vote, ai_vote


if __name__ == "__main__":

    import re

    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument("--bert_model", default="./Sent_CLS_WS", type=str)
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")

    args = parser.parse_args()

    request_file = open('request1.json', 'r')
    request_data = request_file.read()
    jsonobj = json.loads(request_data)

    word_id_lst = []
    teacher_vote_lst = []
    ai_vote_lst = []
    word_lst = []

    result_json = []
    for item in jsonobj:
        word_id, word, teacher_vote, ai_vote = process_input_json(item)

        result_dict = {'word_id': word_id,
                       'word': word,
                       'Teacher_vote': teacher_vote,
                       'AI_vote': ai_vote
                       }
        result_json.append(result_dict)

        word_id_lst.append(word_id)
        word_lst.append(word_lst)
        teacher_vote_lst.append(teacher_vote)
        ai_vote_lst.append(ai_vote)

    results_df = pd.DataFrame(list(zip(word_id_lst, word_lst, teacher_vote_lst, ai_vote_lst)),
                              columns=['word_id', 'word', 'teachers_vote', 'AI_vote'])

    print('Accuracy: ', np.mean(results_df['teachers_vote'] == results_df['AI_vote']))

    with open("WSD_result.json", "w") as write_file:
        json.dump(result_json, write_file)
        write_file.close()