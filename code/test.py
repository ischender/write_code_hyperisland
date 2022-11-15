import numpy as np
import torch
from langdetect import detect

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

from sentence_transformers import SentenceTransformer, util
# MODEL = "all-mpnet-base-v2"
MODEL_SENT = 'paraphrase-MiniLM-L6-v2'
# MODEL = 'all-MiniLM-L6-v2'
MODEL = "roberta-large-mnli"

sent_model = SentenceTransformer(MODEL_SENT)


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = RobertaTokenizer.from_pretrained(MODEL)
model = RobertaForSequenceClassification.from_pretrained(
    MODEL, num_labels=3).to(device)


leaving_sents = ["the user will not renew the contract",
                 "the user is cancelling the contract",
                 "the user wants to cancel the contract",
                 "the user wants to deactivate the auto-renewal",
                 "the user is leaving the service",
                 "the user will move to a different platform",
                 ]

signing = ["the user is signing the contract",
           "the user is signing the agreement",
           "the user wants  the contract",
           ]

changes = ["the user wants to modify the contract",
           "a feature was removed",
           "a feature was changed",
           "a user has been deleted",
           "a login is removed",
           ]

users = ["the user is changing job",
         "the user is leaving their job",
         "the user is on holiday",
         "the user is taking over a role",
         "the user is travelling",
         "a user is on leave",
         "a user has been deleted",
         "a login is removed",
         ]

meetings = ["the user is re-scheduling a meeting",
            "the user is cancelling a meeting",
            ]

problems = ["the service is not working", ]

all_fp = signing + changes + users + meetings + problems


false_positives = ["the user is signing the contract",
                   "the user is signing the agreement",
                   "the user is out of the office",
                   "the user wants to modify the contract",
                   "the user is changing job",
                   "the user is leaving their job",
                   "the user is wants  the contract",
                   "the user is on holiday",
                   "the user is re-scheduling a meeting",
                   "the user is cancelling a meeting",
                   "the user is taking over a role",
                   "the service is not working",
                   "the user is travelling",
                   "a feature was removed",
                   "a feature was changed",
                   "a user is on leave",
                   "a user has been deleted",
                   "a login is removed",
                   ]

checks_fp = {4: [2, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17]}

checks_false_positives = {
    0: signing + changes,
    1: signing + changes + meetings,
    2: signing + changes + meetings,
    3: signing + changes + meetings,
    4: signing + changes + meetings + users + problems
}

LEN = 5


topics = ["contract", "billing", "renewal", "payment", "order form", "meeting"]
topics_emb = sent_model.encode(topics)


def featurize(sent, match_sents, model=model):
    sent_list = [f"{sent} </s></s> {m}" for m in match_sents]
    inputs = tokenizer(sent_list, return_tensors="pt",
                       max_length=512, padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    print(logits)
    smax = sigmoid(logits.cpu().detach().numpy())
    print(smax)
    # for i, t in enumerate(logits):
    #     amax = torch.argmax(t)
    #     if amax == 2:
    #         smax = sigmoid(t.cpu().detach().numpy())
    #         result[i] = smax[2]
    #         found = True
    # torch.cuda.empty_cache()
    # return found, result


def sigmoid(_outputs):
    return 1.0 / (1.0 + np.exp(-_outputs))


def predict(sent, match_sents, model=model):
    found = False
    result = {}
    sent_list = [f"{sent} </s></s> {m}" for m in match_sents]
    inputs = tokenizer(sent_list, return_tensors="pt",
                       max_length=512, padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    for i, t in enumerate(logits):
        amax = torch.argmax(t)
        if amax == 2:
            smax = sigmoid(t.cpu().detach().numpy())
            result[i] = smax[2]
            found = True
    torch.cuda.empty_cache()
    return found, result


def check_false_(text, matches):
    if "unsubscribe" in text.lower():
        return True
    for match, confidence in matches.items():
        if match in checks_fp:
            # if True:
            # current_false_positives = [false_positives[i] for i in checks_fp[match]]
            current_false_positives = checks_false_positives[match]
            # current_false_positives = false_positives
            found_, result = predict(text, current_false_positives)
            if found_:
                for idx, conf in result.items():
                    print(leaving_sents[match], confidence, '->',
                          current_false_positives[idx], conf)
                    if conf > confidence:
                        return True
    return False


def check_text_(text, check=True):
    sent_text = nltk.sent_tokenize(text)
    results = [{}]*LEN
    found = False
    if len(sent_text) < 2 or detect(text) != 'en':
        return(found, results)
    for i, sent in enumerate(sent_text[:LEN]):
        if len(sent) < 10 or len(sent.split()) < 4:
            continue
        found_, result = predict(sent, leaving_sents)
        if found_:
            if check:
                false_positive = check_false_(sent, result)
                if false_positive:
                    continue
            found = found_
        print(i, results)
        results[i] = result
    # if results and sum(True for el in results if el)/len(results) < 0.21:
    #   print(sum(True for el in results if el)/len(results))
    #   print(results)
    #   return False, []*LEN

    return found, results


def check_all_text(text, check=True):
    sent_text = nltk.sent_tokenize(text)
    results = []
    found = False
    if len(sent_text) < 2 or detect(text) != 'en':
        return(found, results)

    found_, result = predict(text, leaving_sents)
    if found_:
        if check:
            false_positive = check_false_(text, result)
            if false_positive:
                found = false
    return found, results


# check_text_(ttt)
check_all_text(ttt)
