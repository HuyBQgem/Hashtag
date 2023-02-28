import math
import re
from collections import Counter
from data_utils import convert

WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def evaluate(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    cosine = get_cosine(vector1, vector2)
    return cosine


def count_true_false(out_tag, test_tag):
    true = 0
    false = 0
    for index in range(len(out_tag)):
        converted = [convert(x) for x in out_tag[index]]
        test = test_tag[index]
        for item in converted:
            if item.replace(' ', '') in test:
                true += 1
            else:
                false += 1

    print('true prediction:', true)
    print('false prediction:', false)
    print('precision:', true / (true + false))


def cos_true_false(out_tag, test_tag, threshold=0.5):
    true = 0
    false = 0
    for index in range(len(out_tag)):
        for item in out_tag[index]:
            final_score = 0
            for test_item in test_tag[index]:
                s = evaluate(convert(item), test_item)
                if s > final_score:
                    final_score = s
            if final_score > threshold:
                true += 1
            else:
                false += 1

    print('cosine true:', true)
    print('cosine false:', false)
    print('precision:', true / (true + false))
