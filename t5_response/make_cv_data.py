
import json
import copy


from t5_response.source import NUM_FOLD


file_ori_dir = '/data3/private/fengtao/Projects/cqr_t5/t5_response/data/processed_data_v4.json'






data_set = []
with open(file_ori_dir, encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        data_set.append(record)


# Split eval data into K-fold
topic_per_fold = len(data_set) // NUM_FOLD
for i in range(NUM_FOLD):
    with open('data/eval_topics_v4.jsonl.%d' % i, 'w') as fout:
        for idx, item in enumerate(data_set):

            if idx // topic_per_fold == i:
                json_str = json.dumps(item, ensure_ascii=False)
                fout.write(json_str + '\n')

