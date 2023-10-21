import json

all_labels = set()


def process_data(source, target, mode):
    with open(source, "r", encoding="utf-8") as fp:
        data = fp.read().strip().split("\n")
    with open(target, "r", encoding="utf-8") as fp:
        labels = fp.read().strip().split("\n")
    res = []
    text_id = 0
    for s, t in zip(data, labels):
        text = s.split(" ")
        label = t.split(" ")
        assert len(text) == len(label)
        length = len(text)
        ent_id = 0
        tmp = {}
        text = "".join(text)
        tmp["id"] = text_id
        tmp["text"] = text
        tmp["labels"] = []
        for i in range(length):
            if 'B-' in label[i]:
                j = i + 1
                ent_type = label[i].split("-")[-1]
                all_labels.add(ent_type)
                if j == length:
                    ent_des = [str(ent_id), ent_type, i, i + 1, text[i:i + 1]]
                    tmp["labels"].append(ent_des)
                    ent_id += 1
                else:
                    while j <= length - 1 and "I-" in label[j]:
                        j += 1
                    ent_des = [str(ent_id), ent_type, i, j, text[i:j]]
                    tmp["labels"].append(ent_des)
                    ent_id += 1
                i += 1
        res.append(tmp)

    with open("../mid_data/{}.json".format(mode), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False)


process_data("source.txt", "target.txt", mode="train")
process_data("dev.txt", "dev-label.txt", mode="dev")
process_data("test1.txt", "testtgt.txt", mode="test")

with open("../mid_data/labels.json", "w", encoding="utf-8") as fp:
    json.dump(list(all_labels), fp, ensure_ascii=False)
