import fasttext
import csv

model = fasttext.load_model('/workspace/datasets/fasttext/title_model.bin')
threshold = 0.75

with open("/workspace/datasets/fasttext/top_words.txt") as top_words_file:
    csv_lines=[]
    for line in top_words_file:
        word = line.rstrip()
        nearest_neighbors = model.get_nearest_neighbors(word)
        filtered_neighbor_words = [nearest_neighbor[1] for nearest_neighbor in nearest_neighbors if nearest_neighbor[0] > threshold]
        if len(filtered_neighbor_words):
            csv_lines.append(tuple([word] + filtered_neighbor_words))
    with open("/workspace/datasets/fasttext/synonyms.csv", "wt") as synonyms_file:
        writer = csv.writer(synonyms_file, delimiter=",")
        writer.writerows(csv_lines)
