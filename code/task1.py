# 1. Read the training data
import json


with open('train.json', 'r') as f:
    train_data = json.load(f)


# 2. Count word frequency
from collections import defaultdict

word_frequency = defaultdict(int)

for record in train_data:
    for word in record["sentence"]:
        word_frequency[word] += 1



# 3. Sort by frequency
UNKNOWN_THRESHOLD = 3
unknown_count = sum(freq for word, freq in word_frequency.items() if freq < UNKNOWN_THRESHOLD)

sorted_vocab = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

# 4. Filter out words with frequency below the threshold
filtered_vocab = [(word, freq) for word, freq in sorted_vocab if freq >= UNKNOWN_THRESHOLD]

# 5. Write to vocab.txt
with open('/home/ubuntu/gxy/hw2/result/vocab.txt', 'w') as f:
    f.write("<unk>\t0\t" + str(unknown_count) + "\n")
    
    index = 1
    for word, freq in filtered_vocab:
        f.write(f"{word}\t{index}\t{freq}\n")
        index += 1



