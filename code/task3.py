
# Step 1: Load the model and data.


import json

# Load the learned model from hmm.json
with open('/home/ubuntu/gxy/hw2/result/hmm.json', 'r') as f:
    model = json.load(f)

transition = model['transition']
emission = model['emission']

# Load the development data
with open('/home/ubuntu/gxy/hw2/CSCI544_HW2/data/dev.json', 'r') as f:
    dev_data = json.load(f)

# Step 2: Implement the Decode function.


def Decode(sentence, transition, emission):
    S = set([key.split(',')[0] for key in transition.keys()])
    
    T = len(sentence)
   
    tags = []

    # Get y1
    y1_probs = {}
    for s in S:
        init_prob = 1  # Since π(si) is not provided, assume uniform distribution
        emission_key = f"{s},{sentence[0]}"
        y1_probs[s] = init_prob * emission.get(emission_key, 0)  # Using dict.get() to handle missing emission keys
    y1 = max(y1_probs, key=y1_probs.get)
    tags.append(y1)

    # Get the rest of the tags
    for i in range(1, T):
        word = sentence[i]
        yi_probs = {}
        for s in S:
            transition_key = f"{tags[-1]},{s}"
            emission_key = f"{s},{word}"
            yi_probs[s] = transition.get(transition_key, 0) * emission.get(emission_key, 0)
        yi = max(yi_probs, key=yi_probs.get)
        tags.append(yi)
    
    return tags

# Step 3: Evaluate on the dev data and generate `greedy.json`.


predictions = []
correct_tags = 0
total_tags = 0

for record in dev_data:
    sentence = record["sentence"]
    labels = record["labels"]

    predicted_tags = Decode(sentence, transition, emission)
    predictions.append({
        "index": record["index"],
        "sentence": sentence,
        "labels": predicted_tags
    })

    correct_tags += sum([1 for true, pred in zip(labels, predicted_tags) if true == pred])
    total_tags += len(labels)

accuracy = correct_tags / total_tags

# Save predictions to greedy.json
with open('/home/ubuntu/gxy/hw2/result/greedy.json', 'w') as f:
    json.dump(predictions, f, indent=4)


# Answer to the question:


print(f"Accuracy on dev data: {accuracy:.4f}")
