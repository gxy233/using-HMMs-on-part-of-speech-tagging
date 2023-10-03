
# Step 1: Load the training data.


import json

with open('/home/ubuntu/gxy/hw2/CSCI544_HW2/data/train.json', 'r') as f:
    train_data = json.load(f)


# Step 2: Count transitions, emissions, and initial states.


from collections import defaultdict

transition_counts = defaultdict(int)
emission_counts = defaultdict(int)
initial_state_counts = defaultdict(int)
total_sentences = len(train_data)

for record in train_data:
    sentence = record["sentence"]
    labels = record["labels"]
    
    # Count the initial state
    initial_state_counts[labels[0]] += 1
    
    for i in range(len(labels)):
        # Emission counts
        emission_key = (labels[i], sentence[i])
        emission_counts[emission_key] += 1
        
        # Transition counts (skip the last word as there's no transition after it)
        if i < len(labels) - 1:
            transition_key = (labels[i], labels[i+1])
            transition_counts[transition_key] += 1



# Step 3: Calculate transition and emission probabilities.


transition_probs = {}
emission_probs = {}

# Total counts for each state/label
state_counts = defaultdict(int)
for label, _ in emission_counts.keys():
    state_counts[label] += emission_counts[(label, _)]

for key, count in transition_counts.items():
    s, s_prime = key
    transition_probs[f"{s},{s_prime}"] = count / state_counts[s]

for key, count in emission_counts.items():
    s, x = key
    emission_probs[f"{s},{x}"] = count / state_counts[s]


# Step 4: Save the model to `hmm.json`.

model = {
    'transition': transition_probs,
    'emission': emission_probs
}

with open('/home/ubuntu/gxy/hw2/result/hmm.json', 'w') as f:
    json.dump(model, f, indent=4)



# num_transition_params` gives the number of transition parameters, and `num_emission_params` gives the number of emission parameters.

num_transition_params = len(transition_probs)
num_emission_params = len(emission_probs)

print(f'{num_transition_params=}_{num_emission_params=}')