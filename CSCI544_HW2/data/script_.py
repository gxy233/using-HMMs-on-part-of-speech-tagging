# 1. Read the training data
import json
from collections import defaultdict

# train_path='/home/ubuntu/gxy/hw2/CSCI544_HW2/data/train.json'
# hmmpara_path='/home/ubuntu/gxy/hw2/result/hmm.json'
# dev_path='/home/ubuntu/gxy/hw2/CSCI544_HW2/data/dev.json'
# test_path='/home/ubuntu/gxy/hw2/CSCI544_HW2/data/test.json'
# vocab_path='/home/ubuntu/gxy/hw2/result/vocab.txt'

#load path
train_path='train.json'
dev_path='dev.json'
test_path='test.json'

#save path
vocab_path='vocab.txt'
hmmpara_path='hmm.json'
greedy_path='greedy.json'
viterbi_path='viterbi.json'

def task1():
    with open(train_path, 'r') as f:
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
    print(f'{unknown_count=}')
    
    
    sorted_vocab = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)

    # 4. Filter out words with frequency below the threshold
    filtered_vocab = [(word, freq) for word, freq in sorted_vocab if freq >= UNKNOWN_THRESHOLD]
    # 5. Write to vocab.txt
    with open(vocab_path, 'w') as f:
        f.write("<unk>\t0\t" + str(unknown_count) + "\n")
        
        index = 1
        for word, freq in filtered_vocab:
            f.write(f"{word}\t{index}\t{freq}\n")
            index += 1
    print(f'overall size of your vocabulary is {len(filtered_vocab)+1}')

def task2():
    
    # Step 1: Load the training data.

    with open(train_path, 'r') as f:
        train_data = json.load(f)


    # Step 2: Count transitions, emissions, and initial states.



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
    
    #initial var
    transition_probs = {}
    emission_probs = {}

    #count the number of states and words
    all_states = set(label for record in train_data for label in record["labels"])
    all_words = set(word for record in train_data for word in record["sentence"])
    N_states = len(all_states)
    N_words = len(all_words)

    # Total counts for each state/label
    state_counts = defaultdict(int)
    for label, _ in emission_counts.keys():
        state_counts[label] += emission_counts[(label, _)]

    
    # ---------set probability =0 if missing the key
    for key, count in transition_counts.items():
        s, s_prime = key
        transition_probs[f"{s},{s_prime}"] = count / state_counts[s]

    for key, count in emission_counts.items():
        s, x = key
        emission_probs[f"{s},{x}"] = count / state_counts[s]
    
    
    
    # # ---------laplacian smoothing
    
    # # For transition probabilities
    # for key, count in transition_counts.items():
    #     s, s_prime = key
    #     transition_probs[f"{s},{s_prime}"] = (count) / (state_counts[s])

    # counter_dict=defaultdict(int)
    # # For emission probabilities
    # for key, count in emission_counts.items():
    #     s, x = key
    #     if s not in counter_dict.keys():
    #         for key1, count1 in emission_counts.items():
    #             s1, x1 = key1
    #             if s==s1:
    #                 counter_dict[s]+=count1
             
             
    #     # emission_probs[f"{s},{x}"] = (count + 1) / (state_counts[s] + N_words)
    #     emission_probs[f"{s},{x}"] = (count) / (counter_dict[s])


    ini_count=0
    for key, count in initial_state_counts.items():
        ini_count=ini_count+count
   
    
    # for key, count in initial_state_counts.items():
    #     initial_state_counts[key]=initial_state_counts[key]/ini_count
    
    # for key, count in initial_state_counts.items():
    #     initial_state_counts[key] = (count + 1) / (total_sentences + N_states)


    initial_state_pro=defaultdict(int)
    for key in state_counts:
        v1=initial_state_counts[key]
        initial_state_pro[key]+=(v1+1)/(total_sentences + N_states)

   
    
    # Step 4: Save the model to `hmm.json`.

    model = {
        'transition': transition_probs,
        'emission': emission_probs
    }

    with open(hmmpara_path, 'w') as f:
        json.dump(model, f, indent=4)



    # num_transition_params` gives the number of transition parameters, and `num_emission_params` gives the number of emission parameters.

    num_transition_params = len(transition_probs)
    num_emission_params = len(emission_probs)

    print(f'{num_transition_params=}_{num_emission_params=}')

    paradict = {
        'initial_state_pro': initial_state_pro,
        'N_words': N_words,
        'N_states': N_states,
        'state_counts': state_counts
    }

    return paradict
    
def task3(paradict):
    
    # Load the learned model from hmm.json
    with open(hmmpara_path, 'r') as f:
        model = json.load(f)

    transition = model['transition']
    emission = model['emission']

    # Load the development data
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)

    with open(test_path, 'r') as f:
        test_data = json.load(f)

    # Step 2: Implement the Decode function.


    def Decode(piror_prob, transition, emission):
        S = piror_prob
        
        T = len(sentence)
    
        tags = []

        # Get y1
        y1_probs = {}
        for s, ini_pro in S.items():
            emission_key = f"{s},{sentence[0]}"
            y1_probs[s] = ini_pro * emission.get(emission_key, 0)  # Using dict.get() to handle missing emission keys
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

    # Step 3: Evaluate on the dev data .

    predictions = []
    correct_tags = 0
    total_tags = 0

    for record in dev_data:
        sentence = record["sentence"]
        labels = record["labels"]

        predicted_tags = Decode(paradict['initial_state_pro'], transition, emission)
        predictions.append({
            "index": record["index"],
            "sentence": sentence,
            "labels": predicted_tags
        })

        correct_tags += sum([1 for true, pred in zip(labels, predicted_tags) if true == pred])
        total_tags += len(labels)

    accuracy = correct_tags / total_tags


    # generate `greedy.json`
    predictions_ = []


    for record in test_data:
        sentence = record["sentence"]

        predicted_tags = Decode(paradict['initial_state_pro'], transition, emission)
        predictions_.append({
            "index": record["index"],
            "sentence": sentence,
            "labels": predicted_tags
        })

  

    # Save predictions to greedy.json
    with open(greedy_path, 'w') as f:
        json.dump(predictions_, f, indent=4)


    # Answer to the question:


    print(f"Accuracy on dev data using Greedy algorithm: {accuracy:.4f}")






def task4(paradict):
    # for Laplace smoothing

    VOCAB_SIZE = paradict['N_words']  # Total number of unique words in your training data
    SMOOTHING = 1  # for Laplace smoothing
    
    
    
    def viterbi_decode(prior_prob, transition, emission, sentence):
        T = len(sentence)
        states = list(prior_prob.keys())

        # Initialization
        V = [{}]   # probability table
        path = {}  # path table

        # Initialize base cases (t == 0)


        for s in states:
            emission_key = f"{s},{sentence[0]}"
            # 使用拉普拉斯平滑
            emission_prob = (emission.get(emission_key, 0) + 1) / (paradict['state_counts'][s] + VOCAB_SIZE)
            V[0][s] = prior_prob[s] * emission_prob
            path[s] = [s]



        # For t > 0
        for t in range(1, T):
            V.append({})
            newpath = {}

            for s in states:
                # Use max to find the maximum probability for previous state that leads to current state
                emission_prob = (emission.get(f"{s},{sentence[t]}", 0)*paradict['state_counts'][s] + SMOOTHING) / (paradict['state_counts'][s] + VOCAB_SIZE)
                
                # Applying smoothing for transition probabilities as well
                (prob, state) = max(
                    (
                        V[t-1][y] * (transition.get(f"{y},{s}", 0)*paradict['state_counts'][y] + SMOOTHING) / (paradict['state_counts'][y] + len(states)) * emission_prob, 
                        y
                    ) for y in states
                )
                V[t][s] = prob
                newpath[s] = path[state] + [s]

            path = newpath
                
        # Look for the most probable final state
        (prob, state) = max((V[T - 1][s], s) for s in states)

        return path[state]



    def viterbi_decode2(prior_prob, transition, emission, sentence):
        T = len(sentence)
        states = list(prior_prob.keys())

        # Initialization
        V = [{}]   # probability table
        path = {}  # path table

        # Initialize base cases (t == 0)
        for s in states:
            emission_key = f"{s},{sentence[0]}"
            V[0][s] = prior_prob[s] * emission.get(emission_key,0.000000001)
            path[s] = [s]

        # Laplacian Smoothing
       

        # For t > 0
        # for t in range(1, T):
        #     V.append({})
        #     newpath = {}

        #     for s in states:
        #         # Use max to find the maximum probability for previous state that leads to current state
        #         # emission_prob = (emission.get(f"{s},{sentence[t]}", 0)*paradict['state_counts'][s] + SMOOTHING) / (paradict['state_counts'][s] + VOCAB_SIZE)
                
        #         emission_prob = (emission.get(f"{s},{sentence[t]}", 0))+ SMOOTHING / (paradict['state_counts'][s] + VOCAB_SIZE)
        #         print(f"{SMOOTHING / (paradict['state_counts'][s] + VOCAB_SIZE)=}")
                
        #         (prob, state) = max((V[t-1][y] * transition.get(f"{y},{s}", 0) * emission_prob, y) for y in states)
        #         V[t][s] = prob
        #         newpath[s] = path[state] + [s]

        #     path = newpath
           
        for t in range(1, T):
            V.append({})
            newpath = {}

            for s in states:
                # Use max to find the maximum probability for previous state that leads to current state
                (prob, state) = max((V[t-1][y] * transition.get(f"{y},{s}", 0.000000001) * emission.get(f"{s},{sentence[t]}", 0.000000001), y) for y in states)
                
                V[t][s] = prob
                newpath[s] = path[state] + [s]

            path = newpath
            
            
            
        # Look for the most probable final state
        (prob, state) = max((V[T - 1][s], s) for s in states)

        return path[state]
   
    # Load the learned model from hmm.json
    with open(hmmpara_path, 'r') as f:
        model = json.load(f)

    transition = model['transition']
    emission = model['emission']

    # Load the development data and test data
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        
        
    predictions = []
    correct_tags = 0
    total_tags = 0

    for record in dev_data:
        sentence = record["sentence"]
        labels = record["labels"]

        predicted_tags = viterbi_decode2(paradict['initial_state_pro'], transition, emission, sentence)
        predictions.append({
            "index": record["index"],
            "sentence": sentence,
            "labels": predicted_tags
        })

        correct_tags += sum([1 for true, pred in zip(labels, predicted_tags) if true == pred])
        total_tags += len(labels)

    accuracy = correct_tags / total_tags

    
    
    predictions_ = []


    for record in test_data:
        sentence = record["sentence"]

        predicted_tags = viterbi_decode2(paradict['initial_state_pro'], transition, emission, sentence)
        predictions_.append({
            "index": record["index"],
            "sentence": sentence,
            "labels": predicted_tags
        })


    # Save predictions to viterbi.json
    with open(viterbi_path, 'w') as f:
        json.dump(predictions_, f, indent=4)

    print(f"Accuracy on dev data using Viterbi decoding: {accuracy:.4f}")


if __name__ == "__main__":
    task1()
    paradict=task2()
    task3(paradict)
    task4(paradict)
   
