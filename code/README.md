# **`script.py`: A Step-by-Step Guide**

---

## 🚀 Getting Started

### **1. Configure Your File Paths**
- **Training Data**: 
  - Path: `train_path`
  - Default: `train.json` *(load path)*
  
- **Development Data**: 
  - Path: `dev_path`
  - Default: `dev.json` *(load path)*
  
- **Testing Data**: 
  - Path: `test_path`
  - Default: `test.json` *(load path)*

- **Vocabulary File**:
  - Path: `vocab_path`
  - Default: `vocab.txt` *(save path)*

- **HMM Parameters**: 
  - Path: `hmmpara_path`
  - Default: `hmm.json` *(save path)*
  
- **Greedy alg prediction**: 
  - Path: `greedy_path`
  - Default: `greedy.json` *(save path)*
  
- **Viterbi alg prediction**: 
  - Path: `viterbi_path`
  - Default: `viterbi.json` *(save path)*
  
### **2. Execute The Script**

Run `script.py` and observe the output. The console will display the following metrics:

1. **Frequency of `<unk>` Token**
2. **Overall Vocabulary Size**
3. **Number of Transition and Emission Parameters**
4. **Dev Data Accuracy** (Using the Greedy algorithm - Task 3)
5. **Dev Data Accuracy** (Using the Viterbi decoding - Task 4)

---

**📌 Note:** Ensure the data files are placed in their respective directories as mentioned.

Happy Coding! 🎉