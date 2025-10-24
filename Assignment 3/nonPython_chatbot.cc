#include<iostream>
#include<map>
#include<string>
#include<vector>
#include<sstream>
#include<fstream>
#include<regex>
#include<locale>
#include<algorithm>
#include<cwctype>
#include<torch/script.h>
#include<memory>
#include<tuple>

using namespace std;

class Voc {
private:
    unordered_map<string, int> word2index;
    unordered_map<int, string> index2word;
    int next_index;
    int PAD_token = 0;  // Used for padding short sentences
    int SOS_token = 1;  // Start-of-sentence token
    int EOS_token = 2;  // End-of-sentence token
    int UNK_token = -1;

    vector<string> splitString(const string& str, char delimiter) {
        vector<string> tokens;
        stringstream ss(str);
        string token;
        
        while (getline(ss, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }

    void addWord(const string& word) {
        auto it = word2index.find(word);
        if (it == word2index.end()) {
            word2index[word] = next_index;
            index2word[next_index] = word;
            next_index++;
        }
    }

public:
    Voc(){}

    Voc(string &vocabPath) {
        next_index = 3;
        index2word[PAD_token] = "PAD";
        index2word[SOS_token] = "SOS";
        index2word[EOS_token] = "EOS";
        index2word[UNK_token] = "<UNK>";

        ifstream file(vocabPath);
        string line;
        vector<string> tokens;

        while (getline(file, line)) {
            tokens = splitString(line, ' ');
            index2word[stoi(tokens[1])] = tokens[0];
            word2index[tokens[0]] = stoi(tokens[1]);
            next_index++;
        }
    }

    // Add sentence to vocabulary
    void addSentence(string &sentence) {
        auto tokens = splitString(sentence, ' ');
        for(string token: tokens) {
            addWord(token);
        }
    }

    // Convert sentence to tensor
    vector<int> sentenceToTensor(const string& sentence) {
        vector<int> tensor;
        istringstream iss(sentence);
        string word;
        while (iss >> word) {
            int index = wordToIndex(word);
            tensor.push_back(index);
        }
        tensor.push_back(EOS_token);
        return tensor;
    }

    // Convert tensor to sentence
    string tensorToSentence(const vector<int>& tensor) {
        string sentence;
        for (int index : tensor) {
            string word = indexToWord(index);
            if(word != "EOS" && word != "PAD")
                sentence += word + " ";
        }
        return sentence;
    }

    int vocabSize() {
        return index2word.size();
    }

private:
    // Get index of word in vocabulary
    int wordToIndex(const string& word) {
        auto it = word2index.find(word);
        if (it != word2index.end()) {
            return it->second;
        } else {
            return UNK_token; // Out of vocabulary
        }
    }

    // Get word from index
    string indexToWord(int index) {
        auto it = index2word.find(index);
        if (it != index2word.end()) {
            return it->second;
        } else {
            return "<UNK>"; // Unknown word
        }
    }
};

class Utils {
public:

    Utils() {}

    static bool isNotMn(wchar_t c) {
        return iswprint(c) && !std::iswcntrl(c);
    }

    static string unicodeToAscii(const string& s) {
        string result;
        for (char c : s) {
            if (iswspace(static_cast<wchar_t>(c)) || iswalnum(static_cast<wchar_t>(c)) || iswpunct(static_cast<wchar_t>(c))) {
                // Preserve spaces, alphanumeric characters and punctuations
                result += c;
            } else {
                // Convert accented characters to their base form
                wchar_t wc = static_cast<wchar_t>(c);
                wchar_t normalized = towlower(wc);
                if(isNotMn(normalized))
                    result += static_cast<char>(normalized);
            }
        }
        return result;
    }

    static string normalizeString(const string& s) {

        // Remove leading and trailing whitespaces
        string trimmed = regex_replace(s, regex("^\\s+|\\s+$"), "");

        // Convert to lowercase
        locale loc;
        string lowercase;
        for (char c : trimmed) {
            lowercase += tolower(c, loc);
        }
        string ascii = unicodeToAscii(lowercase);

        // Insert space before punctuation
        string withSpaces = regex_replace(ascii, regex("([.!?])"), " $1");

        // Remove non-letter characters
        string lettersOnly = regex_replace(withSpaces, regex("[^a-zA-Z.!?]+"), " ");

        // Remove consecutive whitespaces
        string finalString = regex_replace(lettersOnly, regex("\\s+"), " ");

        // Remove leading and trailing whitespaces
        finalString = regex_replace(finalString, regex("^\\s+|\\s+$"), "");

        return finalString;
    }

    //Function to read query/response pairs
    static vector<pair<string, string> > readVocs(const string& datafile) {
        cout << "Reading lines...\n";
        ifstream file(datafile);
        vector<pair<string, string> > pairs;
        string line;
        while (getline(file, line)) {
            istringstream iss(line);
            string query, response;
            if (getline(iss, query, '\t') && getline(iss, response)) {
                pairs.push_back(make_pair(normalizeString(query), normalizeString(response)));
            }
        }
        return pairs;
    }

    static int splitStringLen(const string& str) {
        int len = 0;
        istringstream iss(str);
        string token;
        while (iss >> token) {
            len++;
        }
        return len;
    }

    static vector<pair<string, string> > filterPairs(vector<pair<string, string> > &pairs, int len) {
        vector<pair<string, string> > newPair;
        for(auto p: pairs) {
            int l1 = splitStringLen(p.first);
            int l2 = splitStringLen(p.second);
            if(l1 < len && l2 < len) {
                newPair.push_back(p);
            }
        }
        return newPair;
    }
};

class TorchModel {
public:
    Voc vocab;
    torch::jit::script::Module module;
    int max_len;

    TorchModel(Voc &vocab, torch::jit::script::Module &module, int max_len) {
        this->vocab = vocab;
        this->module = module;
        this->max_len = max_len;
    }

    string getResponse(string query) {
        string inputSentence = Utils::normalizeString(query);
        vector<int> inputVec = vocab.sentenceToTensor(inputSentence);
        vector<int> inputLen;
        inputLen.push_back(inputVec.size());

        // torch::Tensor input_seq_tensor = torch::from_blob(inputVec.data(), {(int)inputVec.size(), 1}, torch::kInt);
        torch::Tensor input_seq_tensor = torch::tensor(inputVec);
        input_seq_tensor = input_seq_tensor.unsqueeze(0);
        input_seq_tensor = input_seq_tensor.permute({1, 0});
        input_seq_tensor = input_seq_tensor.to(torch::kInt);
        torch::Tensor input_len_tensor = torch::tensor(inputLen);
        input_len_tensor = input_len_tensor.to(torch::kInt);

        vector<torch::jit::IValue> arguments;
        arguments.push_back(input_seq_tensor);
        arguments.push_back(input_len_tensor);
        arguments.push_back(max_len);
        auto outputs_tuple = module.forward(arguments).toTuple();
        auto output1 = outputs_tuple->elements()[0].toTensor();
        auto output2 = outputs_tuple->elements()[1].toTensor();

        auto detached_tensor = output1.cpu().detach();
        vector<int> out_vec;

        #ifdef __linux__
        out_vec = vector<int>(detached_tensor.data_ptr<long>(), detached_tensor.data_ptr<long>() + detached_tensor.numel());
        #elif __APPLE__
        out_vec = vector<int>(detached_tensor.data_ptr<long long>(), detached_tensor.data_ptr<long long>() + detached_tensor.numel());
        #else
        out_vec = vector<int>(detached_tensor.data_ptr<long>(), detached_tensor.data_ptr<long>() + detached_tensor.numel());
        #endif

        // vector<int> out_vec(detached_tensor.data_ptr<long long>(), detached_tensor.data_ptr<long long>() + detached_tensor.numel());
        string result = vocab.tensorToSentence(out_vec);
        return result;
    }
};

int main(int argc, char* argv[]) {
    
    // string fileName = "formatted_movie_lines.txt";
    string modelPath = "scripted_chatbot_cpu.pt";
    string vocabPath = "vocab.csv";
    if(argc == 3) {
        modelPath = argv[1];
        vocabPath = argv[2];
    }

    int MAX_LENGTH = 10;

    Voc vocab(vocabPath);
    cout << "Vocab Size: " << vocab.vocabSize() << endl;

    c10::InferenceMode guard(true);
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(modelPath);
        std::cerr << "Model Loaded" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model" << e.what() << std::endl;
        return -1;
    }
    TorchModel seqModel(vocab, module, MAX_LENGTH);

    string inputSentence = "Hello";
    cerr << "User: " << inputSentence << std::endl;
    cerr << "Bot: " << seqModel.getResponse(inputSentence) << endl;
    
    while(true) {
        cout << "User: ";
        string userInput;
        getline(std::cin, userInput);
        if(userInput == "quit" || userInput == "q")
            break;
        cout << "Bot: " << seqModel.getResponse(userInput) << endl;
    }
    return 0;
}
