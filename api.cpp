using namespace std;
class Generator {

};

class Pipeline {

};

class LLM : Pipeline {

};

class VLM : LLM {
public:
    void add_context(const std::string&) {}

};

class Whisper : Pipeline {
public:
    Generator generate(?);  // return word and probability or logit for greedy
    State state_generate();  // return all beams
};

class Sampler {

};

class Tokenizer {

};

class ChatHistory {

};

class Streamer {

};

class Thread {

};

// Take compiled model or ireq or ov model or whatever comes from ovAutoModel. Chat history outside so the cancel and stop behave persosly without extra tail. Pair ireq and it's tokenized history. Single threaded iterabke streamer. Don't distinguish llm and vlm. Don't require tokenizer. How to use this unterface with CB, image gen, promt lookup and speculative decoding. Corutines. All should be applicable for c api. Background thread. Handle stop strings. Clear cache and abort the last infer. Aborting rhe last infer also enables abort in the middle. Allow using only llm part of vlm to build class hierarchy: embeddings-language-visual-withTokenizer-async. Should the be dynamism: load whatever is in my folder and I may pass an image and you should be able to process it. It's like MaybeVisual. Callback wrapper should be able to wrap withTokenizer or async. But better to be able to wrap all possible classes. What about llm that loops on last layer state instead of tokens. What about generating an image like a text. How is usually written conversation loop in transformers. Does ovmodelforcausallm use ov tokenizers or transformers? Expose samplers? Match with rag api. Whisper? TTS? Video and music generation? Video as an input. Проверить, что ещё нужно для апи 2.0 в трелло и эксэле. Return probability for accuracy calculation. Speculative decoding is just an addition of infer loop, but need to add api for verification by bigger model. Prompt lookup is a regular generation with internal verification. Sampler is usually inside to enable async infer but would still be nice to expose it if async isn't needed. Serving scenario. Termite tensor. Batch support. Fionas requests. Chaining pipelines. Different devices for vlm. How many chars would llm wait from whisper before detokenization to ensure tokens don't change? Take ov midrl, compilrd model or ireqs in constructir?
// Is there RAG from VLM, Whisper?
// scenarios:
// beam search
// token stream
// str stream
// no sampler
// passing one to another
// multiple conversations
// minimal example to make async callback for any of the Pipelines in one line
// most verbose example
// most verbose without Sampler
// wrap with tokenizer
// Wrap with async queue
// wrap with callback

void minimal_example() {
    ChatHistory conversation;
    VLM vlm([](string&& word) {cout << word << flush; return RUNNING;});
    string prompt;
    while (getline(cin, prompt)) {
        vlm.generate();
    }
}

void verbose_example() {
    Tokenizer whisper_detokenizer;
    Whisper whisper{?};
    VLM vlm;
    ChatHistory conversation;
    Generator decoded = Thread.wrap(whisper_detokenizer.wrap(whisper.iterate_generated()));
    for (auto& [word, problog] : generator) {  // not logprob
        conversation.push_back(word);
        vlm.extend_context(conversation);  // This should stack requests asyncroniously internally concatinating them in a row if was'nt able to process in time
    }
    std::string prompt;
    Streamer streamer;
    while (std::getline(std::cin, prompt)) {
        State generator = vlm.state_generate(conversation);
        for (Beams& beams : generator) {
            streamer.put(beams);
        }
    }
}

int main() {
    minimal_example();
    verbose_example();
}
