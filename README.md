# Instruction Following Robustness Evaluation


![Illustration of our instruction-following robustness evaluation](./pics/illustration.png)

The objective of our benchmark is to 
1. quantify the extent to which instruction-following LLMs are influenced by injected adversarial instructions;
2. assess their ability to differentiate between these injected adversarial instructions and original user instructions.
   
As shown in the illustration above, we focus on the real-world scenario that current commercial conversational agents (e.g., Bing chat) face, where LLMs are tasked to answer user questions based on the web search results/retrieved knowledge. In this scenario, the web search results can potentially be pre-injected with adversarial instructions by third-party attackers in the websites.

To avoid being guardrailed by LLM vendors and exclude the influence of malicious content, we examine two types of benign adversarial instructions: 
1. random task instructions generated by self-instruct, which is irrelevant to the original user questions and search results
2. automatically generated relevant questions about the web search results, but different from the original user queries. 


## How to evaluate

### Injection (Evaluation data preparation)
Insert the adversarial instructions into the question-answering evaluation samples. Adjust the injection using the following parameters:

- position: Where to embed the adversarial instruction in the context (web search results).
- attack_type: Choose between direct (no prefix) or ignore (prepend with "ignore previous prompt").
- task_type: Type of instruction to inject – relevant for context-appropriate questions or irrelevant for random directives via self-instruct.
- qg_model: Model used for generating the relevant adversarial questions.
- test_mode: Set evaluation mode: original (original query only), injected (injected query only), or original+injected (both).
- dataset: Use either NaturalQuestions or TriviaQA.


### Evaluation
Once the data is prepared, proceed with evaluating your LLM.

* LLM Inference Code: Check `llm_utils.py` for the necessary code for LLM inference.
* LLM Configuration: Add the model you intend to evaluate in `llm_configs.py` and assign an alias.
* Evaluation Parameters: Specify the following arguments:
1. All parameters from the Injection section to indicate the data you intend to evaluate on.
2. `model`: specify the alias of the model that you intend to evaluate (recorded in the `llm_configs.py` file).

We provide the inference results in the `./evaluation/` directory. The human evaluation protocol and evaluation results are provided in the `./human_evals/` directory.
