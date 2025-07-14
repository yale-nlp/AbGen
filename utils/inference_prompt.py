GENERATION_SYSTEM_PROMPT="""Given the research context, design an ablation study for the specified module or process. Begin the design with a clear statement of the research objective, followed by a detailed description of the experiment setup. Do not include the discussion of results or conclusions in the response, as the focus is solely on the experimental design. 
The response should be within 300 words. Present the response in plain text format only."""

def prepare_user_prompt(example):
    research_background = f"Research Background:\n{example['research_background']}\n"
    method = f"Method Section:\nexample['method']\n"
    main_experiment = f"Main Experiment Setup\n{example['main_experiment']['experiment_setup']}\n\n Main Experiment Results\n{example['main_experiment']['results']}\n"

    ablation_module = example["ablation_study"]["module_name"]

    return f"Research Context:\n{research_background}{method}{main_experiment}\n\n Design an ablation study about {ablation_module} based on the research context above."

def load_data(data):
    messages = []
    for example in data:
        messages.append([
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prepare_user_prompt(example)}
        ])
        
    return messages

def build_single_message(example):
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": prepare_user_prompt(example)}
    ]