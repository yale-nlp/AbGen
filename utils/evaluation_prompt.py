from string import Template

IMPORTANCE_CRITERIA = Template("""
Based on the provided research context—including background, methodology, main experimental setup, and results—evaluate the proposed ablation study focused on $module.
Rate its importance on a scale from 1 (least important) to 5 (most important) by judging how much new, decision-changing insight the ablation would contribute to understanding the role of $module in the overall findings. 
The criteria for scoring are as follows:

- **Score 5**
  • Directly tests the core function or a make-or-break assumption about the $module that earlier experiments did not isolate
  • Without this test, the true importance of $module remains uncertain
  • Likely to guide future work or practical adoption of the overall method

- **Score 4**
  • Targets a component tightly linked to performance or a key theoretical role of the $module
  • Results would materially strengthen or weaken confidence in the contribution of $module, though the overall method could still stand  
  • Adds insight most readers would consider essential follow-up

- **Score 3**
  • Examines a secondary but still relevant aspect within the $module 
  • Refines understanding or best practices but is unlikely to change the headline view of the $module's role
  • Provides useful nuance rather than pivotal evidence

- **Score 2**
  • Focuses on a peripheral detail whose impact on the significance of $module is expected to be small  
  • Outcomes would be nice to know but not influence the main narrative about the $module

- **Score 1**
  • Tests something unrelated to the $module or reiterates well-established knowledge.  
  • Whatever the result, it would not meaningfully inform understanding of the $module’s contribution.

Please begin by clearly explaining the reasoning behind your rating. Conclude with a separate sentence stating the numerical score from 1 to 5, using the format: "Therefore, the score is: X", where X is the number you have assigned.
""")


FAITHFULNESS_CRITERIA = Template("""
Based on the provided research context—including background, methodology, main experimental setup, and results—evaluate the **faithfulness** of the proposed ablation study focused on $module. 
Rate its faithfulness on a scale from 1 (least faithful) to 5 (most faithful). Your rating should capture how accurately the ablation reflects the original research framework, constraints, and assumptions surrounding $module, without introducing contradictions or unsupported additions.

- **Score 5**  
  • Every element of the ablation (datasets, preprocessing, hyper-parameters, model variants, evaluation metrics, training budget, etc.) matches the stated methodology for $module with *zero* contradictions or speculative extensions.  
  • References results and baseline behaviors involving $module exactly as reported, maintaining full logical consistency with prior findings.  

- **Score 4**  
  • Fully consistent with all major methodological details and prior results involving $module, though it may omit a minor parameter choice that does **not** affect interpretability.  
  • No contradictions; any assumptions are explicitly justified by the context or common practice.  
  • A knowledgeable reader would trust the design to reproduce comparable conditions for assessing $module.

- **Score 3**  
  • Largely faithful but exhibits one of the following:  
    – A small mismatch in secondary settings (e.g., batch size, learning-rate schedule) *or*  
    – An implied procedural step related to $module not spelled out in the paper but commonly inferred.  
  • These gaps are unlikely to flip conclusions but could affect exact reproducibility.

- **Score 2**  
  • Contains noticeable inconsistencies or unsupported extensions—e.g., adds a dataset not used in the paper, changes the evaluation metric, or modifies network depth—in ways that obscure $module's original context.  
  • While the ablation might still shed some insight, its results would be difficult to interpret against the original findings.

- **Score 1**  
  • Clearly contradicts the published methodology or results—e.g., uses a different problem formulation, ignores critical control conditions for $module.  
  • Introduces speculative modules or assumptions absent from the research context.  
  • Outcomes would not be comparable or informative regarding $module’s contribution.

Please begin by clearly explaining the reasoning behind your rating. Conclude with a separate sentence stating the numerical score from 1 to 5, using the format: "Therefore, the score is: X", where X is the number you have assigned.
""")

SOUNDNESS_CRITERIA = Template("""
Based on the provided research context—including background, methodology, and main experimental setup—evaluate the **soundness of the *design*** of the proposed ablation study focused on $module.   
Rate its soundness on a scale from 1 (least sound) to 5 (most sound). Your rating should consider only the *planned* experimental design—controls, variable isolation, statistical plan, and reproducibility—not any hypothetical or reported results.

- **Score 5**   
  • Evaluation metrics and analysis plan are fully specified and directly address the research question, leaving no unexplained degrees of freedom.  
  • Procedures are clearly described and reproducible; if executed, the study would unambiguously attribute effects to $module.

- **Score 4**  
  • Robust design that controls all key variables but omits a minor secondary check (e.g., reduced hyper-parameter sweep or fewer seeds).  
  • Any limitations are explicitly acknowledged and unlikely to compromise interpretability.  
  • Most readers would trust the study to yield reliable insight into $module's role.

- **Score 3**  
  • Generally sound but with noticeable gaps—single-seed runs, minimal hyper-parameter tuning, or partial baseline coverage.  
  • Potential confounds exist, yet the design should still reveal meaningful trends about $module.  
  • Conclusions would be suggestive rather than definitive.

- **Score 2**  
  • Lacks one or more critical controls or baselines, making it difficult to attribute changes solely to $module.  
  • Statistical power appears low or evaluation metrics are only loosely justified.  
  • Design risks producing ambiguous evidence without further verification.

- **Score 1**  
  • Severely flawed: no controls, multiple variables changed simultaneously, or use of incompatible data/metrics.  
  • Cannot support any reliable conclusion about the role of $module, even if executed.

Please begin by clearly explaining the reasoning behind your rating. Conclude with a separate sentence stating the numerical score from 1 to 5, using the format: "Therefore, the score is: X", where X is the number you have assigned.
""")

OVERALL_CRITERIA = Template("""
Based on the provided research context—including background, methodology, and main experimental setup—give an **overall evaluation** of the proposed ablation study focused on $module.  
Integrate all relevant dimensions (the importance of the insight it could yield, faithfulness to the original framework, and soundness of the experimental design) to judge how convincingly the study would clarify the role of $module.

Rate the design on a scale from **1 (weakest) to 5 (strongest)**. Your rating should consider only the *planned* experimental design—controls, variable isolation, statistical plan, and reproducibility—not any hypothetical or reported results.

- **Score 5**  
  • Simultaneously critical (high prospective impact), fully aligned with the original methodology, and rigorously designed.  
  • If executed, it would deliver decisive, trustworthy evidence about $module, likely shaping future research or practical adoption.  
  • No substantive weaknesses detected across any dimension.

- **Score 4**  
  • Strong on at least two dimensions and solid on the third (e.g., very important and faithful, with only a minor design limitation).  
  • Insights would substantially influence interpretation of $module, albeit with a small caveat or acknowledged trade-off.

- **Score 3**  
  • Adequate but mixed: clear value in addressing $module, yet moderate gaps in relevance, methodological match, or design rigor.  
  • Findings would refine understanding but are unlikely to overturn headline conclusions or set a new standard.

- **Score 2**  
  • Noticeable deficiencies in two dimensions—e.g., peripheral importance and design weaknesses—even if one aspect remains acceptable.  
  • Any knowledge gained would be tentative, with limited effect on the overall narrative about $module.

- **Score 1**  
  • Lacks meaningful importance, diverges from the original framework, and/or is methodologically unsound to the point of being uninterpretable.  
  • Whatever the outcome, it would not advance understanding of $module’s contribution.

Please begin by clearly explaining the reasoning behind your rating. Conclude with a separate sentence stating the numerical score from 1 to 5, using the format: "Therefore, the score is: X", where X is the number you have assigned.
""")



def prepare_user_prompt(example):
    research_background = f"Research Background:\n{example['research_background']}\n"
    method = f"Method Section:\nexample['method']\n"
    main_experiment = f"Main Experiment Setup\n{example['main_experiment']['experiment_setup']}\n\n Main Experiment Results\n{example['main_experiment']['results']}\n"

    references = example["ablation_study"]["experiment_setup"]
    ablation_module = example["ablation_study"]["module_name"]

    response = example["model_response"]

    return f"""
Research Context:
{research_background}{method}{main_experiment}
Based on the above research context, the following ablation study has been designed:
{response}

Using the provided scoring criteria, please evaluate this ablation study, with a particular focus on the {ablation_module}.

For reference, here is an example ablation study setup from the original paper:
{references}

Provide your evaluation according to the specified criteria.
"""

def load_data(data, aspect):
    messages = []
    for example in data:
        messages.append([
            {"role": "system", "content": aspect_dict[aspect].substitute(module=example["ablation_study"]["module_name"])},
            {"role": "user", "content": prepare_user_prompt(example)}
        ])
        
    return messages

def build_faithfulness_single_message(example):
    return [
        {"role": "system", "content": FAITHFULNESS_CRITERIA.substitute(module=example["ablation_study"]["module_name"])},
        {"role": "user", "content": prepare_user_prompt(example)}
    ]

def build_soundness_single_message(example):
    return [
        {"role": "system", "content": SOUNDNESS_CRITERIA.substitute(module=example["ablation_study"]["module_name"])},
        {"role": "user", "content": prepare_user_prompt(example)}
    ]

def build_importance_single_message(example):
    return [
        {"role": "system", "content": IMPORTANCE_CRITERIA.substitute(module=example["ablation_study"]["module_name"])},
        {"role": "user", "content": prepare_user_prompt(example)}
    ]
    
def build_overall_single_message(example):
    return [
        {"role": "system", "content": OVERALL_CRITERIA.substitute(module=example["ablation_study"]["module_name"])},
        {"role": "user", "content": prepare_user_prompt(example)}
    ]