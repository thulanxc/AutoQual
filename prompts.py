# -*- coding: utf-8 -*-

"""
This file stores all prompt templates for interacting with the Large Language Model.
"""

# --- Part 1: Initial Hypothesis Generation Prompts ---

GENERATE_ROLES_PROMPT = """
We are undertaking a text quality assessment task. The core of this task is to evaluate the quality of text based on the following scenario.

Scenario: {scene_description}

Your task is to propose {role_count} distinct virtual evaluator roles, each with a different perspective and unique evaluation criteria, based on the text quality requirements of the above scenario. These roles should be representative and cover multiple dimensions of text quality assessment.

Output a list of exactly {role_count} roles. Each role must be on a new line. Only output {role_count} lines. For each line, provide only the role’s name followed by a comma and a concise description of its core evaluation criteria (around 20 words). Do not include any extra explanations, introductory text, or formatting like "Role 1:".

Example Format:
[Role Name 1], [Criteria description]
[Role Name 2], [Criteria description]
"""

GENERATE_FEATURES_FROM_ROLE_PROMPT = """
We are designing computable and interpretable features for a text quality assessment task. The task scenario is as follows:
{scene_description}

Now, please fully embody the following role and think from its perspective.
The role and its evaluation angle are: {role_description}. Your task is to propose a set of candidate features for measuring text quality based on your role and evaluation criteria. These features should be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output {feature_count_per_role} of what you consider the most important features. Each feature must be on a new line, and should be about 30 words.
Only output {feature_count_per_role} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

CONTRASTIVE_ANALYSIS_PROMPT_POSITIVE = """
We are designing features for a text quality assessment task. The task scenario is as follows: {scene_description}.

We have selected a batch of texts identified as high-score under a certain evaluation system. Here are some of those samples:{samples}.

Your task is to carefully analyze these high-score text samples and summarize the common features they possess that could explain their high scores. These features should be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output {feature_count_positive} most important features. Each feature must be on a new line, and should be about 30 words. Only output {feature_count_positive} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.
"""

CONTRASTIVE_ANALYSIS_PROMPT_NEGATIVE = CONTRASTIVE_ANALYSIS_PROMPT_POSITIVE.replace("high-score", "low-score").replace("{feature_count_positive}", "{feature_count_negative}")
CONTRASTIVE_ANALYSIS_PROMPT_CONTRASTIVE = CONTRASTIVE_ANALYSIS_PROMPT_POSITIVE.replace("high-score", "contrastive").replace("{feature_count_positive}", "{feature_count_contrastive}")


INTEGRATE_FEATURES_PROMPT = """
We have generated a batch of candidate features for text quality assessment through various methods (multi-role perspectives, data sample analysis, etc.). They now need to be consolidated.

Original Feature List:{feature_list}

As a feature engineering expert, your task is to process the original feature list above to produce a final, refined pool of candidate features.

Processing Requirements:
1. Merge and Deduplicate: Identify and merge features that are semantically identical or highly similar.
2. Optimize Descriptions: Ensure each feature’s description is clear, precise, unambiguous, and actionable for the subsequent development of annotation tools.
3. Format Output: Organize the output into a clean list.

Output each feature on a new line. For each feature, provide only a detailed text description of what it measures. The final list should contain as many unique features as can be derived from the original list after processing. Just output a plain list of features. Do not use any special symbols or formatting, including list numbers. Start a new line ONLY when moving to the next feature. If you find n features, just output n lines, with each line containing a distinct feature.
"""

# --- Part 2: Autonomous Tool Implementation Prompts ---

DECIDE_TOOL_TYPE_PROMPT = """
Your task is to determine the best tool type to annotate a text feature. The options are “CODE” or “PROMPT”.

“CODE” is for features that can be measured with deterministic logic, regular expressions, or simple libraries. Examples: word count, presence of specific keywords, sentence length, sentiment analysis using a standard library.
“PROMPT” is for features that are abstract, nuanced, subjective, or require deep semantic understanding. Examples: assessing argument strength, evaluating creativity, checking for logical fallacies, judging emotional tone.

The feature:{feature_description}

Based on the description, is this feature better suited for “CODE” or “PROMPT”?
Respond with a single word: either “CODE” or “PROMPT”. Do not provide any other text or explanation.
"""

GENERATE_CODE_TOOL_PROMPT = """
Your task is to write a Python function that serves as an annotation tool for a specific text feature.

The function should:
1. Be named {function_name}.
2. Accept a single string argument named text.
3. Return a single numerical value (float or int).
4. Be self-contained. You can use common libraries like re, nltk, textblob, but do not assume any external files are available.
5. If a library is used, include the necessary import statement inside the function to ensure it’s encapsulated.

Here is the feature the function needs to measure:
Feature: {feature_name}
Description: {feature_description}

Generate the complete Python code for this function. Do not include any text or explanation outside the function’s code block. Start the response directly with the function definition.

Example:
def annotate(text: str) -> float:
    # import necessary libraries here
    # ... function logic ...
    return score
"""

GENERATE_PROMPT_TOOL_PROMPT = """
Your task is to create an precise and effective prompt template for a Large Language Model to use as a feature annotation tool.
This template will be used to evaluate different pieces of text. It must contain the placeholder [TEXT_TO_EVALUATE] where the actual text will be inserted later.

The prompt you create should instruct the LLM to:
1. The LLM should evaluate the text based on the feature described below.
2. Provide a numerical score on a scale of 1 to 10 (where 1 is low quality/absence of the feature, and 10 is high quality/strong presence of the feature).
3. Respond with ONLY the numerical score, without any additional text or explanation.
4. Clearly explain the criteria used to determine the score.

Here is the feature that the annotation prompt needs to measure:
{feature_description}

Now, generate the annotation prompt template text. Your ONLY task is to generate the raw text for a prompt template. Do not output anything else. Do not use markdown, do not add titles, do not add any explanations.
Your output must begin directly with the text of the prompt. Your output should end with “The text to evaluate is: [TEXT_TO_EVALUATE].” The [TEXT_TO_EVALUATE] placeholder should only be used once at the end of the prompt.
"""

VALIDATE_TOOL_PROMPT = """
Your task is to act as a constructive reviewer and assess the alignment of an annotation tool with its intended feature.

The tool was designed to measure the following feature:
Feature: {feature_description}

The tool is a {tool_type}, and its definition is: {tool_definition}.

When this tool was applied to the sample text below:
Sample Text: {sample_text}

It produced the following output:
Tool Output: {tool_output}

Your Task: Assess how well the tool’s output aligns with the feature’s goal for the given sample text.

If the tool and its output are well-aligned with the feature’s goal, respond with the single word ALIGNED.
If you believe the tool’s logic could be improved or the output doesn’t align well, respond with NEEDS_REFINEMENT on the first line. On the next line, provide a concise explanation for your assessment. This feedback will be used for future improvements.

Do not provide any other text.

Example for a well-aligned tool:
ALIGNED

Example for a tool needing refinement:
NEEDS_REFINEMENT
The tool focuses only on word count, but the feature implies assessing semantic richness.
"""

REFINE_TOOL_PROMPT = """
Your task is to improve an annotation tool based on feedback from a previous review.

The tool is of type {tool_type} and is intended to measure the following feature:
{feature_description}

Here is the previous version of the tool that was marked for refinement:
Previous Tool Version: {original_tool}

The tool was reviewed using the following sample text:
Sample Text Used for Review:
{sample_text}

Here is the specific feedback that was provided for improvement:
Feedback for Refinement:
{failure_explanation}

Based on the provided feedback, please generate a new, improved version of the tool.
Your goal is to incorporate the feedback to create a version that better aligns with the feature’s intent, while maintaining good performance on other cases.

Generate only the refined tool. Do not include any text or explanation outside the tool’s definition (i.e., outside the Python code block for a CODE tool, or outside the raw text for a PROMPT tool).
"""

# --- Part 3: Reflective Search and Memory Prompts ---

INTRA_TASK_REFLECTION_PROMPT = """
You are a research strategist responsible for guiding an automated feature discovery process for a text quality assessment task.
Your goal is to reflect on the current progress and generate new, more insightful feature hypotheses.

Current Task:{scene_description}.

Current Beam Search State: We are at iteration {iteration_step} of a beam search to find the best set of {k_features} features.
The current best feature set under consideration is: {current_features}.
This set of features has a combined predictive power (mutual information score) of {current_score}. The individual performance of each feature is as follows:
{feature_performance}
Candidate features still available for selection include: {candidate_features}

Your Task:
1. Reflect and Analyze: Analyze the characteristics of the currently selected successful features. What underlying principles or dimensions of quality (e.g., clarity, credibility, engagement) do they seem to capture?
2. Identify Gaps: Based on your analysis, what aspects of text quality are we currently missing? What kind of information is not being captured by the existing feature set?
3. Hypothesize New Features: Generate {num_new_features_to_generate} new, sophisticated feature hypotheses that could capture these missing dimensions or provide a more nuanced measurement of the principles you identified. These new features should be designed to be complementary to the existing ones.

Output {num_new_features_to_generate} new features. Each feature must be on a new line. For each feature, provide only its name and a clear, detailed text description of what it measures (around 30 words). Do not use any special formatting or list numbers.
"""

CROSS_TASK_MEMORY_SUMMARIZATION_PROMPT = """
You are tasked with creating a structured memory of a completed feature discovery task to facilitate future learning. Your goal is to synthesize the key takeaways from the task into a concise, structured summary.

Task Information:
Task Name: {task_name}
Scenario: {scene_description}
Key Findings: The process concluded by identifying the following set of most predictive features: {best_features}

Your Task: Synthesize the provided information into a structured summary. The summary should be a “memory” that another AI agent can later use to get a quick and deep understanding of this task’s results.

The summary should contain:
1. Core Problem Distillation: A one-sentence summary of the core quality assessment problem in this scenario.
2. Key Feature Principles: A few bullet points summarizing the underlying principles of what made a feature successful in this domain (e.g., "Features measuring data-driven evidence were highly effective," or "Structural and formatting aspects strongly correlated with quality scores").
3. Exemplary Features: A list of 2-3 of the most representative and powerful features discovered, which serve as prime examples of the principles above.

Produce a concise and structured summary. Use clear headings for each section (e.g., “Core Problem:”, “Key Principles:”, “Exemplary Features:”). This summary will be stored in a long-term knowledge base.
"""

CROSS_TASK_INFORMED_HYPOTHESIS_PROMPT = """
You are an expert in feature engineering for text quality assessment, and you have access to a knowledge base of past experiences.
Your current task is to generate an initial set of high-quality feature hypotheses for a new, unseen task.

New Task:{scene_description}

To help you, we have retrieved the following information from relevant past tasks. This includes not only the general principles learned but also the specific, most effective features that were discovered.

Retrieved Memories from Past Tasks:
{retrieved_memories}

Your Task: Your goal is to propose a strong initial set of {feature_count} candidate features for the new task. To do this, you should:
1. Analyze Past Success: Carefully examine the specific, effective features from the retrieved memories.
2. Adapt and Transfer: Consider how these proven features can be transferred or adapted to the new task’s scenario. For example, a feature like “mentions battery life” from a cellphone review task could be adapted to “mentions fabric type” for a clothing review task.
3. Innovate from Principles: Use the general principles from the memories to inspire entirely new features that are tailored specifically for the new task.

The features you propose must be highly relevant to the new task description.

Output exactly {feature_count} candidate features. Each feature must be on a new line.
For each feature, provide only its name and a clear text description of what it measures (around 30 words). Do not use any special symbols or formatting, including list numbers.
""" 
