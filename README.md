# VQA-X Task: Question Answering and Explanation Generation

## Task

The goal of this project is to predict answers to questions based on image descriptions and to generate explanations for those answers. 

- **Dataset**: VQA-X, based on **V2_MSCOCO**
- **Data Points**: 32,886

## Captioning

- **Model**: `BLIP-2-FLAN-T5-XXL`
- **Components**:
  - Q-Transformer
  - LLM
  - Image Encoder
- **Image Pre-processing**: LAVIS

### Prompt for Captioning

- **Prompt**: "Give a caption for this image along with details on object specifics"

## Prompting

- **Model**: `FLAN-T5-XXL`

### Task 1: Answer Prediction

- **Inputs**: Questions and Captions
- **Prompt**: “Given the caption, provide an answer for the question provided below.”

### Task 2: Generate Explanations

- **Inputs**: Questions, Captions, and `answers_generated_from_flan`
- **Prompt**: “Based on the following data provided, answer the following question: Why is that the answer? Give an explanation/rationale for the answer provided.”

### Task 3: Generate Explanations Using Ground Truth Answers

- **Inputs**: Questions, Captions, and Ground Truth Answers
- **Prompt**: “Based on the following data provided, answer the following question: Why is that the answer? Give an explanation/rationale for the answer provided.”

## Prompting Results

Metrics used for evaluation: **METEOR** and **ROUGEL**

| Task | METEOR | ROUGEL |
|------|--------|--------|
| Task 1 | 36.57  | 70.66  |
| Task 2 | 21.04  | 21.12  |
| Task 3 | 20.35  | 20.54  |

## Annotation Scheme

Based on the evaluation criteria from:

- Carvalho, Diogo V., Eduardo M. Pereira, and Jaime S. Cardoso. 2019. ["Machine Learning Interpretability: A Survey on Methods and Metrics"](https://doi.org/10.3390/electronics8080832) Electronics 8, no. 8: 832.

### Selected Criteria

- Dropped model-oriented and infeasible criteria
- Seven columns with weighted averages applied to the following:
  - **Truthful**: 0.4
  - **Understandable**: 0.3
  - **Efficiency**: 0.1
  - **Prior Beliefs**: 0.05
  - **Social**: 0.05
  - **Contrastiveness**: 0.05
  - **Focus Abnormal**: 0.05

### Results

- **Overall Annotation Score**: 3.37

## Issues

- Captions often lack sufficient detail.
- The explainer tends to answer "no" or hallucinate when the question cannot be answered from the given caption.
- Explanations relevant for the caption may not always be relevant for the image, and vice versa.
- In some cases, the answer is already implied by the captioning, making it difficult for the text model to provide a suitable explanation.

## Key Learnings

- Annotation criteria are highly application-specific.
- Feasibility must be considered in annotation tasks.
- Criteria can evolve during the annotation process.
- Developing adequate prompts for caption generation is complex and requires fine-tuning.
