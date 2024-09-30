import argparse
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import metric_computation

from models import load_blip_2, generate_captions, load_flan_model, find_flan_output_answers, \
    find_flan_output_explainations_gen
from utils import load_dataset, make_final_questions_df, make_final_answers_df, make_final_df, select_sample_dataset, \
    save_file, convert_to_string


def main():
    parser = argparse.ArgumentParser(
        description='VQA-X Flan and Blip'
    )

    parser.add_argument(
        "--prepare_data", dest="prepare_data",
        help="Prepares the dataset",
        action="store_true",
        default=None
    )

    parser.add_argument(
        "--generate_captions", dest="generate_captions",
        help="uses blip2 model to generate captions for the images",
        action="store_true",
        default=None
    )

    parser.add_argument(
        "--get_outputs", dest="get_outputs",
        help="uses flan model to generate answers and explanations for the images",
        action="store_true",
        default=None
    )
    parser.add_argument(
        "--eval_answer", dest="eval_answer",
        help="evaluates the answer prediction model",
        action="store_true",
        default=None

    )

    parser.add_argument(
        "--eval_exp", dest="eval_exp",
        help="evaluates the explanation prediction model",
        action="store_true",
        default=None

    )

    parser.add_argument(
        "--eval_gen", dest="eval_gen",
        help="evaluates the explanation prediction model based on the generated answers",
        action="store_true",
        default=None

    )


    args = parser.parse_args()

    if args.prepare_data:
        #number of samples for evaluation
        num_samples = 100

        print('Preparing the dataset!')

        #paths to the dataset
        file_path_questions = "data/v2_OpenEnded_mscoco_train2014_questions.json"
        file_path_answers = "data/v2_mscoco_train2014_annotations.json"
        file_path_explain = "data/train_exp_anno.json"

        #load the dataset
        df_questions, df_answers, df_explain = load_dataset(file_path_questions, file_path_answers, file_path_explain)

        #make the final dataframe
        df_question = make_final_questions_df(df_questions)
        df_answer = make_final_answers_df(df_answers)
        final_df = make_final_df(df_question, df_answer, df_explain)
        #select the sample dataset
        df = select_sample_dataset(num_samples, final_df)
        #save the file
        save_file('data/final_dataset.csv', df)

    if args.generate_captions:

        print('Generating captions!')
        df = pd.read_csv("data/final_dataset.csv")

        #loads the model and the vis_processors
        model_blip, vis_processors = load_blip_2(name="blip2_t5", model_type="caption_coco_flant5xl")
        #generates the captions
        df = generate_captions(model_blip, vis_processors, df)
        df['captions_BLIP'] = df['captions_BLIP'].apply(convert_to_string)
        df.to_csv("data/final_dataset.csv")

    if args.get_outputs:

        df = pd.read_csv("data/final_dataset.csv")
        df.drop(columns=["image", "image_id_y"], inplace=True)
        #loads the model and the tokenizer, generates the answers and explanations
        tokenizer, model_flan = load_flan_model()
        df = find_flan_output_answers(model_flan, tokenizer, df)
        df = find_flan_output_explainations_gen(model_flan, tokenizer, df)
        df.to_csv("data/final_output.csv")

    if args.eval_answer:

        print('Evaluation in progress!')
        df= pd.read_csv("data/final_output.csv")

        #meteor and rouge scores for the answer prediction model
        meteor = []
        rouge = {}

        references = list(df["answer"])
        prediction = list(df["answer_flan"])

        meteor_score = metric_computation(prediction, references, metric='meteor')["meteor"]
        rouge_score = list(metric_computation(prediction, references, metric='rouge').values())


        meteor.append(meteor_score)
        rouge["answer_flan"] = rouge_score

        print("The METEOR and ROUGE scores for the answer prediction model")
        print("METEOR score:", meteor_score)
        print("ROUGE score:", rouge_score)





    if args.eval_exp:
        #meteor and rouge scores for the explanation generation model based on true answers

        df= pd.read_csv("data/final_output.csv")
        meteor = []
        rouge = {}
        references = list(df["explainations"])
        prediction = list(df["explain_flan"])

        meteor_score = metric_computation(prediction, references, metric='meteor')["meteor"]
        rouge_score = list(metric_computation(prediction, references, metric='rouge').values())

        meteor.append(meteor_score)
        rouge["explain_flan"] = rouge_score

        print("""The METEOR and ROUGE scores for the explanation prediction model 
        where ground truth answers where used to generate explnanation.""")
        print("METEOR score:", meteor_score)
        print("ROUGE score:", rouge_score)


    if args.eval_gen:
        #meteor and rouge scores for the explanation generation model based on generated answers
        df= pd.read_csv("data/final_output.csv")

        meteor = []
        rouge = {}

        references = list(df["explainations"])
        prediction = list(df["explain_flan_gen"])

        meteor_score = metric_computation(prediction, references, metric='meteor')["meteor"]
        rouge_score = list(metric_computation(prediction, references, metric='rouge').values())

        meteor.append(meteor_score)
        rouge["explain_flan_gen"] = rouge_score

        print("""The METEOR and ROUGE scores for the explanation prediction model
        where answers from flan_xxl model where used to generate explnanation""")
        print("METEOR score:", meteor_score)
        print("ROUGE score:", rouge_score)

if __name__ == "__main__":
    main()