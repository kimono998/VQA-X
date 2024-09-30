# Description: This file contains the functions to load the dataset,
# make the final dataframe, select the sample dataset, and save the file

import pandas as pd
def convert_to_string(lst):
    return ' '.join(lst)

#unpacks the files and loads them into a dataframe
def load_dataset(file_path_questions, file_path_answers, file_path_explain):
    df_questions = pd.read_json(file_path_questions, lines=True)
    df_answers = pd.read_json(file_path_answers, lines=True)
    df_explain = pd.read_json(file_path_explain, lines=True)

    df_explain = df_explain.transpose().copy()
    df_explain.reset_index(inplace=True)
    df_explain.rename(columns={"index": "question_id", 0: "explainations"}, inplace=True)
    df_explain["explainations"] = df_explain.apply(lambda row: row["explainations"][0], axis=1)

    return df_questions, df_answers, df_explain

#creates a question dataframe
def make_final_questions_df(df_questions):
    df_question = df_questions['questions'].iloc[0].copy()

    image_id = []
    question = []
    question_id = []
    for i in df_question:
        x = list(i.values())
        image_id.append(x[0])
        question.append(x[1])
        question_id.append(x[2])

    df_question = pd.DataFrame({
        "image_id": image_id,
        "question": question,
        "question_id": question_id
    })

    return df_question

#creates an answers dataframe
def make_final_answers_df(df_answers):
    df_answers = df_answers['annotations'].iloc[0]

    image_id = []
    answer = []
    question_id = []
    question_type = []
    for i in df_answers:
        x = list(i.values())
        image_id.append(x[3])
        question_type.append(x[0])
        answer.append(x[1])
        question_id.append(x[5])

    df_answer = pd.DataFrame({
        "image_id": image_id,
        "question_type": question_type,
        "question_id": question_id,
        "answer": answer
    })

    return df_answer

#merges all relevant dataframes
def make_final_df(df_question, df_answer, df_explain):
    final_df = pd.merge(pd.merge(df_question, df_answer, how="outer", on="question_id"),
                        df_explain, how="outer", on="question_id").dropna(axis=0)
    final_df.drop(columns=["image_id_y"])
    final_df.rename(columns={"image_id_x": "image_id"}, inplace=True)

    return final_df

#selects a sample dataset
def select_sample_dataset(num_samples, final_df):
    random_samples = final_df.sample(n=num_samples, random_state=42)
    random_samples_df = pd.DataFrame(random_samples)

    return random_samples_df

#saves file
def save_file(file_name, df):
    df.to_csv(file_name)

#identifies image paths
def find_image_list(df):
    image_list = []
    for index, rows in df.iterrows():
        string = "data/images/train2014/COCO_train2014_" + "".join([str(0)] * (12 - len(str(rows["image_id"])))) + str(
            rows["image_id"]) + ".jpg"
        #Image_data = Image.open(string).convert("RGB")
        #image_list.append(Image_data)
        image_list.append(string)

    return image_list
