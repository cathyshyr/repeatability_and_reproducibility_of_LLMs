from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
from multiprocessing import Process, Queue, set_start_method
import time

endpoint = ""
api_key = ""

# Download stop words if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def compute_score_per_run(prompt, model_embed, client, model_name="gpt-4o-mini",
                                    num_runs=100, max_tokens=256, top_k=30, temperature=0.5):
    outputs_per_run = []
    outputs_content_per_run = []
    embeddings_per_run = []
    embeddings_content_per_run = []
    probs_per_position = defaultdict(list)
    token_probs_per_run = []
    token_content_probs_per_run = []
    top_k_probs_per_run = []

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs} ({(run_idx + 1) / num_runs:.0%} complete)", flush=True)
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_k,
            messages=[{"role": "user", "content": prompt}]
        )

        logprobs = response.choices[0].logprobs.content
        decoded_tokens = []
        run_token_probs = []
        run_token_content_probs = []
        run_topk_probs = []

        for i, token_logprob in enumerate(logprobs):
            token = token_logprob.token.strip()
            token_prob = np.exp(token_logprob.logprob)
            decoded_tokens.append(token)
            run_token_probs.append((token, token_prob))

            if token.lower() not in stop_words and token.isalpha():
                run_token_content_probs.append((token, token_prob))

            prob_dist = {
                top.token.strip(): np.exp(top.logprob)
                for top in token_logprob.top_logprobs
            }
            total_prob = sum(prob_dist.values())
            prob_dist = {tok: prob / total_prob for tok, prob in prob_dist.items()}
            probs_per_position[i].append(prob_dist)
            run_topk_probs.append(list(prob_dist.items()))

        full_output = " ".join(decoded_tokens).strip()
        content_output = " ".join([tok for tok, _ in run_token_content_probs])

        outputs_per_run.append(full_output)
        outputs_content_per_run.append(content_output)
        token_probs_per_run.append(run_token_probs)
        token_content_probs_per_run.append(run_token_content_probs)
        top_k_probs_per_run.append(run_topk_probs)
        embeddings_per_run.append(model_embed.encode(full_output, convert_to_numpy=True))
        embeddings_content_per_run.append(model_embed.encode(content_output, convert_to_numpy=True))

    # Entropy computation
    H_r_list = []
    H_r_list_content = []
    entropy_per_position_per_run = []
    entropy_per_position_content_per_run = []

    for r in range(num_runs):
        H_i_list = []
        H_i_content_list = []
        run_entropy_values = []

        for i in range(len(probs_per_position)):
            if r < len(probs_per_position[i]):
                run_probs = probs_per_position[i][r]
            else:
                continue

            entropy = 0.0
            for p in run_probs.values():
                if p > 0:
                    entropy -= p * np.log(p)
            H_i_list.append(entropy)
            run_entropy_values.append(entropy)

            token_str = list(run_probs.keys())[0].lower()
            if token_str not in stop_words and token_str.isalpha():
                H_i_content_list.append(entropy)

        entropy_per_position_per_run.append(run_entropy_values)
        entropy_per_position_content_per_run.append(H_i_content_list)
        H_r_list.append(np.mean(H_i_list))
        if H_i_content_list:
            H_r_list_content.append(np.mean(H_i_content_list))

    H_bar = np.mean(H_r_list)
    H_bar_content = np.mean(H_r_list_content) if H_r_list_content else None

    return {
        # === Semantic ===
        "semantic_repeatability": {
            "embeddings_per_run": embeddings_content_per_run,  # e_r from output (excluding stop words)
        },

        # === Internal ===
        "internal_repeatability": {
            "H_bar_per_run": H_bar_content,  # -1 * average H_r (excluding stop words)
        },

        # === Raw Outputs (Y_{r,1:L_r}) ===
        "outputs": {
            "full": outputs_per_run,  # raw token outputs
            "content": outputs_content_per_run,  # filtered token outputs (excluding stop words)
        }
    }


example_q1 = """Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old
woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient
maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee.
Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her
temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95mm Hg. Pulse oximetry on room air shows an
oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of
10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is
the most likely diagnosis?"""

example_q2 = """A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he
was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of
hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5Â°C (101.3 °F),
pulse is 92/min, respirations are 22/min, and blood pressure is 130/80mm Hg. Cardiac examination shows a high-pitched scratching sound best
heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations.
Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?"""

traditional_cot_prompt = """Provide a step-by-step deduction that identifies the correct response.

Example Question 1: 
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is the most likely diagnosis?

Example Rationale 1: 
The patient had a surgical repair of a displaced femoral neck fracture. The patient has petechiae. The patient has a new oxygen requirement, meaning they are having difficulty with their breathing. This patient most likely has a fat embolism.

Example Question 2: 
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5 °C (101.3 °F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?

Example Rationale 2: 
This patient is having chest pain. They recently had a heart attack and has new chest pain, suggesting he may have a problem with his heart. The EKG has diffuse ST elevations and he has a scratching murmur. This patient likely has Dressler Syndrome.

Question:
{question}

Rationale:"""


ddx_cot_prompt = """Use step by step deduction to create a differential diagnosis and then use step by step deduction to determine the correct response.

Example Question 1: 
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is the most likely diagnosis?

Example Rationale 1: 
This patient has shortness of breath after a long bone surgery. The differential for this patient is pulmonary embolism, fat embolism, myocardial infarction, blood loss, anaphylaxis, or a drug reaction. The patient has petechiae which makes fat embolism more likely. This patient most likely has a fat embolism.

Example Question 2: 
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5 °C (101.3 °F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?

Example Rationale 2: 
This patient has chest pain with diffuse ST elevations after a recent myocardial infarction. The differential for this patient includes: myocardial infarction, pulmonary embolism, pericarditis, Dressler syndrome, aortic dissection, and costochondritis. This patient likely has a high-pitched scratching sound on auscultation associated with pericarditis and Dressler Syndrome. This patient has diffuse ST elevations associated with Dressler Syndrome. This patient most likely has Dressler Syndrome.

Question:
{question}

Rationale:"""

intuitive_reasoning_cot_prompt = """Use symptoms, signs, and laboratory disease associations to step by step deduce the correct response.

Example Question 1: 
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is the most likely diagnosis?

Example Rationale 1: 
This patient has findings of petechiae, altered mental status, shortness of breath, and recent surgery suggesting a diagnosis of fat emboli. The patient most likely has a fat embolism.

Example Question 2: 
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5 °C (101.3 °F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?

Example Rationale 2: 
This patient had a recent myocardial infarction with new development of diffuse ST elevations, chest pain, and a high pitched scratching murmur which are found in Dressler's syndrome. This patient likely has Dressler's syndrome.

Question:
{question}

Rationale:"""

analytic_reasoning_cot_prompt = """Use analytic reasoning to deduce the physiologic or biochemical pathophysiology of the patient and step by step identify the correct response.

Example Question 1: 
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is the most likely diagnosis?

Example Rationale 1: 
The patient recently had large bone surgery making fat emboli a potential cause because the bone marrow was manipulated. Petechiae can form in response to capillary inflammation caused by fat emboli. Fat micro globules cause CNS microcirculation occlusion causing confusion and altered mental status. Fat obstruction in the pulmonary arteries can cause tachycardia and shortness of breath as seen in this patient. This patient most likely has a fat embolism.

Example Question 2: 
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5 °C (101.3 °F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?

Example Rationale 2: 
This patient had a recent myocardial infarction which can cause myocardial inflammation that causes pericarditis and Dressler Syndrome. The diffuse ST elevations and high pitched scratching murmur can be signs of pericardial inflammation as the inflamed pericardium rubs against the pleura as seen with Dressler Syndrome. This patient likely has Dressler Syndrome

Question:
{question}

Rationale:"""

bayesian_reasoning_cot_prompt = """Use step-by-step Bayesian Inference to create a prior probability that is updated with new information in the history to produce a posterior probability and determine the final diagnosis.

Example Question 1: 
Shortly after undergoing a bipolar prosthesis for a displaced femoral neck fracture of the left hip acquired after a fall the day before, an 80-year-old woman suddenly develops dyspnea. The surgery under general anesthesia with sevoflurane was uneventful, lasting 98 min, during which the patient maintained oxygen saturation readings of 100% on 8 l of oxygen. She has a history of hypertension, osteoporosis, and osteoarthritis of her right knee. Her medications include ramipril, naproxen, ranitidine, and a multivitamin. She appears cyanotic, drowsy, and is oriented only to person. Her temperature is 38.6 °C (101.5 °F), pulse is 135/min, respirations are 36/min, and blood pressure is 155/95 mm Hg. Pulse oximetry on room air shows an oxygen saturation of 81%. There are several scattered petechiae on the anterior chest wall. Laboratory studies show a hemoglobin concentration of 10.5 g/dl, a leukocyte count of 9000/mm3, a platelet count of 145,000/mm3, and a creatine kinase of 190 U/l. An ECG shows sinus tachycardia. What is the most likely diagnosis?

Example Rationale 1: 
The prior probability of fat embolism is 0.05% however the patient has petechiae on exam which is seen with fat emboli, which increases the posterior probability of fat embolism to 5%. Altered mental status increases the probability further to 10%. Recent orthopedic surgery increases the probability of fat emboli syndrome to 60%. This patient most likely has a fat embolism.

Example Question 2: 
A 55-year-old man comes to the emergency department because of a dry cough and severe chest pain beginning that morning. Two months ago, he was diagnosed with inferior wall myocardial infarction and was treated with stent implantation of the right coronary artery. He has a history of hypertension and hypercholesterolemia. His medications include aspirin, clopidogrel, atorvastatin, and enalapril. His temperature is 38.5 °C (101.3 °F), pulse is 92/min, respirations are 22/min, and blood pressure is 130/80 mm Hg. Cardiac examination shows a high-pitched scratching sound best heard while sitting upright and during expiration. The remainder of the examination shows no abnormalities. An ECG shows diffuse ST elevations. Serum studies show a troponin I of 0.005 ng/ml (N < 0.01). What is the most likely cause of this patient’s symptoms?

Example Rationale 2: 
The prior probability of Dressler Syndrome is 0.01%. The patient has diffuse ST elevations, increasing the probability of Dressler Syndrome to 5%. The patient has a scratching murmur which increases the probability to 10%. In the setting of a recent MI the posterior probability of myocardial infarction is 55%. This patient likely has Dressler Syndrome.

Question:
{question}

Rationale:"""


MedQA = pd.read_csv("Diagnostic reasoning prompts reveal the potential for large language model interpretability in medicine Supplement 1.csv", encoding='latin1')
MedQA = MedQA.head(2)

# Create list of tasks
prompt_templates = [traditional_cot_prompt, ddx_cot_prompt,
                    intuitive_reasoning_cot_prompt, analytic_reasoning_cot_prompt,
                    bayesian_reasoning_cot_prompt]


def worker_loop(task_queue: Queue, gpu_id: int, endpoint: str, api_key: str, prompt_templates, MedQA):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] Starting worker...", flush=True)

    client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
        api_key=api_key
    )
    # Change this to your embedding model of choice
    model_embed = SentenceTransformer("/home/models/medembed_local")

    while True:
        try:
            prompt_id, question_id = task_queue.get_nowait()
        except Exception:
            break  # Queue is empty

        prompt_template = prompt_templates[prompt_id]
        question_text = MedQA["question"].iloc[question_id]
        prompt = prompt_template.format(question=question_text)

        prompt_id1 = prompt_id + 1
        question_id1 = question_id + 1
        output_path = f"./prompt{prompt_id1}_q{question_id1}.pkl"

        if os.path.exists(output_path):
            print(f"[GPU {gpu_id}] SKIP Prompt {prompt_id1}, question {question_id1}", flush=True)
            continue

        print(f"[GPU {gpu_id}] START Prompt {prompt_id1}, question {question_id1}", flush=True)

        try:
            result = compute_score_per_run(
                prompt=prompt,
                model_embed=model_embed,
                client=client,
                model_name="gpt-4o-mini",
                num_runs=100,
                max_tokens=256,
                top_k=30,
                temperature=0.5
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(result, f)

            print(f"[GPU {gpu_id}] DONE Prompt {prompt_id1}, question {question_id1}", flush=True)

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR Prompt {prompt_id1}, question {question_id1}: {type(e).__name__} - {e}", flush=True)

        time.sleep(0.1)

if __name__ == '__main__':
    from multiprocessing import Queue

    set_start_method('spawn')

    num_gpus = 3
    num_workers_per_gpu = 3  # You can increase this if your GPU has enough VRAM

    # Fill task queue: (prompt_id, question_id)
    task_queues = [Queue() for _ in range(num_gpus)]

    num_questions = len(MedQA)
    num_prompts = len(prompt_templates)

    all_tasks = [
        (p, q)
        for q in range(num_questions)
        for p in range(num_prompts)
    ]

    print(f"Launching {len(all_tasks)} tasks using {num_prompts} prompt types...")

    # Round-robin distribute tasks to GPU queues
    for idx, task in enumerate(all_tasks):
        task_queues[idx % num_gpus].put(task)

    # Launch workers
    processes = []
    for gpu_id, task_queue in enumerate(task_queues):
        for _ in range(num_workers_per_gpu):
            proc = Process(
                target=worker_loop,
                args=(task_queue, gpu_id, endpoint, api_key, prompt_templates, MedQA)
            )
            proc.start()
            processes.append(proc)

    for proc in processes:
        proc.join()

    print("All tasks completed.")