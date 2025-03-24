# %% Import Libs
from huggingface_hub import login
#from google.colab import userdata
#HF_TOKEN = userdata.get('HuggingFace')
#login(HF_TOKEN)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BertTokenizer, BertForQuestionAnswering, logging
from sentence_transformers import SentenceTransformer
import os
#os.chdir('/home/hlee981/LAM-LLM/')

import re
import torch
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
import json
import pickle
import lam_adapt
from difflib import get_close_matches
import warnings
logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)
# %% Set up
"""
Extract a substring before a user indicated marker in a full string
"""
def extract_before_substring(full_string, marker):
    return full_string.split(marker)[0]

def extract_after(full_string, marker):
    return full_string.split(marker)[1]

def extract_between(full_string, start_marker, end_marker):
    pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
    match = re.search(pattern, full_string)
    return match.group(1) if match else None


"""
Set up retriever
"""
class aspire_retriever:
    """ """
    def __init__(self, git_token, show_debug_logs=False, load_save=None,):
        # Debug?
        self.show_debug_logs = show_debug_logs
        self.git_token = git_token
        # Read and create context snippets from ASPIRE database
        if self.show_debug_logs:
            print('[DEBUG] Reading all csv files from ASPIRE database...' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        if load_save is None:
            self.data_csv = [] # initialize data_csv
            self.get_cp_files_from_aspire() # fill data_csv
            self.data_contexts, self.file_list, self.airfoil_contexts = self.create_context_snippets(self.data_csv)
            self.save_contexts_to_pickle('temp_model.pkl');
        else:
            self.load_contexts_from_pickle(load_save)
        self.airfoil_name_list = list(dict.fromkeys([entry["airfoil"].lower() for entry in self.data_csv]))
        self.lambert = self.load_bert_pipeline()
        # self.af_score_thresh = 5
        self.data_score_thresh = 0.5

    """ Load in BERT pipeline """
    def load_bert_pipeline(self):
        # Load LAMBERT pipeline using pretrained weights
        print('[DEBUG] Loading in LAMBERT pipeline...' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        # Utilize GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"[DEBUG] LAMBERT is using device: {device}" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        return qa_pipeline

    def extract_query_parameters(self, query, qa_pipeline):
        #background context for the model to use when answering questions
        structured_context = f"""
        The user wants to extract aerodynamic data.
        The airfoil name, angle of attack, Mach number, and Reynolds number are mentioned in the query.
        Extract these values from the following user request: {query}
        """

        #questions for the model to find the various values in the user's prompt
        parameters = {
            "Airfoil": "What is the airfoil name?",
            "Angle of attack": "What is the angle of attack in degrees, allowing for optional negative degrees?",
            "Mach": "What is the Mach number? It may be written as m or mach.",
            "Reynolds": "What is the Reynolds number or re? It may be written as reynolds or re."
        }

        extracted_parameters = {}

        for param, question in parameters.items():
            try:
                #uses BERT to extract the data from the query
                response = qa_pipeline(question=question, context=structured_context)
                extracted_value = response["answer"]

                print(f"[DEBUG] BERT raw output for {param}: {extracted_value}" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')

                #remove any commas from the extracted data
                extracted_value = extracted_value.replace(",", "").strip()

                if param == "Airfoil":
                    extracted_parameters[param] = extracted_value
                    if "Mach" in extracted_value.lower():
                        extracted_parameters[param] = None
                    if "Reynolds" in extracted_value.lower():
                        extracted_parameters[param] = None
                    if "Angle of attack" in extracted_value.lower() or "aoa" in extracted_value.lower() or "degrees" in extracted_value.lower():
                        extracted_parameters[param] = None
                else:
                    #extract only the numerical values
                    if any(char.isdigit() for char in extracted_value) and extracted_parameters["Airfoil"] not in extracted_value:
                        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", extracted_value)
                        extracted_parameters[param] = float(match.group(0)) if match else None
                    else:
                        extracted_parameters[param] = None

            except Exception as e:
                #handle errors if unable to extract value
                print(f"Error extracting {param}: {e}" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
                extracted_parameters[param] = None
        return extracted_parameters

    def extract_metadata_from_filename(self, filename):
        match = re.search(r'A(m?[-+]?\d*\.\d+|\d+)_M([-+]?\d*\.\d+|\d+)_Re((?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)', filename, re.IGNORECASE)

        if match:
            aoa_value = match.group(1)
            if aoa_value.startswith("m"):
                aoa_value = -float(aoa_value[1:])
            else:
                aoa_value = float(aoa_value)

            extracted_data = {
                "angle_of_attack": aoa_value,
                "mach_number": float(match.group(2)),
                "reynolds_number": float(match.group(3)),
                "filename": filename
            }
            # print(f"Extracted from {filename}: {extracted_data}")
            return extracted_data

        print(f"No match found for {filename}" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        return None

    def retrieve_context(self, query, angle_threshold, mach_threshold, re_threshold):
        # retrieve the extracted query parameters
        query_params = self.extract_query_parameters(query, self.lambert)

        # find appropriate list using airfoil match index
        if not query_params["Airfoil"]:
            return ''

        af_match_idx = self.best_match(query_params["Airfoil"].lower(), self.airfoil_name_list)
        if af_match_idx is not None:
            retrieved_airfoil = self.airfoil_contexts[af_match_idx] # context for airfoil
            file_list = self.file_list[af_match_idx] # list of files
        else:
            return 'There is no matching airfoil available in the database. You cannot retrieve the airfoil coordinates from the database. Maybe the attachment has information you can use.' # return empty string for no context

        best_score = float('inf')
        metadata_list = [self.extract_metadata_from_filename(file) for file in file_list if self.extract_metadata_from_filename(file)]
        best_match_idx = None
        ct = 0
        num_none = 0

        if query_params["Angle of attack"] is None:
            num_none += 1
        if query_params["Mach"] is None:
            num_none += 1
        if query_params["Reynolds"] is None:
            num_none += 1
            query_params["Reynolds"] = 1e6 # place holder value

        if num_none <= 1:
            for metadata in metadata_list: # DO REYNOLDS AND FIX
                angle_score = abs(metadata["angle_of_attack"] - query_params["Angle of attack"])
                mach_score = abs(metadata["mach_number"] - query_params["Mach"])
                re_score = abs(metadata["reynolds_number"] - query_params["Reynolds"])

                if angle_score < angle_threshold and mach_score < mach_threshold : # and re_score < re_threshold
                    score = angle_score + mach_score + re_score
                    if score < best_score:
                        best_score = score
                        best_match_idx = ct
                ct += 1
        else:
            best_match_idx = None

        print(query_params)
        if best_match_idx is not None:
            return retrieved_airfoil + self.data_contexts[af_match_idx][best_match_idx]
        elif "predict" not in query.lower() and ((query_params["Angle of attack"] is not None) or (query_params["Mach"] is not None)):
            change_thresholds = input("No relevant pressure distribution file was found in the database. \n The current thresholds are " + str(angle_threshold) + " degrees for angle of attack, " + str(mach_threshold) + " for mach number, " + str(re_threshold) + " for reynold's number. Would you like to change these? (y/n)")
            if (change_thresholds.lower() == "y"):
                angle_threshold_1 = (input("New angle of attack threshold: "))
                try:
                    angle_threshold_new = float(angle_threshold_1)
                except ValueError:
                    angle_threshold_new = angle_threshold
                mach_threshold_1 = (input("New mach number threshold: "))
                try:
                    mach_threshold_new = float(mach_threshold_1)
                except ValueError:
                    mach_threshold_new = mach_threshold
                re_threshold_1 = (input("New reynolds number threshold: "))
                try:
                    re_threshold_new = float(re_threshold_1)
                except ValueError:
                    re_threshold_new = re_threshold
                return self.retrieve_context(query, angle_threshold_new, mach_threshold_new, re_threshold_new)
            else:
                return retrieved_airfoil + ' No relevant pressure distribution file was found in the database. You should instead predict the results using the ADAPT module.'
        else:
            return retrieved_airfoil + ' No relevant pressure distribution file was found in the database. You should instead predict the results using the ADAPT module.'

    def best_match(self, target, choices):
        matches = get_close_matches(target, choices, n=1, cutoff=0.9)
        return choices.index(matches[0]) if matches else None

    def load_contexts_from_pickle(self, save_name):
        with open(save_name, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.data_csv = loaded_dict['data_csv']
        self.data_contexts = loaded_dict['data_contexts']
        self.airfoil_contexts = loaded_dict['airfoil_contexts']
        self.file_list = loaded_dict['file_list']

    def save_contexts_to_pickle(self, save_name):
        save_dict = {
            'data_csv': self.data_csv,
            'data_contexts': self.data_contexts,
            'airfoil_contexts': self.airfoil_contexts,
            'file_list': self.file_list,
        }

        with open(save_name, 'wb') as f:
            pickle.dump(save_dict, f)


    """
    Process Cp files from ASPIRE
    """
    def get_cp_files_from_aspire(self, api_url="https://api.github.com/repos/hwlee924/Large-Airfoil-Model/contents/ASPIRE/Airfoils"):
        headers = {"Authorization": f"token {self.git_token}"}
        response = requests.get(api_url, headers=headers)
        # Check if the request fails
        if response.status_code != 200:
            raise ValueError(f"Error fetching content from {api_url}")

        files = response.json()
        # csv_files = []

        for file in files:
            #
            if (
                file["name"].endswith(".csv")
                and "Re" in file["name"]
                and "M" in file["name"]
            ):
                case_data = self.extract_file_information(file["name"])
                self.data_csv.append(
                    {
                        # meta data
                        "name": file["name"],
                        "path": file["path"],
                        "download_url": file["download_url"],
                        "nav_url": file["html_url"],
                        # case data
                        "airfoil": case_data[0].replace(" ", ""), # remove space
                        "alpha": case_data[1],
                        "mach": case_data[2],
                        "reynolds": case_data[3],
                        # uncertainty data
                        # add later!
                        # tag data
                    }
                )
            elif file["type"] == "dir":  # Recursively search in subdirectories
                self.get_cp_files_from_aspire(file["url"])

    def extract_file_information(self, file_name):
        airfoil_name = extract_before_substring(file_name, "_A")
        match = re.search(
            r"A(m?[-+]?\d*\.?\d+|\d+)_M([-+]?\d*\.\d+|\d+)_Re((?:\d+\.\d+|\d+)(?:e[+-]?\d+)?)",
            file_name,
            re.IGNORECASE,
        )
        if match:
            aoa_value = match.group(1)
            if aoa_value.startswith("m"):
                # adds a negative sign to angle of attack values starting with 'm'
                aoa_value = -float(aoa_value[1:])
            else:
                aoa_value = float(aoa_value)
            mach_value = float(match.group(2))
            reynolds_value = float(match.group(3))
            return [airfoil_name, aoa_value, mach_value, reynolds_value]
        else:
            print('Issue extracting file info')
            return None

# MOVE THE TAG DATA STUFF
    """ Generate snippets based on the database for RAG """
    def create_context_snippets(self, csv_list):
        counter = 0

        if self.show_debug_logs:
            print('[DEBUG] Converting all csv files to context snippets for RAG...')
        file_list, file_list_ = [], []
        cp_snippets = [] # for pressure distribution files
        cp_snippets_ = [] # per airfoil
        airfoil_snippets = [] # for airfoils
        prev_airfoil = '11PercentThicknessSupercriticalAirfoil' # for tracking

        for entry in csv_list:
            base_url = 'https://raw.githubusercontent.com/hwlee924/Large-Airfoil-Model/main/ASPIRE/Airfoils/'
            # This is regarding the airfoil
            file_str = f"{entry['name']} describes the Cp, or pressure coefficient, distribution of the {entry['airfoil']} airfoil. "
            airfoil_str = f"The following is information about the {entry['airfoil']} airfoil. "
            geom_str_1 = f"Your functions should retrieve the airfoil geometry (or shape) file with the download url {extract_before_substring(entry['download_url'], '_A') + '_coordinates.csv'}. "
            geom_str_2 = f"The {entry['airfoil']} airfoil's geometry can be accessed by the user through the html of {extract_before_substring(entry['nav_url'], '_A') + '_coordinates.csv'}. "

            # This is regarding the operating conditions
            alph_str = f"The {entry['airfoil']} airfoil is at the angle of attack, or alpha, of {entry['alpha']}, degrees. "
            mach_str = f"The {entry['airfoil']} airfoil is at a Mach number, or M, of {entry['mach']}, "
            reyn_str = (
                f"The {entry['airfoil']} airfoil is at a Reynolds number, or Re, of {entry['reynolds']}. "
            )
            dl_str = f"Your functions should retrieve the pressure distribution file with the download url {entry['download_url']}. "
            nav_str = f"The pressure distribution can be accessed by the user through the html of {entry['nav_url']}. "

            # This is regarding the tag information
            tag_locs = entry['download_url'].split('/')#extract_between(entry['download_url'], base_url, '/')
            tag_loc = "/".join(tag_locs[:-1])
            tag_info = self.read_tag_file(tag_loc + '/tags.json')

            camber_str = f"The {entry['airfoil']} airfoil is {tag_info['camber']}. "
            super_str = f"The {entry['airfoil']} airfoil is {tag_info['super']}. "
            usage_str = f"The {entry['airfoil']} airfoil is used for {tag_info['apply']} applications. "
            # This is regarding Cp files
            cp_snippets__ = file_str + alph_str + mach_str + reyn_str + dl_str + nav_str + camber_str + super_str + usage_str
            airfoil_snippets_ = airfoil_str + geom_str_1 + geom_str_2 + camber_str + super_str + usage_str

            if counter == 0:
                airfoil_snippets.append(airfoil_snippets_)

            if prev_airfoil == entry['airfoil']:
                cp_snippets_.append(cp_snippets__)
                file_list_.append(entry['name'])
            else:  # need to submit iter 1
                cp_snippets.append(cp_snippets_)
                file_list.append(file_list_)
                airfoil_snippets.append(airfoil_snippets_)
                cp_snippets_, file_list_ = [], []
                cp_snippets_.append(cp_snippets__)
                file_list_.append(entry['name'])
            prev_airfoil = entry['airfoil']
            counter += 1
        cp_snippets.append(cp_snippets_)
        return cp_snippets, file_list, airfoil_snippets

    def read_tag_file(self, tag_file_url):
        response = requests.get(tag_file_url)
        if response.status_code != 200:
            # print("Error downloading file.")
            tag_info = {
            "camber": 'unknown if cambered due to missing tag file',
            "super": 'unknown if supercritical due to missing tag file',
            "apply":  'unknown application due to missing tag file',
            }
            return tag_info
        tag = json.loads(response.text)

        # camber info
        if tag['airfoil']['camber'] == 'Y':
            camber_str = 'cambered'
        else:
            camber_str = 'symmetric'

        # supercriticality info
        if tag['airfoil']['supercritical'] == 'Y':
            supercritical_str = 'supercritical airfoil'
        else:
            supercritical_str = 'not supercritical airfoil'

        tag_info = {
            "camber": camber_str,
            "super": supercritical_str,
            "apply":  tag['airfoil']['application'],
        }
        return tag_info

class rag_model:
    def __init__(self, git_token, show_debug_logs=False, context_save_file=None, use_gpu=True):
        # Show debugging logs if true
        self.show_debug_logs = show_debug_logs
        self.use_gpu = use_gpu
        # Load pretrained Llama model
        if self.show_debug_logs:
            print('[DEBUG] Initializing Llama model...')
        checkpoint = "meta-llama/Llama-3.2-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token_id = tokenizer.eos_token_id  # to suppress warning
        llama_model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
        self.generator = pipeline("text-generation", model=llama_model, tokenizer=tokenizer, device=0 if use_gpu else -1)

        # initialize model instructions
        self.init_function_definitions()
        self.init_model_instructions() # this calls the BASE instructions

        # initialize session memory
        if self.show_debug_logs:
            print('[DEBUG] Initializing session memory...' if show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        self.memory = session_memory_tracker()

        # call data retriever for RAG
        print('[DEBUG] Initializing ASPIRE retriever for RAG...' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        self.retriever = aspire_retriever(git_token, show_debug_logs=self.show_debug_logs, load_save=context_save_file)

        # initialize ADAPT prediction module
        print('[DEBUG] Initializing ADAPT prediction module...' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        self.predictor = adapt_predictor(show_debug_logs=self.show_debug_logs, use_gpu=self.use_gpu)

        # show complete message
        print('[DEBUG] Model initialized!' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')

    def ask_query(self, query, attachment=None):
        if self.show_debug_logs:
            print(textwrap.fill(f"[DEBUG] Query: {query}", width=90))
            # print("\n" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')

        if self.memory.memory_snippets is not None: # if this is not the first question
            # retrieve memory and include in the instructions
            retrieved_memory = self.memory.retrieve_recent_memory()
            # query = retrieved_memory + query
            # print(query)
            self.append_to_instructions(retrieved_memory)

        retrieved_context = self.retriever.retrieve_context(query, 0.1, 0.01, 100000) # get context for RAG
        self.append_to_instructions(f"When answering the user inquiries, use ONLY the context provided by the following sentences. {retrieved_context}") # add context to underlying instruction for model
        if attachment:
            attachment_str = f" This is the airfoil coordinates as an attachment: {attachment}."
            # self.append_to_instructions(attachment_str)
            query += attachment_str
            print('[DEBUG]' + attachment_str if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        # assemble message
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query},
        ]

        # Generate a response using the text generation pipeline
        response = self.generator(messages, temperature=0.1, max_new_tokens=128 * 3)[-1]["generated_text"][-1]["content"]
        responses = {
            'PLACEHOLDER': ['hiya']
        }

        # Debugging outputs
        if self.show_debug_logs:
            print(textwrap.fill(f"[DEBUG] Context: {retrieved_context}", width=90))
            # print("\n" )
            if self.memory.memory_snippets is not None:
                print(textwrap.fill(f"[DEBUG] Memory: {retrieved_memory}", width=90))
                # print("\n")
            print(textwrap.fill(f"[DEBUG] Uncut Answer: {response}", width=90))
            # print("\n")

        # Parse
        if "[" in response and "]" in response:
            response = self.parse_function_call(response)

        print(textwrap.fill(f"Answer: {response}", width=90), flush=True)
        print('\n')

        # Retain memory and clear instructions
        self.memory.add_entry(query, response, retrieved_context)
        self.init_model_instructions()

    def init_function_definitions(self):
        self.function_definitions = """[
        {
            "name": "retrieve_and_plot_cp_from_csv",
            "description": "Visualizes the pressure distribution over an airfoil by reading in an existing file from the database and plotting it. This should only be invoked if it is evident that the airfoil coordinate files and the pressure distribution csv file exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cp_file_url": {
                        "type": "string",
                        "description": "The pressure distribution file download url defined that is only used by the function. This should be a real url or a real file directory provided by the user."
                    },
                },
                "required": ["cp_file_url"]
            }
        },
        {
            "name": "retrieve_and_plot_geometry_from_csv",
            "description": "Visualizes the airfoil geometry or shape by reading in an existing coordinates file from the database and plotting it. This should be only invoked if the user asks for the airfoil geometry and the airfoil coordinate files exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coordinates_file_url": {
                        "type": "string",
                        "description": "The airfoil coordinates/geometry/shape file defined that is only used by the function. This should be a real url or a real file directory provided by the user."
                    },
                },
                "required": ["coordinates_file_url"]
            }
        },
        {
            "name": "predict_using_adapt",
            "description": "Predicts the pressure distribution using the ADAPT module if there is a matching coordinate file and/or pressure data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "float",
                        "description": The angle of attack for the predicted cp distribution
                    },
                    "mach": {
                        "type": "float",
                        "description": The Mach number for the predicted cp distribution
                    },
                    "coordinates_file_url": {
                        "type": "string",
                        "description": "The airfoil coordinates/geometry/shape file download url defined that is only used by the function."
                    },
                },
                "required": ["alpha", "mach", "coordinates_file_url"]
            }
        }
        ]"""

    def init_model_instructions(self):
        self.instructions = f"""You are an AI-driven airfoil aerodynamics analysis digital assistant, named Large Airfoil Model Digital Assistant (LAMDA).
                You have access to a set of tasks. Your job is to identify the user-defined task and act accordingly. You also have the capacity to make function/tool calls when necessary.
                You should identify all of the tasks within the user query and invoke the appropriate functions if necessary.
                Here is a list of functions in JSON format that you can invoke: {self.function_definitions}

                Task 1: If the user asks for the pressure distribution of an airfoil under different operating conditions, retrieve the appropriate file name within the ASPIRE database and plot it only if the file exists.
                Task 2: If there is a matching airfoil, but the pressure distribution file does not exist, generate a prediction of it using the ADAPT module.
                Task 3: If there is no matching airfoil, but user provided the coordinates attachment, and the user asks for a prediction of the pressure distribution of an airfoil under different operating condtions, generate a prediction using the ADAPT module.
                Task 4: If the user asks for the geometry of an airfoil (what it looks like), retrieve the appropriate coordinates file within the ASPIRE database or attachment and plot it.
                Note that if the user directly asks for the link, you will output the html. However, if you are to invoke functions, you should use the download url but only utilize the context in retrieving the url.

                If there is no matching airfoil and no attached coordinates file, then your tasks cannot be performed.

                If you decide to invoke any of the function(s), you MUST put it in the format like this within brackets [function1(params_name1=params_value1, params_name2=params_value2...), function2(params)].
                ONLY use URLs and HTML links when invoking a function within the brackets. Link should not be located outside these brackets.
                Please explain the outputs you generate by explaning what it is and what file you obtained the information but don't explicitly mention evoking a function.
                Please speak like you are a kind assistant.
                If you invoke a function, make sure the output brackets that includes ALL the invoked function are the very LAST sentence.

                Only provide the answer to the user questions, or invoke a function to the following query per the task workflow defined above.
                Answer in one or two sentences and do not be repetitive in your response.
                Prioritize the existing attachment over the database files.
                """
#                 However, if there is not matching airfoil file, Do not invoke any functions and let the user know.z
    """
    Append new strings to the existing instructions for the model
    new_text: string
    """
    def append_to_instructions(self, new_text):
        self.instructions = self.instructions + new_text # predict_using_adapt

    """
    Parses the function call string and dynamically executes the function if allowed.
    """
    def parse_function_call(self, raw_response):
        function_calls = (
                    raw_response.strip().split("[")[1].split("]")[0]
                )  # get the function call within []
        function_calls_list = re.findall(r'\w+\(.*?\)', function_calls) #[item.strip() for item in function_calls.split('),')]
        try:
            for function_calls in function_calls_list:
                print(f'[DEBUG] Running internal function {function_calls}' if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
                eval('self.' + function_calls)
        except Exception as e:
            print(f"[DEBUG] Error in parsing function: {e}" if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
        # output everything except the function call.
        return raw_response.split("[")[0]

    """
    Visualizes the pressure distribution over an airfoil by reading in an existing file from the database and plotting it
    """
    def retrieve_and_plot_cp_from_csv(self, cp_file_url):
        # Credit to AC's LAMBERT
        # Retrieve file from the file ULR
        response = requests.get(cp_file_url)
        if response.status_code != 200:
            print("[DEBUG] Error downloading file." if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
            return

        # obtain data
        data = response.text.split("\n")  # split the file by lines
        data = data[1:]  # skips the first line
        # removes commas from file
        data = [line.replace(",", " ") for line in data if line.strip()]
        df = pd.DataFrame(
            [line.split() for line in data], dtype=float
        )  # creates a data frame object

        df = df.iloc[:, :2]  # only uses first two columns

        # plots the data
        plt.figure(figsize=(8, 6))
        plt.plot(
            df.iloc[:, 0],
            df.iloc[:, 1],
            "ko",
            linestyle="None",
            markersize=8,
            label="Measured $C_p$",
        )
        plt.xlabel("Chordwise location, $x/c$", fontsize=20)
        plt.ylabel("Pressure Coefficient, $C_p$", fontsize=20)
        plt.gca().tick_params(axis="both", labelsize=16)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.show()

    """
    Visualizes the airfoil geometry by reading in an existing coordinates from the database and plotting it
    """
    def retrieve_and_plot_geometry_from_csv(self, coordinates_file_url):
        # Retrieve file from the file URL or directory
        if os.path.isfile(coordinates_file_url): # utilizing attachment file if it exists
            df = pd.read_csv(coordinates_file_url, header=None)
        else: # directory option
            response = requests.get(coordinates_file_url)
            if response.status_code != 200:
                print("[DEBUG] Error downloading file." if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
                return

            # obtain data
            data = response.text.split("\n")  # split the file by lines
            data = data[1:]  # skips the first line
            # removes commas from file
            data = [line.replace(",", " ") for line in data if line.strip()]
            df = pd.DataFrame(
                [line.split() for line in data], dtype=float
            )  # creates a data frame object

            df = df.iloc[:, :2]  # only uses first two columns

        # plots the data
        plt.figure(figsize=(8, 6))
        plt.plot(
            np.hstack((df.iloc[:, 0], df.iloc[0, 0])),
            np.hstack((df.iloc[:, 1], df.iloc[0, 1])),
            "ko-",
            markersize=4,
        )
        plt.xlabel("$x/c$", fontsize=20)
        plt.ylabel("$z/c$", fontsize=20)
        plt.gca().tick_params(axis="both", labelsize=16)
        plt.gca().set_aspect(2)
        plt.grid()
        plt.show()

    """
    Predicts
    """
    def predict_using_adapt(self, alpha, mach, coordinates_file_url):
        # Download CSV coordinate file file
        if os.path.isfile(coordinates_file_url): # utilizing attachment file if it exists
            airfoil_nparray = pd.read_csv(coordinates_file_url, header=None).values
        else:
            response = requests.get(coordinates_file_url)
            if response.status_code != 200:
                print("[DEBUG] Error downloading file." if self.show_debug_logs else '', end='' if not self.show_debug_logs else '\n')
                return

            # obtain data
            data = response.text.split("\n")  # split the file by lines
            # removes commas from file
            data = [line.replace(",", " ") for line in data if line.strip()]
            df = pd.DataFrame(
                [line.split() for line in data], dtype=float
            )  # creates a data frame object
            airfoil_nparray = df.iloc[:, :2].values  # only uses first two columns

        airfoil_input = self.predictor.create_input(airfoil_nparray, alpha, mach) # create input tensor from LLM input
        predictions = self.predictor.predict(airfoil_input) # Generate prediction using ADAPT module
        self.predictor.plot_prediction(predictions)  # Plot above prediciton

    def reset_memory(self):
        self.memory = session_memory_tracker()

class session_memory_tracker:
    def __init__(self):
        self.memory = {"query": [], "response": [], "context": []}
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # embedding model
        self.memory_snippets = None # initialize as none
        self.memory_embeddings = None

    def add_entry(self, query, response, context):
        # update memory entries
        self.memory["query"].append(query)
        self.memory["response"].append(response)
        self.memory["context"].append(context)

        # update snippets
        self.create_snippets_from_memory() # create updated snippets
        self.memory_embeddings = self.model.encode(self.memory_snippets) # create updated embeddings from snippets

    def create_snippets_from_memory(self):
        final_str = []
        for i in range(0, len(self.memory["query"])):
            query_str_ = "The user queried: " + self.memory['query'][i] + ' '
            context_str_ = "The following context was retrieved to help answer the question: " + self.memory['context'][i]+ ' '
            response_str_ = "This was your response: " + self.memory['response'][i] + ' '
            final_str_ = query_str_ + context_str_ + response_str_
            final_str.append(final_str_)
        self.memory_snippets = final_str

    def retrieve_relevant_memory(self, query):     # dont use this for
        query_embedded = self.model.encode([query])  # Encode the query to obtain its embedding
        similarities = self.model.similarity(self.memory_embeddings, query_embedded)  # Calculate cosine similarities between the query embedding and the snippet embeddings
        retrieved_memory = self.memory_snippets[similarities.argmax().item()] # Retrieve the text snippet with the highest similarity
        return retrieved_memory

    def retrieve_recent_memory(self, num_entries=3):
        retrieval_num = np.min([len(self.memory["query"]), num_entries])
        retrieved_memory = ''
        for i in range(-retrieval_num, -1+1):
            retrieved_memory += self.memory_snippets[i]
        intro_str = "The following sentences refer to what was discussed before this query. You can choose to use these facts or not."
        return intro_str + retrieved_memory


"""
Wrapper class for lam_adapt
"""
class adapt_predictor():
    def __init__(self, show_debug_logs=False, use_gpu=True):
        self.use_gpu = use_gpu # gpu usage, inherit from main class
        self.show_debug_logs = show_debug_logs
        self.adapt_model, _ = lam_adapt.unpack_model() # unpack the model
        self.num_points_per_surface = 120 # default number of points per surface


    """
    Create input tensor based on LLM input
    """
    def create_input(self, airfoil_nparray, angle_of_attack, mach_number):
        airfoil_input = lam_adapt.input_data(airfoil_nparray, angle_of_attack, mach_number, num_auto_points=self.num_points_per_surface, use_gpu=self.use_gpu)
        return airfoil_input

    """
    Generate prediction given the input
    """
    def predict(self, airfoil_input):
        predictions = self.adapt_model.predict(airfoil_input, get_coeff=True)
        return predictions

    """
    Plot the prediction from adapt module
    """
    def plot_prediction(self, pred_dict):
        # Extract info from dict
        prediction_mean = pred_dict['cp_distribution'].mean.cpu().detach().numpy()
        prediction_sig = np.sqrt(np.diag(pred_dict['cp_distribution'].covariance_matrix.cpu().detach().numpy()))
        pred_xc = pred_dict['xc'].cpu().detach().numpy()

        # Plot below
        plt.figure(figsize=(8, 6))
        # Upper surface, mean
        plt.plot(
            pred_xc[:self.num_points_per_surface],
            prediction_mean[:self.num_points_per_surface],
            "r",
            linestyle="-",
            linewidth=3,
            label="Predicted $C_p$",
        )
        plt.fill_between(
            pred_xc[:self.num_points_per_surface],
            prediction_mean[:self.num_points_per_surface] - 2 * prediction_sig[:self.num_points_per_surface],
            prediction_mean[:self.num_points_per_surface] + 2 * prediction_sig[:self.num_points_per_surface],
            color="lightgray",
            label="Predicted 2$sigma$",
        )
        # Lower surface, mean
        plt.plot(
            pred_xc[self.num_points_per_surface:],
            prediction_mean[self.num_points_per_surface:],
            "r",
            linestyle="--",
            linewidth=3,
        )
        plt.fill_between(
            pred_xc[self.num_points_per_surface:],
            prediction_mean[self.num_points_per_surface:] - 2 * prediction_sig[self.num_points_per_surface:],
            prediction_mean[self.num_points_per_surface:] + 2 * prediction_sig[self.num_points_per_surface:],
            color="lightgray",
        )
        plt.xlabel("Chordwise location, $x/c$", fontsize=20)
        plt.ylabel("Pressure Coefficient, $C_p$", fontsize=20)
        plt.legend(fontsize=12)
        plt.gca().tick_params(axis="both", labelsize=16)
        plt.gca().invert_yaxis()
        plt.grid()
        plt.show()

#%% Run the model
if __name__ == "__main__":
    #token_str = userdata.get("github")
    use_gpu = True
    if use_gpu:  # 6 is for personal use
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #print(os.environ["CUDA_VISIBLE_DEVICES"])
        output_device = torch.device("cuda:0")
        n_devices = torch.cuda.device_count()
        print("Using {} GPUs".format(n_devices))
    elif use_gpu == False:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        output_device = torch.device("cpu")
        print("Using CPU")
    else:
        raise ValueError("Incorrect value of useGPU variable")

    if os.path.exists("temp_model.pkl") and os.path.getsize("temp_model.pkl") > 0:
        lam_da = rag_model(git_token=token_str, show_debug_logs=False, context_save_file='temp_model.pkl')
    else:
    # File not valid; build fresh
        lam_da = rag_model(git_token=token_str, show_debug_logs=False, context_save_file=None)

    while True:
        lam_da.reset_memory() # reseting memory everytime cuz memory is kinda jank right now
        query = input("Enter your query: ")
        print(f'Query: {query}', flush=True)
        if query.lower() == "exit":
            print(f'Answer: Good bye!', flush=True)
            break
        attachment = input("Enter directory to any attachments: ")
        print(f'Attachment: {attachment}', flush=True)

        lam_da.ask_query(query, attachment)

#%% List of useful test questions
query = 'What is the NACA0012 airfoil?' # General question about airfoil
query = 'Can you plot what the NACA0012 airfoil look like?' # Specific inquiry about airfoil geometric profile
query = 'Can you plot the SC1095 airfoil Cp distribution at angle of attack = 6.2 and Mach number = 0.6?' # Retrival of existing Cp distribution
query = 'Can you predict the SC1095 airfoil Cp distribution at angle of attack = 10.2 and Mach number = 0.6?' # Prediction of Cp distribution
# "Bad" questions
query = 'Can you predict the SC1115 airfoil Cp distribution at angle of attack = 10.2 and Mach number = 0.6?' # Bad user input wrt airfoil
query = 'Cp distribution over SC1095 at Angle of X and Mach Y?' # Bad user input wrt operatinfg conditions
#%%
lam_da.reset_memory()
query = 'Predict this airfoil Cp distribution at AoA = 5 degrees, Mach = 0.08 please?'
attachment = '/home/hlee981/LAM-LLM/WT 180_coordinates.csv'
print(f'Query: {query}', flush=True)
print(f'Attachment: {attachment}', flush=True)
lam_da.ask_query(query, attachment)