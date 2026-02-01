import re
import json 


def parse_output(input_text, pstyle=None):
    input_text = input_text.encode('utf-8').decode('unicode_escape')
    # Define the keys you want in your dictionary
    keys = ['Temporal Lobe', 'Frontal Lobe', 'Cingulate Gyrus', 'Parietal Lobe', 'Occipital Lobe', 'Insula', 'Hypothalamus']
    # Initialize an empty dictionary
    dict_obj = {}
    dict_roi = {}
    # Extract the likelihoods for each brain region
    pattern = r'(?:["\']?(' + '|'.join(re.escape(key) for key in keys) + r')["\']?): ["\']?(\d+%)["\']?(?: - ["\']?(\d+%)["\']?)?'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    for match in matches:
        dict_roi[match[0]] = match[1] if match[1] else '0%'
    dict_obj['predicted_likelihood'] = dict_roi
    return dict_obj

def parse_output_type2(input_text, pstyle=None):
    dict_obj = {}
    # Extract the likelihood  
    likelihood_match = re.search(r'Likelihood: \s*\{([^}]+)\}', input_text, re.DOTALL)
    if likelihood_match:
        likelihood_string = likelihood_match.group(1)
        likelihood_dict = dict(re.findall(r'"([^"]+)": (\d+)%', likelihood_string))
        dict_obj['predicted_likelihood'] = {k: f"{v}%" for k, v in likelihood_dict.items()}
    return dict_obj


def form_json(input_text, pstyle=None):    
    # Find the start and end of the dictionary in the string
    start = input_text.find("{")
    end = input_text.rfind("}") + 1  # +1 to include the closing brace
  
    # Extract the dictionary string
    dict_string = input_text[start:end]
    # Add double quotes around percentage values
    dict_string = re.sub(r': (\d+%)', r': "\1"', dict_string)  
    try:
        # Load the dictionary from the string
        dict_obj = json.loads(dict_string)

        # Rename keys and provide default values if keys are not found
        
        dict_obj['predicted_likelihood'] = dict_obj.pop('likelihood', dict_obj.pop('Likelihood', None))

        # If 'predicted_likelihood' is not found, set dict_obj to None
        if dict_obj['predicted_likelihood'] is None:
            dict_obj = None
    except json.JSONDecodeError as e:
        try:
            dict_obj = parse_output(input_text, pstyle)
            if not dict_obj['predicted_likelihood']:
                dict_obj = parse_output_type2(input_text, pstyle)
                
        except:
            print(f"Error decoding JSON: {e} \n check input text {input_text}")
            dict_obj = None
    return dict_obj

#similar code is implementaed for other languages
def parse_output_german(input_text, pstyle):
    # Define the keys you want in your dictionary
    keys = ['Temporallappen', 'Frontallappen', 'Gyrus Cinguli', 'Parietallappen', 'Occipitallappen', 'Insula', 'Hypothalamus']
    # Initialize an empty dictionary
    dict_obj = {}
    dict_roi = {}
    # Extract the likelihoods for each brain region
    for key in keys:
        match = re.search(f'(?:["\']?{key}["\']?): ["\']?(\d+%)["\']?(?: - ["\']?(\d+%)["\']?)?', input_text, re.IGNORECASE)
        if match:
            if match.group(2):
                dict_roi[key] = match.group(2)
            else:
                dict_roi[key] = match.group(1)
        else:
            dict_roi[key] = '0%'
    dict_obj['predicted_likelihood'] = dict_roi
    return dict_obj

def parse_output_french(input_text, pstyle):
    input_text = input_text.encode('utf-8').decode('unicode_escape')
    # Define the keys you want in your dictionary
    keys = ['Lobe Temporal', 'Lobe Frontal', 'Gyrus Cingulaire', 'Lobe Pariétal', 'Lobe Occipital', 'Insula', 'Hypothalamus']
    # Initialize an empty dictionary
    dict_obj = {}
    dict_roi = {}
    pattern = r'["\']?\b(' + '|'.join(re.escape(key) for key in keys) + r')\b["\']?\s*:\s*["\']?(\d+%)["\']?'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    for match in matches:
        dict_roi[match[0]] = match[1] if match[1] else '0%'
    dict_obj['predicted_likelihood'] = dict_roi
    return dict_obj

def parse_output_spanish(input_text, pstyle):
    input_text = input_text.encode('utf-8').decode('unicode_escape')
    # Define the keys you want in your dictionary
    keys = ['Lóbulo Temporal', 'Lóbulo Frontal', 'Circunvolución Cingulada', 'Lóbulo Parietal', 'Lóbulo Occipital', 'Ínsula', 'Hipotálamo']
    # Initialize an empty dictionary
    dict_obj = {}
    dict_roi = {}
    # Extract the likelihoods for each brain region
    pattern = r'(?:["\']?(' + '|'.join(re.escape(key) for key in keys) + r')["\']?): ["\']?(\d+%)["\']?(?: - ["\']?(\d+%)["\']?)?'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    for match in matches:
        dict_roi[match[0]] = match[1] if match[1] else '0%'
    dict_obj['predicted_likelihood'] = dict_roi
    return dict_obj

def parse_output_chinese(input_text, pstyle):
    input_text = input_text.encode('utf-8').decode('unicode_escape')
    # Define the keys you want in your dictionary
    keys = ['颞叶', '额叶', '扣带回', '顶叶', '枕叶', '脑岛', '下丘脑']
    # Initialize an empty dictionary
    dict_obj = {}
    dict_roi = {}
    pattern = r'(?:["\']?(' + '|'.join(re.escape(key) for key in keys) + r')["\']?): ["\']?(\d+%)["\']?(?: - ["\']?(\d+%)["\']?)?'
    matches = re.findall(pattern, input_text, re.IGNORECASE)
    for match in matches:
        dict_roi[match[0]] = match[1] if match[1] else '0%'
    dict_obj['predicted_likelihood'] = dict_roi
    return dict_obj