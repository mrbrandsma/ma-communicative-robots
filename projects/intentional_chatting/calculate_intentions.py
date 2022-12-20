import json

# Map 23 speech acts to Searle's 5 speech acts + 'other'
_MAPPING={'open_question_factual': ['assertive'],
        'pos_answer': ['expressive', 'assertive'],
        'command': ['directive'],
        'opinion': ['expressive'],
        'statement': ['assertive', 'expressive'],
        'back-channeling': ['other'],
        'yes_no_question': ['assertive', 'expressive'],
        'appreciation': ['expressive'],
        'other_answers': ['assertive', 'expressive'],
        'thanking': ['expressive'],
        'open_question_opinion': ['expressive'],
        'hold': ['other'],
        'closing': ['declarative'],
        'comment': ['assertive', 'expressive'],
        'neg_answer': ['assertive', 'expressive'],
        'complaint': ['expressive'],
        'abandon': ['other'],
        'dev_command': ['directive'],
        'apology': ['expressive'],
        'nonsense': ['other'],
        'other': ['other'],
        'opening': ['declarative'],
        'respond_to_apology': ['expressive']}

def json_loader(path):
    """
    Gets the needed .json file.
    """
    with open(path) as json_file:
        data = json.load(json_file)
    
    return(data)


def get_act_scores(conversations, key):
    """
    Store the amount of every speech act
    """
    speech_act_labels={'open_question_factual': 0,
        'pos_answer': 0,
        'command': 0,
        'opinion': 0,
        'statement': 0,
        'back-channeling': 0,
        'yes_no_question': 0,
        'appreciation': 0,
        'other_answers': 0,
        'thanking': 0,
        'open_question_opinion': 0,
        'hold': 0,
        'closing': 0,
        'comment': 0,
        'neg_answer': 0,
        'complaint': 0,
        'abandon': 0,
        'dev_command': 0,
        'apology': 0,
        'nonsense': 0,
        'other': 0,
        'opening': 0,
        'respond_to_apology': 0}

    for turn in conversations[key]:
        speech_act_labels[turn['speech-act']] += 1
    
    return(speech_act_labels)


def get_prevalent_intention(total, intention_scores):
    """
    Calculate the speech act(s) that occur the most.
    """
    calculated_half = []
    calculated_third = []
    calculated_small = []

    found_intention = False
    for speech_act in intention_scores:
        if intention_scores[speech_act] >= 0.5*total:
            calculated_half.append(speech_act)
            found_intention = True
        elif intention_scores[speech_act] >= 0.3*total:
            calculated_third.append(speech_act)
            found_intention = True
        elif intention_scores[speech_act] >= 0.1*total:
            calculated_small.append(speech_act)
    
    if found_intention == False:
        calculated_intention = calculated_small
    elif len(calculated_half) == 0:
        calculated_intention = calculated_third
    else:
        calculated_intention = calculated_half
    
    return(calculated_intention)

def get_speech_act_analysis(conv_act_scores, mapping):
    """
    Finds most common speech act according to mapping
    """
    searle_speech_acts = {}
    for label in conv_act_scores:
        # Find mapped label
        for searle_act in mapping[label]:
            if searle_act in searle_speech_acts:
                searle_speech_acts[searle_act] = searle_speech_acts[searle_act] + conv_act_scores[label] 
            else:
                searle_speech_acts[searle_act] = conv_act_scores[label]
    
    # Calculate total found intentions
    total = 0
    for speech_act in searle_speech_acts:
        total = total + searle_speech_acts[speech_act]
    
    calculated_intention = get_prevalent_intention(total, searle_speech_acts)

    return(searle_speech_acts, calculated_intention) 


def get_dataset_intention(dataset_info):
    """
    Get the intentions that are most prevalent in the dataset
    """
    intention_scores = {'assertive': 0,
                        'commissive': 0,
                        'directive': 0,
                        'declarative': 0,
                        'expressive': 0,
                        'other': 0}

    for conversation in dataset_info:
        for intention in intention_scores:
            if intention in conversation['conv_intention']:
                intention_scores[intention] += 1
    
    total = len(dataset_info)

    calculated_intention = get_prevalent_intention(total, intention_scores)

    return(intention_scores, calculated_intention)


############################RUN#####################

# Get the conversations from the data
conversations_data = json_loader('processed_data\conv_ai_2.json') # Change database name here
data_name = "conv_ai_2" # Change database name here

# Loop through everything to find the turn
conversation_list = []
for conversation in conversations_data:
    i = list(conversation.keys())[0]
    current_act_score = get_act_scores(conversation, str(i))
    get_speech_act_analysis(current_act_score, _MAPPING)
    current_searle_acts, current_intentions = get_speech_act_analysis(current_act_score, _MAPPING)

    # Store information
    current_info = {}
    current_info['distribution_classifier'] = current_act_score
    current_info['distribution_searle'] = current_searle_acts
    current_info['conv_intention'] = current_intentions
    conversation_list.append(current_info)

# Write conversation info to .txt
file_name = 'intentions_analysis\\' + data_name + '_conversations.json'
with open(file_name, "w") as writer:
    json.dump(conversation_list, writer, indent=2)

# Calculate dataset distributions and write to file
distribution_intentions, data_intention = get_dataset_intention(conversation_list)
file_name = 'intentions_analysis\\' + data_name + '_dataset.txt'
with open(file_name, "w") as writer:
    writer.write("DISTRIBUTION SEARLE SPEECH ACTS:\n")
    for speech_act in distribution_intentions:
        writer.write(speech_act + ': ' + str(distribution_intentions[speech_act]) + '\n')
    writer.write("\n")
    writer.write("This dataset is characterized by the intention(s): " + str(data_intention))