import json
import logging
import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.entity_linking.label_linker import LabelBasedLinker
from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.spacy_analyzer import spacyAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules

from create_context_capsule import create_context_capsule
from loaders import DailyDialogueLoader, ConvAI2Loader, CommonsenseLoader  # , EmoryLoader


def collect_speakers(conversation):
    speakers = set()
    for speaker in conversation:
        speakers.add(speaker["Speaker"])

    return [s for s in speakers]


def expand_author(capsule):
    if type(capsule['author']) == str:
        capsule['author'] = {
            'label': capsule['author'],
            'type': ['person'],
            'uri': []
        }

    capsule['subject']['uri'] = []
    capsule['predicate']['uri'] = []
    capsule['object']['uri'] = []
    capsule['timestamp'] = datetime.now()

    return capsule


def dataset_to_rdf(data, name, out_dir='./rdf_files/'):
    print(f'----------DATASET:{name}---------')

    capsules_skipped = 0
    for idx, row in enumerate(data):
        print(f"Processing conversation {idx}/{len(data) - 1}")

        key = list(row.keys())[0]  # obtain the ID for the dialog
        conversation = row[key]
        speakers = collect_speakers(conversation)  # collect the list of speakers
        if len(speakers) != 2:
            continue
        assert len(speakers) == 2, f'{len(speakers)} speakers found. Can only parse 2 speakers.'

        file_loc = Path(out_dir) / name / str(key) / 'rdf'
        turn_to_trig_file = Path(out_dir) / name / str(key) / 'turn_to_trig_file.json'
        if not os.path.exists(file_loc):
            os.makedirs(Path(file_loc))  # create directory to store the rdf files if need be

        # Initialize brain, Chat, and context capsule
        brain = LongTermMemory(address="http://localhost:7200/repositories/test",
                               log_dir=file_loc,
                               clear_all=False)
        chat = Chat(speakers[0], speakers[-1])
        context_capsule = create_context_capsule(key)
        brain.capsule_context(context_capsule)

        capsules = []
        for i, turn in enumerate(conversation):
            print(f"\tProcessing turn {i}/{len(conversation) - 1}")

            # switch around speakers every turn
            speakers = [speakers[-1], speakers[0]]
            utterance = turn['Response']

            # add utterance to chat and use spacy analyzer to analyze
            print("\t\tAnalyzing utterance")
            chat.add_utterance(utterance)
            subj, obj = speakers
            analyzer.analyze(chat.last_utterance, subj, obj)
            capsules += utterance_to_capsules(chat.last_utterance)

            # add statement capsules to brain
            for capsule in capsules:
                capsule = expand_author(capsule)
                linker.link(capsule)

                try:
                    # Add capsule to brain
                    print("\t\t\tAdding capsule to brain")
                    response = brain.capsule_statement(capsule)
                    row[key][i]['rdf_file'].append(str(response['rdf_log_path'].stem) + '.trig')
                except:
                    capsules_skipped += 1
                    print(f"\t\t\tCapsule skipped. Total skipped: {capsules_skipped}")

        # Save conversation turn to trig
        with open(turn_to_trig_file, 'w') as f:
            js = json.dumps(row[key])
            f.write(js)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-D', '--dataset',
                        dest='data',
                        help='Which dataset to use. [daily_dialogue, conv_ai_2, commonsense, emory_nlp]',
                        default='conv_ai_2', type=str)

    args = parser.parse_args()

    chat_logger.setLevel(logging.ERROR)
    brain_logger.setLevel(logging.ERROR)

    analyzer = spacyAnalyzer()
    linker = LabelBasedLinker()

    if args.data == 'daily_dialogue':
        dataset = DailyDialogueLoader()
    elif args.data == 'conv_ai_2':
        dataset = ConvAI2Loader()
    elif args.data == 'commonsense':
        dataset = CommonsenseLoader()
    elif args.data == 'emory_nlp':
        print('EmoryNLP not properly implemented yet, loading Dailydialog instead.')
        dataset = DailyDialogueLoader()

    data = dataset.data
    name = dataset.name
    dataset.save_to_file(f'{name}')
    dataset_to_rdf(data, name)
