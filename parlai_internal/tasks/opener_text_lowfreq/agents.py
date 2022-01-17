#!/usr/bin/env python3

from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager

import json
import os


def _path(opt):
    #build(opt) commented because local
    dt = opt['datatype'].split(':')[0]
    
    if(dt == "train"):
        data_path = os.path.join(opt['datapath'], 'opener_text' + dt + '_dataset_lowfreq.json')
    else:
        data_path = os.path.join(opt['datapath'], 'opener_text' + dt + '_dataset.json')
    return data_path


class TextOpenerTeacher(DialogTeacher):
    """
    This teacher inherits from the core Dialog Teacher, which just requires it
    to define an iterator over its data `setup_data` in order to inherit basic metrics,
    a `act` function, and enables Hogwild training with shared memory with no extra
    work.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype'].split(':')[0]
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'opener_text_lowfreq'
	
        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            raw_data = json.load(data_file)

        episode_done = True
        
        if("train" in path):
            for i in raw_data["Answer.question1"].keys():
                if(raw_data["Answer.question1"][i] != ""):
                    yield (raw_data["dot_string"][i], raw_data["Answer.question1"][i], None, None, None), episode_done
                if(raw_data["Answer.question2"][i] != ""):
                    yield (raw_data["dot_string"][i], raw_data["Answer.question2"][i], None, None, None), episode_done
                if(raw_data["Answer.question3"][i] != ""):
                    yield (raw_data["dot_string"][i], raw_data["Answer.question3"][i], None, None, None), episode_done
        else:
            for i in raw_data["Answer.question1"].keys():
                yield (raw_data["dot_string"][i], [raw_data["Answer.question1"][i], raw_data["Answer.question2"][i], raw_data["Answer.question3"][i]], None, None, None), episode_done

class DefaultTeacher(TextOpenerTeacher):
    pass
