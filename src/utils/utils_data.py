#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''	
    Written by H.Turbé, October 2020.
'''
import os
import sys
import time
import yaml
import json
import xml.etree.ElementTree as ET
import pathlib

def extract_pathData(DATA_PATH):
    dict_path_data = []
    for dir_name in os.listdir(DATA_PATH):
        isFile = os.path.isdir(os.path.join(DATA_PATH,dir_name))
        if dir_name != 'tmp' and isFile:
            dir_tmp = os.listdir(os.path.join(DATA_PATH,dir_name))
            for elem in dir_tmp:
                path_tmp =os.path.join(DATA_PATH,dir_name,elem)
                if os.path.isdir(path_tmp):
                    dict_path_data.append(path_tmp)
    
    return dict_path_data

def parse_yaml(path_file):
    with open(path_file) as f:
        my_file = yaml.safe_load(f)

    return my_file

def parse_json(path_file):
    with open(path_file) as config_file:
        data = json.load(config_file)
    
    return data

def parse_xml(path_file):
    """
    Read the xml file and returns dictionary tree with all parameters.
    Note: maximum depth-level is 2 (grandchildren). Key bulding block
            is the keyword param.
    """
    tree = ET.parse(path_file)
    root = tree.getroot()

    # Loop over <tag> children and get dictionary of parameters
    xml__ = dict()
    for child in root:
        cnt = 0
        xml__[child.tag] = dict()
        for param in child.findall('param'):
            k = param.get('key')
            v = param.get('value')
            t = param.get('type')
            xml__[child.tag][k] = eval(t)(v)
            cnt = cnt + 1

        # if no tag children are found, explore next level (grandchildren)
        if cnt == 0:
            for grandchild in child:
                xml__[child.tag][grandchild.tag] = dict()
                for param in grandchild.findall('param'):
                    k = param.get('key')
                    v = param.get('value')
                    t = param.get('type')
                    xml__[child.tag][grandchild.tag][k] = eval(t)(v)
    return xml__
def parse_config(config_path):
    """Function to choose corect parser for config file

    Args:
        config_path ([str]): [path to the config file to be parsed]

    Raises:
        ValueError: [Unsupported format of the config file]

    Returns:
        [dict]: [configuration dictionary]
    """
    extension = pathlib.Path(config_path).suffix.lstrip('.')
    if extension == 'yaml':
        config = parse_yaml(config_path)
    elif extension == 'xml':
        config = parse_xml(config_path)
    elif extension == 'json':
        config = parse_json(config_path)
    else:
        raise ValueError('Config format not supported')
    
    return config
def resize_len(np_signal,new_len, cropping='end'):
        
        if cropping.lower() == 'end':
            np_new = np_signal[0:new_len,:]

        elif cropping.lower() == 'start':
            tmp_length = np_signal.shape[0]
            difference = tmp_length - new_len
            np_new = np_signal[difference:,:]

        elif cropping.lower() == 'both':
            tmp_length = np_signal.shape[0]
            difference = tmp_length - new_len
            splits = difference // 2
            res = difference % 2
            lb = splits
            ub = tmp_length - splits
            if res != 0:
                lb = splits+1
            np_new = np_signal[lb:ub,:]
         
        return np_new