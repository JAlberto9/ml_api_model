from collections import defaultdict
import logging

vectors_names = [
    'total_earning',
    'to_user_distance',
    'to_user_elevation'
]


def transform(data):
    features = []

    feature_map = defaultdict(int)

    feature_map['total_earning'] = data['total_earning']
    feature_map['to_user_distance'] = data['to_user_distance']
    feature_map['to_user_elevation'] = data['to_user_elevation']

    for vector_name in vectors_names:
        try:
            feature_map[vector_name] = vector_name.lower()
        except:
            pass

    for feature in vectors_names:
        print("Feature %s value %s", feature, feature_map[feature])
        logging.debug("Feature %s value %s", feature, feature_map[feature])
        features.append(feature_map[feature])

    return features
